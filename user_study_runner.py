from argparse import ArgumentParser, Namespace

from src.env import PandaEnv
from src.llm import LLM, DoubleSimRAG as RAG
import time
from datetime import datetime
import os
import logging
import yaml

from main import cprint, try_identify_and_execute
import atexit

USER_STUDY_DIRNAME = "./user_study_data"
STUDY_DURATION_SECONDS = 0.0


def diff_and_destroy(rag_primary, rag_secondary, rag_diff_out, delete_primary=True):
    """
    Identifies entries in rag_primary that are NOT in rag_secondary.
    Saves those entries to rag_diff_out.
    Deletes the chosen instance without triggering a file save.
    """
    # 1. Identify uniqueness based on (key, value) pairs
    secondary_set = set((m['key'], m['value']) for m in rag_secondary.metadata)
    
    # 2. Add unique items from primary to the third instance
    count = 0
    for meta in rag_primary.metadata:
        pair = (meta['key'], meta['value'])
        if pair not in secondary_set:
            rag_diff_out.add(meta['key'], meta['value'])
            count += 1
    
    print(f"Added {count} unique entries to the difference RAG.")

    target_to_kill = rag_primary if delete_primary else rag_secondary
    
    # The base class registers a closure 'cleanup_function'
    # We unregister by searching the atexit registry if possible, 
    target_to_kill.save_to_file = lambda x: None # Save bypassed for deleted instance.
    
    del target_to_kill
    print("Instance deleted successfully.")

def read_next_user_info() -> tuple[list[str], int]:
    config_filename = os.path.join(USER_STUDY_DIRNAME, "next", "config.yml")
    with open(config_filename, "r") as fh:
        data = yaml.load(fh)
    next_task = data["task"]
    tasks_filename = "config/tasks/tasks.yml"
    with open(tasks_filename, "r") as fh:
        tasks = yaml.load(tasks_filename)
        tasks = tasks["tasks"]  # list of str
    # need to reorder tasks so that next_task is first and it wraps around
    idx = tasks.index(next_task)
    new_tasks = tasks[idx:] + tasks[:idx]
    return new_tasks, int(data["user_id"])


def write_next_user_info(next_task: str, next_id: int):
    config_filename = os.path.join(USER_STUDY_DIRNAME, "next", "config.yml")
    data = {"task": next_task, "user_id": next_id}
    with open(config_filename, "w") as fh:
        yaml.dump(data, fh)
    return config_filename


def run_task(
    task: str,
    scene_file: str,
    video_file: str,
    llm: LLM,
    global_lorebook: RAG,
    user_lorebook: RAG,
    start_time: float,
):
    env = PandaEnv(scene_config=scene_file)
    env.set_recorder(video_file)
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""

    while subtask != "DONE()" and start_time + STUDY_DURATION_SECONDS > time.time():
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"

        messages = []
        messages.append({"role": "system", "content": llm.generate_system_prompt()})

        subtask_done, subtask, code, code_output, messages, new_global_lorebook = (
            try_identify_and_execute(
                env,
                llm,
                messages,
                global_lorebook,
                task=task,
                open_or_closed=open_or_closed,
                python_code_called_history=code_history,
                python_code_output_history=code_output_history,
                subtasks=subtasks,
            )
        )
        
        if subtask_done:
            subtasks.append(subtask)
            code_history += "\n" + code
            code_output_history += "\n" + code_output
            # identify diff in global lorebook and save it to local
            diff_and_destroy(global_lorebook, new_global_lorebook, user_lorebook)
            global_lorebook = new_global_lorebook

    if start_time + STUDY_DURATION_SECONDS < time.time():
        raise TimeoutError
    return


def main(args: Namespace):
    api_key = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
    llm = LLM(api_key, args.api_url, args.prompt_filename, args.model)

    new_tasks, id = read_next_user_info()
    user_folder = os.path.join(USER_STUDY_DIRNAME, f"user_{id}")
    scene_folder = os.path.join("config", "scene", "user_study")
    os.makedirs(user_folder, exist_ok=True)
    global_lorebook = RAG(filename=args.rag_filename)
    user_lorebook = RAG(filename=os.path.join(user_folder, "lorebook.pkl"))
    session_start_time = time.time()
    session_data = {"user_id": id, "n_feedbacks": 0, "tasks": []}
    idx = 0
    for i, task in enumerate(new_tasks):
        task_urlfriendly = task.strip().replace(" ", "_")
        try:
            scene_file = os.path.join(scene_folder, f"{task_urlfriendly}.yml")
            video_file = os.path.join(user_folder, f"{task_urlfriendly}.mp4")
            run_task(task, scene_file, video_file, llm, global_lorebook, user_lorebook, session_start_time)
        except TimeoutError:
            idx = i
            break
        except Exception as e:
            cprint(f"An error occured {e}", "red")
            logging.exception(f"Fatal Error: {e}")
        finally:
            idx = i
    # save user data
    session_data["n_feedbacks"] = len(user_lorebook.metadata)
    session_data["tasks"] = new_tasks[:idx]
    info_file = os.path.join(user_folder, "info.yml")
    with open(info_file, "w") as fh:
        yaml.dump(session_data, fh)
    # update next user data
    write_next_user_info(new_tasks[idx], id+1)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(f"logs/{log_filename}.log")],
    )
    API_URL = "https://llm-api.arc.vt.edu/api/v1"
    MODEL = "gemini-3-flash-preview"  # "gpt"  # "gemini-3-flash-preview"
    GEN_CONF = "config/prompts/llm_unified_function_store.yml"
    VIDEO_PATH = f"videos/{log_filename}.mp4"

    parser = ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--video-path", default=VIDEO_PATH)
    parser.add_argument("--prompt-filename", default=GEN_CONF)
    parser.add_argument(
        "--rag-filename",
        default=os.path.join(USER_STUDY_DIRNAME, "global_lorebook.pkl"),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--study-duration", default=30, help="minutes")
    # parser.add_argument("--api-key", default=API_KEY)

    args = parser.parse_args()
    STUDY_DURATION_SECONDS = args.study_duration * 60
    main(args)
