import threading
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import time
import yaml
from argparse import Namespace, ArgumentParser
from datetime import datetime
import logging

# Imports from your existing codebase
from src.llm import LLM, DoubleSimRAG as RAG
from src.env import PandaEnv
from main_utils import try_identify_and_execute, cprint
from src.shared import state_manager
import _thread

# --- CONFIGURATION (Adapted from runner) ---
USER_STUDY_DIRNAME = "./user_study_data"
STUDY_DURATION_SECONDS = 30 * 60  # Default 30 mins

app = FastAPI()

# Setup templates (assumes a folder named 'templates')
templates = Jinja2Templates(directory="templates")

# --- DATA MODELS ---
class FeedbackRequest(BaseModel):
    feedback: str

class StateResponse(BaseModel):
    logs: list
    current_subtask: str
    reasoning: str
    model_output: str
    is_waiting_for_feedback: bool
    current_task: str
    server_start_time: float

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/state", response_model=StateResponse)
async def get_state():
    return {
        "logs": state_manager.logs,
        "current_subtask": state_manager.current_subtask,
        "model_output": state_manager.model_output,
        "reasoning": state_manager.reasoning,
        "is_waiting_for_feedback": state_manager.is_waiting_for_feedback,
        "current_task": state_manager.current_task,
        "server_start_time": state_manager.start_time,
    }


@app.post("/api/interrupt")
async def trigger_interrupt():
    state_manager.add_log("Interrupting Robot Thread...", "magenta")
    # This raises KeyboardInterrupt in the main thread 
    # where your robot logic is running.
    _thread.interrupt_main() 
    return {"status": "ok"}


@app.post("/api/feedback")
async def receive_feedback(req: FeedbackRequest):
    state_manager.feedback_text = req.feedback
    state_manager.feedback_event.set()
    state_manager.add_log(f"Feedback received: {req.feedback}", "cyan")
    return {"status": "ok"}

# --- ROBOT LOGIC (Adapted from user_study_runner.py) ---

# def diff_and_destroy(rag_primary, rag_secondary, rag_diff_out, delete_primary=True):
#     secondary_set = set((m['key'], m['value']) for m in rag_secondary.metadata)
#     count = 0
#     for meta in rag_primary.metadata:
#         pair = (meta['key'], meta['value'])
#         if pair not in secondary_set:
#             rag_diff_out.add(meta['key'], meta['value'])
#             count += 1
#     print(f"Added {count} unique entries to the difference RAG.")
#     target_to_kill = rag_primary if delete_primary else rag_secondary
#     target_to_kill.save_to_file = lambda *args, **kwargs: None 
#     del target_to_kill

def read_next_user_info():
    config_filename = os.path.join(USER_STUDY_DIRNAME, "next", "config.yml")
    with open(config_filename, "r") as fh:
        data = yaml.safe_load(fh)
    next_task = data["task"]
    tasks_filename = "config/tasks/tasks.yml"
    with open(tasks_filename, "r") as fh:
        tasks = yaml.safe_load(fh)
        tasks = tasks["tasks"]
    idx = tasks.index(next_task)
    new_tasks = tasks[idx:] + tasks[:idx]
    return new_tasks, int(data["user_id"])

def write_next_user_info(next_task: str, next_id: int):
    config_filename = os.path.join(USER_STUDY_DIRNAME, "next", "config.yml")
    data = {"task": next_task, "user_id": next_id}
    with open(config_filename, "w") as fh:
        yaml.dump(data, fh)

def run_task(task, scene_file, video_file, llm, global_lorebook, user_lorebook, start_time):
    env = PandaEnv(scene_config=scene_file)
    env.set_recorder(video_file)
    
    state_manager.current_task = task
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""
    
    while subtask != "DONE()" and start_time + STUDY_DURATION_SECONDS > time.time():
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"

        messages = []
        messages.append({"role": "system", "content": llm.generate_system_prompt()})

        # 1. Capture the number of items BEFORE execution
        prev_lore_count = len(global_lorebook.metadata)

        # 2. Execute (modifies global_lorebook in-place)
        subtask_done, subtask, code, code_output, messages, _ = (
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
            
            # 3. Identify what was added and sync to user_lorebook
            new_lore_count = len(global_lorebook.metadata)
            if new_lore_count > prev_lore_count:
                cprint(f"Syncing {new_lore_count - prev_lore_count} new items to user lorebook...", "cyan")
                for i in range(prev_lore_count, new_lore_count):
                    new_item = global_lorebook.metadata[i]
                    # Add to local user history
                    user_lorebook.add(new_item['key'], new_item['value'])
            
            # Note: No need to 'destroy' anything. global_lorebook persists naturally.
            
    env.p.disconnect()

    if start_time + STUDY_DURATION_SECONDS < time.time():
        raise TimeoutError

def runner_thread(args):
    """The main loop running in a background thread."""
    time.sleep(2) # Give server time to start
    cprint("Starting Study Session...", "green")
    
    api_key = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
    llm = LLM(api_key, args.api_url, args.prompt_filename, args.model)
    
    new_tasks, id = read_next_user_info()
    user_folder = os.path.join(USER_STUDY_DIRNAME, f"user_{id}")
    os.makedirs(user_folder, exist_ok=True)
    
    global_lorebook = RAG(filename=args.rag_filename)
    user_lorebook = RAG(filename=os.path.join(user_folder, "lorebook.pkl"))
    
    session_start_time = time.time()
    session_data = {"user_id": id, "n_feedbacks": 0, "tasks": []}
    idx = 0
    
    state_manager.start_time = time.time()
    
    try:
        for i, task in enumerate(new_tasks):
            task_urlfriendly = task.strip().replace(" ", "_")
            try:
                scene_file = os.path.join("user_study", f"{task_urlfriendly}.yml")
                video_file = os.path.join(user_folder, f"{task_urlfriendly}.mp4")
                run_task(task, scene_file, video_file, llm, global_lorebook, user_lorebook, session_start_time)
            except TimeoutError:
                cprint("Study duration exceeded.", "red")
                state_manager.add_log("[SYSTEM]: Study duration reached, exiting.", "white")
                idx = i
                break
            finally:
                idx = i
    except Exception as e:
        cprint(f"FATAL ERROR IN RUNNER: {e}", "red")
        logging.exception(e)
    finally:
        session_data["n_feedbacks"] = len(user_lorebook.metadata)
        session_data["tasks"] = new_tasks[:idx]
        info_file = os.path.join(user_folder, "info.yml")
        with open(info_file, "w") as fh:
            yaml.dump(session_data, fh)
        write_next_user_info(new_tasks[idx], id+1)
        cprint("Study Session Complete. You may close the server.", "green")
        
def start_fastapi():
    # Run uvicorn in this worker thread
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--api-url", default=None) #default="https://llm-api.arc.vt.edu/api/v1")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--prompt-filename", default="config/prompts/llm_unified_function_store.yml")
    parser.add_argument("--rag-filename", default=os.path.join(USER_STUDY_DIRNAME, "global_lorebook.pkl"))
    parser.add_argument("--study-duration", default=30, type=int, help="minutes")
    
    args = parser.parse_args()
    
    STUDY_DURATION_SECONDS = 60 * args.study_duration

    # 1. Start FastAPI in a background thread
    threading.Thread(target=start_fastapi, daemon=True).start()

    # 2. Run the Robot Runner in the MAIN THREAD
    # This allows it to receive the interrupt_main() signal
    try:
        runner_thread(args) 
    except KeyboardInterrupt:
        # This handles the case where the user hits Ctrl+C in terminal
        print("Manual exit detected.")
