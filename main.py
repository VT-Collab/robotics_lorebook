from src.env import PandaEnv
from src.llm import LLM, RAG
from src.objects import objects
from termcolor import cprint as termcolor_cprint
import time
from datetime import datetime
import os
import json
import logging

os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)
log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(f"logs/{log_filename}.log")],
)


def cprint(text, color="white", **kwargs):
    clean_text = str(text).strip()
    logging.info(clean_text)
    termcolor_cprint(text, color=color, **kwargs)


API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gpt" #"gemini-3-flash-preview" #"gpt" #"qwen3:32b"
GEN_CONF = "config/prompts/llm_decomp_gem.yml"
# GEN_CONF = "config/prompts/llm_decomposer.yml"
# DICT_CONF = "config/prompts/llm_dict_gem.yml"
DICT_CONF = "config/prompts/llm_dictionary.yml"
TASK = "put the block in the cabinet. the cabinet door is closed at the beginning. the cabinet door opens prismatically TOWARDS the robot along the negative x direction"
VIDEO_PATH = f"videos/{log_filename}.mp4"

QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()


def generate_objects_table(env: PandaEnv) -> str:
    """
    Generates a markdown table of objects in the scene.
    Dimensions are absolute axis-aligned extents (X, Y, Z).
    """
    info = []
    context_header = (
        "### Scene Object Manifest\n"
        "Values are in meters. Coordinates are in the World Frame.\n"
        "- **Position**: [x, y, z] center of the bounding box.\n"
        "- **Orientation**: [roll, pitch, yaw] in radians.\n"
        "- **Extents**: Absolute size along the [X (Width), Y (Depth), Z (Height)] axes.\n\n"
    )
    
    for obj_entry in env.objects:
        body_id = obj_entry["id"]
        t = obj_entry["type"]
        if t == "plane":
            continue
        _, quat = env.p.getBasePositionAndOrientation(body_id)
        euler = [round(x, 2) for x in env.p.getEulerFromQuaternion(quat)]
        aabb_min, aabb_max = env.p.getAABB(body_id)
        x_size = round(aabb_max[0] - aabb_min[0], 3)
        y_size = round(aabb_max[1] - aabb_min[1], 3)
        z_size = round(aabb_max[2] - aabb_min[2], 3)
        dims = [x_size, y_size, z_size]
        center_pos = [round((aabb_min[i] + aabb_max[i]) / 2, 2) for i in range(3)]
        info.append({
            "type": t,
            "pos": center_pos,
            "orn": euler,
            "dims": dims,
        })
        if isinstance(obj_entry["ref"], objects.CollabObject):
            state = obj_entry["ref"].get_state()
            h_pos = [round(x, 2) for x in state["handle_position"]]
            h_orn = [round(x, 2) for x in state["handle_euler"]]
            h_min, h_max = env.p.getAABB(body_id, linkIndex=1)
            h_dims = [round(h_max[i] - h_min[i], 3) for i in range(3)]
            info.append({
                "type": t + " handle",
                "pos": h_pos,
                "orn": h_orn,
                "dims": h_dims,
            })
    table = "| Object | Position (X,Y,Z) | Orientation (R,P,Y) | Size [X, Y, Z] |\n"
    table += "| :--- | :--- | :--- | :--- |\n"
    for obj in info:
        table += f'| {obj["type"]} | {obj["pos"]} | {obj["orn"]} | {obj["dims"]} |\n'
    return context_header + table

def generate_rag_key(env: PandaEnv, subtask: str) -> str:
    key = f"{subtask}"
    for obj_entry in env.objects:
        t = obj_entry["type"]
        if t == "plane":
            continue
        key += f" {t}"
    return key


def get_model_output(model: LLM, messages: list[dict], verbose=True):
    global TIME_SINCE_LAST_QUERY
    global QUERY_TIMEOUT
    while time.time() < TIME_SINCE_LAST_QUERY + QUERY_TIMEOUT:
        pass
    reasoning, subtask = model.query(messages)
    if verbose:
        for m in messages:
            content = m["content"]
            content = "\t" + content.replace("\n", "\n\t")
            cprint(f"[{m['role']}]: {content}", "yellow")
        cprint(reasoning, "green")
        cprint(subtask, "red")
    TIME_SINCE_LAST_QUERY = time.time()
    return reasoning, subtask


def try_identify_and_execute(
    env: PandaEnv,
    gen: LLM,
    disc: LLM,
    messages: list[dict],
    lorebook: RAG,
    verbose=True,
    **prompt_kwargs,
) -> tuple[bool, str, str, str, list[dict], RAG]:
    python_code_called_history = prompt_kwargs.get("python_code_called_history", "")
    python_code_output_history = prompt_kwargs.get("python_code_output_history", "")
    task = prompt_kwargs.get("task", "")
    subtasks_list = prompt_kwargs.get("subtasks", [])
    open_or_closed = prompt_kwargs.get("open_or_closed", "open")
    next_thing_to_do = "identify the next subtask to execute"

    og_messages_len = len(messages)
    subtask = ""
    subtask_done = False
    code = ""
    code_output = ""

    og_code_len = len(python_code_called_history)
    og_output_len = len(python_code_output_history)

    while True:
        ckpt = env.get_checkpoint()
        try:
            position, orn = env.get_print_state()
            objs_table = generate_objects_table(env)

            message = gen.generate_followup_prompt(
                python_code_called_history=python_code_called_history,
                python_code_output_history=python_code_output_history,
                task=task,
                subtasks_list=subtasks_list,
                objects_table=objs_table,
                position=position,
                angle=orn[2],
                open_or_closed=open_or_closed,
                next_thing_to_do=next_thing_to_do,
            )

            messages.append({"role": "user", "content": message})
            _, subtask = get_model_output(gen, messages, verbose=verbose)
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return subtask_done, subtask, code, code_output, messages, lorebook
            messages.append({"role": "assistant", "content": subtask})

            rag_query_key = generate_rag_key(env, subtask)
            retrieved_lore = lorebook.query(rag_query_key, top_k=10)
            cprint(f"n lore: {len(retrieved_lore)}")
            for lore in retrieved_lore:
                cprint(lore, "white")

            current_task_instruction = f"output code to accomplish {subtask}"
            if retrieved_lore:
                feedback_str = "\n".join(
                    [f"- {item['value']}" for item in retrieved_lore]
                )
                current_task_instruction += (
                    f". **Important:** Use the following past experience as feedback:\n{feedback_str}"
                )

            code_prompt = gen.generate_followup_prompt(
                next_thing_to_do=current_task_instruction,
                python_code_called_history=python_code_called_history,
                python_code_output_history=python_code_output_history,
                task=task,
                subtasks_list=subtasks_list,
                objects_table=objs_table,
                position=position,
                angle=orn[2],
                open_or_closed=open_or_closed,
            )
            messages.append({"role": "user", "content": code_prompt})
            _, code = get_model_output(gen, messages, verbose=verbose)
            messages.append({"role": "assistant", "content": code})
            code_output = env.run_code(code)
            python_code_called_history += code + "\n"
            python_code_output_history += code_output + "\n"

            print("=" * 50)
            print("Waiting 5s for feedback interruption", end="", flush=True)
            for i in range(5):
                print(".", end="", flush=True)
                time.sleep(1)
            print("\n" + "=" * 50)
            env.p.removeState(ckpt)
            subtask_done = True
            break

        except KeyboardInterrupt:
            print("\n" + "=" * 50)
            print("Ctrl+C detected. Resetting subtask...")
            feedback = input("Provide your feedback here: ")
            print("=" * 50)

            disc_messages = disc.generate_initial_message()
            disc_input = f"Feedback for the current subtask {subtask} is: {feedback}."
            disc_messages.append({"role": "user", "content": disc_input})
            _, response_content = disc.query(disc_messages)

            try:
                new_lore = json.loads(response_content)
                cprint(f"[disc]: adding to vector database: {new_lore}", "red")
                for k, v in new_lore.items():
                    rag_query_key = generate_rag_key(env, k)
                    lorebook.add(rag_query_key, v)
            except (AttributeError, json.JSONDecodeError):
                cprint("Failed to parse discriminator feedback.", "red")

            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            del messages[og_messages_len:]
            python_code_called_history = python_code_called_history[:og_code_len]
            python_code_output_history = python_code_output_history[:og_output_len]
            print("Environment reset. Retrying with vector-retrieved feedback...")

    return subtask_done, subtask, code, code_output, messages, lorebook


def main():
    env = PandaEnv()
    if VIDEO_PATH:
        env.set_recorder(VIDEO_PATH)
    lorebook = RAG(filename="data/lorebook.pkl")
    gen = LLM(API_KEY, API_URL, GEN_CONF, MODEL)
    disc = LLM(API_KEY, API_URL, DICT_CONF, MODEL)

    print("=" * 50)
    print(
        "To provide feedback, hit Ctrl+C. This will interrupt execution and revert the environment."
    )
    input("Press any button to proceed: ")
    print("=" * 50)

    messages = []
    messages.append({"role": "system", "content": gen.generate_system_prompt()})
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""

    while subtask != "DONE()":
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"
        subtask_done, subtask, code, code_output, messages, lorebook = (
            try_identify_and_execute(
                env,
                gen,
                disc,
                messages,
                lorebook,
                task=TASK,
                open_or_closed=open_or_closed,
                python_code_called_history=code_history,
                python_code_output_history=code_output_history,
                subtasks=subtasks,
            )
        )
        subtasks.append(subtask)
        code_history += "\n" + code
        code_output_history += "\n" + code_output
    env.set_recorder()


if __name__ == "__main__":
    main()
