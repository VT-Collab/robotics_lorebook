from src.env import PandaEnv
from src.llm import LLM, upload_file
from src.objects import objects
from termcolor import cprint as termcolor_cprint
import time
from datetime import datetime
import os
import json
import logging
import re

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
GEN_CONF = "config/prompts/llm_unified.yml" 
TASK = "put the block in the cabinet. the cabinet door is closed at the beginning. the cabinet door opens prismatically TOWARDS the robot along the negative x direction"
VIDEO_PATH = f"videos/{log_filename}.mp4"
MODEL = "gpt"

QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()

def generate_objects_table(env: PandaEnv) -> str:
    info = []
    for obj_entry in env.objects:
        body_id = obj_entry["id"]
        t = obj_entry["type"]
        if t == "plane":
            continue
        pos, quat = env.p.getBasePositionAndOrientation(body_id)
        euler = [round(x, 2) for x in env.p.getEulerFromQuaternion(quat)]
        pos = [round(x, 2) for x in pos]
        aabb_min, aabb_max = env.p.getAABB(body_id)
        dims = [round(aabb_max[i] - aabb_min[i], 3) for i in range(3)]
        info.append({"type": t, "pos": pos, "orn": euler, "dims": dims})
        
        # Handle handles/sub-parts
        if isinstance(obj_entry["ref"], objects.CollabObject):
            state = obj_entry["ref"].get_state()
            handle_pos = [round(x, 2) for x in state["handle_position"]]
            handle_orn = [round(x, 2) for x in state["handle_euler"]]
            h_min, h_max = env.p.getAABB(body_id, linkIndex=1)
            h_dims = [round(h_max[i] - h_min[i], 3) for i in range(3)]
            info.append({"type": t + " handle", "pos": handle_pos, "orn": handle_orn, "dims": h_dims})

    table = "| Object | Position | Orientation | Dimensions (WxLxH) |\n| ------ | -------- | ----------- | ------------------ |"
    for obj in info:
        table += f'\n| {obj["type"]} | {obj["pos"]} | {obj["orn"]} | {obj["dims"]} |'
    return table

def generate_rag_key(env: PandaEnv, subtask: str) -> str:
    key = f"{subtask}"
    for obj_entry in env.objects:
        t = obj_entry["type"]
        if t != "plane":
            key += f" {t}"
    return key

def get_model_output(model: LLM, messages: list[dict], file_ids: list | None = None, verbose=True):
    global TIME_SINCE_LAST_QUERY
    while time.time() < TIME_SINCE_LAST_QUERY + QUERY_TIMEOUT:
        pass
    lorebook_content = {}
    if file_ids:
        lorebook_content["files"] = [
            {"type": "file", "id": fid} for fid in file_ids
        ]
    reasoning, content = model.query(messages, lorebook_content=lorebook_content)
    if verbose:
        for m in messages:
            cprint(f"[{m['role']}]: {m['content']}", "yellow")
        cprint(reasoning, "green")
        cprint(content, "red")
    TIME_SINCE_LAST_QUERY = time.time()
    return reasoning, content

def extract_json(content: str):
    """Helper to extract JSON from markdown code blocks"""
    try:
        # Try finding standard json block
        match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Fallback to plain text parsing
        return json.loads(content)
    except Exception:
        return None

def try_identify_and_execute(
    env: PandaEnv,
    gen: LLM,
    messages: list[dict],
    file_ids: list | None = None,
    verbose=True,
    **prompt_kwargs,
) -> tuple[bool, str, str, str, list[dict]]:
    python_code_called_history = prompt_kwargs.get("python_code_called_history", "")
    python_code_output_history = prompt_kwargs.get("python_code_output_history", "")
    task = prompt_kwargs.get("task", "")
    subtasks_list = prompt_kwargs.get("subtasks", [])
    open_or_closed = prompt_kwargs.get("open_or_closed", "open")
    
    # Store indices to revert history on failure
    og_messages_len = len(messages)
    og_code_len = len(python_code_called_history)
    og_output_len = len(python_code_output_history)

    subtask = ""
    subtask_done = False
    code = ""
    code_output = ""

    while True:
        ckpt = env.get_checkpoint()
        try:
            position, orn = env.get_print_state()
            objs_table = generate_objects_table(env)

            # --- PHASE 1: SUBTASK ID ---
            id_prompt = gen.generate_followup_prompt(
                python_code_called_history=python_code_called_history,
                python_code_output_history=python_code_output_history,
                task=task,
                subtasks_list=subtasks_list,
                objects_table=objs_table,
                position=position,
                angle=orn[2],
                open_or_closed=open_or_closed,
                next_thing_to_do="identify the next subtask to execute",
            )
            messages.append({"role": "user", "content": id_prompt})
            
            _, subtask = get_model_output(gen, messages, file_ids, verbose=verbose)
            
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return subtask_done, subtask, code, code_output, messages
            
            messages.append({"role": "assistant", "content": subtask})

            # --- RAG RETRIEVAL ---
            try:
                file_ids = [upload_file("lorebook.txt")]
            except Exception as e:
                cprint(f"Failed to upload file for RAG retrieval: {e}", "red")
                file_ids = None

            # --- PHASE 2: CODE GENERATION ---
            code_gen_instruction = f"output code to accomplish {subtask}, content ptrovides past experience as feedback"
            
            code_prompt = gen.generate_followup_prompt(
                next_thing_to_do=code_gen_instruction,
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
            
            _, code = get_model_output(gen, messages, file_ids, verbose=verbose)
            messages.append({"role": "assistant", "content": code})
            
            # Execute Code
            code_output = env.run_code(code)
            
            # Update history tentatively
            python_code_called_history += code + "\n"
            python_code_output_history += code_output + "\n"

            print("=" * 50)
            print("Waiting 5s for feedback interruption", end="", flush=True)
            for i in range(5):
                print(".", end="", flush=True)
                time.sleep(1)
            print("\n" + "=" * 50)
            
            # Success path
            env.p.removeState(ckpt)
            subtask_done = True
            break

        except KeyboardInterrupt:
            # --- PHASE 3: FEEDBACK LOOP (Unified Instance) ---
            print("\n" + "=" * 50)
            print("Ctrl+C detected. Entering Feedback Mode...")
            feedback = input("Provide your feedback here: ")
            print("=" * 50)

            # We construct a prompt for the SAME instance that already has the context of the failure
            feedback_prompt = (
                f"You just attempted the action `{subtask}`. The user has intervened with the following feedback: "
                f"\"{feedback}\". \n\n"
                "Please analyze this feedback according to PHASE 3 instructions. "
                "Output the JSON memory update."
            )
            
            messages.append({"role": "user", "content": feedback_prompt})
            
            cprint("[System]: Reasoning about feedback...", "magenta")
            _, response_content = get_model_output(gen, messages, verbose=verbose)
            
            new_lore = extract_json(response_content)
            
            if new_lore:
                cprint(f"[Memory]: Adding to vector database: {new_lore}", "red")
                for k, v in new_lore.items():
                    # Generate a key that mixes the specific action and general concepts
                    key = generate_rag_key(env, k)
                    # append to lorebook.txt
                    with open("lorebook.txt", "a") as fh:
                        fh.write(f"{key}\t{v}\n")
            else:
                cprint("Failed to parse feedback JSON from model response.", "red")

            # --- RESET ---
            print("Restoring checkpoint and retrying subtask...")
            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            
            # Slice messages back to start of this subtask attempt to avoid pollution
            # But the RAG (lorebook) IS updated, so the next retrieval will be smarter.
            del messages[og_messages_len:]
            python_code_called_history = python_code_called_history[:og_code_len]
            python_code_output_history = python_code_output_history[:og_output_len]

    return subtask_done, subtask, code, code_output, messages

def main():
    env = PandaEnv()
    if VIDEO_PATH:
        env.set_recorder(VIDEO_PATH)
    # Only one LLM instance needed now
    gen = LLM(API_KEY, API_URL, GEN_CONF, MODEL)

    print("=" * 50)
    print("To provide feedback, hit Ctrl+C during the 5s wait period.")
    input("Press any button to proceed: ")
    print("=" * 50)

    messages = []
    messages.append({"role": "system", "content": gen.generate_system_prompt()})

    # --- RAG RETRIEVAL ---
    try:
        file_ids = [upload_file("lorebook.txt")]
    except Exception as e:
        cprint(f"Failed to upload file for RAG retrieval: {e}", "red")
        file_ids = None
    
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""
    
    while subtask != "DONE()":
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"
        
        # We only pass 'gen', no 'disc'
        subtask_done, subtask, code, code_output, messages = (
            try_identify_and_execute(
                env,
                gen,
                messages,
                file_ids,
                task=TASK,
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
            
    env.set_recorder()

if __name__ == "__main__":
    main()