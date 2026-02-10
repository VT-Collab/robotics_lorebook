from src.env import PandaEnv
from src.llm import LLM, RAG, GenericKeyRAG
from src.utils import generate_objects_table
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
LOREBOOK_PATH = "data/lorebook_generic.pkl"#"data/lorebook_generic.pkl"

QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()


def generate_rag_key(env: PandaEnv, subtask: str) -> str:
    key = f"{subtask}"
    for obj_entry in env.objects:
        t = obj_entry["type"]
        if t != "plane":
            key += f" {t}"
    return key

def get_model_output(model: LLM, messages: list[dict], verbose=True):
    global TIME_SINCE_LAST_QUERY
    while time.time() < TIME_SINCE_LAST_QUERY + QUERY_TIMEOUT:
        pass
    reasoning, content = model.query(messages)
    # reasoning = output.choices[0].message.reasoning
    # content = output.choices[0].message.content
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
    lorebook: GenericKeyRAG, #GenericKeyRAG,
    verbose=True,
    **prompt_kwargs,
) -> tuple[bool, str, str, str, list[dict], GenericKeyRAG]:
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
            
            _, subtask = get_model_output(gen, messages, verbose=verbose)
            
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return subtask_done, subtask, code, code_output, messages, lorebook
            
            messages.append({"role": "assistant", "content": subtask})

            # --- RAG RETRIEVAL ---
            rag_query_key = generate_rag_key(env, subtask)
            retrieved_lore = lorebook.query(rag_query_key, top_k=5)
            cprint(f"n lore: {len(retrieved_lore)}")
            print(retrieved_lore)
            cprint(f"Retrieved lore with scores: {[item['score'] for item in retrieved_lore]}")
            feedback_context = ""
            if retrieved_lore:
                cprint("Integrating past feedback...", "cyan")
                feedback_items = "\n".join([f"- {item['value']}" for item in retrieved_lore])
                feedback_context = f". Use the following past experience as feedback:\n{feedback_items}"

            # --- PHASE 2: CODE GENERATION ---
            code_gen_instruction = f"output code to accomplish {subtask}{feedback_context}"
            
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
            
            _, code = get_model_output(gen, messages, verbose=verbose)
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
                    lorebook.add(key, v)
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

    return subtask_done, subtask, code, code_output, messages, lorebook

def main():
    env = PandaEnv()
    if VIDEO_PATH:
        env.set_recorder(VIDEO_PATH)
    lorebook = GenericKeyRAG(filename=LOREBOOK_PATH)
    # Only one LLM instance needed now
    gen = LLM(API_KEY, API_URL, GEN_CONF)

    print("=" * 50)
    print("To provide feedback, hit Ctrl+C during the 5s wait period.")
    input("Press any button to proceed: ")
    print("=" * 50)
    
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""
    
    while subtask != "DONE()":
        messages = []
        messages.append({"role": "system", "content": gen.generate_system_prompt()})
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"
        
        # We only pass 'gen', no 'disc'
        subtask_done, subtask, code, code_output, messages, lorebook = (
            try_identify_and_execute(
                env,
                gen,
                messages,
                lorebook,
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