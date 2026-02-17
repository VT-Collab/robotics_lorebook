from src.env import PandaEnv
from src.llm import LLM
from src.llm import DoubleSimRAG as RAG
from src.utils import generate_objects_table, extract_json
from termcolor import cprint as termcolor_cprint
import time
from datetime import datetime
import os
import logging
import signal

# --- NEW IMPORT ---
from src.shared import state_manager

# Custom exception for web flow
class WebInterruption(Exception):
    pass

os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)
log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(f"logs/{log_filename}.log")],
)

API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gemini-3-flash-preview"
GEN_CONF = "config/prompts/llm_unified_function_store.yml"
QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()

# --- REPLACED CPRINT ---
def cprint(text, color="white", **kwargs):
    clean_text = str(text).strip()
    logging.info(clean_text)
    # Print to terminal
    termcolor_cprint(text, color=color, **kwargs)
    # Push to web interface
    # state_manager.add_log(clean_text, color)

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
    while time.time() < TIME_SINCE_LAST_QUERY + QUERY_TIMEOUT:
        pass
    cprint("[QUERY] Querying model...", "blue")
    state_manager.add_log("[QUERY] Querying model...", color="blue")
    
    reasoning, content = model.query(messages)
    
    # Update state for web view
    state_manager.reasoning = reasoning
    state_manager.model_output = content
    
    if verbose:
        for m in messages:
            cprint(f"[{m['role']}]: {m['content']}", "yellow")
        cprint(reasoning, "green")
        cprint(content, "red")
    TIME_SINCE_LAST_QUERY = time.time()
    return reasoning, content

def identify_next_subtask(env, gen, messages, verbose, **kwargs) -> str:
    position, orn = env.get_print_state()
    objs_table = generate_objects_table(env)
    state_manager.add_log("[MODEL] Identifying subtask...", color="red")
    
    
    # Update web state
    state_manager.current_subtask = "Identifying next step..."

    id_prompt = gen.generate_followup_prompt(
        python_code_called_history=kwargs.get("python_code_called_history", ""),
        python_code_output_history=kwargs.get("python_code_output_history", ""),
        task=kwargs.get("task", ""),
        subtasks_list=kwargs.get("subtasks", []),
        objects_table=objs_table,
        position=position,
        angle=orn[2],
        open_or_closed=kwargs.get("open_or_closed", "open"),
        next_thing_to_do="identify the next subtask to execute",
    )

    messages.append({"role": "user", "content": id_prompt})
    _, subtask = get_model_output(gen, messages, verbose=verbose)
    
    state_manager.current_subtask = subtask
    return subtask

def retrieve_feedback_context(env, gen, lorebook, messages, subtask, verbose) -> str:
    rag_query_key = generate_rag_key(env, subtask)
    retrieved_lore = lorebook.query(rag_query_key, top_k=20)
    rag_general_key = generate_rag_key(env, "GENERAL")
    retrieved_lore += lorebook.query(rag_general_key, top_k=10)

    cprint(f"n lore: {len(retrieved_lore)}")

    if not retrieved_lore:
        return ""

    cprint("Integrating past feedback...", "cyan")
    feedback_items = "\n".join([f"- {item['value']}" for item in retrieved_lore])
    raw_context = f"Use the following past experience as feedback:\n{feedback_items}"

    template_key = f"TEMPLATE {subtask}"
    template_lore = lorebook.query(template_key, top_k=-1, min_score=0.95)
    template_items = "\n\n".join([f"{item['value']}" for item in template_lore])
    if template_lore:
        raw_context += f"\n\nThe following templated functions have solved similar subtasks in the past:\n{template_items}"

    return f" You should refer to human's feedback to accomplish the task:\n{raw_context}"

def generate_and_execute_code(env, gen, messages, subtask, feedback_context, verbose, **kwargs) -> tuple[str, str]:
    position, orn = env.get_print_state()
    objs_table = generate_objects_table(env)

    code_gen_instruction = f"output code to accomplish {subtask}. {feedback_context}"
    code_gen_instruction += "\n\n\n**You are now in PHASE 2: Code Generation.**"
    
    state_manager.add_log("[MODEL] Generating code...", color="red")

    code_prompt = gen.generate_followup_prompt(
        next_thing_to_do=code_gen_instruction,
        python_code_called_history=kwargs.get("python_code_called_history", ""),
        python_code_output_history=kwargs.get("python_code_output_history", ""),
        task=kwargs.get("task", ""),
        subtasks_list=kwargs.get("subtasks", []),
        objects_table=objs_table,
        position=position,
        angle=orn[2],
        open_or_closed=kwargs.get("open_or_closed", "open"),
    )

    messages.append({"role": "user", "content": code_prompt})
    _, code = get_model_output(gen, messages, verbose=verbose)
    messages.append({"role": "assistant", "content": code})

    state_manager.model_output = code

    # Rollout
    code_output = env.run_code(code)
    return code, code_output

# --- NEW FEEDBACK HANDLER ---
def handle_web_interruption(env, gen, lorebook, messages, subtask, verbose):
    """
    Called when WebInterruption is raised.
    Pauses execution and waits for feedback via the shared state.
    """
    cprint("Interrupt detected! Waiting for feedback from web interface...", "magenta")
    
    # 1. Signal Web that we are waiting
    state_manager.is_waiting_for_feedback = True
    
    # 2. Block until feedback arrives
    state_manager.feedback_event.wait()
    
    # 3. Retrieve feedback
    feedback = state_manager.feedback_text
    
    # 4. Reset flags
    state_manager.feedback_event.clear()
    state_manager.interrupt_event.clear()
    state_manager.is_waiting_for_feedback = False
    
    if not feedback:
        cprint("Empty feedback received, resuming...", "yellow")
        return

    feedback_prompt = (
        f"You just attempted the action `{subtask}`. The user has intervened with the following feedback: "
        f'"{feedback}". \n\n'
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
            key = generate_rag_key(env, k)
            lorebook.add(key, v)
    else:
        cprint("Failed to parse feedback JSON from model response.", "red")

def consolidate_success(env, gen, lorebook, messages, subtask, code_history, verbose):
    key = f"TEMPLATE {subtask}"
    results = lorebook.query(key, top_k=1, min_score=0.999)
    if results:
        cprint(f"lorebook already contains TEMPLATE {subtask}")
        return
    
    cprint(f"\n[System]: Subtask {subtask} successful. Generalizing...")
    
    state_manager.add_log("[MODEL] Generalizing function...", color="red")
    
    generalize_prompt = (
        f"The code for subtask `{subtask}` executed successfully. "
        "Please enter PHASE 4. "
        "Take the executed code below and convert it into a generalized Python function template. "
        "Replace specific object coordinates with parameters. "
        f"\n\nEXECUTED CODE:\n{code_history}"
    )

    messages.append({"role": "user", "content": generalize_prompt})
    _, generalized_func = get_model_output(gen, messages, verbose=verbose)
    clean_func = generalized_func.replace("```python", "").replace("```", "").strip()

    cprint(f"[Memory]: Saving generalized skill to RAG under key: '{key}'", "green")
    lorebook.add(key, clean_func)
    messages.pop()

# --- REPLACED WAIT LOGIC ---
def wait_for_user_approval(seconds=5):
    """
    Non-blocking wait (loop) that checks for web interrupt signal.
    """
    print("=" * 50)
    print(f"Waiting {seconds}s for feedback interruption", end="", flush=True)
    
    start = time.time()
    while time.time() - start < seconds:
        # Check if the "Interrupt" button was pressed on the web
        if state_manager.interrupt_event.is_set():
            print("\nInterrupt signal received from Web!")
            raise WebInterruption()
        
        print(".", end="", flush=True)
        time.sleep(0.1)
        
    print("\n" + "=" * 50)

def try_identify_and_execute(env, gen, messages, lorebook, verbose=True, **prompt_kwargs):
    subtask = ""
    code = ""
    code_output = ""
    og_messages_len = len(messages)

    while True:
        ckpt = env.get_checkpoint()
        try:
            # Check interrupt at start of loop too
            if state_manager.interrupt_event.is_set():
                 raise WebInterruption()

            subtask = identify_next_subtask(env, gen, messages, verbose, **prompt_kwargs)
            
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return False, subtask, code, code_output, messages, lorebook
            
            messages.append({"role": "assistant", "content": subtask})
            feedback_context = retrieve_feedback_context(env, gen, lorebook, messages, subtask, verbose)
            code, code_output = generate_and_execute_code(env, gen, messages, subtask, feedback_context, verbose, **prompt_kwargs)
            
            prompt_kwargs["python_code_called_history"] += code + "\n"
            prompt_kwargs["python_code_output_history"] += code_output + "\n"
            
            wait_for_user_approval()
            
        except (KeyboardInterrupt, WebInterruption):
            state_manager.add_log("[SYSTEM] Handling interrupt.", color="white")
            handle_web_interruption(env, gen, lorebook, messages, subtask, verbose)
            print("Restoring checkpoint and retrying subtask...")
            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            
            del messages[og_messages_len:]
            # Clean history
            prompt_kwargs["python_code_called_history"] = prompt_kwargs.get("python_code_called_history", "").replace(code + "\n", "")
            prompt_kwargs["python_code_output_history"] = prompt_kwargs.get("python_code_output_history", "").replace(code_output + "\n", "")
            continue
        else:
            state_manager.add_log("[SYSTEM] Subtask successful!", color="white")
            consolidate_success(env, gen, lorebook, messages, subtask, code, verbose)
            env.p.removeState(ckpt)
            return True, subtask, code, code_output, messages, lorebook
