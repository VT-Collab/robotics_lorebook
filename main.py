from src.env import PandaEnv
from src.llm import LLM
from src.llm import DoubleSimRAG as RAG
from src.utils import generate_objects_table, extract_json
from termcolor import cprint as termcolor_cprint
import time
from datetime import datetime
import os
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


API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gemini-3-flash-preview"  # "gpt"  # "gemini-3-flash-preview"
GEN_CONF = "config/prompts/llm_unified_function_store.yml"
# TASK = "put the block in the cabinet. the cabinet door is closed at the beginning. the cabinet door opens prismatically TOWARDS the robot along the negative x direction"
# TASK = "put the block in the oven. the oven door is closed at the beginning"
TASK = "put the food in the microwave"
VIDEO_PATH = f"videos/{log_filename}.mp4"
SCENE = "cooking.yml"  


QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()


def cprint(text, color="white", **kwargs):
    clean_text = str(text).strip()
    logging.info(clean_text)
    termcolor_cprint(text, color=color, **kwargs)


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


def identify_next_subtask(
    env: PandaEnv, gen: LLM, messages: list[dict], verbose: bool, **kwargs
) -> str:
    """Phase 1: Analyzes the environment to determine the next text-based subtask."""
    position, orn = env.get_print_state()
    objs_table = generate_objects_table(env)

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
    return subtask


def retrieve_feedback_context(
    env: PandaEnv,
    gen: LLM,
    lorebook: RAG,
    messages: list[dict],
    subtask: str,
    verbose: bool,
) -> str:
    """Retrieves past failures/successes from RAG and summarizes them for the current context."""
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
    template_lore = lorebook.query(template_key, top_k=-1)
    template_items = "\n\n".join([f"{item['value']}" for item in template_lore])
    if template_lore:
        raw_context += f"\nThe following templated functions have solved similar subtasks in the past:\n{template_items}"

    return (
        f" You should refer to human's feedback to accomplish the task:\n{raw_context}"
    )


def generate_and_execute_code(
    env: PandaEnv,
    gen: LLM,
    messages: list[dict],
    subtask: str,
    feedback_context: str,
    verbose: bool,
    **kwargs,
) -> tuple[str, str]:
    """Phase 2: Generates Python code for the subtask and executes it."""
    position, orn = env.get_print_state()
    objs_table = generate_objects_table(env)

    code_gen_instruction = f"output code to accomplish {subtask}. {feedback_context}"
    code_gen_instruction += "\n\n\n**You are now in PHASE 2: Code Generation.**"

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

    # Rollout
    code_output = env.run_code(code)
    return code, code_output


def handle_human_interruption(
    env: PandaEnv,
    gen: LLM,
    lorebook: RAG,
    messages: list[dict],
    subtask: str,
    verbose: bool,
):
    """Phase 3: Feedback Loop. Analyzes user feedback and updates the vector DB."""
    print("\n" + "=" * 50)
    print("Ctrl+C detected. Entering Feedback Mode...")
    feedback = input("Provide your feedback here: ")
    print("=" * 50)

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


def consolidate_success(
    env: PandaEnv,
    gen: LLM,
    lorebook: RAG,
    messages: list[dict],
    subtask: str,
    code_history: str,
    verbose: bool,
):
    """
    Phase 4: Success Consolidation.
    Generalizes the successful code into a template and saves it to RAG.
    """
    affirm = (
        input(
            f"The system solved {subtask}. Continue? [Y/s/n] (yes/store_function/no) "
        )
        or "y"
    )
    if affirm.lower() == "y":
        return
    elif affirm.lower() == "n":
        # exit with failure
        raise KeyboardInterrupt
    else:
        pass  # carry on with storing function
    print(f"\n[System]: Subtask {subtask} successful. Generalizing...")

    # We create a specific prompt for generalization
    generalize_prompt = (
        f"The code for subtask `{subtask}` executed successfully. "
        "Please enter PHASE 4. "
        "Take the executed code below and convert it into a generalized Python function template. "
        "Replace specific object coordinates with parameters. "
        f"\n\nEXECUTED CODE:\n{code_history}"
    )

    # We append this to history so the LLM sees what it just wrote
    messages.append({"role": "user", "content": generalize_prompt})

    _, generalized_func = get_model_output(gen, messages, verbose=verbose)

    # Clean up the function (remove markdown ticks if present)
    clean_func = generalized_func.replace("```python", "").replace("```", "").strip()

    # Store in RAG
    # We use a special key prefix "TEMPLATE" so Phase 1/2 can find it easily later
    key = f"TEMPLATE {subtask}"
    cprint(f"[Memory]: Saving generalized skill to RAG under key: '{key}'", "green")
    lorebook.add(key, clean_func)

    # Clean up messages so the generalization conversation doesn't pollute the next task
    messages.pop()


def wait_for_user_approval(seconds=5):
    """Blocking wait to allow user to trigger KeyboardInterrupt."""
    print("=" * 50)
    print(f"Waiting {seconds}s for feedback interruption", end="", flush=True)
    for _ in range(seconds):
        print(".", end="", flush=True)
        time.sleep(1)
    print("\n" + "=" * 50)


def try_identify_and_execute(
    env: PandaEnv,
    gen: LLM,
    messages: list[dict],
    lorebook: RAG,
    verbose=True,
    **prompt_kwargs,
) -> tuple[bool, str, str, str, list[dict], RAG]:

    subtask = ""
    code = ""
    code_output = ""
    og_messages_len = len(messages)

    while True:
        ckpt = env.get_checkpoint()
        try:
            subtask = identify_next_subtask(
                env, gen, messages, verbose, **prompt_kwargs
            )
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return False, subtask, code, code_output, messages, lorebook
            messages.append({"role": "assistant", "content": subtask})
            feedback_context = retrieve_feedback_context(
                env, gen, lorebook, messages, subtask, verbose
            )
            code, code_output = generate_and_execute_code(
                env, gen, messages, subtask, feedback_context, verbose, **prompt_kwargs
            )
            prompt_kwargs["python_code_called_history"] += code + "\n"
            prompt_kwargs["python_code_output_history"] += code_output + "\n"
            wait_for_user_approval()
        except KeyboardInterrupt:
            handle_human_interruption(env, gen, lorebook, messages, subtask, verbose)
            print("Restoring checkpoint and retrying subtask...")
            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            # Rollback history to avoid polluting the next attempt
            del messages[og_messages_len:]
            prompt_kwargs["python_code_called_history"] = prompt_kwargs.get(
                "python_code_called_history", ""
            ).replace(code + "\n", "")
            prompt_kwargs["python_code_output_history"] = prompt_kwargs.get(
                "python_code_output_history", ""
            ).replace(code_output + "\n", "")
            continue
        else:
            consolidate_success(env, gen, lorebook, messages, subtask, code, verbose)
            env.p.removeState(ckpt)
            return True, subtask, code, code_output, messages, lorebook


def main():
    env = PandaEnv(scene_config=SCENE)
    if VIDEO_PATH:
        env.set_recorder(VIDEO_PATH)
    # lorebook = RAG(filename="data/lorebook.pkl")
    # lorebook = RAG(filename="data/lorebook_pruned.pkl")
    lorebook = RAG(filename="data/lorebook_double_pruned.pkl")
    # Only one LLM instance needed now
    gen = LLM(API_KEY, API_URL, GEN_CONF, MODEL)

    print("=" * 50)
    print("To provide feedback, hit Ctrl+C during the 5s wait period.")
    input("Press any button to proceed: ")
    print("=" * 50)

    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""

    while subtask != "DONE()":
        gripper_state = env.get_state()["gripper"][0]
        open_or_closed = "open" if gripper_state > 0.039 else "closed"

        messages = []
        messages.append({"role": "system", "content": gen.generate_system_prompt()})

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
