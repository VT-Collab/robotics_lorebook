from src.env import PandaEnv
from src.llm import LLM, RAG
from src.objects import objects
from termcolor import cprint
import time
import os
import json


API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
GEN_CONF = "config/prompts/llm_decomposer.yml"
DICT_CONF = "config/prompts/llm_dictionary.yml"
TASK = "Put the block in the cabinet. The cabinet is initially closed. the handle of the cabinet is oriented horizontally across the cabinet and is defined by the offset 0.04 0.1 0.015"

QUERY_TIMEOUT = 0.5
TIME_SINCE_LAST_QUERY = time.time()


def generate_objects_table(env: PandaEnv) -> str:
    info = []
    for obj_entry in env.objects:
        body_id = obj_entry["id"]
        pos, quat = env.p.getBasePositionAndOrientation(body_id)
        euler = env.p.getEulerFromQuaternion(quat)
        pos = [round(x, 2) for x in pos]
        euler = [round(x, 2) for x in euler]
        t = obj_entry["type"]
        if t == "plane":
            continue
        info.append({"type": t, "pos": pos, "orn": euler})
        if isinstance(obj_entry["ref"], objects.CollabObject):
            state = obj_entry["ref"].get_state()
            handle_pos = [round(x, 2) for x in state["handle_position"]]
            handle_orn = [round(x, 2) for x in state["handle_euler"]]
            info.append({"type": t + " handle", "pos": handle_pos, "orn": handle_orn})
    table = (
        """| Object | Position | Orientation |\n| ------ | -------- | ----------- |"""
    )
    for obj in info:
        table += "\n"
        table += f'| {obj["type"]} | {str(obj["pos"])} | {obj["orn"]} |'
    return table


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
    output = model.query(messages)
    reasoning = output.choices[0].message.reasoning
    subtask = output.choices[0].message.content
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

            for lore in retrieved_lore:
                cprint(lore, "white")

            current_task_instruction = f"output code to accomplish {subtask}"
            if retrieved_lore:
                feedback_str = "\n".join(
                    [f"- {item['value']}" for item in retrieved_lore]
                )
                current_task_instruction += (
                    f". Use the following past experience as feedback: {feedback_str}"
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
            output = disc.query(disc_messages)

            try:
                response_content = output.choices[0].message.content
                new_lore = json.loads(response_content)
                cprint(f"[disc]: adding to vector database: {new_lore}", "red")
                for k, v in new_lore.items():
                    objs_table = generate_objects_table(env)
                    vector_key = f"Subtask: {k}. Environment: {objs_table}"
                    lorebook.add(vector_key, v)
            except (AttributeError, json.JSONDecodeError):
                print("Failed to parse discriminator feedback.")

            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            del messages[og_messages_len:]
            python_code_called_history = python_code_called_history[:og_code_len]
            python_code_output_history = python_code_output_history[:og_output_len]
            print("Environment reset. Retrying with vector-retrieved feedback...")

    return subtask_done, subtask, code, code_output, messages, lorebook


def main():
    env = PandaEnv()
    lorebook = RAG()
    gen = LLM(API_KEY, API_URL, GEN_CONF)
    disc = LLM(API_KEY, API_URL, DICT_CONF)

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
        open_or_closed = "open" if gripper_state > 0.02 else "closed"
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


if __name__ == "__main__":
    main()
