## for env
from env import PandaEnv

## for llm
import yaml
from llm import LLM
from termcolor import cprint
import time
import os
from copy import deepcopy
import json

API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
GEN_CONF = "config/prompts/llm_decomposer.yml"
DICT_CONF = "config/prompts/llm_dictionary.yml"
TASK = "move the block closer to the base of the robot. make sure the block ends on the table. you may need to move very close to the block to pick it up, as the gripper is very small."
VIDEO_PATH = None

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
    table = (
        """| Object | Position | Orientation |\n| ------ | -------- | ----------- |"""
    )
    for obj in info:
        table += "\n"
        table += f'| {obj["type"]} | {str(obj["pos"])} | {obj["orn"]} |'
    return table


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
    lorebook: dict,
    verbose=True,
    **prompt_kwargs,
) -> tuple[bool, str, str, str, list[dict], dict]:
    """
    Resets environment on Ctrl+C and retries with updated lorebook.
    """
    python_code_called_history = prompt_kwargs.get("python_code_called_history", "")
    python_code_output_history = prompt_kwargs.get("python_code_output_history", "")
    task = prompt_kwargs.get("task", "")
    subtasks_list = prompt_kwargs.get("subtasks", [])
    open_or_closed = prompt_kwargs.get("open_or_closed", "open")
    next_thing_to_do = "identify the next subtask to execute"

    og_messages_len = len(messages)

    # these are returned later
    subtask = ""
    subtask_done = False
    code = ""
    code_output = ""

    # begin execution
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

            # query for subtask identification
            messages.append({"role": "user", "content": message})
            _, subtask = get_model_output(gen, messages, verbose=verbose)
            if subtask == "DONE()":
                env.p.removeState(ckpt)
                return subtask_done, subtask, code, code_output, messages, lorebook
            messages.append({"role": "assistant", "content": subtask})
            potential_feedback = lorebook.get(subtask, None)
            current_task_instruction = f"output code to accomplish {subtask}"
            if potential_feedback:
                current_task_instruction += f". Feedback is {potential_feedback}"
            # query for Code Execution
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
                cprint(f"[disc]: adding new lore: {new_lore}", "red")
                lorebook.update(new_lore)
            except (AttributeError, json.JSONDecodeError):
                print("Failed to parse discriminator feedback.")
            env.restore_checkpoint(ckpt)
            env.p.removeState(ckpt)
            del messages[og_messages_len:]
            print("Environment reset. Retrying with updated lorebook...")

    return subtask_done, subtask, code, code_output, messages, lorebook


def main():
    env = PandaEnv()
    lorebook = {}
    if VIDEO_PATH:
        env.set_recorder(VIDEO_PATH)
    gen = LLM(API_KEY, API_URL, GEN_CONF)
    disc = LLM(API_KEY, API_URL, DICT_CONF)

    print("=" * 50)
    print(
        "To provide feedback, hit Ctrl+C. This will interrupt execution and revert the environment to the previous state."
    )
    input("Press any button to proceed: ")
    print("=" * 50)

    messages = []
    messages.append({"role": "system", "content": gen.generate_system_prompt()})
    subtask = ""
    subtasks = []
    code_history = ""
    code_output_history = ""
    open_or_closed = (
        "open"  # TODO: idk how to determine if the gripper is open or closed rn
    )
    while subtask != "DONE()":
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
