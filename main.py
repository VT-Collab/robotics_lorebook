## for env
from env import PandaEnv

## for llm
import yaml
from llm import LLM
from termcolor import cprint
import time
import os

API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
GEN_CONF = "config/prompts/llm_decomposer.yml"
DICT_CONF = "config/prompts/llm_dictionary.yml"
TASK = "move the block closer to the base of the robot. make sure the block ends on the table. you may need to move very close to the block to pick it up, as the gripper is very small."

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
    output = model.query_llm(messages)
    reasoning = output.choices[0].message.reasoning
    subtask = output.choices[0].message.content
    if verbose:
        for m in messages:
            content = m['content']
            content = "\t" + content.replace("\n", "\n\t")
            cprint(f"[{m['role']}]: {content}", "yellow")
        cprint(reasoning, "green")
        cprint(subtask, "red")
    TIME_SINCE_LAST_QUERY = time.time()
    return reasoning, subtask

def main():
    env = PandaEnv()
    subtask_dict = {}
    gen = LLM(API_KEY, API_URL, GEN_CONF)
    disc = LLM(API_KEY, API_URL, DICT_CONF)
    disc_messages = disc.generate_initial_message()

    # 1. prepare initial prompts
    initial_position, initial_orientation = env.get_print_state()
    initial_position = [round(x, 2) for x in initial_position]
    initial_orientation = [round(x, 2) for x in initial_orientation]
    messages = gen.generate_initial_message(
        position=initial_position,
        angle=initial_orientation[2],
        open_or_closed="open",
        task=TASK,
        objects_table=generate_objects_table(env),
        next_thing_to_do="identify the next subtask to complete",
    )
    # 2. begin interaction (repeat until finished)
    # use a do while for now
    subtasks = []
    python_code_called_history = ""
    python_code_output_history = ""
    _, subtask = get_model_output(gen, messages)
    messages.append({"role": "assistant", "content": subtask})
    next_thing_to_do = f"output code to accomplish {subtask}"
    next_prompt = gen.generate_initial_prompt(
        position=initial_position,
        angle=initial_orientation[2],
        open_or_closed="open",
        task=TASK,
        objects_table=generate_objects_table(env),
        next_thing_to_do=next_thing_to_do,
    )
    messages.append({"role": "user", "content": next_prompt})
    _, code = get_model_output(gen, messages)
    messages.append({"role": "assistant", "content": code})
    code_output = env.run_code(code)
    python_code_called_history += code + "\n"
    python_code_output_history += code_output + "\n"
    subtasks.append(subtask)
    while True:
        next_thing_to_do = "identify the next subtask to complete"
        pos, orn = env.get_print_state()
        followup_prompt = gen.generate_followup_prompt(
            python_code_called_history=python_code_called_history,
            python_code_output_history=python_code_output_history,
            task=TASK,
            subtasks_list=subtasks,
            objects_table=generate_objects_table(env),
            position=pos,
            angle=orn[2],
            open_or_closed="open", # TODO: fix
            next_thing_to_do=next_thing_to_do,
        )
        messages.append({"role": "user", "content": followup_prompt})
        # 2a. identify subtask
        _, subtask = get_model_output(gen, messages)
        if subtask == "DONE()":
            break
        messages.append({"role": "assistant", "content": subtask})
        # 2b. perform actions for subtask
        next_thing_to_do = f"output code to accomplish {subtask}"
        followup_prompt = gen.generate_followup_prompt(
            python_code_called_history=python_code_called_history,
            python_code_output_history=python_code_output_history,
            task=TASK,
            subtasks_list=subtasks,
            objects_table=generate_objects_table(env),
            position=pos,
            angle=orn[2],
            open_or_closed="open", # TODO: fix
            next_thing_to_do=next_thing_to_do,
        )
        messages.append({"role": "user", "content": followup_prompt})
        _, code = get_model_output(gen, messages)
        messages.append({"role": "assistant", "content": code})
        code_output = env.run_code(code)
        python_code_called_history += code + "\n"
        python_code_output_history += code_output + "\n"

if __name__ == "__main__":
    main()
