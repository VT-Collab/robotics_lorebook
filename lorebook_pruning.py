import os 
import json
import time
import concurrent.futures
import pickle
from termcolor import cprint, colored
from src.llm import LLM

LOREBOOK_PATH = "data/lorebook.pkl"
API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gemini-3-flash-preview"  # "gpt"  # "gemini-3-flash-preview"
GEN_CONF = "config/prompts/llm_pruning.yml"

QUERY_TIMEOUT = 0.5
QUERY_EXPIRY_SECONDS = 600
QUERY_MAX_RETRIES = 3
TIME_SINCE_LAST_QUERY = time.time()

def get_model_output(model: LLM, messages: list[dict], verbose=True):
    global TIME_SINCE_LAST_QUERY
    while time.time() < TIME_SINCE_LAST_QUERY + QUERY_TIMEOUT:
        time.sleep(0.05)    
    last_error = None
    for attempt in range(QUERY_MAX_RETRIES):
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(model.query, messages)
            timeout = None if attempt == QUERY_MAX_RETRIES - 1 else QUERY_EXPIRY_SECONDS
            try:
                reasoning, content = future.result(timeout=timeout)
            finally:
                executor.shutdown(wait=False)

            if verbose:
                for m in messages:
                    cprint(f"[{m['role']}]: {m['content']}", "yellow")
                cprint(reasoning, "green")
                cprint(content, "red")

            TIME_SINCE_LAST_QUERY = time.time()
            return reasoning, content
        except concurrent.futures.TimeoutError as exc:
            last_error = exc
            cprint("[system] Model query timed out, retrying...", "red")
        except Exception as exc:
            last_error = exc
            cprint(f"[system] Model query failed: {exc}. Retrying...", "red")

        TIME_SINCE_LAST_QUERY = time.time()
    raise RuntimeError("Model query failed after retries") from last_error

def parse_model_output(output: str):
    cleaned = output.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    payload = json.loads(cleaned)
    feedbacks = payload.get("feedbacks", [])
    return [str(item).strip() for item in feedbacks if str(item).strip()]

def add_pruned_entry(entry_key: str, entry_value: str) -> None:
    pruned_lorebook["metadata"].append({"key": entry_key, "value": entry_value})
    if key_to_vector:
        vector = key_to_vector.get(entry_key)
        if vector is None:
            raise KeyError(f"Missing vector for key: {entry_key}")
        pruned_lorebook["vectors"].append(vector)

llm = LLM(API_KEY, API_URL, GEN_CONF, MODEL)
with open(LOREBOOK_PATH, "rb") as f:
    data = pickle.load(f)
pruned_lorebook = data.copy()
pruned_lorebook["metadata"] = []
pruned_lorebook["vectors"] = []

key_to_vector = {}
original_vectors = data.get("vectors", [])
if original_vectors and len(original_vectors) == len(data.get("metadata", [])):
    for entry, vector in zip(data["metadata"], original_vectors):
        key = entry.get("key")
        if key and key not in key_to_vector:
            key_to_vector[key] = vector

# find all unique keys
keys_list = []
for fd in data["metadata"]:
    if fd["key"] not in keys_list:
        keys_list.append(fd["key"])

for key in keys_list:
    print(colored("[system] ", "blue") + f"Processing key: {key}")
    if "TEMPLATE" not in key: 
        instances = [] # grab all instances of the key
        for fb in data["metadata"]:
            if fb["key"] == key:
                instances.append(fb["value"])
        
        all_feedback = "\n".join([f"- {instance}" for instance in instances])
        abstract = key.split(" ")[0]
        code_template = None
        if f"TEMPLATE {abstract}" in keys_list: # Get the template code as a reference
            for fb in data["metadata"]:
                if fb["key"] == f"TEMPLATE {abstract}":
                    code_template = fb["value"]
                    break
        messages = [
            {"role": "system", "content": llm.generate_system_prompt()},
            {"role": "user", "content": llm.generate_followup_prompt(feedback = all_feedback, template = code_template)},
        ]
        print(colored("[system] ", "blue") + f"Querying LLM to prune {len(instances)} feedbacks...")
        reasoning, content = get_model_output(llm, messages)
        assert content is not None, "Model did not return any content"

        new_feedback = parse_model_output(content)
        for fb in new_feedback:
            add_pruned_entry(key, fb)
        
    elif "TEMPLATE" in key: # directly copy over the code template
        code_template = None
        for fb in data["metadata"]:
            if fb["key"] == key:
                code_template = fb["value"]
                break
        assert code_template is not None, "Code template not found in lorebook"
        add_pruned_entry(key, code_template)

print(colored("[system] ", "blue") + f"new lorebook has {len(pruned_lorebook['metadata'])} entries, down from {len(data['metadata'])} entries.")
print(colored("[system] ", "blue") + "Pruned lorebook metadata:")
for entry in pruned_lorebook["metadata"]:
    print(f"- {entry['key']}: {entry['value']}")

time = time.strftime("%Y%m%d-%H%M%S")
with open(f"data/lorebook_pruned_{time}.pkl", "wb") as f:
    pickle.dump(pruned_lorebook, f)