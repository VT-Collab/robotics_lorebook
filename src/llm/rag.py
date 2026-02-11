from dataclasses import dataclass
from typing import Literal
import json
import re
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import atexit
import os
import pickle
import json
import re
from dataclasses import dataclass
from typing import Literal
from termcolor import colored
from src.llm import LLM


class SimpleRAG:
    def __init__(self, filename="data/lorebook.pkl", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.metadata = []
        if filename:
            try:
                self.load_from_file(filename)
            except FileNotFoundError:
                pass

        def cleanup_function():
            if filename:
                print("saving lorebook before closing...")
                self.save_to_file(filename)

        atexit.register(cleanup_function)  # called when python exits

    def save_to_file(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="wb") as fh:
            pickle.dump(
                {
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                },
                fh,
            )

    def load_from_file(self, filename: str):
        with open(filename, mode="rb") as fh:
            data = pickle.load(fh)
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]

    def add(self, key: str, value: str):
        vector = self.model.encode(key)
        self.vectors.append(vector)
        self.metadata.append({"key": key, "value": value})

    def query(self, key, top_k=1, min_score=0.8):
        if not self.vectors:
            return ""
        query_vec = self.model.encode(key)
        # use cosine similarity as lookup
        similarities = [np.dot(query_vec, v) for v in self.vectors]
        if top_k != -1:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        else:
            top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            if float(similarities[idx]) < min_score:
                # the top_indices are sorted, so we can break early
                break
            results.append(
                {
                    "key": self.metadata[idx]["key"],
                    "value": self.metadata[idx]["value"],
                    "score": float(similarities[idx]),
                }
            )

        return results


@dataclass(frozen=True)
class Abstracts:
    action_type: Literal["PICK", "PLACE", "PULL", "PUSH"] = "PICK"
    object_type: Literal["stick", "sphere", "cube"] = "stick"
    # joint_type: Literal["free", "fixed", "revolute", "prismatic"] = "free"

    def __post_init__(self) -> None:
        if self.action_type not in {"PICK", "PLACE", "PULL", "PUSH"}:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        if self.object_type not in {"stick", "sphere", "cube"}:
            raise ValueError(f"Unsupported object_type: {self.object_type}")
        # if self.joint_type not in {"free", "fixed", "revolute", "prismatic"}:
        #     raise ValueError(f"Unsupported joint_type: {self.joint_type}")


API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gemini-3-flash-preview"  # "gemini-3-flash-preview" "gpt" "qwen3:32b"
RAG_CONFIG = "config/prompts/llm_rag.yml"


class HierarchicalRAG:
    def __init__(self, filename="data/lorebook_hier.json", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.data = {}
        self.vectors = {}
        if filename:
            try:
                self.load_from_file(filename)
            except FileNotFoundError:
                pass
        self.rag_llm = LLM(API_KEY, API_URL, RAG_CONFIG, MODEL)

        def cleanup_function():
            if filename:
                print("saving lorebook before closing...")
                self.save_to_file(filename)

        atexit.register(cleanup_function)  # called when python exits

    def save_to_file(self, filename: str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode="w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)

    def load_from_file(self, filename: str):
        with open(filename, mode="r", encoding="utf-8") as fh:
            self.data = json.load(fh)
        self._rebuild_vectors()

    def _rebuild_vectors(self):
        self.vectors = {}
        for action, primitives in self.data.items():
            self.vectors[action] = {}
            for primitive, objects in primitives.items():
                self.vectors[action][primitive] = {}
                for obj_name in objects.keys():
                    self.vectors[action][primitive][obj_name] = self.model.encode(obj_name)

    def _find_existing(self, obj_name: str):
        for action, primitives in self.data.items():
            for primitive, objects in primitives.items():
                if obj_name in objects:
                    return action, primitive, objects.get(obj_name, [])
        return None, "None", []

    def _find_existing_in_action(self, action: str, obj_name: str):
        for primitive, objects in self.data.get(action, {}).items():
            if obj_name in objects:
                return primitive, objects.get(obj_name, [])
        return "None", []

    def _split_key(self, key: str) -> tuple[str, str]:
        print(colored(f"Parsing key: {key}", "yellow"))
        match = re.search(r"\b([^\s(]+)\s*\(([^)]+)\)", key)
        if not match:
            raise ValueError("Key must be formatted as 'action(object)'.")
        action, obj_name = match.group(1).strip(), match.group(2).strip()
        if not action or not obj_name:
            raise ValueError("Key must be formatted as 'action(object)'.")
        return action, obj_name

    def _parse_abstract(self, abstract_text: str) -> Abstracts:
        cleaned = abstract_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        data = None
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        if data is None:
            parts = [
                part.strip()
                for part in cleaned.replace("\n", ",").split(",")
                if ":" in part
            ]
            data = {}
            for part in parts:
                key, value = part.split(":", 1)
                data[key.strip()] = value.strip().strip('"').strip("'")
        if not data:
            return Abstracts()
        return Abstracts(
            action_type=data.get("action_type", "PICK"),
            object_type=data.get("object_type", "stick"),
        )

    def add(self, key: str, value: str):
        _, obj_name = self._split_key(key)
        followup_prompt = self.rag_llm.generate_followup_prompt(
            task=key + value,
            current_abstract=current_abstract,
        )
        messages = [
            {"role": "system", "content": self.rag_llm.generate_system_prompt()},
            {"role": "user", "content": followup_prompt},
        ]
        reason, abstract_text = self.rag_llm.query(messages)
        # print(colored(f"Reasoning for {key}: {reason}", "yellow"))
        print(colored(f"Generated abstract for {key}: {abstract_text}", "cyan"))
        abstract = self._parse_abstract(abstract_text)
        primitive = abstract.object_type
        action = abstract.action_type

        current_abstract, existing_values = self._find_existing_in_action(action, obj_name)
        if current_abstract != "None" and current_abstract != primitive:
            del self.data[action][current_abstract][obj_name]
            if not self.data[action][current_abstract]:
                del self.data[action][current_abstract]
            if action in self.vectors and current_abstract in self.vectors[action]:
                self.vectors[action][current_abstract].pop(obj_name, None)
                if not self.vectors[action][current_abstract]:
                    del self.vectors[action][current_abstract]

        self.data.setdefault(action, {})
        self.data[action].setdefault(primitive, {})
        if obj_name not in self.data[action][primitive]:
            self.data[action][primitive][obj_name] = []
        if existing_values and current_abstract != primitive:
            self.data[action][primitive][obj_name].extend(existing_values)
        self.data[action][primitive][obj_name].append(value)

        self.vectors.setdefault(action, {})
        self.vectors[action].setdefault(primitive, {})
        self.vectors[action][primitive][obj_name] = self.model.encode(obj_name)

    def query(self, key, top_k=1, min_score=0.7):
        _, obj_name = self._split_key(key)
        _, current_abstract, _ = self._find_existing(obj_name)
        followup_prompt = self.rag_llm.generate_followup_prompt(
            task=key,
            current_abstract=current_abstract,
        )
        messages = [
            {"role": "system", "content": self.rag_llm.generate_system_prompt()},
            {"role": "user", "content": followup_prompt},
        ]
        _, abstract_text = self.rag_llm.query(messages)
        abstract = self._parse_abstract(abstract_text)
        print(colored(f"Generated abstract for query {key}: {abstract_text}", "cyan"))
        action = abstract.action_type
        if action not in self.vectors:
            return ""
        query_vec = self.model.encode(obj_name)

        best_primitive = None
        best_score = None
        for primitive, obj_vectors in self.vectors[action].items():
            if not obj_vectors:
                continue
            similarities = [float(np.dot(query_vec, v)) for v in obj_vectors.values()]
            avg_score = float(np.mean(similarities))
            if best_score is None or avg_score > best_score:
                best_score = avg_score
                best_primitive = primitive

        if best_primitive is None or best_score is None or best_score < min_score:
            return ""

        results = []
        for obj_key, feedback in self.data[action][best_primitive].items():
            results.append(
                {
                    "key": obj_key,
                    "value": feedback,
                    "score": float(best_score),
                }
            )
        return results
    
class FunctionStoreRAG(SimpleRAG):
    def __init__(self, filename="data/lorebook.pkl", model_name="all-MiniLM-L6-v2"):
        super().__init__(filename=filename, model_name=model_name)

    def add(self, key: str, value: str = None, function: str = None):
        vector = self.model.encode(key)
        self.vectors.append(vector)
        if function:
            self.metadata.append({"key": key, "function": function})
        else:
            self.metadata.append({"key": key, "value": value})

    def query(self, key, top_k=1, min_score=0.8):
        if not self.vectors:
            return ""
        query_vec = self.model.encode(key)
        # use cosine similarity as lookup
        similarities = [np.dot(query_vec, v) for v in self.vectors]
        if top_k != -1:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        else:
            top_indices = np.argsort(similarities)[::-1]
        results = []
        for idx in top_indices:
            if float(similarities[idx]) < min_score:
                # the top_indicies are sorted, so we can break early
                break
            results.append(
                {
                    "key": self.metadata[idx]["key"],
                    "value": self.metadata[idx]["value"],
                    "score": float(similarities[idx]),
                }
            )
        return results



api_key = os.environ.get("ARC_API_KEY")


def upload_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        response = requests.post(
            "https://llm-api.arc.vt.edu/api/v1/files/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            files={"file": file},
        )

    if response.status_code == 200:
        data = response.json()
        file_id = data.get("id")
        if file_id:
            print(f"Uploaded {file_path} successfully! File ID: {file_id}")
            return file_id
        else:
            raise RuntimeError("Upload succeeded but no file id returned.")
    else:
        raise RuntimeError(
            f"Failed to upload {file_path}. Status code: {response.status_code}"
        )


def wait_for_file_processing(file_id, timeout=300, poll_interval=2):
    """
    Wait for a file to finish processing.

    Returns:
        dict: Final status with 'status' key ('completed' or 'failed')

    Raises:
        TimeoutError: If processing doesn't complete within timeout
    """
    url = f"https://llm-api.arc.vt.edu/api/v1/files/{file_id}/process/status"
    headers = {"Authorization": f"Bearer {api_key}"}

    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get("status")
        print(colored(f"File processing status: {status}", "cyan"))

        if status == "completed":
            return result
        elif status == "failed":
            raise Exception(f"File processing failed: {result.get('error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"File processing did not complete within {timeout} seconds")
