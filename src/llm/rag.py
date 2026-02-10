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
from termcolor import colored
from src.llm import LLM
from termcolor import cprint


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

    def query(self, key, top_k=1, min_score=0.7):
        if not self.vectors:
            return ""
        query_vec = self.model.encode(key)
        # use cosine similarity as lookup
        similarities = [np.dot(query_vec, v) for v in self.vectors]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
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
    object_type: Literal["stick", "sphere", "cube"] = "stick"
    joint_type: Literal["free", "fixed", "revolute", "prismatic"] = "free"
    interaction_type: Literal["pick", "place", "push", "pull"] = "pick"

    def __post_init__(self) -> None:
        if self.object_type not in {"stick", "sphere", "cube"}:
            raise ValueError(f"Unsupported object_type: {self.object_type}")
        if self.joint_type not in {"free", "fixed", "revolute", "prismatic"}:
            raise ValueError(f"Unsupported joint_type: {self.joint_type}")
        if self.interaction_type not in {"pick", "place", "push", "pull"}:
            raise ValueError(f"Unsupported interaction_type: {self.interaction_type}")
        
API_KEY = os.environ.get("ARC_API_KEY", "YOUR_API_KEY_HERE")
API_URL = "https://llm-api.arc.vt.edu/api/v1"
MODEL = "gpt" #"gemini-3-flash-preview" #"gpt" #"qwen3:32b"
RAG_CONFIG = "config/prompts/llm_rag.yml"
        
class GenericKeyRAG: #test
    def __init__(self, filename="data/lorebook_generic.pkl", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.metadata = {}
        self.keys = []
        if filename:
            try:
                self.load_from_file(filename)
            except FileNotFoundError:
                pass
        # initiate LLM 
        self.rag_llm = LLM(API_KEY, API_URL, RAG_CONFIG, MODEL)
        


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
                    "keys": self.keys,
                },
                fh,
            )

    def load_from_file(self, filename: str):
        with open(filename, mode="rb") as fh:
            data = pickle.load(fh)
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]
        self.keys = data.get("keys", [])
        if self.keys and self.vectors:
            seen = set()
            deduped_keys = []
            deduped_vectors = []
            for key, vector in zip(self.keys, self.vectors):
                if key in seen:
                    continue
                seen.add(key)
                deduped_keys.append(key)
                deduped_vectors.append(vector)
            self.keys = deduped_keys
            self.vectors = deduped_vectors

    def _parse_abstract(self, abstract_text: str) -> Abstracts:
        cprint(f"Raw abstract text: {abstract_text}", "red")
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
            object_type=data.get("object_type", "stick"),
            joint_type=data.get("joint_type", "free"),
            interaction_type=data.get("interaction_type", "pick"),
        )

    def add(self, key: str, value: str):
        current_abstract = self.metadata.get(key, "None")
        followup_prompt = self.rag_llm.generate_followup_prompt(task=key + value,
                                                                current_abstract=current_abstract)
        messages = [
            {"role": "system", "content": self.rag_llm.generate_system_prompt()},
            {"role": "user", "content": followup_prompt},
        ]
        _, abstract_text = self.rag_llm.query(messages)
        abstract = self._parse_abstract(abstract_text)
        self.metadata[key] = abstract
        if abstract not in self.metadata.keys():
            self.metadata[abstract] = [value]
            if current_abstract != "None":
                self.metadata[abstract] += self.metadata.get(current_abstract, [])
        else:
            self.metadata[abstract].append(value)
            if current_abstract != abstract and current_abstract != "None":
                self.metadata[abstract] += self.metadata.get(current_abstract, [])
        
        vector = self.model.encode(key)
        if key not in self.keys:
            self.vectors.append(vector)
            self.keys.append(key)

    def query(self, key, top_k=1, min_score=0.8):
        if not self.vectors:
            return ""
        query_vec = self.model.encode(key)
        # use cosine similarity as lookup
        similarities = [np.dot(query_vec, v) for v in self.vectors]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if float(similarities[idx]) < min_score:
                # the top_indices are sorted, so we can break early
                break
            if idx >= len(self.keys):
                continue
            matched_key = self.keys[idx]
            abstract = self.metadata.get(matched_key)
            results.append(
                {
                    "key": matched_key,
                    "abstract": abstract,
                    "value": self.metadata.get(abstract, []),
                    "score": float(similarities[idx]),
                }
            )

        return results
    

api_key= os.environ.get("ARC_API_KEY")
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
        raise RuntimeError(f"Failed to upload {file_path}. Status code: {response.status_code}")
    

def wait_for_file_processing(file_id, timeout=300, poll_interval=2):
    """
    Wait for a file to finish processing.
    
    Returns:
        dict: Final status with 'status' key ('completed' or 'failed')
    
    Raises:
        TimeoutError: If processing doesn't complete within timeout
    """
    url = f'https://llm-api.arc.vt.edu/api/v1/files/{file_id}/process/status'
    headers = {'Authorization': f'Bearer {api_key}'}
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(url, headers=headers)
        result = response.json()
        status = result.get('status')
        print(colored(f"File processing status: {status}", 'cyan'))
        
        if status == 'completed':
            return result
        elif status == 'failed':
            raise Exception(f"File processing failed: {result.get('error')}")
        
        time.sleep(poll_interval)
    
    raise TimeoutError(f"File processing did not complete within {timeout} seconds")
        
