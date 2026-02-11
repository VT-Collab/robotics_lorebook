import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import atexit
import os
import pickle
from termcolor import colored


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
