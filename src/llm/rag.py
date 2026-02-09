import os
import requests
import numpy as np 
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.metadata = []
        
    def add(self, key: str, value: str):
        vector = self.model.encode(key)
        self.vectors.append(vector)
        self.metadata.append({
            "key": key,
            "value": value
        })
    
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
                # the top_indicies are sorted, so we can break early
                break
            results.append({
                "key": self.metadata[idx]["key"],
                "value": self.metadata[idx]["value"],
                "score": float(similarities[idx])
            })
            
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