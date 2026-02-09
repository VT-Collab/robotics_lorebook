import numpy as np
from sentence_transformers import SentenceTransformer
import atexit
import os
import pickle


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
                filename,
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
