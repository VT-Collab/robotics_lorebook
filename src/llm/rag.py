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
                # the top_indicies are sorted, so we can break early
                break
            results.append({
                "key": self.metadata[idx]["key"],
                "value": self.metadata[idx]["value"],
                "score": float(similarities[idx])
            })
            
        return results