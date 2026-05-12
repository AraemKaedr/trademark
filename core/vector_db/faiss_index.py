import faiss
import numpy as np
import pickle
from pathlib import Path

class FaissIndex:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine после нормализации
        self.id_to_path = {}
        
    def add(self, embeddings, paths):
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        for i, path in enumerate(paths):
            self.id_to_path[len(self.id_to_path)] = path
            
    def search(self, query_embedding, k=5):
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]