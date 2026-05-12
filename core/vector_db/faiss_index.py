import faiss
import numpy as np
import pickle
from pathlib import Path

class FaissIndex:
    def __init__(self, dimension=768, index_path="indexes/logo_index.faiss"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index = faiss.IndexFlatIP(dimension)  # косинусное сходство
        self.mapping = {}  # id -> filename
        
    def add(self, embeddings: np.ndarray, filenames: list):
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        start_id = len(self.mapping)
        for i, name in enumerate(filenames):
            self.mapping[start_id + i] = name
            
    def search(self, query_embedding: np.ndarray, k=5):
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]
    
    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.mapping, f)

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.index_path.with_suffix('.pkl'), 'rb') as f:
                self.mapping = pickle.load(f)