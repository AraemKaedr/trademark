import faiss
import numpy as np
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, dimension=768, index_path="indexes/logo_index.faiss"):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index = faiss.IndexFlatIP(dimension)  # Скалярное произведение = косинусное подобие
        self.mapping = {}  # index_id -> image_path
        
    def add(self, embeddings: np.ndarray, image_paths: list):
        """Добавление эмбеддингов в индекс"""
        if len(embeddings) == 0:
            logger.warning("Пустой массив эмбеддингов")
            return
        
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        start_idx = len(self.mapping)
        self.index.add(embeddings)
        
        for i, path in enumerate(image_paths):
            self.mapping[start_idx + i] = str(path)

        self.save()
        logger.info(f"Добавлено {len(embeddings)} эмбеддингов в индекс")
            
    def search(self, query_embedding: np.ndarray, k=5):
        """Поиск k ближайших соседей"""
        if self.index.ntotal == 0:
            logger.warning("Индекс пустой!")
            return np.array([]), np.array([])
        
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]
    
    def save(self):
        """Сохранение индекса и маппинга"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.mapping, f)
        logger.info(f"Индекс сохранён: {self.index_path}")

    def load(self):
        """Загрузка индекса"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            pkl_path = self.index_path.with_suffix('.pkl')
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    self.mapping = pickle.load(f)
            logger.info(f"Индекс загружен ({self.index.ntotal} векторов)")
            return True
        return False

    def get_path_by_index(self, idx: int):
        """Получить путь к изображению по индексу"""
        return self.mapping.get(int(idx))