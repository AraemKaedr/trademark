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
        self.embeddings_path = self.index_path.with_suffix('.npy')
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.embeddings = None  # np.array всех векторов
        self.mapping = {}       # index_id -> image_path
        self.is_loaded = False

        self.load()

    def load(self) -> bool:
        """Загрузка индекса существующего индекса или создание нового, а также загрузка сохранённых эмбеддингов"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Пытаемся загрузить существующий индекс
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                
                # Загружаем сохранённые эмбеддинги
                if self.embeddings_path.exists():
                    self.embeddings = np.load(self.embeddings_path)
                else:
                    self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
                
                # Загружаем mapping (соответствие id → путь к файлу)
                pkl_path = self.index_path.with_suffix('.pkl')
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        self.mapping = pickle.load(f)
                else:
                    self.mapping = {}   
                
                self.is_loaded = True
                logger.info(f"{self.index_path.name} успешно загружен ({self.index.ntotal} векторов)")
                return True
            except Exception as e:
                logger.warning(f"Ошибка! Не удалось загрузить индекс {self.index_path.name}: {e}")
                logger.info(" Создаётся новый индекс...")

        # Создаём новый пустой индекс
        logger.info(f"Создаётся новый FAISS индекс (dimension={self.dimension})")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
        self.mapping = {}
        self.is_loaded = True

        return False

    def add(self, embeddings: np.ndarray, image_paths: list):
        """Добавление эмбеддингов + сохранение оригинальных векторов"""
        if len(embeddings) == 0:
            logger.warning("Пустой массив эмбеддингов")
            return

        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        start_idx = len(self.mapping)
        self.index.add(embeddings)

        # Сохраняем оригинальные эмбеддинги
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        for i, path in enumerate(image_paths):
            self.mapping[start_idx + i] = str(path)

        self.save()
        logger.info(f"Добавлено {len(embeddings)} эмбеддингов")

    def search(self, query_embedding: np.ndarray, k=5):
        """Поиск k ближайших соседей"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Индекс пустой!")
            return np.array([]), np.array([])
        
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def save(self):
        """Сохранение всего"""
        faiss.write_index(self.index, str(self.index_path))
        np.save(self.embeddings_path, self.embeddings)
        
        with open(self.index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.mapping, f)
        
        logger.info(f"Сохранён индекс + эмбеддинги ({len(self.embeddings)} векторов)")

    def get_all_embeddings(self) -> np.ndarray:
        """Возвращает все сохранённые эмбеддинги"""
        return self.embeddings if self.embeddings is not None else np.empty((0, self.dimension))

    def get_path_by_index(self, idx: int):
        """Получить путь к изображению по его индексу в базе"""
        return self.mapping.get(int(idx))

    def get_total_vectors(self) -> int:
        """Количество векторов в индексе"""
        return self.index.ntotal if self.index else 0