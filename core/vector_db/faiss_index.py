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
        self.pkl_path = self.index_path.with_suffix('.pkl')
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.embeddings = None  # np.array всех векторов
        self.mapping = {}       # index_id -> image_path
        self.is_loaded = False

        self.load()

    def load(self) -> bool:
        """Загрузка существующего индекса или создание нового, а также загрузка сохранённых эмбеддингов"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Пытаемся загрузить существующий индекс
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                
                # Загружаем сохранённые эмбеддинги
                if self.embeddings_path.exists():
                    self.embeddings = np.load(self.embeddings_path)
                    logger.info(f"Загружено {len(self.embeddings)} эмбеддингов из файла {self.embeddings_path.name}")
                else:
                    self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
                    logger.warning(f"Файл эмбеддингов {self.embeddings_path.name} не найден")
                
                # Загружаем mapping (соответствие id и путь к файлу)
                if self.pkl_path.exists():
                    with open(self.pkl_path, 'rb') as f:
                        self.mapping = pickle.load(f)
                else:
                    self.mapping = {}   
                
                self.is_loaded = True
                logger.info(f"{self.index_path.name} успешно загружен ({self.index.ntotal} векторов)")
                return True
            except Exception as e:
                logger.error(f"Ошибка! Не удалось загрузить индекс {self.index_path.name}: {e}")
                logger.info(" Создаётся новый пустой индекс...")
        
        # Если индекс не существует или ошибка — создаём новый
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
        logger.info(f"Добавлено {len(embeddings)} эмбеддинговв {self.index_path.name}")

    def save(self):
        """Сохранение всего"""
        faiss.write_index(self.index, str(self.index_path))
        if self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)
        
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(self.mapping, f)
        
        logger.info(f"Сохранён индекс {self.index_path.name} + эмбеддинги ({len(self.embeddings)} векторов)")

    def get_all_embeddings(self) -> np.ndarray:
        """Возвращает все сохранённые эмбеддинги (с повторной попыткой загрузки)"""
        if self.embeddings is None or len(self.embeddings) == 0:
            # Если эмбеддинги не загружены — пытаемся загрузить ещё раз
            if self.embeddings_path.exists():
                try:
                    self.embeddings = np.load(self.embeddings_path)
                    logger.info(f"Повторно загружено {len(self.embeddings)} эмбеддингов")
                except Exception as e:
                    logger.error(f"Не удалось загрузить эмбеддинги: {e}")
                    self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
            else:
                self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
        return self.embeddings

    def get_path_by_index(self, idx: int):
        """Получить путь к изображению по его индексу в базе"""
        return self.mapping.get(int(idx))

    def get_total_vectors(self) -> int:
        """Количество векторов в индексе"""
        return self.index.ntotal if self.index is not None else 0