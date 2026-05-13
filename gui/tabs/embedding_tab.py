from gui.tabs.base_tab import BaseTab
from core.embedders.sam_embedder import SAMEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
import numpy as np
from pathlib import Path


class EmbeddingTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.sam_embedder = SAMEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        
        # Отдельные индексы для каждой модели
        self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        
        self.init_ui()

    def init_ui(self):
        title = QLabel("Извлечение эмбеддингов (SAM + YOLO)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, title)

        # Кнопки
        btn_layout = QHBoxLayout()

        self.btn_sam = QPushButton("Извлечь эмбеддинги SAM")
        self.btn_sam.clicked.connect(self.extract_sam)
        btn_layout.addWidget(self.btn_sam)

        self.btn_yolo = QPushButton("Извлечь эмбеддинги YOLO")
        self.btn_yolo.clicked.connect(self.extract_yolo)
        btn_layout.addWidget(self.btn_yolo)

        self.btn_both = QPushButton("Извлечь эмбеддинги обеих моделей")
        self.btn_both.clicked.connect(self.extract_both)
        btn_layout.addWidget(self.btn_both)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

    def extract_sam(self):
        self._extract_embeddings("sam")

    def extract_yolo(self):
        self._extract_embeddings("yolo")

    def extract_both(self):
        self._extract_embeddings("both")

    def _extract_embeddings(self, mode: str):
        self.log_message(f"Начато извлечение эмбеддингов: {mode.upper()}")
        self.progress.setVisible(True)
        self.progress.setValue(5)

        try:
            from core.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            processed_dir = preprocessor.processed_dir

            if not processed_dir.exists() or len(list(processed_dir.glob("*.jpg"))) == 0:
                self.log_message("Сначала выполните предобработку датасета на вкладке 1!", "ERROR")
                return

            image_paths = sorted(list(processed_dir.glob("*.jpg")))
            self.log_message(f"Найдено изображений: {len(image_paths)}")

            sam_embeddings = []
            yolo_embeddings = []
            paths_to_save = []

            for i, img_path in enumerate(image_paths):
                progress = 10 + int((i / len(image_paths)) * 80)
                self.progress.setValue(progress)
                self.log_message(f"Обработка {i+1}/{len(image_paths)}: {img_path.name}")

                if mode in ["sam", "both"]:
                    sam_emb = self.sam_embedder.get_embedding(str(img_path))
                    sam_embeddings.append(sam_emb)

                if mode in ["yolo", "both"]:
                    yolo_emb = self.yolo_embedder.get_embedding(str(img_path))
                    yolo_embeddings.append(yolo_emb)

                paths_to_save.append(img_path)

            # Сохранение в индексы
            if mode in ["sam", "both"] and sam_embeddings:
                self.sam_index.add(np.array(sam_embeddings), paths_to_save)
                self.log_message(f"SAM эмбеддинги сохранены ({len(sam_embeddings)})", "УСПЕХ")

            if mode in ["yolo", "both"] and yolo_embeddings:
                self.yolo_index.add(np.array(yolo_embeddings), paths_to_save)
                self.log_message(f"YOLO эмбеддинги сохранены ({len(yolo_embeddings)})", "УСПЕХ")

            self.log_message("Извлечение эмбеддингов успешно завершено!", "УСПЕХ")

        except Exception as e:
            self.log_message(f"Критическая ошибка: {e}", "ERROR")
            import traceback
            self.log_message(traceback.format_exc(), "ERROR")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)