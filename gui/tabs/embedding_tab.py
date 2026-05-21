from gui.tabs.base_tab import BaseTab
from core.embedders.resnet_embedder import ResNetEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from core.vector_db.faiss_index import FaissIndex
from gui.workers import EmbeddingWorker
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt
import numpy as np
from pathlib import Path
import time


class EmbeddingTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.resnet_embedder = ResNetEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        
        # Отдельные индексы для каждой модели
        self.resnet_index = FaissIndex(dimension=2048, index_path="indexes/resnet_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        
        self.current_worker = None
        self.start_time = None
        self.init_ui()

    def init_ui(self):
        title = QLabel("Извлечение эмбеддингов (ResNet50 + YOLO)")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, title)

        # Кнопки
        btn_layout = QHBoxLayout()

        self.btn_resnet = QPushButton("Извлечь эмбеддинги ResNet50")
        self.btn_resnet.clicked.connect(self.extract_resnet)
        btn_layout.addWidget(self.btn_resnet)

        self.btn_yolo = QPushButton("Извлечь эмбеддинги YOLO")
        self.btn_yolo.clicked.connect(self.extract_yolo)
        btn_layout.addWidget(self.btn_yolo)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)
    
    def _start_extraction(self, mode: str):
        if self.current_worker and self.current_worker.isRunning():
            self.log_message("Извлечение уже выполняется! Подождите завершения.", "WARNING")
            return

        from core.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        processed_dir = preprocessor.processed_dir

        image_paths = sorted(list(processed_dir.glob("*.jpg")))
        if not image_paths:
            self.log_message("Нет обработанных изображений в data/processed/ !", "ERROR")
            return

        self.start_time = time.time()
        self.log_message(f"Запущено извлечение {mode.upper()} для {len(image_paths)} изображений...")
        self.progress.setVisible(True)
        self.progress.setValue(0)

        try:
            if mode == "resnet":
                embedder = self.resnet_embedder
                index = self.resnet_index
            elif mode == "yolo":
                embedder = self.yolo_embedder
                index = self.yolo_index
        except Exception as e:
            self.log_message(f"Ошибка кластеризации: {e}", "ERROR")
        
        # Запускаем извлечение эмбеддингов в отдельном потоке
        self.current_worker = EmbeddingWorker(embedder, image_paths, mode)
        self.current_worker.progress.connect(self._update_progress)
        self.current_worker.log.connect(self.log_message)
        self.current_worker.finished.connect(lambda embs: self._on_extraction_finished(mode, embs, image_paths, index))
        
        self.current_worker.start()

    def _update_progress(self, value: int):
        """Обновление прогресса + оценка времени"""
        self.progress.setValue(value)
        
        if value > 0 and self.start_time:
            elapsed = time.time() - self.start_time
            if value > 0:
                estimated_total = elapsed * (100 / value)
                remaining = estimated_total - elapsed
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                self.log_message(f"Прогресс: {value}% | Осталось ~{mins} мин {secs} сек")
    
    def extract_resnet(self):
        self._start_extraction("resnet")

    def extract_yolo(self):
        self._start_extraction("yolo")

    def _on_extraction_finished(self, mode: str, embeddings, image_paths, index):
        """Вызывается после завершения потока"""
        self.progress.setValue(100)
        self.progress.setVisible(False)
        image_paths = image_paths
        
        if not embeddings:
            self.log_message("Не удалось извлечь эмбеддинги", "ERROR")
            return

        try:
            if mode == "resnet":
                self.resnet_index.add(np.array(embeddings), self.current_worker.image_paths)
            elif mode == "yolo":
                self.yolo_index.add(np.array(embeddings), self.current_worker.image_paths)
            self.log_message(f"Кластеризация {mode.upper()} успешно завершена!", "УСПЕХ")
            self.log_message(f"{mode.upper()} эмбеддинги успешно сохранены ({len(embeddings)} векторов)", "УСПЕХ")
        except Exception as e:
            self.log_message(f"Ошибка сохранения в индекс: {e}", "ERROR")
        finally:
            self.current_worker = None   # Освобождаем worker