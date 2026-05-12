from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from core.embedders.sam_embedder import SAMEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from core.vector_db.faiss_index import FaissIndex


class SearchTab(QWidget):
    def __init__(self, results_tab=None):
        super().__init__()
        self.sam_embedder = SAMEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        self.index = FaissIndex()
        self.results_tab = results_tab  # Ссылка на вкладку Результаты
        self.current_image_path = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Группа загрузки изображения
        group = QGroupBox("Поиск графического товарного знака")
        group_layout = QVBoxLayout()

        self.btn_load = QPushButton("Загрузить изображение для проверки")
        self.btn_load.clicked.connect(self.load_image)
        group_layout.addWidget(self.btn_load)

        self.lbl_preview = QLabel("Изображение не загружено")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumHeight(260)
        group_layout.addWidget(self.lbl_preview)

        self.btn_search = QPushButton("Запустить поиск")
        self.btn_search.clicked.connect(self.start_search)
        self.btn_search.setEnabled(False)
        group_layout.addWidget(self.btn_search)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        group_layout.addWidget(self.progress)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Лог
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def load_image(self):
        """Загрузка изображения для проверки"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_path = file_path
            filename = Path(file_path).name
            pixmap = QPixmap(file_path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_preview.setPixmap(pixmap)
            self.btn_search.setEnabled(True)
            self.log.append(f"Изображение загружено: {filename}")

    def start_search(self):
        """Запуск поиска"""
        if not self.current_image_path:
            self.log.append("Изображение не загружено!")
            return

        self.log.append("Запуск поиска...")
        self.progress.setVisible(True)
        self.progress.setValue(20)

        try:
            # 1. Извлечение эмбеддингов
            self.log.append("Извлечение эмбеддингов (SAM + YOLO)...")
            self.progress.setValue(50)

            # Реальное извлечение
            query_emb = self._get_combined_embedding(self.current_image_path)

            # 2. Поиск похожих изображений в векторной базе (в индексе)
            self.log.append("Поиск ближайших соседей в векторной базе...")
            self.progress.setValue(75)
            
            distances, indices = self.index.search(query_emb, k=5)

            # 3. Передаём результаты во вкладку "Результаты"
            if self.results_tab:
                self.results_tab.show_results(self.current_image_path, distances, indices)
                self.log.append("Результаты переданы и будут доступны во вкладке «6. Результаты»")
            else:
                self.log.append("Вкладка Результатов не подключена")
            
            self.log.append("Поиск успешно завершён!")

        except Exception as e:
            self.log.append(f"Ошибка при поиске: {e}")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)
    
    def _get_combined_embedding(self, image_path: str) -> np.ndarray:
        """Комбинированный эмбеддинг от двух моделей SAM и YOLO"""
        sam_emb = self.sam_embedder.get_embedding(image_path)
        yolo_emb = self.yolo_embedder.get_embedding(image_path)
        
        # Объединяем и нормализуем
        combined = np.concatenate([sam_emb, yolo_emb])
        combined = combined.astype(np.float32)
        
        # Нормализация
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
            
        return combined