from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt
from PIL import Image
import numpy as np

from core.embedders.sam_embedder import SAMEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from core.vector_db.faiss_index import FaissIndex


class SearchTab(QWidget):
    def __init__(self):
        super().__init__()
        self.sam_embedder = None
        self.yolo_embedder = None
        self.index = None
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
        self.lbl_preview.setMinimumHeight(200)
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
            self.lbl_preview.setText(f"Загружено: {filename}")
            self.btn_search.setEnabled(True)
            self.log.append(f"Изображение загружено: {filename}")

    def start_search(self):
        """Запуск поиска"""
        if not self.current_image_path:
            self.log.append("Изображение не загружено!")
            return

        self.log.append("Запуск поиска...")
        self.progress.setVisible(True)
        self.progress.setValue(30)

        try:
            # Здесь будет реальная логика поиска (пока заглушка)
            self.log.append("Извлечение эмбеддингов...")
            self.progress.setValue(60)

            # Заглушка результата
            self.log.append("Выполняется поиск в векторной базе...")
            self.progress.setValue(90)
            
            # Заглушка результата
            self.log.append("Поиск завершён. Найдено 5 похожих знаков.")
        except Exception as e:
            self.log.append(f"Ошибка при поиске: {e}")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)