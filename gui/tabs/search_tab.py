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
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Группа загрузки
        group = QGroupBox("Поиск графического товарного знака")
        group_layout = QVBoxLayout()

        self.btn_load = QPushButton("Загрузить изображение для проверки")
        self.btn_load.clicked.connect(self.load_image)
        group_layout.addWidget(self.btn_load)

        self.lbl_preview = QLabel("Изображение не загружено")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if file_path:
            self.current_image_path = file_path
            self.lbl_preview.setText(f"Загружено: {Path(file_path).name}")
            self.btn_search.setEnabled(True)
            self.log.append(f"Изображение загружено: {file_path}")

    def start_search(self):
        self.log.append("Запуск поиска...")
        self.progress.setVisible(True)
        self.progress.setValue(30)

        # Здесь будет реальная логика
        self.log.append("Эмбеддинги извлечены. Выполняется поиск...")
        self.progress.setValue(70)

        # Заглушка результата
        self.log.append("Найдено 5 похожих знаков.")
        self.progress.setValue(100)
        self.progress.setVisible(False)