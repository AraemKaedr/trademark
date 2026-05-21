from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from core.embedders.resnet_embedder import ResNetEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from core.vector_db.faiss_index import FaissIndex


class SearchTab(QWidget):
    def __init__(self, results_tab=None):
        super().__init__()
        self.resnet_embedder = ResNetEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        self.resnet_index = FaissIndex(dimension=2048, index_path="indexes/resnet_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        self.results_tab = results_tab # Ссылка на вкладку Результаты
        self.current_image_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        group = QGroupBox("Поиск графического товарного знака")
        g_layout = QVBoxLayout()

        self.btn_load = QPushButton("Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)
        g_layout.addWidget(self.btn_load)

        self.lbl_preview = QLabel("Изображение не загружено")
        self.lbl_preview.setMinimumHeight(260)
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        g_layout.addWidget(self.lbl_preview)

        self.btn_search = QPushButton("Запустить поиск")
        self.btn_search.clicked.connect(self.start_search)
        self.btn_search.setEnabled(False)
        g_layout.addWidget(self.btn_search)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        g_layout.addWidget(self.progress)

        group.setLayout(g_layout)
        layout.addWidget(group)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if path:
            self.current_image_path = path
            pix = QPixmap(path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
            self.lbl_preview.setPixmap(pix)
            self.btn_search.setEnabled(True)
            self.log.append(f"Загружено: {Path(path).name}")

    def start_search(self):
        """Запуск поиска"""
        if not self.current_image_path:
            self.log.append("Изображение не загружено!")
            return

        self.log.append("Запуск поиска...")
        self.progress.setVisible(True)
        self.progress.setValue(30)

        try:
            # Извлечение эмбеддингов ResNet
            self.log.append("Извлечение эмбеддингов ResNet...")
            resnet_emb = self.resnet_embedder.get_embedding(self.current_image_path)
            self.progress.setValue(45)

            # Извлечение эмбеддингов YOLO
            self.log.append("Извлечение эмбеддингов YOLO...")
            yolo_emb = self.yolo_embedder.get_embedding(self.current_image_path)
            self.progress.setValue(70)

            # Поиск в обоих индексах
            self.log.append("Поиск по ResNet...")
            resnet_dist, resnet_idx = self.resnet_index.search(resnet_emb, k=6)  # берём на 1 больше
            
            self.log.append("Поиск по YOLO...")
            yolo_dist, yolo_idx = self.yolo_index.search(yolo_emb, k=6)

            # Оригинальность
            # Берём второй максимум (или максимум, если self не найден)
            resnet_max_sim = max(resnet_dist)
            if resnet_max_sim > 0.98:  # очень высокая схожесть = self
                originality = (1 - sorted(resnet_dist)[-2]) * 100 if len(resnet_dist) > 1 else 85.0
            else:
                originality = (1 - resnet_max_sim) * 100

            # Передаём результаты во вкладку "Результаты"
            if self.results_tab:
                self.results_tab.show_results(
                    query_path=self.current_image_path,  # Убирает из результата лого запроса
                    resnet_dist=resnet_dist,
                    resnet_idx=resnet_idx,
                    yolo_dist=yolo_dist,
                    yolo_idx=yolo_idx,
                    originality=originality,
                    resnet_index=self.resnet_index,
                    yolo_index=self.yolo_index
                )
                self.log.append(f"Оригинальность: {originality:.1f}%")
                
                self.log.append("Результаты переданы во вкладку 6. Результаты")
            else:
                self.log.append("Вкладка Результатов не подключена")
            
            self.log.append("Поиск успешно завершён!")

        except Exception as e:
            self.log.append(f"Ошибка поиска: {e}")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)