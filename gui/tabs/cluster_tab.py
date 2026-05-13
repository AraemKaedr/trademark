from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClusterTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        self.init_ui()

    def init_ui(self):
        label = QLabel("Кластеризация векторных представлений")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, label)

        btn_layout = QHBoxLayout()

        self.btn_sam = QPushButton("Кластеризация SAM")
        self.btn_sam.clicked.connect(lambda: self.run_clustering("sam"))
        btn_layout.addWidget(self.btn_sam)

        self.btn_yolo = QPushButton("Кластеризация YOLO")
        self.btn_yolo.clicked.connect(lambda: self.run_clustering("yolo"))
        btn_layout.addWidget(self.btn_yolo)

        self.btn_both = QPushButton("Кластеризация обоих")
        self.btn_both.clicked.connect(lambda: self.run_clustering("both"))
        btn_layout.addWidget(self.btn_both)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

    def run_clustering(self, mode: str):
        self.log_message(f"Запуск кластеризации: {mode.upper()}")
        self.progress.setVisible(True)
        self.progress.setValue(10)

        try:
            if mode in ["sam", "both"]:
                self._perform_clustering(self.sam_index, "SAM")
            if mode in ["yolo", "both"]:
                self._perform_clustering(self.yolo_index, "YOLO")

            self.log_message(f"Кластеризация {mode.upper()} успешно завершена", "УСПЕХ")
        except Exception as e:
            self.log_message(f"Ошибка кластеризации: {e}", "ERROR")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)

    def _perform_clustering(self, index: FaissIndex, model_name: str):
        embeddings = index.get_all_embeddings()
        
        if len(embeddings) < 3:
            self.log_message(f"Недостаточно данных для {model_name} кластеризации (нужно минимум 3 изображения)", "WARNING")
            return

        # Определяем оптимальное число кластеров
        n_clusters = min(10, max(3, len(embeddings) // 30))
        self.log_message(f"{model_name}: {len(embeddings)} векторов → {n_clusters} кластеров")

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Оценка качества
        try:
            silhouette = silhouette_score(embeddings, labels)
            self.log_message(f"Оценка качества ({model_name}): {silhouette:.3f}")
        except:
            pass

        # Распределение по кластерам
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            self.log_message(f"  Кластер {u}: {c} логотипов")

        self.log_message(f"Кластеризация {model_name} завершена")