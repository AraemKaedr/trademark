from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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
        self.log_message(f"Запуск кластеризации для: {mode.upper()}")
        self.progress.setVisible(True)
        self.progress.setValue(10)

        try:
            if mode in ["sam", "both"]:
                self._perform_clustering(self.sam_index, "SAM")
            if mode in ["yolo", "both"]:
                self._perform_clustering(self.yolo_index, "YOLO")

            self.log_message(f"Кластеризация {mode.upper()} успешно завершена!", "УСПЕХ")
        except Exception as e:
            self.log_message(f"Ошибка кластеризации: {e}", "ERROR")
        finally:
            self.progress.setValue(100)
            self.progress.setVisible(False)

    def _perform_clustering(self, index: FaissIndex, model_name: str):
        """Полноценная кластеризация с реальными эмбеддингами"""
        embeddings = index.get_all_embeddings()
        n_vectors = len(embeddings)

        self.log_message(f"{model_name} индекс содержит {n_vectors} векторов")
        
        count_vectors = 10
        if n_vectors < count_vectors:
            self.log_message(f"Недостаточно данных для {model_name} кластеризации (нужно минимум {count_vectors} векторов изображений)", "WARNING")
            self.log_message("   Сначала перейдите во вкладку 2. Эмбеддинги и извлеките эмбеддинги", "WARNING")
            return

        # Нормализация данных перед кластеризацией (важно!)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Определяем оптимальное число кластеров
        n_clusters = min(15, max(4, n_vectors // 45))   # количество кластеров
        self.log_message(f"Выполняется кластеризация K-Means для {model_name}: {n_vectors} векторов | {n_clusters} кластеров")

        # K-Means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=1000)
        labels = kmeans.fit_predict(embeddings_scaled)

        # Подсчёт уникальных кластеров
        unique_labels = np.unique(labels)
        n_unique = len(unique_labels)

        if n_unique <= 2:
            self.log_message(f"KMeans не смог разделить данные и нашёл только {n_unique} кластеров (все точки в одном кластере)", "WARNING")
            self.log_message("   Причина: эмбеддинги слишком похожи друг на друга и данные слабо разделимы", "WARNING")
            self.log_message("   Рекомендация: улучшить качество эмбеддингов YOLO/SAM", "WARNING")
        else:
            self.log_message(f"Найдено {n_unique} уникальных кластеров из оптимальных {n_clusters}")
            
            # Оценка качества кластеризации (только если больше 1 кластера)
            if n_unique > 1:
                try:
                    silhouette = silhouette_score(embeddings_scaled, labels)
                    self.log_message(f"Оценка качества ({model_name}): {silhouette:.3f} (чем ближе к 1 — тем лучше кластеризация) (хорошо > 0.5, отлично > 0.7)")
                except Exception as e:
                    self.log_message(f"Не удалось рассчитать Оценку качества: {e}")

        # Распределение по кластерам
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            percentage = (c / n_vectors) * 100
            self.log_message(f"  Кластер {u:2d}: {c:4d} логотипов ({percentage:.1f}%)")

        self.log_message(f"Кластеризация {model_name} завершена.")