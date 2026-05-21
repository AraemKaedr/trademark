from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ClusterTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.resnet_index = FaissIndex(dimension=2048, index_path="indexes/resnet_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        self.init_ui()

    def init_ui(self):
        label = QLabel("Кластеризация векторных представлений")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, label)

        btn_layout = QHBoxLayout()

        self.btn_resnet = QPushButton("Кластеризация ResNet50")
        self.btn_resnet.clicked.connect(lambda: self.run_clustering("resnet"))
        btn_layout.addWidget(self.btn_resnet)

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
            if mode in ["resnet", "both"]:
                self._perform_clustering(self.resnet_index, "ResNet")
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
        embeddings = index.get_all_embeddings() if hasattr(index, 'get_all_embeddings') else index.embeddings
        n_vectors = len(embeddings)

        self.log_message(f"{model_name} индекс содержит {n_vectors} векторов")
        
        count_vectors = 15
        if n_vectors < count_vectors:
            self.log_message(f"Недостаточно данных для {model_name} кластеризации ({n_vectors} векторов) нужно минимум {count_vectors} векторов изображений", "WARNING")
            self.log_message("   Сначала перейдите во вкладку 2. Эмбеддинги и извлеките эмбеддинги", "WARNING")
            return

        # Улучшение качества кластеризации
        # Нормализация данных перед кластеризацией (важно!)
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # UMAP (в 2D для визуализации)
        reducer_2d = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        embeddings_2d = reducer_2d.fit_transform(embeddings_scaled)

        self.log_message(f"UMAP 2D выполнен для визуализации")

        # Используем KMeans как основной (стабильнее для твоих данных)
        n_clusters = 8
        self.log_message(f"Выполняется KMeans кластеризация - {n_clusters} кластеров")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
        labels = kmeans.fit_predict(embeddings_scaled)

        # Метрики
        try:
            sil = silhouette_score(embeddings_scaled, labels)

            if sil > 0.6:
                quality = "ОТЛИЧНО"
            elif sil > 0.4:
                quality = "ХОРОШО"
            elif sil > 0.25:
                quality = "УДОВЛЕТВОРИТЕЛЬНО"
            else:
                quality = "СЛАБО"
            self.log_message(f"Оценка качества: {sil:.3f} — {quality}")
        except Exception as e:
            self.log_message(f"Ошибка. Недостаточно точек для расчёта метрик: {e}")

        # Визуализация
        self._save_cluster_visualization(embeddings_2d, labels, model_name)

        # Распределение
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            perc = (c / n_vectors) * 100
            self.log_message(f"  Кластер {u}: {c:4d} логотипов ({perc:.1f}%)")

        self.log_message(f"Кластеризация {model_name} завершена")
    
    def _save_cluster_visualization(self, embeddings_2d, labels, model_name):
        """Сохраняет график кластеров"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab20', s=30, alpha=0.8)
        plt.colorbar(scatter, label='Кластер')
        plt.title(f'Визуализация кластеров {model_name} (UMAP 2D)')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, alpha=0.3)
        
        save_path = Path("logs") / f"clusters_{model_name.lower()}.png"
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        self.log_message(f"График кластеров сохранён: {save_path}")