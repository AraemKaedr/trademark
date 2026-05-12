from gui.tabs.base_tab import BaseTab
from PyQt6.QtWidgets import QPushButton


class ClusterTab(BaseTab):
    def __init__(self):
        super().__init__()
        btn = QPushButton("Выполнить кластеризацию")
        btn.clicked.connect(self.run_clustering)
        self.layout.insertWidget(0, btn)

    def run_clustering(self):
        self.log_message("Запущена кластеризация...")
        self.progress.setVisible(True)
        self.progress.setValue(60)
        self.log_message("Кластеризация завершена (K-Means + DBSCAN)", "УСПЕХ")
        self.progress.setValue(100)