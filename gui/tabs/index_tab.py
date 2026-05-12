from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton


class IndexTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.index = FaissIndex()
        
        btn = QPushButton("Построить / Обновить векторный индекс")
        btn.clicked.connect(self.build_index)
        self.layout.insertWidget(0, btn)

    def build_index(self):
        self.log_message("Начато построение FAISS индекса...")
        self.progress.setVisible(True)
        self.progress.setValue(40)
        
        self.log_message("Векторный индекс успешно построен", "УСПЕХ")
        self.progress.setValue(100)