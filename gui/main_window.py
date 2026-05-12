from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget
from gui.tabs import (DatasetTab, EmbeddingTab, IndexTab, ClusterTab, 
                     SearchTab, ResultsTab, ManagerTab)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrademarkSearch - Поиск графических товарных знаков")
        self.resize(1200, 800)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tabs.addTab(DatasetTab(), "1. Датасет")
        self.tabs.addTab(EmbeddingTab(), "2. Эмбеддинги")
        self.tabs.addTab(IndexTab(), "3. Индекс")
        self.tabs.addTab(ClusterTab(), "4. Кластеризация")
        self.tabs.addTab(SearchTab(), "5. Поиск")
        self.tabs.addTab(ResultsTab(), "6. Результаты")
        self.tabs.addTab(ManagerTab(), "7. Менеджер моделей")