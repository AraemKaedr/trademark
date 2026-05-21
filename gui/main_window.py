from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from gui.tabs.dataset_tab import DatasetTab
from gui.tabs.embedding_tab import EmbeddingTab
from gui.tabs.index_tab import IndexTab
from gui.tabs.cluster_tab import ClusterTab
from gui.tabs.search_tab import SearchTab
from gui.tabs.results_tab import ResultsTab
from gui.tabs.manager_tab import ManagerTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TrademarkSearch - Поиск графических товарных знаков")
        self.resize(1400, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs)
        
        self._create_tabs()
        
        # По умолчанию открываем вкладку "Поиск"
        self.tabs.setCurrentIndex(4)
        
    def _create_tabs(self):
        """Создание вкладок с правильными зависимостями"""
        
        # Вкладка Результатов создаётся первой
        self.results_tab = ResultsTab()

        self.tabs.addTab(DatasetTab(), "1. Датасет")
        self.tabs.addTab(EmbeddingTab(), "2. Эмбеддинги")
        self.tabs.addTab(IndexTab(), "3. Индекс")
        self.tabs.addTab(ClusterTab(), "4. Кластеризация")
        
        # Передаём ResultsTab в SearchTab
        self.search_tab = SearchTab(results_tab=self.results_tab)
        self.tabs.addTab(self.search_tab, "5. Поиск")
        self.tabs.addTab(self.results_tab, "6. Результаты")
        # self.tabs.addTab(ManagerTab(), "7. Менеджер")