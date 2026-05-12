"""
Интеллектуальная система векторного поиска графических товарных знаков
TrademarkSearch — основной файл запуска приложения
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# Импорт вкладок
from gui.tabs.dataset_tab import DatasetTab
from gui.tabs.embedding_tab import EmbeddingTab
from gui.tabs.index_tab import IndexTab
from gui.tabs.cluster_tab import ClusterTab
from gui.tabs.search_tab import SearchTab
from gui.tabs.results_tab import ResultsTab
from gui.tabs.manager_tab import ManagerTab

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LogoSimSearch — Поиск графических товарных знаков")
        self.resize(1280, 820)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Вкладки
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        layout.addWidget(self.tabs)
        
        # Добавление вкладок
        self._add_tabs()
        
        logger.info("Главное окно приложения успешно инициализировано")
        
    def _add_tabs(self):
        """Добавление всех вкладок"""
        self.tabs.addTab(DatasetTab(), "1. Датасет")
        self.tabs.addTab(EmbeddingTab(), "2. Извлечение эмбеддингов")
        self.tabs.addTab(IndexTab(), "3. Векторный индекс")
        self.tabs.addTab(ClusterTab(), "4. Кластеризация")
        self.tabs.addTab(SearchTab(), "5. Поиск")
        self.tabs.addTab(ResultsTab(), "6. Результаты")
        self.tabs.addTab(ManagerTab(), "7. Менеджер моделей")
        
        # Устанавливаем начальную вкладку
        self.tabs.setCurrentIndex(4)  # Поиск — основная вкладка


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    # Иконка (опционально)
    # app.setWindowIcon(QIcon("icons/app_icon.png"))
    
    window = MainWindow()
    window.show()

    logger.info("Приложение запущено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()