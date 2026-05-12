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

def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    # Иконка (опционально)
    # app.setWindowIcon(QIcon("icons/app_icon.png"))
    
    logger.info("Приложение запущено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()