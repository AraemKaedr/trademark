"""
main.py - Главный файл приложения TrademarkSearch
Интеллектуальная система векторного поиска графических товарных знаков
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# Импорт вкладок
from gui.main_window import MainWindow

# Настройка логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    logger.info("Запуск приложения TrademarkSearch")
    
    window = MainWindow()
    window.show()

    logger.info("Приложение запущено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()