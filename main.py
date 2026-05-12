"""
main.py - Главный файл приложения TrademarkSearch
Интеллектуальная система векторного поиска графических товарных знаков
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Импорт вкладок
from gui.main_window import MainWindow

# Настройка пути логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка сообщений логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_and_download_models():
    """Проверка наличия моделей и автоматическая загрузка при необходимости"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    sam_path = models_dir / "sam_vit_b.pth"
    yolo_path = models_dir / "yolov8n.pt"

    missing = []

    if not sam_path.exists():
        missing.append("SAM (sam_vit_b.pth)")
    if not yolo_path.exists():
        missing.append("YOLOv8n (yolov8n.pt)")

    if missing:
        logger.info(f"Отсутствуют модели: {', '.join(missing)}")
        
        reply = QMessageBox.question(
            None,
            "Отсутствуют модели",
            f"В папке models/ не найдены следующие модели:\n\n" +
            "\n".join(missing) +
            "\n\nЗапустить скачивание сейчас?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Запуск download_models.py...")
            try:
                import subprocess
                result = subprocess.run([sys.executable, "download_models.py"], 
                                      capture_output=True, text=True, check=True)
                logger.info("Модели успешно скачаны.")
                print(result.stdout)
            except Exception as e:
                logger.error(f"Ошибка при скачивании моделей: {e}")
                QMessageBox.critical(None, "Ошибка", 
                                   f"Не удалось скачать модели:\n{str(e)}")
                return False
    else:
        logger.info("Все необходимые модели найдены.")

    return True


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    logger.info("Запуск приложения TrademarkSearch")
    
    window = MainWindow()
    window.show()

    logger.info("Приложение успешно запущено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()