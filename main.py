"""
main.py - Главный файл приложения TrademarkSearch
Интеллектуальная система векторного поиска графических товарных знаков
"""

import sys
import logging
from pathlib import Path
import subprocess

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

# Импорт вкладок
from gui.main_window import MainWindow

# Настройка пути логирования
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Настройка сообщений логирования (пишет и в файл, и в терминал)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "app.log", encoding='utf-8', mode='a'),  # 'a' - режим append. Нужен для того, чтобы логи гарантированно писались в logs/app.log
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Перезаписываем конфигурацию, если она уже была
)
logger = logging.getLogger(__name__)


def check_and_download_models():
    """Проверка наличия моделей и автоматическая загрузка при необходимости"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    yolo_path = models_dir / "yolov8n.pt"
    # ResNet не требует скачивания

    missing = []

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


def check_and_prepare_data():
    """Проверка наличия датасета и автоматическая подготовка"""
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    need_download = False
    if not raw_dir.exists() or len(list(raw_dir.glob("*.*"))) < 10:
        need_download = True
        logger.info("Папка data/raw пуста или не существует.")
    
    if need_download:
        logger.info("Запускается подготовка датасета...")
        
        reply = QMessageBox.question(
            None,
            "Датасет не найден",
            "Датасет не обнаружен или не подготовлен.\n\n"
            "Запустить скачивание и распаковку сейчас?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                result = subprocess.run(
                    [sys.executable, "download_dataset.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("Датасет успешно подготовлен.")
                print(result.stdout)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка при подготовке датасета:\n{e.stderr}")
                QMessageBox.critical(None, "Ошибка", 
                                   "Не удалось подготовить датасет.\n"
                                   "Пожалуйста, запустите download_dataset.py вручную.")
                return False
            except FileNotFoundError:
                logger.error("Файл download_dataset.py не найден!")
                QMessageBox.critical(None, "Ошибка", 
                                   "Файл download_dataset.py не найден в корне проекта.")
                return False
    else:
        logger.info("Датасет уже подготовлен.")
        return True


def main():
    """Точка входа в приложение"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Современный стиль
    
    logger.info("Запуск приложения TrademarkSearch")
    
    # Проверка и скачивание моделей перед запуском
    if not check_and_download_models():
        logger.warning("Приложение запущено без некоторых моделей.")
    
    # Автоматическая проверка и подготовка датасета
    if not check_and_prepare_data():
        logger.warning("Приложение запущено без датасета.")
    
    window = MainWindow()
    window.show()

    logger.info("Приложение успешно запущено")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()