"""
download_models.py
Скрипт для автоматической загрузки модели YOLOv8 в папку models/
ResNet50 загружается автоматически из torchvision
Для проекта TrademarkSearch
"""

import sys
from pathlib import Path
import requests
from tqdm import tqdm

# НАСТРОЙКИ
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Прямая ссылка на модель YOLOv8n
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
YOLO_FILENAME = "yolov8n.pt"


def download_file(url: str, filename: str):
    """Скачивание файла с прогресс-баром"""
    filepath = MODELS_DIR / filename
    
    if filepath.exists():
        print(f"{filename} уже существует ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"Скачивание {filename}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"{filename} успешно скачан")
        return True
        
    except Exception as e:
        print(f"Ошибка скачивания {filename}: {e}")
        if filepath.exists():
            filepath.unlink()  # удаляем частично скачанный файл
        return False


def main():
    print("Скачивание моделей для проекта TrademarkSearch (ResNet50 + YOLOv8n)")

    # YOLOv8n
    success = download_file(YOLO_URL, YOLO_FILENAME)

    print("\n-----")
    print(f"Завершено. Успешно скачано: {success}/1 моделей")
    print("\nResNet50 будет загружаться автоматически из torchvision.")

    if success < 1:
        print("\nЕсли скачивание зависло — скачайте YOLO вручную:")
        print("Ссылка на YOLO: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
        print("переименуйте файл в yolov8n.pt и положите его в папку models/")
    
    print(f"Модели сохранены в: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()