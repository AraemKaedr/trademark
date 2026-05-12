"""
download_models.py
Скрипт для автоматической загрузки моделей SAM и YOLOv8 в папку models/
Для проекта TrademarkSearch
"""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error
from tqdm import tqdm

# НАСТРОЙКИ
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Модель SAM ViT-B (Segment Anything Model)
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILENAME = "sam_vit_b.pth"

# Модель YOLOv8n
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
YOLO_FILENAME = "yolov8n.pt"


def download_with_progress(url: str, filepath: Path):
    """Скачивание файла с прогресс-баром"""
    try:
        print(f"Скачивание: {filepath.name}")
        
        def reporthook(count, block_size, total_size):
            progress = int(count * block_size * 100 / total_size)
            print(f"\rПрогресс: {progress}%", end="")
        
        urllib.request.urlretrieve(url, filepath, reporthook=reporthook)
        print(f"\nУспешно скачано: {filepath.name} ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        return True
        
    except Exception as e:
        print(f"\nОшибка при скачивании {filepath.name}: {e}")
        return False


def main():
    print("Скачивание моделей для проекта TrademarkSearch")

    success_count = 0

    # 1. SAM ViT-B
    sam_path = MODELS_DIR / SAM_FILENAME
    if not sam_path.exists():
        if download_with_progress(SAM_URL, sam_path):
            success_count += 1
    else:
        print(f"SAM модель уже существует: {sam_path.name}")

    # 2. YOLOv8n
    yolo_path = MODELS_DIR / YOLO_FILENAME
    if not yolo_path.exists():
        if download_with_progress(YOLO_URL, yolo_path):
            success_count += 1
    else:
        print(f"YOLOv8n модель уже существует: {yolo_path.name}")

    print(f"\nГотово! Скачано моделей: {success_count}/2")
    print(f"Папка: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()