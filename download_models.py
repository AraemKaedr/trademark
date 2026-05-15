"""
download_models.py
Скрипт для автоматической загрузки моделей SAM и YOLOv8 в папку models/¶
Для проекта TrademarkSearch
"""

import sys
from pathlib import Path
import requests
from tqdm import tqdm

# НАСТРОЙКИ
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Прямая ссылка на модель SAM ViT-B (Segment Anything Model)
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILENAME = "sam_vit_b.pth"

# Прямая ссылка на модель YOLOv8n
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
YOLO_FILENAME = "yolov8n.pt"


def download_file(url: str, filename: str):
    """Скачивание файла с прогресс-баром"""
    filepath = MODELS_DIR / filename
    
    if filepath.exists():
        print(f"{filename} уже существует ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"Скачивание {filename} (~350 МБ для SAM)...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
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
    print("Скачивание моделей для проекта TrademarkSearch")

    success = 0

    # SAM
    if download_file(SAM_URL, SAM_FILENAME):
        success += 1

    # YOLOv8n
    if download_file(YOLO_URL, YOLO_FILENAME):
        success += 1

    print("\n-----")
    print(f"Завершено. Успешно скачано: {success}/2 моделей")
    print(f"Папка: {MODELS_DIR.resolve()}")

    if success < 2:
        print("\nЕсли скачивание зависло — скачайте SAM и YOLO вручную:")
        print("Ссылка на SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print("Ссылка на YOLO: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
        print("переименуйте файлы в sam_vit_b.pth и yolov8n.pt и положите их в папку models/")


if __name__ == "__main__":
    main()