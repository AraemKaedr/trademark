"""
Автоматическая загрузка и распаковка датасета Popular Brand Logos
Для проекта TrademarkSearch
"""

import os
import zipfile
from pathlib import Path
import urllib.request
import shutil
import sys

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DATASET_URL = "https://www.kaggle.com/datasets/kkhandekar/popular-brand-logos-image-dataset/download?datasetVersionNumber=1"
ZIP_FILENAME = "popular-brand-logos-image-dataset.zip"


def download_kaggle_dataset():
    """Скачивание датасета с Kaggle"""
    DATA_DIR.mkdir(exist_ok=True)
    
    zip_path = DATA_DIR / ZIP_FILENAME
    
    if zip_path.exists():
        print("ZIP-архив датасета уже существует.")
        return zip_path

    print("Скачивание датасета с Kaggle...")
    print("Если скачивание не начнётся автоматически, скачайте вручную по ссылке:")
    print("https://www.kaggle.com/datasets/kkhandekar/popular-brand-logos-image-dataset")
    
    try:
        # Простой способ — открыть браузер
        import webbrowser
        webbrowser.open("https://www.kaggle.com/datasets/kkhandekar/popular-brand-logos-image-dataset")
        print("\nПосле скачивания положите файл popular-brand-logos-image-dataset.zip в папку data/ и запустите скрипт снова.")
        return None
    except:
        print("Не удалось открыть браузер. Скачайте датасет вручную.")
        return None


def extract_dataset(zip_path: Path):
    """Распаковка датасета и переименование папки Logos → raw"""
    print("Распаковка датасета...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Переименовываем папку Logos в raw
    logos_dir = DATA_DIR / "Logos"
    if logos_dir.exists():
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
        logos_dir.rename(RAW_DIR)
        print(f"Папка Logos переименована в raw → {RAW_DIR}")
    else:
        print("Папка Logos не найдена после распаковки.")

    # Перемещаем LogoDatabase.csv в корень data/
    csv_source = DATA_DIR / "LogoDatabase.csv"
    if csv_source.exists():
        csv_target = DATA_DIR / "LogoDatabase.csv"
        if csv_source != csv_target:
            shutil.move(csv_source, csv_target)
            print("LogoDatabase.csv перемещён в data/")
    
    print("Датасет успешно подготовлен!")


def main():
    print("Скачивание и подготовка датасета для TrademarkSearch")

    zip_path = DATA_DIR / ZIP_FILENAME
    
    if not RAW_DIR.exists() or not (DATA_DIR / "LogoDatabase.csv").exists():
        if not zip_path.exists():
            download_kaggle_dataset()
            print("\nПосле скачивания запустите скрипт снова.")
            sys.exit(0)
        else:
            extract_dataset(zip_path)
    else:
        print("Датасет уже подготовлен (data/raw/ и LogoDatabase.csv существуют).")

    print("\nСтруктура данных готова:")
    print(f"   - data/raw/          — изображения логотипов")
    print(f"   - data/LogoDatabase.csv — метаданные")


if __name__ == "__main__":
    main()