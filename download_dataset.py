"""
Автоматическая загрузка и распаковка датасета Flickr Logos 27
Для проекта TrademarkSearch
"""

from pathlib import Path
import urllib.request
import tarfile
import shutil
import sys
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "flickr_logos_27_dataset"


def download_flickr27():
    """Скачивание и распаковка датасета Flickr Logos 27"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    url = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"
    outer_tar = DATA_DIR / "flickr_logos_27.tar.gz"
    inner_tar_name = "flickr_logos_27_dataset_images.tar.gz"
    
    # Проверка — достаточно ли уже изображений
    existing_images = len(list(RAW_DIR.glob("*.jpg")))
    if existing_images >= 100:
        logger.info(f"Датасет уже подготовлен ({existing_images} изображений в data/raw/).")
        return True
        
    logger.info(f"Найдено только {existing_images} изображений. Начинаем подготовку...")

    # Скачивание
    if not outer_tar.exists():
        logger.info("Скачивание Flickr Logos 27 датасета. (Это может занять несколько минут)...")
        try:
            urllib.request.urlretrieve(url, outer_tar)
            logger.info("Скачивание завершено.")
        except Exception as e:
            logger.error(f"Ошибка скачивания: {e}")
            logger.info(f"\nНе удалось скачать автоматически. Скачайте вручную по ссылке: {url}")
            logger.info(f"и положите файл как {outer_tar}")
            return False
    else:
        logger.info("Внешний архив датасета уже скачан.")
    
    # Распаковка внешнего архива
    logger.info("Распаковка внешнего архива .tar.gz...")
    try:
        with tarfile.open(outer_tar, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        logger.info("Внешний архив успешно распакован.")
    except Exception as e:
        logger.error(f"Ошибка распаковки внешнего архива: {e}")
        return False
    
    # Поиск внутреннего архива с изображениями
    inner_tar = None
    for p in DATA_DIR.rglob(inner_tar_name):
        inner_tar = p
        break

    if not inner_tar or not inner_tar.exists():
        logger.error("Не найден внутренний архив с изображениями!")
        return False

    # Распаковка внутреннего архива
    logger.info(f"Распаковка изображений из {inner_tar.name}...")
    try:
        with tarfile.open(inner_tar, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        logger.info("Внутренний архив успешно распакован. Изображения успешно извлечены.")
    except Exception as e:
        logger.error(f"Ошибка распаковки внутреннего архива: {e}")
        return False
    
    # Поиск папки с изображениями и перемещение в data/raw/
    images_source = None
    for p in DATA_DIR.rglob("flickr_logos_27_dataset_images"):
        if p.is_dir():
            images_source = p
            break

    if images_source and images_source.exists():
        logger.info(f"Перемещение изображений из {images_source} в {RAW_DIR}")
        copied = 0
        for img in images_source.glob("*.jpg"):
            shutil.copy2(img, RAW_DIR / img.name)
            copied += 1
        logger.info(f"Изображения успешно скопированы в data/raw/ ({copied} файлов)")
    else:
        logger.warning("Не удалось найти папку с изображениями после распаковки.")

    final_count = len(list(RAW_DIR.glob("*.jpg")))
    logger.info(f"Подготовка завершена! В data/raw/ теперь {final_count} изображений.")
    logger.info("Датасет Flickr Logos 27 успешно подготовлен!")
    return True

if __name__ == "__main__":
    success = download_flickr27()
    if not success:
        logger.info("\nЕсли возникли ошибки — скачайте датасет вручную:")
        logger.info("1. http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz")
        logger.info("2. Положите файл в папку data/ как flickr_logos_27.tar.gz")
        logger.info("3. Распакуйте дважды .tar.gz")
        logger.info("4. Изображения из папки flickr_logos_27_dataset_images положите в data/raw/")
    sys.exit(0 if success else 1)