from pathlib import Path
import pandas as pd
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, data_dir="data", raw_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        
    def run(self) -> int:
        """Полная очистка и подготовка датасета"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0

        # Поддерживаемые расширения
        image_files = list(self.raw_dir.glob("*.jpg")) + list(self.raw_dir.glob("*.png")) + \
                      list(self.raw_dir.glob("*.jpeg")) + list(self.raw_dir.glob("*.webp"))
        
        logger.info(f"Найдено {len(image_files)} изображений для обработки")

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    if img.width < 64 or img.height < 64:
                        continue
                    
                    # Конвертируем в RGB (GIF и PNG с альфа-каналом)
                    if img.mode in ('RGBA', 'P', 'LA'):
                        img = img.convert('RGB')
                    else:
                        img = img.convert('RGB')
                    
                    save_path = self.processed_dir / f"{img_path.stem}.jpg"
                    img.save(save_path, quality=95)
                    valid_count += 1
            except Exception as e:
                logger.warning(f"Ошибка обработки {img_path.name}: {e}")
                
        logger.info(f"Предобработка завершена. Успешно обработано и сохранено {valid_count} изображений (из {len(image_files)})")
        return valid_count