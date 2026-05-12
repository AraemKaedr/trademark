import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.csv_path = self.raw_dir / "LogoDatabase.csv"
        
    def run(self):
        """Полная очистка и подготовка датасета"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Загружено {len(df)} записей из CSV")
        
        valid_count = 0
        for _, row in df.iterrows():
            img_path = self.raw_dir / row['filename']
            if not img_path.exists():
                # Попытка исправить расширение
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = self.raw_dir / (row['filename'].rsplit('.', 1)[0] + ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
                else:
                    continue
                    
            try:
                with Image.open(img_path) as img:
                    if img.size[0] < 100 or img.size[1] < 100:
                        continue
                    img = img.convert('RGB')
                    save_path = self.processed_dir / f"{row['filename'].split('.')[0]}.jpg"
                    img.save(save_path, quality=95)
                    valid_count += 1
            except:
                continue
                
        logger.info(f"Успешно обработано {valid_count} изображений (из 1471)")
        return valid_count