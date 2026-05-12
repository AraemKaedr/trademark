from pathlib import Path
import pandas as pd
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, data_dir="data/", raw_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.csv_path = self.data_dir / "LogoDatabase.csv"
        
    def run(self) -> int:
        """Полная очистка и подготовка датасета"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists():
            logger.error("Файл LogoDatabase.csv не найден!")
            return 0
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Загружено {len(df)} записей из CSV")
        
        valid_count = 0
        all_count = 0
        for _, row in df.iterrows():
            img_path = self.raw_dir / row['filename']
            all_count += 1
            if not img_path.exists():
                continue
                    
            try:
                with Image.open(img_path) as img:
                    if img.width < 100 or img.height < 100:
                        continue
                    img = img.convert('RGB')
                    save_path = self.processed_dir / f"{Path(row['filename']).stem}.jpg"
                    img.save(save_path, quality=95)
                    valid_count += 1
            except Exception as e:
                logger.warning(f"Ошибка обработки {row['filename']}: {e}")
                
        logger.info(f"Успешно обработано {valid_count} изображений (из {all_count})")
        return valid_count