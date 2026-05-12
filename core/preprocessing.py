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
        self.csv_path = self.data_dir / "LogoDatabase.csv"
        
    def run(self) -> int:
        """Полная очистка и подготовка датасета"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists():
            logger.error("Файл LogoDatabase.csv не найден в папке data/ !")
            return 0
        
        df = pd.read_csv(self.csv_path)
        logger.info(f"Загружено {len(df)} записей из CSV")
        logger.info(f"Доступные колонки: {list(df.columns)}")
        
        valid_count = 0
        all_count = 0

        # Поддерживаемые расширения
        supported_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff'}

        for _, row in df.iterrows():
            filename = str(row['fileName']).strip()
            all_count += 1
            
            img_path = None
            for ext in supported_extensions:
                candidate = self.raw_dir / (Path(filename).stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            
            if not img_path:
                continue
                    
            try:
                with Image.open(img_path) as img:
                    if img.width < 100 or img.height < 100:
                        continue
                    
                    # Конвертируем в RGB (GIF и PNG с альфа-каналом)
                    if img.mode in ('RGBA', 'P', 'LA'):
                        img = img.convert('RGB')
                    else:
                        img = img.convert('RGB')
                    
                    save_path = self.processed_dir / f"{Path(filename).stem}.jpg"
                    img.save(save_path, quality=95)
                    valid_count += 1
            except Exception as e:
                logger.warning(f"Ошибка обработки {filename}: {e}")
                
        logger.info(f"Успешно обработано {valid_count} изображений (из {all_count})")
        return valid_count