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

        # Пути к аннотациям
        self.annotation_file = self.data_dir / "flickr_logos_27_dataset" / "flickr_logos_27_dataset_training_set_annotation.txt"
        
    def run(self) -> int:
        """Полная очистка и подготовка датасета"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        skipped = 0
        processed_files = set() # Чтобы не дублировать изображения

        # Собираем все изображения из data/raw/ (поддерживаемые расширения)
        image_files = list(self.raw_dir.glob("*.jpg")) + list(self.raw_dir.glob("*.png")) + \
                      list(self.raw_dir.glob("*.jpeg")) + list(self.raw_dir.glob("*.webp"))
        
        logger.info(f"Найдено {len(image_files)} изображений в {self.raw_dir} для обработки")

        if not self.annotation_file.exists():
            logger.warning("Файл аннотаций не найден. Используем простой режим.")
            return self._simple_process()

        logger.info("Извлечение логотипов по аннотациям...")

        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                filename = parts[0]
                class_id = parts[1]
                try:
                    x1, y1, x2, y2 = map(int, parts[3:7])
                except ValueError:
                    continue

                img_path = self.raw_dir / filename
                if not img_path.exists():
                    continue

                # Чтобы не обрабатывать одно изображение много раз
                if filename in processed_files:
                    continue
                processed_files.add(filename)

                try:
                    with Image.open(img_path) as img:
                        # Конвертируем в RGB
                        if img.mode != "RGB":
                            img = img.convert('RGB')

                        # Кроп логотипа
                        crop_box = (max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2))
                        cropped = img.crop(crop_box)

                        # Пропускаем слишком маленькие изображения
                        if cropped.width < 80 or cropped.height < 80:
                            skipped += 1
                            continue
                        
                        save_path = self.processed_dir / f"{Path(filename).stem}_{class_id}.jpg"
                        cropped.save(save_path, quality=95, optimize=True)
                        valid_count += 1

                        if valid_count % 200 == 0:
                            logger.info(f"Обработано: {valid_count} изображений...")

                except Exception as e:
                    logger.warning(f"Ошибка обработки {filename} (строка {line_num}): {e}")
                    skipped += 1
                
        final_count = len(list(self.processed_dir.glob("*.jpg")))

        logger.info(f"Предобработка завершена.!")
        logger.info(f"Успешно вырезано и сохранено: {valid_count} логотипов (из {final_count})")
        logger.info(f"Пропущено (маленькие/ошибки): {skipped} логотипов (из {final_count})")
        logger.info(f"Фактически в папке data/processed: {final_count} файлов")
        logger.info(f"Изображения сохранены в папку: {self.processed_dir}")
        return valid_count

    def _simple_process(self):
        """Fallback — простой режим"""
        valid_count = 0
        skipped = 0
        for img_path in list(self.raw_dir.glob("*.jpg")):
            try:
                with Image.open(img_path) as img:
                    if img.width < 80 or img.height < 80:
                        skipped += 1
                        continue
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    save_path = self.processed_dir / img_path.name
                    img.save(save_path, quality=95)
                    valid_count += 1
            except:
                pass

        final_count = len(list(self.processed_dir.glob("*.jpg")))
        
        logger.info(f"Успешно вырезано и сохранено: {valid_count} логотипов (из {final_count})")
        logger.info(f"Пропущено (маленькие/ошибки): {skipped} логотипов (из {final_count})")
        logger.info(f"Фактически в папке data/processed: {final_count} файлов")
        logger.info(f"Изображения сохранены в папку: {self.processed_dir}")
        return valid_count

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    count = preprocessor.run()
    print(f"Готово. Обработано {count} логотипов.")