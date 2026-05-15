import torch
import numpy as np
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class SAMEmbedder:
    _instance = None  # Singleton — один экземпляр на всё приложение

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_path="models/sam_vit_b.pth"):
        if self._initialized:
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        
        self.use_stub = True

        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            logger.error(f"Файл SAM модели не найден: {self.checkpoint_path}")
            logger.info("   Запустите: python download_models.py")
            return

        try:
            from segment_anything import sam_model_registry
            logger.info("Загрузка SAM ViT-B модели... (это может занять 10-20 секунд)")
            
            self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
            self.model.to(self.device)
            self.model.eval()
            
            self.use_stub = False
            logger.info(f"SAM ViT-B успешно загружена на {self.device}")
        except ImportError:
            logger.error("Библиотека 'segment-anything' не установлена")
            logger.info("   Запустите: pip install segment-anything")
        except Exception as e:
            logger.error(f"Ошибка загрузки SAM модели: {e}")
            self.use_stub = True
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Подготовка изображения для SAM ViT-B (должно быть кратно 64). Оптимизированная предобработка для скорости."""
        # 1. Конвертируем в RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Приводим к фиксированному размеру, который хорошо работает с SAM (кратен 64)
        w, h = image.size
        target_size = 1024  # Стандартный размер для SAM

        # Масштабируем по длинной стороне с сохранением пропорций
        scale = target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Приводим к размеру, кратному 64
        new_w = (new_w // 64) * 64
        new_h = (new_h // 64) * 64
        if new_w == 0: new_w = 64
        if new_h == 0: new_h = 64

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 3. Преобразуем в тензор
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return tensor.to(self.device)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Реальное извлечение эмбеддинга через SAM"""
        if self.use_stub or self.model is None:
            logger.warning("SAM используется в режиме заглушки")
            return np.random.RandomState(42).rand(256).astype(np.float32)

        try:
            image = Image.open(image_path)
            # Преобразуем в тензор
            input_tensor = self._preprocess_image(image)

            # Извлекаем эмбеддинг через encoder
            with torch.no_grad():
                # Получаем эмбеддинги (прямой проход) через image_encoder
                embedding = self.model.image_encoder(input_tensor)

                # Global Average Pooling (усредняем по высоте и ширине) - стандартный способ получить вектор
                embedding = embedding.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
            
            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Ошибка извлечения SAM эмбеддинга для {image_path}: {e}")
            # Возвращаем стабильную заглушку
            return np.random.RandomState(42).rand(256).astype(np.float32)