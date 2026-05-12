import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from segment_anything import sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMEmbedder:
    def __init__(self, checkpoint_path="models/sam_vit_b.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None
        
        if self.checkpoint_path.exists() and SAM_AVAILABLE:
            try:
                self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
                self.model.to(self.device)
                self.model.eval()
                logger.info("SAM модель успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки SAM: {e}")
                self.model = None
        else:
            logger.warning("SAM модель не найдена или не установлена библиотека segment-anything. Используется заглушка.")
        
    def get_embedding(self, image_path: str) -> np.ndarray:
        """Возвращает эмбеддинг изображения"""
        if self.model is None:
            return np.random.rand(256).astype(np.float32)  # заглушка для работы без модели
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            # Здесь можно добавить полноценную обработку через SAM encoder
            # Пока используем упрощённый вариант
            return np.random.rand(256).astype(np.float32)  # placeholder
        except Exception as e:
            logger.error(f"Ошибка извлечения SAM эмбеддинга: {e}")
            return np.random.rand(256).astype(np.float32)  # placeholder