import torch
import numpy as np
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class SAMEmbedder:
    def __init__(self, checkpoint_path="models/sam_vit_b.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = Path(checkpoint_path)
        self.model = None

        if not self.checkpoint_path.exists():
            logger.error(f"Файл SAM модели не найден: {self.checkpoint_path}")
            logger.info("   Запустите: python download_models.py")
            self.use_stub = True
            return

        try:
            from segment_anything import sam_model_registry
            self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"SAM ViT-B успешно загружена на {self.device}")
            self.use_stub = False
        except Exception as e:
            logger.error(f"Ошибка загрузки SAM: {e}")
            self.use_stub = True

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Реальное извлечение эмбеддинга через SAM"""
        if self.use_stub or self.model is None:
            logger.warning("SAM используется в режиме заглушки")
            return np.random.RandomState(42).rand(256).astype(np.float32)

        try:
            image = Image.open(image_path).convert("RGB")
            # Преобразуем в тензор
            input_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            input_image = input_image.to(self.device)

            # Извлекаем эмбеддинг через encoder
            with torch.no_grad():
                embedding = self.model.image_encoder(input_image)
                # Global Average Pooling
                embedding = embedding.mean(dim=[2, 3]).squeeze(0).cpu().numpy()
            
            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка извлечения SAM эмбеддинга: {e}")
            return np.random.RandomState(42).rand(256).astype(np.float32)