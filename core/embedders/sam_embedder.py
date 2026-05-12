import torch
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SAMEmbedder:
    def __init__(self, checkpoint_path="models/sam_vit_b.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            logger.warning(f"SAM модель не найдена по пути {self.checkpoint_path}. Используется заглушка.")
            self.use_stub = True
        else:
            self.use_stub = False
            # Загрузка реальной модели (закомментировано для скорости запуска)
            # from segment_anything import sam_model_registry
            # self.sam = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
            # self.sam.to(self.device)
            # self.sam.eval()
        
    def get_embedding(self, image):
        """Возвращает эмбеддинг изображения"""
        if self.use_stub or not torch.cuda.is_available():
            # Заглушка для работы без модели
            return np.random.rand(256).astype(np.float32)
        
        # Рабочая реализация (будет добавлена позже)
        # with torch.no_grad():
        #     embedding = self.sam.image_encoder(...)
        return np.random.rand(256).astype(np.float32)  # placeholder