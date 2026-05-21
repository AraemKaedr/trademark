import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class ResNetEmbedder:
    _instance = None # Singleton — один экземпляр на всё приложение

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # # Убираем последний слой (классификатор)
        self.model.eval().to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info(f"ResNet50 успешно загружена на {self.device}")

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Реальное извлечение эмбеддинга через ResNet"""
        try:
            img = Image.open(image_path).convert("RGB")
            # Преобразуем в тензор
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Извлекаем эмбеддинг через encoder
            with torch.no_grad():
                # Получаем эмбеддинги
                emb = self.model(img_tensor).squeeze().cpu().numpy()
            
            # L2-нормализация
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.astype(np.float32)

        except Exception as e:
            logger.warning(f"Ошибка извлечения ResNet эмбеддинга для {Path(image_path).name}: {e}")
            # Возвращаем стабильную заглушку
            return np.zeros(2048, dtype=np.float32)  # fallback