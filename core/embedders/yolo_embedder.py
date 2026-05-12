import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class YOLOEmbedder:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model_path = Path(model_path)
        self.use_stub = not self.model_path.exists()
        
        if self.use_stub:
            logger.warning("YOLO модель не найдена. Используется заглушка.")

    def get_embedding(self, image):
        if self.use_stub:
            return np.random.rand(512).astype(np.float32)
        
        # Реальная реализация позже через ultralytics
        return np.random.rand(512).astype(np.float32)