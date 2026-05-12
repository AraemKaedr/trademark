# core/embedders/sam_embedder.py
from segment_anything import sam_model_registry
import torch
import numpy as np

class SAMEmbedder:
    def __init__(self, model_type="vit_b", checkpoint_path="models/sam_vit_b.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.sam.eval()
        
    def get_embedding(self, image):
        # image: PIL Image или numpy array
        # ... (логика извлечения эмбеддинга)
        return np.random.rand(256).astype(np.float32)  # placeholder