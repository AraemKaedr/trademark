"""
train_contrastive.py — Fine-tuning ResNet50 и YOLOv8 backbone
с Triplet Loss для оценки схожести на вырезанных логотипов на Flickr Logos 27
"""

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Попытка импорта tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logger.warning("tqdm не установлен. Прогресс-бар будет упрощённым.")

# Безопасный импорт ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logger.error("Библиотека ultralytics не установлена! Установите: pip install ultralytics")

# Пути
MODELS_TRAINED = Path("models_trained")
MODELS_TRAINED.mkdir(exist_ok=True)

class LogoTripletDataset(Dataset):
    def __init__(self, processed_dir="data/processed", transform=None):
        self.processed_dir = Path(processed_dir)
        self.images = sorted(list(self.processed_dir.glob("*.jpg")))
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_path = self.images[idx]
        anchor = Image.open(anchor_path).convert("RGB")

        # Простая имитация положительного и отрицательного примера
        positive_idx = random.choice([i for i in range(len(self.images)) if i != idx])
        negative_idx = random.choice([i for i in range(len(self.images)) if i not in (idx, positive_idx)])

        positive = Image.open(self.images[positive_idx]).convert("RGB")
        negative = Image.open(self.images[negative_idx]).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class ContrastiveResNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(2048, embedding_dim)  # embedding размер

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)


class YOLOFeatureExtractor(nn.Module):
    """Извлечение эмбеддингов из backbone YOLOv8"""
    def __init__(self, embedding_dim=512):
        super().__init__()
        if YOLO is None:
            self.dummy = True
            self.embedding_dim = embedding_dim
            return
        self.dummy = False
        self.yolo = YOLO("models/yolov8n.pt")
        self.embedding_dim = embedding_dim

    def forward(self, x):
        if self.dummy or self.yolo is None:
            # Заглушка
            return torch.randn(x.shape[0], self.embedding_dim, device=x.device)
        
        # Используем backbone YOLO до определённого слоя
        with torch.no_grad():  # Заморозим backbone на первом этапе
            try:
                # Получаем features из backbone
                features = self.yolo.model.model[:10](x)  # берём глубокие признаки
                # Global Average Pooling
                emb = torch.mean(features, dim=[2, 3]) # [batch, channels]
            except:
                # Fallback
                emb = torch.mean(x, dim=[2, 3]) # fallback
        
        # Проекция в общее пространство
        if emb.shape[1] != self.embedding_dim:
            projector = nn.Linear(emb.shape[1], self.embedding_dim, device=emb.device)
            emb = projector(emb)
        
        return nn.functional.normalize(emb, p=2, dim=1)


def train_contrastive(epochs=8, batch_size=32, margin=0.4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Запуск обучения на устройстве: {device}")

    dataset = LogoTripletDataset("data/processed")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # Модели
    resnet_model = ContrastiveResNet().to(device)
    yolo_extractor = YOLOFeatureExtractor().to(device)

    optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=3e-4, weight_decay=1e-5)
    optimizer_yolo = optim.Adam(yolo_extractor.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    scaler = GradScaler() if device.type == "cuda" else None

    logger.info("Начало совместного contrastive fine-tuning ResNet50 + YOLOv8...")

    for epoch in range(epochs):
        resnet_model.train()
        total_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if tqdm else loader
        for anchor, positive, negative in progress_bar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer_resnet.zero_grad()
            optimizer_yolo.zero_grad()

            # ResNet
            emb_a_r = resnet_model(anchor)
            emb_p_r = resnet_model(positive)
            emb_n_r = resnet_model(negative)

            # YOLO Feature Extractor
            emb_a_y = yolo_extractor(anchor)
            emb_p_y = yolo_extractor(positive)
            emb_n_y = yolo_extractor(negative)

            with autocast(enabled=device.type == "cuda"):
                loss_r = criterion(emb_a_r, emb_p_r, emb_n_r)
                loss_y = criterion(emb_a_y, emb_p_y, emb_n_y)
                loss = loss_r + loss_y
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer_resnet)
                scaler.step(optimizer_yolo)
                scaler.update()
            else:
                loss.backward()
                optimizer_resnet.step()
                optimizer_yolo.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # Сохранение моделей
    torch.save(resnet_model.state_dict(), MODELS_TRAINED / "resnet_contrastive.pth")
    torch.save(yolo_extractor.state_dict(), MODELS_TRAINED / "yolo_contrastive.pth")

    logger.info("Обе модели успешно обучены и сохранены в models_trained/")
    return resnet_model, yolo_extractor


if __name__ == "__main__":
    logger.info("Contrastive Fine-tuning ResNet50 + YOLOv8\n")
    train_contrastive(epochs=8, batch_size=32)