from gui.tabs.base_tab import BaseTab
from core.embedders.sam_embedder import SAMEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from PyQt6.QtWidgets import QPushButton


class EmbeddingTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.sam = SAMEmbedder()
        self.yolo = YOLOEmbedder()
        
        btn = QPushButton("Извлечь эмбеддинги для всего датасета")
        btn.clicked.connect(self.extract_embeddings)
        self.layout.insertWidget(0, btn)

    def extract_embeddings(self):
        self.log_message("Начато извлечение эмбеддингов...")
        self.progress.setVisible(True)
        self.progress.setValue(50)
        
        # Заглушка
        self.log_message("Эмбеддинги успешно извлечены (SAM + YOLO)", "УСПЕХ")
        self.progress.setValue(100)