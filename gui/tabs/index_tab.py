from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget


class IndexTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        
        self.init_ui()

    def init_ui(self):
        btn_layout = QHBoxLayout()
        
        btn_sam = QPushButton("Перезагрузить SAM индекс")
        btn_sam.clicked.connect(lambda: self._load_index("sam"))
        btn_layout.addWidget(btn_sam)

        btn_yolo = QPushButton("Перезагрузить YOLO индекс")
        btn_yolo.clicked.connect(lambda: self._load_index("yolo"))
        btn_layout.addWidget(btn_yolo)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(0, container)

    def _load_index(self, model_type: str):
        if model_type == "sam":
            success = self.sam_index.load()
            self.log_message("SAM индекс " + ("загружен" if success else "создан заново"), "УСПЕХ")
        else:
            success = self.yolo_index.load()
            self.log_message("YOLO индекс " + ("загружен" if success else "создан заново"), "УСПЕХ")