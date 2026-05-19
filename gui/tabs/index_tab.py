from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel


class IndexTab(BaseTab):
    def __init__(self):
        super().__init__()
        # Создаём отдельные индексы для SAM и YOLO
        self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        
        self.init_ui()

    def init_ui(self):
        title = QLabel("Управление векторными индексами")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, title)

        # Кнопки управления индексами
        btn_layout = QHBoxLayout()
        
        self.btn_reload_sam = QPushButton("Перезагрузить SAM индекс")
        self.btn_reload_sam.clicked.connect(lambda: self._reload_index("sam"))
        btn_layout.addWidget(self.btn_reload_sam)

        self.btn_reload_yolo = QPushButton("Перезагрузить YOLO индекс")
        self.btn_reload_yolo.clicked.connect(lambda: self._reload_index("yolo"))
        btn_layout.addWidget(self.btn_reload_yolo)

        self.btn_clear_sam = QPushButton("Очистить SAM индекс")
        self.btn_clear_sam.clicked.connect(lambda: self._clear_index("sam"))
        btn_layout.addWidget(self.btn_clear_sam)

        self.btn_clear_yolo = QPushButton("Очистить YOLO индекс")
        self.btn_clear_yolo.clicked.connect(lambda: self._clear_index("yolo"))
        btn_layout.addWidget(self.btn_clear_yolo)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

        # Информация о текущем состоянии
        self.status_label = QLabel("Статус индексов будет показан здесь")
        self.layout.insertWidget(2, self.status_label)

    def _reload_index(self, model_type: str):
        """Перезагрузка индекса"""
        if model_type == "sam":
            success = self.sam_index.load()
            status = "SAM"
        else:
            success = self.yolo_index.load()
            status = "YOLO"

        if success:
            self.log_message(f"{status} индекс успешно загружен", "УСПЕХ")
        else:
            self.log_message(f"Создан новый {status} индекс", "УСПЕХ")
        
        self._update_status()

    def _clear_index(self, model_type: str):
        """Очистка индекса (удаление файлов)"""
        if model_type == "sam":
            index = self.sam_index
            status = "SAM"
        else:
            index = self.yolo_index
            status = "YOLO"

        try:
            # Удаляем файлы индекса
            if index.index_path.exists():
                index.index_path.unlink(missing_ok=True)
            if index.embeddings_path.exists():
                index.embeddings_path.unlink(missing_ok=True)
            pkl_path = index.index_path.with_suffix('.pkl')
            if pkl_path.exists():
                pkl_path.unlink(missing_ok=True)

            # Создаём новый пустой индекс
            if model_type == "sam":
                self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
            elif model_type == "yolo":
                self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")

            self.log_message(f"{status} индекс успешно очищен и создан заново", "SUCCESS")
            self._update_status()
        except Exception as e:
            self.log_message(f"Ошибка очистки {status} индекса: {e}", "ERROR")

    def _update_status(self):
        """Обновление информации о состоянии индексов"""
        sam_count = self.sam_index.get_total_vectors()
        yolo_count = self.yolo_index.get_total_vectors()
        
        text = f"SAM индекс: {sam_count} векторов | YOLO индекс: {yolo_count} векторов"
        self.status_label.setText(text)

    def showEvent(self, event):
        """Обновляем статус при открытии вкладки"""
        super().showEvent(event)
        self._update_status()