from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from core.embedders.sam_embedder import SAMEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel


class ManagerTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.sam_embedder = SAMEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
        self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
        
        self.init_ui()

    def init_ui(self):
        label = QLabel("Менеджер моделей и индексов")
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, label)

        btn_layout = QHBoxLayout()

        btn_status = QPushButton("Обновить статус")
        btn_status.clicked.connect(self.update_status)
        btn_layout.addWidget(btn_status)

        btn_clear = QPushButton("Очистить индексы")
        btn_clear.clicked.connect(self.clear_indexes)
        btn_layout.addWidget(btn_clear)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

    def update_status(self):
        self.log_message("=== Статус системы ===")
        
        # Модели
        sam_status = "Загружена" if not getattr(self.sam_embedder, 'use_stub', True) else "Заглушка"
        yolo_status = "Загружена" if not getattr(self.yolo_embedder, 'use_stub', True) else "Заглушка"
        
        self.log_message(f"SAM ViT-B: {sam_status}")
        self.log_message(f"YOLOv8n: {yolo_status}")

        # Индексы
        self.log_message(f"SAM индекс: {self.sam_index.get_total_vectors()} векторов")
        self.log_message(f"YOLO индекс: {self.yolo_index.get_total_vectors()} векторов")

        self.log_message("=== Статус обновлён ===", "SUCCESS")

    def clear_indexes(self):
        reply = self._show_question("Очистить все индексы?", 
                                  "Это действие удалит все сохранённые эмбеддинги.\nПродолжить?")
        if reply:
            try:
                # Удаляем файлы индексов
                for index in [self.sam_index, self.yolo_index]:
                    if index.index_path.exists():
                        index.index_path.unlink(missing_ok=True)
                    pkl = index.index_path.with_suffix('.pkl')
                    if pkl.exists():
                        pkl.unlink(missing_ok=True)
                
                # Пересоздаём индексы
                self.sam_index = FaissIndex(dimension=256, index_path="indexes/sam_index.faiss")
                self.yolo_index = FaissIndex(dimension=512, index_path="indexes/yolo_index.faiss")
                
                self.log_message("Все индексы успешно очищены", "УСПЕХ")
            except Exception as e:
                self.log_message(f"Ошибка очистки: {e}", "ERROR")