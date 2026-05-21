from gui.tabs.base_tab import BaseTab
from core.vector_db.faiss_index import FaissIndex
from core.embedders.resnet_embedder import ResNetEmbedder
from core.embedders.yolo_embedder import YOLOEmbedder
from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QLabel


class ManagerTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.resnet_embedder = ResNetEmbedder()
        self.yolo_embedder = YOLOEmbedder()
        self.resnet_index = FaissIndex(dimension=2048, index_path="indexes/resnet_index.faiss")
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

        btn_clear = QPushButton("Очистить все индексы (сброс)")
        btn_clear.clicked.connect(self.clear_indexes)
        btn_layout.addWidget(btn_clear)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

    def update_status(self):
        self.log_message("Статус системы")
        
        # Модели
        resnet_status = "Загружена" if not getattr(self.resnet_embedder, 'use_stub', False) else "Заглушка"
        yolo_status = "Загружена" if not getattr(self.yolo_embedder, 'use_stub', False) else "Заглушка"
        
        self.log_message(f"ResNet50: {resnet_status}")
        self.log_message(f"YOLOv8n: {yolo_status}")

        # Индексы
        self.log_message(f"ResNet индекс: {self.resnet_index.get_total_vectors()} векторов")
        self.log_message(f"YOLO индекс: {self.yolo_index.get_total_vectors()} векторов")

        self.log_message("Статус обновлён", "УСПЕХ")

    def clear_indexes(self):
        reply = self._show_question("Очистить все индексы?", 
                                  "Это действие удалит все сохранённые эмбеддинги.\nПродолжить?")
        if reply:
            try:
                # Удаляем файлы индексов
                for idx in [self.resnet_index, self.yolo_index]:
                    for p in [idx.index_path, idx.embeddings_path, idx.index_path.with_suffix('.pkl')]:
                        if p.exists():
                            p.unlink(missing_ok=True)
                
                self.log_message("Все индексы успешно очищены", "УСПЕХ")
            except Exception as e:
                self.log_message(f"Ошибка очистки: {e}", "ERROR")