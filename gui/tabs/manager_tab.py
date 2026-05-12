from gui.tabs.base_tab import BaseTab
from PyQt6.QtWidgets import QPushButton, QLabel


class ManagerTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.status = QLabel("Менеджер моделей и индексов")
        self.layout.insertWidget(0, self.status)
        
        btn = QPushButton("Обновить статус моделей")
        btn.clicked.connect(self.update_status)
        self.layout.insertWidget(1, btn)

    def update_status(self):
        self.log_message("Статус моделей обновлён", "УСПЕХ")