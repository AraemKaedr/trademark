# Здесь будет полный код вкладки позже
# gui/tabs/index_tab.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Индекс"))
        # Здесь будет полный код вкладки позже