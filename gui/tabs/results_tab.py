# Здесь будет полный код вкладки позже
# gui/tabs/results_tab.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class DatasetTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Результаты"))
        # Здесь будет полный код вкладки позже