from gui.tabs.base_tab import BaseTab
from core.preprocessing import DataPreprocessor
from PyQt6.QtWidgets import QPushButton, QLabel, QHBoxLayout, QWidget
import subprocess
from pathlib import Path


class DatasetTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.preprocessor = DataPreprocessor()
        self.init_ui()

    def init_ui(self):
        title = QLabel("Предобработка датасета и обучение моделей")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, title)
        
        # Кнопки
        btn_layout = QHBoxLayout()

        self.btn_preprocess = QPushButton("Очистка и предобработка датасета (вырезать логотипы)")
        self.btn_preprocess.clicked.connect(self.run_preprocessing)
        btn_layout.addWidget(self.btn_preprocess)

        container = QWidget()
        container.setLayout(btn_layout)
        self.layout.insertWidget(1, container)

    def run_preprocessing(self):
        self.log_message("Начата предобработка датасета...")
        self.progress.setVisible(True)
        self.progress.setValue(10)
        
        count = self.preprocessor.run()
        
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.log_message(f"Предобработка завершена. Вырезано {count} логотипов.", "УСПЕХ")