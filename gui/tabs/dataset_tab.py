from gui.tabs.base_tab import BaseTab
from core.preprocessing import DataPreprocessor
from PyQt6.QtWidgets import QPushButton, QLabel


class DatasetTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.preprocessor = DataPreprocessor()
        self.init_ui()

    def init_ui(self):
        title = QLabel("Предобработка датасета")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.layout.insertWidget(0, title)
        
        btn = QPushButton("Запустить очистку и предобработку датасета")
        btn.clicked.connect(self.run_preprocessing)
        self.layout.insertWidget(1, btn)

    def run_preprocessing(self):
        self.log_message("Начата предобработка датасета...")
        self.progress.setVisible(True)
        self.progress.setValue(10)
        
        count = self.preprocessor.run()
        
        self.progress.setValue(100)
        self.progress.setVisible(False)
        self.log_message(f"Предобработка завершена. Обработано изображений: {count}", "УСПЕХ")