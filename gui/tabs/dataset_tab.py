from gui.tabs.base_tab import BaseTab
from core.preprocessing import DataPreprocessor


class DatasetTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.preprocessor = DataPreprocessor()
        
        from PyQt6.QtWidgets import QPushButton
        btn = QPushButton("Запустить очистку и предобработку датасета")
        btn.clicked.connect(self.run_preprocessing)
        self.layout.insertWidget(0, btn)

    def run_preprocessing(self):
        self.log_message("Начата предобработка датасета...")
        self.progress.setVisible(True)
        self.progress.setValue(20)
        
        count = self.preprocessor.run()
        
        self.progress.setValue(100)
        self.log_message(f"Предобработка завершена. Обработано изображений: {count}", "УСПЕХ")