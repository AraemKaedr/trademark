from gui.tabs.base_tab import BaseTab
from PyQt6.QtWidgets import QLabel


class ResultsTab(BaseTab):
    def __init__(self):
        super().__init__()
        self.result_label = QLabel("Результаты поиска появятся здесь")
        self.layout.insertWidget(0, self.result_label)

    def show_results(self, results):
        self.log_message("Отображение результатов поиска...")
        self.result_label.setText(f"Найдено совпадений: {len(results)}")