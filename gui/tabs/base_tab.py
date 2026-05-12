from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QProgressBar


class BaseTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.layout.addWidget(self.log)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.layout.addWidget(self.progress)

    def log_message(self, message: str, level: str = "INFO"):
        self.log.append(f"[{level}] {message}")