from PyQt6.QtCore import QThread, pyqtSignal

class Worker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        try:
            self.func()
        except Exception as e:
            self.log.emit(f"Ошибка: {e}")
        finally:
            self.finished.emit()