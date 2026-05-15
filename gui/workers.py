from PyQt6.QtCore import QThread, pyqtSignal


class EmbeddingWorker(QThread):
    """Поток для извлечения эмбеддингов"""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(list)   # возвращаем список эмбеддингов

    def __init__(self, embedder, image_paths, mode="sam"):
        super().__init__()
        self.embedder = embedder
        self.image_paths = image_paths
        self.mode = mode

    def run(self):
        embeddings = []
        total = len(self.image_paths)

        for i, img_path in enumerate(self.image_paths):
            try:
                emb = self.embedder.get_embedding(str(img_path))
                embeddings.append(emb)
                
                # Обновляем прогресс
                progress_value = int((i + 1) / total * 100)
                self.progress.emit(progress_value)
                self.log.emit(f"Обработано {i+1}/{total}: {img_path.name}")
            except Exception as e:
                self.log.emit(f"Ошибка при обработке {img_path.name}: {e}")

        self.finished.emit(embeddings)