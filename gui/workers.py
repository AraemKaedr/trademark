import time
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal


class EmbeddingWorker(QThread):
    """Поток для извлечения эмбеддингов"""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(list)   # возвращаем список эмбеддингов

    def __init__(self, embedder, image_paths, mode="resnet"):
        super().__init__()
        self.embedder = embedder
        self.image_paths = image_paths
        self.mode = mode

    def run(self):
        embeddings = []
        total = len(self.image_paths)
        start_time = time.time()  # Время старта всей обработки

        for i, img_path in enumerate(self.image_paths):
            try:
                # Фиксируем время начала обработки одной картинки
                item_start = time.time()
                
                # Извлекаем эмбеддинг
                emb = self.embedder.get_embedding(str(img_path))
                embeddings.append(emb)

                # Считаем тайминги
                elapsed_total = time.time() - start_time
                avg_speed = elapsed_total / (i + 1)  # Среднее время на 1 картинку (сек)
                remaining_items = total - (i + 1)
                eta_seconds = int(remaining_items * avg_speed)  # Сколько секунд осталось
                
                # Форматируем оставшееся время в ЧЧ:ММ:СС
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
                
                # Обновляем прогресс
                progress_value = int((i + 1) / total * 100)
                self.progress.emit(progress_value)
                
                # Отправляем в лог интерфейса развернутую информацию
                current_img = Path(img_path).name
                self.log.emit(
                    f"[{progress_value}%] Обработано {i+1}/{total} | "
                    f"Скорость: {avg_speed:.2f} сек/изд | "
                    f"Осталось: {eta_str} | Файл: {current_img}"
                )
            except Exception as e:
                self.log.emit(f"Ошибка при обработке {Path(img_path).name}: {e}")

        self.finished.emit(embeddings)