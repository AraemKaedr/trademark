from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QGridLayout, 
    QGroupBox, QLabel, QHBoxLayout
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("Результаты поиска (сравнение ResNet и YOLO)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 12px;")
        layout.addWidget(title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(15)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        self.status = QLabel("Результаты появятся здесь после поиска")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def show_results(self, query_path: str, resnet_dist, resnet_idx, yolo_dist, yolo_idx, originality=0.0, resnet_index=None, yolo_index=None):
        """Отображение сравнительных результатов"""
        # Очистка предыдущих (старых) результатов
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        self.status.setText(f"По ResNet найдено {len(resnet_dist)} похожих изображений")
        self.status.setText(f"По YOLO найдено {len(yolo_dist)} похожих изображений")
        
        row = 0
        displayed = 0

        # Исходное изображение
        orig_group = QGroupBox("Исходное изображение")
        vbox = QVBoxLayout()
        pixmap = QPixmap(query_path).scaled(320, 320, Qt.AspectRatioMode.KeepAspectRatio)
        lbl = QLabel()
        lbl.setPixmap(pixmap)
        vbox.addWidget(lbl)
        vbox.addWidget(QLabel(f"<b>Оригинальность:</b> {originality:.1f}%"))
        orig_group.setLayout(vbox)
        self.grid.addWidget(orig_group, row, 0, 1, 2)
        row += 1

        # Результаты
        max_results = min(len(resnet_dist), len(yolo_dist), 5)
        for i in range(len(resnet_dist)):
            if displayed >= max_results:
                break

            resnet_path = resnet_index.get_path_by_index(resnet_idx[i]) if resnet_index else None
            yolo_path = yolo_index.get_path_by_index(yolo_idx[i]) if yolo_index else None

            # Пропускаем, если найденное изображение — это сам запрос
            if resnet_path and Path(resnet_path).resolve() == Path(query_path).resolve():
                continue
            if yolo_path and Path(yolo_path).resolve() == Path(query_path).resolve():
                continue

            # ResNet колонка
            resnet_sim = (1 - float(resnet_dist[i])) * 100

            group = QGroupBox(f"Результат #{i+1}")
            hbox = QHBoxLayout()

            # ResNet колонка
            resnet_box = QGroupBox("ResNet50")
            resnet_v = QVBoxLayout()
            resnet_v.addWidget(QLabel(f"Схожесть: <b>{resnet_sim:.1f}%</b>"))
            if resnet_path and Path(resnet_path).exists():
                resnet_pix = QPixmap(resnet_path).scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio)
                resnet_lbl = QLabel()
                resnet_lbl.setPixmap(resnet_pix)
                resnet_v.addWidget(resnet_lbl)
            else:
                resnet_v.addWidget(QLabel("Изображение не найдено"))
            resnet_box.setLayout(resnet_v)

            # YOLO колонка
            yolo_sim = (1 - float(yolo_dist[i])) * 100

            yolo_box = QGroupBox("YOLOv8n")
            yolo_v = QVBoxLayout()
            yolo_v.addWidget(QLabel(f"Схожесть: <b>{yolo_sim:.1f}%</b>"))
            if yolo_path and Path(yolo_path).exists():
                yolo_pix = QPixmap(yolo_path).scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio)
                yolo_lbl = QLabel()
                yolo_lbl.setPixmap(yolo_pix)
                yolo_v.addWidget(yolo_lbl)
            else:
                yolo_v.addWidget(QLabel("Изображение не найдено"))
            yolo_box.setLayout(yolo_v)

            hbox.addWidget(resnet_box)
            hbox.addWidget(yolo_box)
            group.setLayout(hbox)

            self.grid.addWidget(group, row, displayed % 2)
            if displayed % 2 == 1:
                row += 1
            displayed += 1

        self.status.setText(f"Сравнение ResNet и YOLO завершено!")
        self.status.setText(f"Показано топ-{displayed} наиболее похожих логотипов (результаты сравнения ResNet50 и YOLO)")