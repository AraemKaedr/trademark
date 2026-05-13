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
        
        title = QLabel("Результаты поиска (сравнение SAM и YOLO)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(15)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        self.status = QLabel("Результаты появятся здесь")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def show_results(self, query_path: str, sam_dist, sam_idx, yolo_dist, yolo_idx):
        """Отображение сравнительных результатов"""
        # Очистка предыдущих (старых) результатов
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()

        self.status.setText(f"По SAM найдено {len(sam_dist)} похожих изображений")
        self.status.setText(f"По YOLO найдено {len(yolo_dist)} похожих изображений")
        
        row = 0

        # Исходное изображение
        orig_group = QGroupBox("Исходное изображение")
        vbox = QVBoxLayout()
        pixmap = QPixmap(query_path).scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio)
        lbl = QLabel()
        lbl.setPixmap(pixmap)
        vbox.addWidget(lbl)
        vbox.addWidget(QLabel("<b>Оригинальность:</b> рассчитывается..."))
        orig_group.setLayout(vbox)
        self.grid.addWidget(orig_group, row, 0, 1, 2)
        row += 1

        # Результаты
        max_results = min(len(sam_dist), len(yolo_dist))
        for i in range(max_results):
            sam_sim = (1 - float(sam_dist[i])) * 100
            yolo_sim = (1 - float(yolo_dist[i])) * 100

            group = QGroupBox(f"Результат #{i+1}")
            hbox = QHBoxLayout()

            # SAM колонка
            sam_box = QGroupBox("SAM ViT-B")
            sam_v = QVBoxLayout()
            sam_v.addWidget(QLabel(f"Схожесть: <b>{sam_sim:.1f}%</b>"))
            sam_box.setLayout(sam_v)

            # YOLO колонка
            yolo_box = QGroupBox("YOLOv8n")
            yolo_v = QVBoxLayout()
            yolo_v.addWidget(QLabel(f"Схожесть: <b>{yolo_sim:.1f}%</b>"))
            yolo_box.setLayout(yolo_v)

            hbox.addWidget(sam_box)
            hbox.addWidget(yolo_box)
            group.setLayout(hbox)

            self.grid.addWidget(group, row, i % 2)
            if i % 2 == 1:
                row += 1

        self.status.setText(f"Сравнение SAM и YOLO завершено!")
        self.status.setText(f"Показано {max_results} результатов сравнения")