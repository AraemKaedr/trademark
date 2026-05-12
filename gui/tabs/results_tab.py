from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea,
    QGridLayout, QGroupBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout  = QVBoxLayout(self)

        title = QLabel("Результаты поиска")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout .addWidget(title)

        # Область с прокруткой
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setSpacing(12)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll)

        self.status = QLabel("Здесь будут показаны результаты поиска")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status)

    def show_results(self, query_path: str, distances, indices):
        """Отображение результатов поиска"""
        # Очистка предыдущих (старых) результатов
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().deleteLater()

        self.status.setText(f"Найдено {len(distances)} похожих изображений")

        row = 0

        # Исходное изображение
        orig_group = QGroupBox("Исходное изображение")
        vbox = QVBoxLayout()

        pixmap = QPixmap(query_path).scaled(280, 280, Qt.AspectRatioMode.KeepAspectRatio)
        lbl = QLabel()
        lbl.setPixmap(pixmap)
        vbox.addWidget(lbl)
        
        vbox.addWidget(QLabel("Оригинальность: рассчитывается..."))
        orig_group.setLayout(vbox)
        self.grid.addWidget(orig_group, row, 0, 1, 2)

        row += 1

        # Найденные изображения
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            similarity = (1 - float(dist)) * 100  # Преобразуем расстояние в процент схожести

            group = QGroupBox(f"#{i+1} — {similarity:.1f}% схожести")
            vbox = QVBoxLayout()

            # Здесь можно добавить реальное изображение по пути из индекса
            img_label = QLabel(f"Изображение {idx}")
            vbox.addWidget(img_label)

            vbox.addWidget(QLabel(f"Схожесть: {similarity:.2f}%"))
            group.setLayout(vbox)

            self.grid.addWidget(group, row + (i // 3), i % 3)

        self.status.setText(f"Результаты поиска обновлены! Найдено {len(distances)} похожих изображений.")