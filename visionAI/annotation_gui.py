"""
This file implements a graphical user interface for the image annotation tool using PyQt5.

Features:
1. Display an image as background
2. Allow drawing of points (as crosses), lines, and rectangles on the image
3. Switch between different annotation modes
4. Temporary shapes in red, finalized shapes in blue
5. All shapes have 1 pixel thickness
6. Temporary point is a larger cross, finalized point is a smaller cross
7. All shapes (including points) have a temporary state while dragging

Constraints and Considerations:
1. Uses PyQt5 for the GUI
2. Implements custom painting for annotations
3. Supports multiple annotation types (point, line, rectangle)
4. Provides a clear and intuitive user interface
5. Allows for easy extension with additional features in the future
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect

class AnnotationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image = QPixmap("path/to/your/image.jpg")
        self.points = []
        self.current_shape = None
        self.lines = []
        self.rectangles = []
        self.mode = "point"  # Default mode
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

        # Draw finalized shapes
        painter.setPen(QPen(QColor(0, 0, 255), 1))  # Blue, 1px thickness

        # Draw points (as smaller crosses)
        for point in self.points:
            painter.drawLine(point.x() - 3, point.y(), point.x() + 3, point.y())
            painter.drawLine(point.x(), point.y() - 3, point.x(), point.y() + 3)

        # Draw lines
        for line in self.lines:
            painter.drawLine(line[0], line[1])

        # Draw rectangles
        for rect in self.rectangles:
            painter.drawRect(QRect(rect[0], rect[1]))

        # Draw temporary shape
        if self.current_shape:
            painter.setPen(QPen(QColor(255, 0, 0), 1))  # Red, 1px thickness
            if self.mode == "point":
                # Draw larger cross for temporary point
                x, y = self.current_shape.x(), self.current_shape.y()
                painter.drawLine(x - 25, y, x + 25, y)
                painter.drawLine(x, y - 25, x, y + 25)
            elif self.mode == "line":
                painter.drawLine(self.current_shape[0], self.current_shape[1])
            elif self.mode == "rectangle":
                painter.drawRect(QRect(self.current_shape[0], self.current_shape[1]))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            if self.mode == "point":
                self.current_shape = event.pos()
            elif self.mode in ["line", "rectangle"]:
                self.current_shape = [event.pos(), event.pos()]
        self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            if self.mode == "point":
                self.current_shape = event.pos()
            elif self.mode in ["line", "rectangle"]:
                self.current_shape[1] = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.mode == "point":
                self.points.append(self.current_shape)
            elif self.mode == "line":
                self.lines.append(self.current_shape)
            elif self.mode == "rectangle":
                self.rectangles.append(self.current_shape)
            self.current_shape = None
        self.update()

    def set_mode(self, mode):
        self.mode = mode

    def clear_annotations(self):
        self.points = []
        self.lines = []
        self.rectangles = []
        self.current_shape = None
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Annotation Tool")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.annotation_widget = AnnotationWidget()
        layout.addWidget(self.annotation_widget)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        point_button = QPushButton("Point")
        line_button = QPushButton("Line")
        rectangle_button = QPushButton("Rectangle")
        clear_button = QPushButton("Clear")

        point_button.clicked.connect(lambda: self.set_mode("point"))
        line_button.clicked.connect(lambda: self.set_mode("line"))
        rectangle_button.clicked.connect(lambda: self.set_mode("rectangle"))
        clear_button.clicked.connect(self.annotation_widget.clear_annotations)

        button_layout.addWidget(point_button)
        button_layout.addWidget(line_button)
        button_layout.addWidget(rectangle_button)
        button_layout.addWidget(clear_button)

        self.status_bar = self.statusBar()
        self.status_label = QLabel()
        self.status_bar.addPermanentWidget(self.status_label)
        self.update_status()

    def set_mode(self, mode):
        self.annotation_widget.set_mode(mode)
        self.update_status()

    def update_status(self):
        status_text = f"Mode: {self.annotation_widget.mode.capitalize()}"
        self.status_label.setText(status_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())