"""
This file implements a self-contained widget for image annotation using PyQt5.

Features:
1. Display an image as background
2. Allow drawing and editing of points (as crosses), lines, and rectangles on the image
3. Switch between different annotation modes
4. Temporary shapes in red, finalized shapes in blue
5. All shapes have 1 pixel thickness
6. Temporary point is a larger cross, finalized point is a smaller cross
7. All shapes (including points) have a temporary state while dragging
8. Integrated controls for mode selection and clearing annotations

Constraints and Considerations:
1. Uses PyQt5 for the GUI
2. Implements custom painting for annotations
3. Supports multiple annotation types (point, line, rectangle)
4. Provides a clear and intuitive user interface
5. Allows for easy extension with additional features in the future
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal

class AnnotationWidget(QWidget):
    annotation_changed = pyqtSignal(list)  # Signal pour indiquer un changement d'annotation

    def __init__(self):
        super().__init__()
        self.image = QPixmap()
        self.points = []
        self.current_shape = None
        self.lines = []
        self.rectangles = []
        self.mode = "point"  # Default mode
        self.drawing = False
        self.current_image = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Controls
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        point_button = QPushButton("Point")
        line_button = QPushButton("Line")
        rectangle_button = QPushButton("Rectangle")
        clear_button = QPushButton("Clear")

        point_button.clicked.connect(lambda: self.set_mode("point"))
        line_button.clicked.connect(lambda: self.set_mode("line"))
        rectangle_button.clicked.connect(lambda: self.set_mode("rectangle"))
        clear_button.clicked.connect(self.clear_annotations)

        button_layout.addWidget(point_button)
        button_layout.addWidget(line_button)
        button_layout.addWidget(rectangle_button)
        button_layout.addWidget(clear_button)

        # Status bar
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        self.update_status()

        # Set minimum size for the widget
        self.setMinimumSize(400, 300)

    def paintEvent(self, event):
        super().paintEvent(event)  # Call the parent's paintEvent
        painter = QPainter(self)
        if self.image and not self.image.isNull():
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
            self.annotation_changed.emit(self.get_annotations())  # Émettre le signal

    def set_mode(self, mode):
        self.mode = mode
        self.update_status()

    def clear_annotations(self):
        self.points = []
        self.lines = []
        self.rectangles = []
        self.current_shape = None
        self.update()

    def load_image(self, image_path):
        self.image = QPixmap(image_path)
        self.current_image = image_path
        self.update()  # Use self.update() instead of self.annotation_area.update()

    def load_annotations(self, annotations):
        self.points = []
        self.lines = []
        self.rectangles = []
        for ann_type, x1, y1, x2, y2 in annotations:
            if ann_type == 'point':
                self.points.append(QPoint(int(x1), int(y1)))
            elif ann_type == 'line':
                self.lines.append([QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))])
            elif ann_type == 'rectangle':
                self.rectangles.append([QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))])
        self.update()

    def get_annotations(self):
        annotations = []
        for point in self.points:
            annotations.append(('point', point.x(), point.y(), None, None))
        for line in self.lines:
            annotations.append(('line', line[0].x(), line[0].y(), line[1].x(), line[1].y()))
        for rect in self.rectangles:
            annotations.append(('rectangle', rect[0].x(), rect[0].y(), rect[1].x(), rect[1].y()))
        return annotations

    def update_status(self):
        status_text = f"Mode: {self.mode.capitalize()}"
        self.status_label.setText(status_text)



if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    import sys
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Image Annotation Tool")
            self.setGeometry(100, 100, 800, 600)

            self.annotation_widget = AnnotationWidget()
            self.setCentralWidget(self.annotation_widget)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())