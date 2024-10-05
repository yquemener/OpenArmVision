"""
This file implements a self-contained widget for image annotation using PyQt5.

Features:
1. Display an image as background using QGraphicsView and QGraphicsScene
2. Allow drawing and editing of points (as crosses), lines, and rectangles on the image
3. Switch between different annotation modes
4. Temporary shapes in red, finalized shapes in blue
5. All shapes have 1 pixel thickness
6. Temporary point is a larger cross, finalized point is a smaller cross
7. All shapes (including points) have a temporary state while dragging
8. Integrated controls for mode selection and clearing annotations
9. Annotations remain correctly positioned when resizing the widget
10. Selection tool to select and move existing annotations
11. Delete tool to delete selected annotations

Constraints and Considerations:
1. Uses PyQt5 for the GUI
2. Implements custom painting for annotations using QGraphicsItems
3. Supports multiple annotation types (point, line, rectangle)
4. Provides a clear and intuitive user interface
5. Allows for easy extension with additional features in the future
6. Uses QGraphicsView and QGraphicsScene for improved rendering and scaling
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup,
                             QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem,
                             QShortcut)
from PyQt5.QtGui import QPixmap, QPen, QColor, QPainter, QKeySequence
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from math import sqrt

class AnnotationItem(QGraphicsItem):
    def __init__(self, item_type, data, color=Qt.blue):
        super().__init__()
        self.item_type = item_type
        self.data = data
        self.color = color
        self.setFlag(QGraphicsItem.ItemIsMovable)
        if self.item_type == "point":
            self.setPos(data)

    def boundingRect(self):
        if self.item_type == "point":
            size = 100 if self.color == Qt.red else 18
            return QRectF(-size/2, -size/2, size, size)
        elif self.item_type == "line":
            return QRectF(self.data[0], self.data[1]).normalized()
        elif self.item_type == "rectangle":
            return QRectF(self.data[0], self.data[1]).normalized()

    def paint(self, painter, option, widget):
        painter.setPen(QPen(self.color, 1))
        if self.item_type == "point":
            size = 100 if self.color == Qt.red else 18
            painter.drawLine(int(-size/2), 0, int(size/2), 0)
            painter.drawLine(0, int(-size/2), 0, int(size/2))
        elif self.item_type == "line":
            painter.drawLine(self.data[0], self.data[1])
        elif self.item_type == "rectangle":
            painter.drawRect(QRectF(self.data[0], self.data[1]).normalized())

class CustomGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent

    def mousePressEvent(self, event):
        self.parent_widget.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.parent_widget.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.parent_widget.mouseReleaseEvent(event)

class AnnotationWidget(QWidget):
    annotation_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.image_item = None
        self.current_item = None
        self.mode = "point"
        self.drawing = False
        self.selected_item = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        main_layout.addWidget(self.view, 1)

        # Controls
        button_layout = QVBoxLayout()
        main_layout.addLayout(button_layout)

        self.button_group = QButtonGroup()
        self.button_group.setExclusive(True)

        point_button = QPushButton("Point")
        line_button = QPushButton("Line")
        rectangle_button = QPushButton("Rectangle")
        select_button = QPushButton("Select")
        delete_button = QPushButton("Delete")
        clear_button = QPushButton("Clear")

        for button in [point_button, line_button, rectangle_button, select_button]:
            button.setCheckable(True)
            self.button_group.addButton(button)

        point_button.setChecked(True)  # Default mode
        
        point_button.clicked.connect(lambda: self.set_mode("point"))
        line_button.clicked.connect(lambda: self.set_mode("line"))
        rectangle_button.clicked.connect(lambda: self.set_mode("rectangle"))
        select_button.clicked.connect(lambda: self.set_mode("select"))
        delete_button.clicked.connect(self.delete_selected_item)
        clear_button.clicked.connect(self.clear_annotations)

        button_layout.addWidget(point_button)
        button_layout.addWidget(line_button)
        button_layout.addWidget(rectangle_button)
        button_layout.addWidget(select_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(clear_button)
        button_layout.addStretch()

        self.setMinimumSize(600, 400)

        # Keyboard shortcuts
        select_button.setShortcut(QKeySequence("S"))
        delete_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self)
        delete_shortcut.activated.connect(self.delete_selected_item)

    def set_mode(self, mode):
        self.mode = mode
        if mode != "select" and self.selected_item:
            self.selected_item.color = Qt.blue
            self.selected_item.update()
            self.selected_item = None

    def mousePressEvent(self, event):
        if self.view.viewport().rect().contains(event.pos()):
            scene_pos = self.view.mapToScene(event.pos())
            if event.button() == Qt.LeftButton:
                if self.mode == "select":
                    self.select_nearest_item(scene_pos)
                else:
                    self.drawing = True
                    if self.mode == "point":
                        self.current_item = AnnotationItem("point", scene_pos, Qt.red)
                    elif self.mode in ["line", "rectangle"]:
                        self.current_item = AnnotationItem(self.mode, [scene_pos, scene_pos], Qt.red)
                    self.scene.addItem(self.current_item)

    def mouseMoveEvent(self, event):
        if self.drawing and self.view.viewport().rect().contains(event.pos()):
            scene_pos = self.view.mapToScene(event.pos())
            if self.mode == "point":
                self.current_item.setPos(scene_pos)
            elif self.mode in ["line", "rectangle"]:
                self.current_item.data[1] = scene_pos
            self.current_item.update()
            self.scene.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            scene_pos = self.view.mapToScene(event.pos())
            if self.mode == "point":
                self.current_item.setPos(scene_pos)
            elif self.mode in ["line", "rectangle"]:
                self.current_item.data[1] = scene_pos
            self.current_item.color = Qt.blue
            self.current_item.update()
            self.scene.update()
            self.current_item = None
            self.annotation_changed.emit(self.get_annotations())

    def select_nearest_item(self, pos):
        min_distance = float('inf')
        nearest_item = None
        for item in self.scene.items():
            if isinstance(item, AnnotationItem):
                distance = self.distance_to_item(item, pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_item = item

        if self.selected_item:
            self.selected_item.color = Qt.blue
            self.selected_item.update()

        if nearest_item:
            self.selected_item = nearest_item
            self.selected_item.color = Qt.red
            self.selected_item.update()

    def distance_to_item(self, item, pos):
        if item.item_type == "point":
            return sqrt((item.pos().x() - pos.x())**2 + (item.pos().y() - pos.y())**2)
        elif item.item_type == "line":
            return min(self.point_to_line_distance(pos, item.data[0], item.data[1]),
                       self.point_to_point_distance(pos, item.data[0]),
                       self.point_to_point_distance(pos, item.data[1]))
        elif item.item_type == "rectangle":
            rect = QRectF(item.data[0], item.data[1]).normalized()
            if rect.contains(pos):
                return 0
            else:
                return min(self.point_to_line_distance(pos, rect.topLeft(), rect.topRight()),
                           self.point_to_line_distance(pos, rect.topRight(), rect.bottomRight()),
                           self.point_to_line_distance(pos, rect.bottomRight(), rect.bottomLeft()),
                           self.point_to_line_distance(pos, rect.bottomLeft(), rect.topLeft()))

    def point_to_line_distance(self, p, a, b):
        # Calculate the distance from point p to line segment ab
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()

        # Calculate the length of the line segment
        length = sqrt((bx - ax)**2 + (by - ay)**2)

        # If the length is zero, return the distance to either point
        if length == 0:
            return sqrt((px - ax)**2 + (py - ay)**2)

        # Calculate the projection of p onto the line
        t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / (length**2)

        # If t is outside [0, 1], the closest point is an endpoint
        if t < 0:
            return sqrt((px - ax)**2 + (py - ay)**2)
        elif t > 1:
            return sqrt((px - bx)**2 + (py - by)**2)

        # Calculate the closest point on the line
        closest_x = ax + t * (bx - ax)
        closest_y = ay + t * (by - ay)

        # Return the distance to the closest point
        return sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def point_to_point_distance(self, p1, p2):
        return sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)

    def clear_annotations(self):
        self.scene.clear()
        self.current_item = None

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        if self.image_item:
            self.scene.removeItem(self.image_item)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        self.view.setSceneRect(self.image_item.boundingRect())
        self.fit_image_in_view()

    def fit_image_in_view(self):
        if self.image_item:
            self.view.fitInView(self.image_item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_image_in_view()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_image_in_view()

    def load_annotations(self, annotations):
        for item in self.scene.items():
            if isinstance(item, AnnotationItem):
                self.scene.removeItem(item)
        for ann_type, x1, y1, x2, y2 in annotations:
            if ann_type == 'point':
                item = AnnotationItem("point", QPointF(x1, y1))
            elif ann_type == 'line':
                item = AnnotationItem("line", [QPointF(x1, y1), QPointF(x2, y2)])
            elif ann_type == 'rectangle':
                item = AnnotationItem("rectangle", [QPointF(x1, y1), QPointF(x2, y2)])
            self.scene.addItem(item)

    def get_annotations(self):
        annotations = []
        for item in self.scene.items():
            if isinstance(item, AnnotationItem):
                if item.item_type == "point":
                    annotations.append(('point', item.pos().x(), item.pos().y(), None, None))
                elif item.item_type in ["line", "rectangle"]:
                    annotations.append((item.item_type, item.data[0].x(), item.data[0].y(), item.data[1].x(), item.data[1].y()))
        return annotations

    def delete_selected_item(self):
        if self.selected_item:
            self.scene.removeItem(self.selected_item)
            self.selected_item = None
            self.annotation_changed.emit(self.get_annotations())

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
            
            # Charger l'image d'exemple
            self.annotation_widget.load_image("test/candidates/2024-08-02_19-33-47.650549.jpg")

            # Ajouter une annotation de type point
            self.annotation_widget.load_annotations([('point', 100, 100, None, None)])

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())