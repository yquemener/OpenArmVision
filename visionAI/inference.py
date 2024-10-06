"""
Inference Module for Object Detection using YOLOv5

This module provides a simple interface for loading a YOLOv5 model,
performing inference on images, and displaying the results.

Features:
- Load YOLOv5 models
- Perform inference on images
- Display results with bounding boxes
- Simple GUI for interaction

Dependencies:
- PyQt5
- PIL
- torch
- YOLOv5 (as a submodule)

Usage:
Run this script directly to launch the GUI application.

Note: Ensure that the YOLOv5 submodule is properly initialized before running.
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import yolov5

# Hack to make PyQt and cv2 load simultaneously
import os
from pathlib import Path
import PyQt5

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)


class InferenceModule:
    def __init__(self):
        self.model = None

    def load_model(self, weights_path):
        try:
            # Assuming YOLOv5 is in a subdirectory named 'yolov5'
            self.model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_image(self, image_path):
        return Image.open(image_path)

    def run_inference(self, image):
        results = self.model(image)
        return results.xyxy[0].cpu().numpy()

class InferenceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.inference_module = InferenceModule()
        self.image_path = None
        self.annotations = []
        self.original_image_size = None  # Pour stocker la taille originale de l'image
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        load_model_btn = QPushButton('Load Model')
        load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(load_model_btn)

        load_image_btn = QPushButton('Load Image')
        load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(load_image_btn)

        run_inference_btn = QPushButton('Run Inference')
        run_inference_btn.clicked.connect(self.run_inference)
        layout.addWidget(run_inference_btn)

        self.setLayout(layout)
        self.setWindowTitle('Inference Module GUI')
        self.setGeometry(300, 300, 400, 400)

    def load_model(self):
        weights_path, _ = QFileDialog.getOpenFileName(self, 'Load Model Weights', '', 'PT files (*.pt)')
        if weights_path:
            self.inference_module.load_model(weights_path)

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Load Image', '', 'Image files (*.jpg *.png)')
        if self.image_path:
            self.annotations = []  # Réinitialiser les annotations
            pixmap = QPixmap(self.image_path)
            self.original_image_size = pixmap.size()  # Stocker la taille originale
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            print(f"Image loaded: {self.image_path}")
            print(f"Original size: {self.original_image_size.width()}x{self.original_image_size.height()}")

    def run_inference(self):
        if self.image_path and self.inference_module.model:
            image = self.inference_module.load_image(self.image_path)
            self.annotations = self.inference_module.run_inference(image)
            print("Annotations:")
            for i, ann in enumerate(self.annotations):
                x1, y1, x2, y2, conf, cls = ann
                print(f"  {i+1}: Class {cls}, Confidence {conf:.2f}, Bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
            self.update()  # Ceci déclenchera un nouveau paintEvent

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image_label.pixmap() is not None and self.annotations is not None:
            painter = QPainter(self.image_label.pixmap())
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.SolidLine))

            # Calculer le facteur d'échelle
            current_size = self.image_label.pixmap().size()
            scale_x = current_size.width() / self.original_image_size.width()
            scale_y = current_size.height() / self.original_image_size.height()

            for ann in self.annotations:
                x1, y1, x2, y2, conf, cls = ann
                # Appliquer l'échelle aux coordonnées
                scaled_x1 = int(x1 * scale_x)
                scaled_y1 = int(y1 * scale_y)
                scaled_width = int((x2 - x1) * scale_x)
                scaled_height = int((y2 - y1) * scale_y)
                painter.drawRect(scaled_x1, scaled_y1, scaled_width, scaled_height)

def main():
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()