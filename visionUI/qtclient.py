import torch
import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QRect, QRectF, QStringListModel, QAbstractListModel
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
import cv2
import numpy as np
from PIL import Image
import time
import sqlite3
import os
from pathlib import Path

# Hack to make PyQt and cv2 load simultaneously
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)


class Capture:
    SOURCE_TYPE_NONE = 0
    SOURCE_TYPE_CAMERA = 1
    SOURCE_TYPE_FILE = 2

    def __init__(self):
        self.source_type = Capture.SOURCE_TYPE_NONE
        self.video_capture = None
        self.file_path = None
        self.last_capture = None
        self.needs_refresh = False
        self.current_np = None

    def open_camera(self, index):
        if self.source_type == Capture.SOURCE_TYPE_CAMERA or self.video_capture is not None:
            self.video_capture.release()
        self.source_type = Capture.SOURCE_TYPE_CAMERA
        self.video_capture = cv2.VideoCapture(index)
        self.needs_refresh = True

    def open_file(self, path):
        self.source_type = Capture.SOURCE_TYPE_FILE
        self.file_path = path
        self.needs_refresh = True

    def refresh(self):
        if self.source_type == Capture.SOURCE_TYPE_CAMERA:
            ctime = time.time()
            if self.last_capture is None or ctime - self.last_capture> 0.016:
                self.needs_refresh=True

        if self.needs_refresh:
            if self.source_type == Capture.SOURCE_TYPE_CAMERA:
                if self.video_capture is not None:
                    _, cv2_im = self.video_capture.read()
                    self.current_np = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            if self.source_type == Capture.SOURCE_TYPE_FILE:
                if self.file_path is not None:
                    self.current_np = np.array(Image.open(self.file_path))
            self.last_capture = time.time()
            self.needs_refresh = False


class TrainingCollection:
    def __init__(self, collection_path):
        self.path = collection_path
        self.connection = sqlite3.connect(self.path+"/collection.db")
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [x[0] for x in cursor.fetchall()]
        cursor.close()
        if "models" not in tables:
                cursor = self.connection.cursor()
                cursor.execute("""CREATE TABLE models 
                                (id integer primary key,
                                rel_path text, abs_path text, name text,
                                creation_date timestamp, last_training timestamp,
                                score real, epochs integer, samples integer)""")
                cursor.close()

        if "classes" not in tables:
            cursor = self.connection.cursor()
            cursor.execute("""CREATE TABLE classes
                                (id INTEGER PRIMARY KEY,
                                name TEXT, class_id INTEGER, model_id INTEGER 
                                rel_path text, abs_path text,
                                creation_date timestamp, last_training timestamp,
                                score real, epochs integer, samples integer,
                                FOREIGN KEY(model_id) REFERENCES model(id))""")
            cursor.close()

        if "training_samples" not in tables:
            cursor = self.connection.cursor()
            cursor.execute("""CREATE TABLE training_samples
                                (id INTEGER PRIMARY KEY,
                                image_rel_path TEXT, image_abs_path TEXT,
                                labels_rel_path TEXT, labels_abs_path)""")
            cursor.close()

        if "candidate_samples" not in tables:
            cursor = self.connection.cursor()
            cursor.execute("""CREATE TABLE candidate_samples
                                (id INTEGER PRIMARY KEY,
                                rel_path TEXT, abs_path TEXT)""")
            cursor.close()


class DetectionResultsModel(QAbstractListModel):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def rowCount(self, parent):
        return len(self.parent.current_yolo_results)

    def data(self, index, role):
        return self.parent.current_yolo_results[index.row()]


class App(QApplication):
    def __init__(self):
        super().__init__([])
        self.collection = TrainingCollection(".")
        self.capture = Capture()
        Form, Window = uic.loadUiType("mainwindow.ui")
        self.window = Window()
        self.form = Form()
        self.form.setupUi(self.window)
        self.form.splitter.setSizes([400, 200])

        self.form.imageView.mousePressEvent = self.imageView_onclick

        self.scene = QGraphicsScene()
        self.background_pixmap = QPixmap(QImage(b'\0\0\0\0', 1, 1, QImage.Format_ARGB32))
        self.background_image_item = self.scene.addPixmap(self.background_pixmap)
        self.background_image_item.setVisible(True)
        self.form.imageView.setScene(self.scene)
        self.form.zoomedView.setScene(self.scene)
        self.form.zoomedView.scale(10,10)
        self.poi_lines = list()

        self.model = None
        self.current_yolo_results = list()

        self.form.listROI.setModel(DetectionResultsModel(self))

        self.refresh_timer = QTimer(self)
        # self.refresh_timer.start(16)
        self.refresh_timer.timeout.connect(self.refresh_video)
        self.form.yoloThresholdSlider.valueChanged.connect(self.change_yolo_threshold)
        self.form.buttonStartVideo.clicked.connect(self.start_video)
        self.form.enableYOLO.clicked.connect(self.enable_yolo)
        # self.form.imageView.clicked.connect(self.imageView_onclick)
        self.window.show()

    def start_video(self):
        ind = self.form.videoIndex.text()
        print(ind)
        try:
            ind = int(ind)
        except:
            ind = -1

        print(ind)
        self.capture.open_camera(ind)
        self.refresh_timer.start(16)
        # self.form.imageView.refresh()

    def refresh_video(self):
        self.refresh_timer.stop()
        if not self.form.pauseButton.isChecked():
            self.capture.refresh()
            img = self.capture.current_np
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
            self.background_image_item.setPixmap(QPixmap(qimg))
        if self.form.enableYOLO.isChecked():
            results = self.model(self.capture.current_np)
            self.current_yolo_results.clear()
            for r in self.poi_lines:
                self.scene.removeItem(r)
            self.poi_lines.clear()
            for r in results.xywh[0]:
                arr = r.detach().tolist()
                self.current_yolo_results.append(f"{int(arr[0])} {int(arr[1])} {int(arr[2])} {int(arr[3])} {arr[4]:.3f} {int(arr[5])} ")
                self.poi_lines.append(self.scene.addLine(r[0]-10, r[1], r[0]+10, r[1],
                                                         QColor(255,0,0)))
                self.poi_lines.append(self.scene.addLine(r[0], r[1]-10, r[0], r[1]+10,
                                                         QColor(255, 0, 0)))
            # self.form.listROI.setModel(QStringListModel(self.current_yolo_results))
        if not self.form.pauseButton.isChecked():
            self.refresh_timer.start(16)

    def change_yolo_threshold(self):
        value = self.form.yoloThresholdSlider.value() / 1000.0
        self.form.yoloThresholdLabel.setText(f"{value:.3f}")
        if self.model is not None:
            self.model.conf = value
        self.refresh_video()

    def enable_yolo(self):
        if self.form.enableYOLO.isChecked() and self.model is None:
            self.model = torch.hub.load('../yolov5/', 'custom',
                                        path='../runs/train/exp11/weights/best.pt',
                                        source='local')

    def imageView_onclick(self, event):
        p = self.form.imageView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        print(x,y)
        print(p)
        self.form.zoomedView.setSceneRect(x - 10, y-10, 20,20)
        objs = self.form.imageView.items(x-5, y-5, x+5, y+5)
        if len(objs) < 2:
            return
        o = objs[0]
        ind = self.poi_lines.index(o)
        ind = int(ind/2)
        qind = self.form.listROI.model().index(ind,0)
        self.form.listROI.setCurrentIndex(qind)





app = App()
app.exec_()


