from datetime import datetime

import torch
import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QRect, QRectF, QStringListModel, QAbstractListModel, QDir, QItemSelectionModel, Qt, \
    QModelIndex
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtSql import QSqlQueryModel, QSqlTableModel, QSqlDatabase, QSqlDriver
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QFileDialog, QFileSystemModel, QListWidgetItem, \
    QErrorMessage
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


class App(QApplication):
    DATASET_DIRECTORY="vui_datasets"
    CANDIDATES_DIR = f"{DATASET_DIRECTORY}/candidates/"
    KEYFRAMES_DIR = f"{DATASET_DIRECTORY}/keyframes/"

    def __init__(self):
        super().__init__([])
        self.db = sqlite3.connect(f"{App.DATASET_DIRECTORY}/visionUI.db")

        self.capture = Capture()

        if not os.path.exists(App.DATASET_DIRECTORY):
            os.mkdir(App.DATASET_DIRECTORY)

        Form, Window = uic.loadUiType("mainwindow.ui")
        self.window = Window()
        self.form = Form()
        self.form.setupUi(self.window)
        self.form.splitter.setSizes([400, 200])
        self.form.splitter_video.setSizes([400, 200])

        self.form.imageView.mousePressEvent = self.imageView_onclick
        self.form.zoomedView.mousePressEvent = self.zoomedView_onclick
        self.scene_selection = None

        self.scene = QGraphicsScene()
        self.background_pixmap = QPixmap(QImage(b'\0\0\0\0', 1, 1, QImage.Format_ARGB32))
        self.background_image_item = self.scene.addPixmap(self.background_pixmap)
        self.background_image_item.setVisible(True)
        self.form.imageView.setScene(self.scene)
        self.form.zoomedView.setScene(self.scene)
        self.form.zoomedView.scale(10,10)
        self.poi_lines = list()
        self.current_image_browser_dir = None
        self.current_image_browser_file = None

        self.ml_model = None
        self.current_yolo_results = list()

        for cl in self.request("SELECT * FROM classes;"):
            self.form.classesList.addItem(f"{cl[3]}\t{cl[1]}:  {cl[2]}")
            self.form.comboROIclass.addItem(f"{cl[3]}:{cl[1]}")

        model = QFileSystemModel(self)
        self.form.candidatesList.setModel(model)
        self.form.candidatesList.setRootIndex(model.setRootPath(App.CANDIDATES_DIR))

        self.form.listROI.setModel(QStringListModel())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_video)

        self.form.yoloThresholdSlider.valueChanged.connect(self.change_yolo_threshold)
        self.form.buttonStartVideo.clicked.connect(self.start_video)
        self.form.enableYOLO.clicked.connect(self.enable_yolo)
        self.form.selectDirImageButton.clicked.connect(self.set_image_browser_directory)
        # self.form.createNewDatasetButton.pressed.connect(self.create_new_dataset)
        # self.form.datasetNameList.activated.connect(self.select_dataset)
        # self.form.datasetNameEdit.editingFinished.connect(self.update_selected_dataset)
        # self.form.datasetType.activated.connect(self.update_selected_dataset)
        self.form.makeCandidateButton.pressed.connect(self.create_candidate)
        self.form.candidatesList.selectionModel().selectionChanged.connect(self.candidate_image_select)
        self.form.newClassButton.clicked.connect(self.create_new_class)
        self.form.deleteClassButton.clicked.connect(self.remove_class)

        # self.form.annotationsTab.setEnabled(False)
        self.window.show()

    def request(self, request, args=[]):
        cursor = self.db.cursor()
        cursor.execute(request, args)
        ret = list(cursor.fetchall())
        cursor.close()
        self.db.commit()
        return ret

    def remove_class(self):
        # TODO: check if no annotation uses this class
        ind = self.form.classesList.selectedIndexes()
        if len(ind)==0:
            return
        ind = ind[0].row()
        self.classes_model.deleteRowFromTable(ind)
        self.classes_model.submitAll()
        self.classes_model.select()

    def create_new_class(self):
        cm = self.classes_model
        row = self.classes_model.rowCount()
        cm.insertRows(row, 1)
        cm.submitAll()
        self.form.classesList.reset()

    @staticmethod
    def make_uid():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

    def create_candidate(self):
        fulldir = f"{App.DATASET_DIRECTORY}/candidates/"
        if not os.path.exists(fulldir):
            os.mkdir(fulldir)
        uid = self.make_uid()

        if self.capture.source_type == Capture.SOURCE_TYPE_CAMERA:
            filename = f"{uid}.jpg"
        if self.capture.source_type == Capture.SOURCE_TYPE_FILE:
            namepart = ".".join(self.current_image_browser_file.split(".")[:-1])
            filename = f"{uid}_was_{namepart}.jpg"

        fullpath = fulldir+"/"+filename
        print(fullpath)
        Image.fromarray(self.capture.current_np).save(fullpath)

    # def create_new_dataset(self):
    #     name = self.form.datasetNameEdit.text()
    #     cursor = self.db.cursor()
    #     cursor.execute("SELECT description FROM datasets WHERE description=?", (name,))
    #     if len(list(cursor.fetchall()))>0:
    #         QErrorMessage().showMessage(f"A dataset with name '{name}' already exists")
    #         return
    #     dataset_type = self.form.datasetType.currentIndex()
    #     cursor.close()
    #     cursor = self.db.cursor()
    #     cursor.execute("INSERT INTO datasets (type, description) VALUES (?, ?)", (dataset_type, name))
    #     cursor.close()
    #     self.db.commit()
    #     self.refresh_datasets_namelist()

    def update_statusbar(self):
        self.window.statusBar().showMessage("Capture mode:"+str(self.capture_mode))

    def image_browser_select(self, index):
        self.current_image_browser_file=index.data()
        fullname = self.current_image_browser_dir + "/" + self.current_image_browser_file
        self.capture.open_file(fullname)

    def candidate_image_select(self):
        filename_index = self.form.candidatesList.currentIndex()
        fullname = self.form.candidatesList.model().filePath(filename_index)
        self.capture.open_file(fullname)
        self.refresh_video()

    def set_image_browser_directory(self):
        dir = QFileDialog.getExistingDirectory(caption="Images directory", directory=".")
        self.current_image_browser_dir=dir
        self.form.labelImageDirectory.setText(dir)
        self.form.labelImageDirectory.setToolTip(dir)
        browse_image_model = QFileSystemModel()
        browse_image_model.setRootPath(dir)
        self.form.BrowseImageList.setModel(browse_image_model)
        self.form.BrowseImageList.setRootIndex(browse_image_model.index(dir))
        self.form.BrowseImageList.selectionModel().currentChanged.connect(self.image_browser_select)

    def start_video(self):
        ind = self.form.videoIndex.text()
        try:
            ind = int(ind)
        except:
            ind = -1
        self.capture.open_camera(ind)
        self.refresh_timer.start(16)

    def refresh_video(self):
        self.refresh_timer.stop()
        if not self.form.pauseButton.isChecked():
            self.capture.refresh()
            img = self.capture.current_np
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
            self.background_image_item.setPixmap(QPixmap(qimg))
        if self.form.enableYOLO.isChecked():
            results = self.ml_model(self.capture.current_np)
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
            self.form.listROI.setModel(QStringListModel(self.current_yolo_results))
            # self.form.listROI.model().dataChanged.emit(QModelIndex(), QModelIndex())
        if not self.form.pauseButton.isChecked():
            self.refresh_timer.start(16)

    def change_yolo_threshold(self):
        value = self.form.yoloThresholdSlider.value() / 1000.0
        self.form.yoloThresholdLabel.setText(f"{value:.3f}")
        if self.ml_model is not None:
            self.ml_model.conf = value
        self.refresh_video()

    def enable_yolo(self):
        if self.form.enableYOLO.isChecked() and self.ml_model is None:
            self.ml_model = torch.hub.load('../yolov5/', 'custom',
                                           path='../runs/train/exp11/weights/best.pt',
                                           source='local')

    def imageView_onclick(self, event):
        p = self.form.imageView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        print(x,y)
        print(p)
        self.form.zoomedView.setSceneRect(x - 10, y-10, 20,20)

    def zoomedView_onclick(self, event):
        p = self.form.zoomedView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())

        if self.form.selectPointButton.isChecked():
            objs = self.form.imageView.items(x-5, y-5, x+5, y+5)
            if len(objs) < 2:
                return
            o = objs[0]
            ind = self.poi_lines.index(o)
            ind = int(ind/2)
            if self.scene_selection:
                old_ind = self.poi_lines.index(self.scene_selection)//2
                self.poi_lines[old_ind * 2].setPen(QColor(0, 0, 255))
                self.poi_lines[old_ind * 2+1].setPen(QColor(0, 0, 255))
            self.poi_lines[ind * 2].setPen(QColor(255, 0, 0))
            self.poi_lines[ind * 2+1].setPen(QColor(255, 0, 0))
            self.scene_selection = o
            qind = self.form.listROI.model().index(ind, 0)
            self.form.listROI.setCurrentIndex(qind)

        if self.form.pointCreateButton.isChecked():
            self.create_point(x, y)

        if self.form.movePointButton.isChecked():
            if self.scene_selection:
                ind = self.poi_lines.index(self.scene_selection)//2
                obj = self.poi_lines[ind * 2]
                obj.setLine(x - 10, y, x + 10, y)
                obj = self.poi_lines[ind * 2+1]
                obj.setLine(x, y - 10, x, y + 10)
                s = self.form.listROI.model().stringList()[ind]
                print(s)
                print(s.rstrip().split(" ")[-1])
                m=self.form.listROI.model()
                m.setData(m.index(ind,0), f"{x} {y} {1} {1} {1.0} {s.rstrip().split(' ')[-1]}")

    def create_point(self, x, y):
        obj_class = self.form.comboROIclass.currentText().split(":")[0]
        print("class:", obj_class)
        self.request("SELECT * FROM classes WHERE encoding=?", [obj_class])
        self.poi_lines.append(
            self.scene.addLine(x - 10, y, x + 10, y, QColor(0, 0, 255)))
        self.poi_lines.append(
            self.scene.addLine(x, y - 10, x, y + 10, QColor(0, 0, 255)))
        m = self.form.listROI.model()
        if m.insertRow(m.rowCount()):
            ind = m.index(m.rowCount() - 1, 0)
            m.setData(ind, f"{x} {y} {1} {1} {1.0} {obj_class}")

    def refresh_roi_scene(self):
        self.request("SELECT * FROM annotated_file,  WHERE name=?", (obj_class))




app = App()
app.exec_()


