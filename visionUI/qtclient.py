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


def debug():
    print("plop")

class App(QApplication):
    DATASETS_DIRECTORY="vui_datasets"

    def __init__(self):
        super().__init__([])
        self.db = sqlite3.connect("visionUI.db")

        self.qdb = QSqlDatabase()
        self.qdb = QSqlDatabase.addDatabase("QSQLITE");
        self.qdb.setDatabaseName("visionUI.db")

        if self.qdb.driver().hasFeature(QSqlDriver.EventNotifications):
            self.qdb.driver().subscribeToNotification("classes")
            self.qdb.driver().notification.connect(debug)
        else:
            print("Driver does NOT support database event notifications");
            return


        self.capture = Capture()

        if not os.path.exists(App.DATASETS_DIRECTORY):
            os.mkdir(App.DATASETS_DIRECTORY)

        Form, Window = uic.loadUiType("mainwindow.ui")
        self.window = Window()
        self.form = Form()
        self.form.setupUi(self.window)
        self.form.splitter.setSizes([400, 200])
        self.form.splitter_video.setSizes([400, 200])

        self.form.imageView.mousePressEvent = self.imageView_onclick
        self.form.zoomedView.mousePressEvent = self.zoomedView_onclick

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

        self.classes_model = QSqlTableModel()
        self.classes_model.setTable("classes")
        self.form.classesList.setModel(self.classes_model)
        self.classes_model.select()

        model = QSqlQueryModel()
        model.setQuery("SELECT name FROM classes")
        self.form.comboROIclass.setModel(model)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_video)

        self.form.yoloThresholdSlider.valueChanged.connect(self.change_yolo_threshold)
        self.form.buttonStartVideo.clicked.connect(self.start_video)
        self.form.enableYOLO.clicked.connect(self.enable_yolo)
        self.form.selectDirImageButton.clicked.connect(self.set_image_browser_directory)
        self.form.createNewDatasetButton.pressed.connect(self.create_new_dataset)
        self.form.datasetNameList.activated.connect(self.select_dataset)
        self.form.datasetNameEdit.editingFinished.connect(self.update_selected_dataset)
        self.form.datasetType.activated.connect(self.update_selected_dataset)
        self.form.makeCandidateButton.pressed.connect(self.create_candidate)
        self.form.candidatesList.currentItemChanged.connect(self.candidate_image_select)
        self.form.newClassButton.clicked.connect(self.create_new_class)
        self.form.deleteClassButton.clicked.connect(self.remove_class)

        self.refresh_datasets_namelist()
        self.form.annotationsTab.setEnabled(False)
        self.window.show()

    def request(self, request, args):
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

    def create_candidate(self):
        fulldir = App.DATASETS_DIRECTORY + "/" + self.form.datasetNameList.currentText()
        if not os.path.exists(fulldir):
            os.mkdir(fulldir)
        filelist = os.listdir(fulldir)
        frame_num = 0

        if self.capture.source_type == Capture.SOURCE_TYPE_CAMERA:
            filename = f"frame_{frame_num:08}.jpg"
            while filename in filelist:
                frame_num += 1
                filename = f"frame_{frame_num:08}.jpg"
        if self.capture.source_type == Capture.SOURCE_TYPE_FILE:
            namepart = ".".join(self.current_image_browser_file.split(".")[:-1])
            filename = f"{namepart}.jpg"
            while filename in filelist:
                filename = f"{namepart}_{frame_num:04}.jpg"
                frame_num += 1

        fullpath = fulldir+"/"+filename
        Image.fromarray(self.capture.current_np).save(fullpath)
        self.form.candidatesList.addItem(filename)

    def update_selected_dataset(self):
        desc = self.form.datasetNameList.currentText()
        cursor = self.db.cursor()
        cursor.execute("""UPDATE datasets 
                           SET type=?, description=? 
                           WHERE description=?""", (self.form.datasetType.currentIndex(),
                                                    self.form.datasetNameEdit.text(),
                                                    desc))
        cursor.close()
        self.db.commit()
        self.refresh_datasets_namelist()

    def select_dataset(self):
        desc = self.form.datasetNameList.currentText()
        if desc == "":
            return
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM datasets WHERE description=?", (desc,))
        (did, dtype, _) = cursor.fetchall()[0]
        self.form.datasetNameEdit.setText(desc)
        self.form.datasetType.setCurrentIndex(dtype)
        cursor.close()
        self.form.annotationsTab.setEnabled(True)

    def create_new_dataset(self):
        name = self.form.datasetNameEdit.text()
        cursor = self.db.cursor()
        cursor.execute("SELECT description FROM datasets WHERE description=?", (name,))
        if len(list(cursor.fetchall()))>0:
            QErrorMessage().showMessage(f"A dataset with name '{name}' already exists")
            return
        dataset_type = self.form.datasetType.currentIndex()
        cursor.close()
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO datasets (type, description) VALUES (?, ?)", (dataset_type, name))
        cursor.close()
        self.db.commit()
        self.refresh_datasets_namelist()

    def refresh_datasets_namelist(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT description FROM datasets ORDER BY id;")
        datasets=list(cursor.fetchall())
        print(datasets)
        index = self.form.datasetNameList.currentIndex()
        self.form.datasetNameList.clear()
        for ds in datasets:
            self.form.datasetNameList.addItem(ds[0])
        if self.form.datasetNameList.count() > index:
            self.form.datasetNameList.setCurrentIndex(index)
        cursor.close()

    def update_statusbar(self):
        self.window.statusBar().showMessage("Capture mode:"+str(self.capture_mode))

    def image_browser_select(self, index):
        self.current_image_browser_file=index.data()
        fullname = self.current_image_browser_dir + "/" + self.current_image_browser_file
        self.capture.open_file(fullname)

    def candidate_image_select(self):
        filename = self.form.candidatesList.currentItem().data(Qt.DisplayRole)
        fullname = App.DATASETS_DIRECTORY + "/" + self.form.datasetNameList.currentText() + "/" + filename
        self.capture.open_file(fullname)

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
        print(x,y)
        print(p)

        if self.form.selectPointButton.isChecked():
            objs = self.form.imageView.items(x-5, y-5, x+5, y+5)
            if len(objs) < 2:
                return
            o = objs[0]
            print(objs)
            ind = self.poi_lines.index(o)
            ind = int(ind/2)
            print(ind)
            qind = self.form.listROI.ml_model().index(ind, 0)
            self.form.listROI.setCurrentIndex(qind)
        if self.form.pointCreateButton.isChecked():
            self.create_point(x,y)

    def create_point(self, x, y):
        obj_class = self.form.comboROIclass.currentText()
        self.request("SELECT * FROM classes WHERE name=?", (obj_class))
        self.scene.addLine(x - 10, y, x + 10, y, QColor(0, 0, 255))
        self.scene.addLine(x, y - 10, x, y + 10, QColor(0, 0, 255))

    def refresh_roi_scene(self):
        self.request("SELECT * FROM annotated_file,  WHERE name=?", (obj_class))




app = App()
app.exec_()


