from datetime import datetime

import torch
import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QRect, QRectF, QStringListModel, QAbstractListModel, QDir, QItemSelectionModel, Qt, \
    QModelIndex
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtSql import QSqlQueryModel, QSqlTableModel, QSqlDatabase, QSqlDriver
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QFileDialog, QFileSystemModel, QListWidgetItem, \
    QErrorMessage, QMessageBox
import cv2
import sys
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

    def __init__(self):
        super().__init__([])
        self.db = sqlite3.connect(f"{App.DATASET_DIRECTORY}/visionUI.db")
        self.capture = Capture()

        if len(sys.argv)>1:
            App.DATASET_DIRECTORY = sys.argv[1]
        App.CANDIDATES_DIR = f"{App.DATASET_DIRECTORY}/candidates/"
        App.KEYFRAMES_DIR = f"{App.DATASET_DIRECTORY}/keyframes/"
        if not os.path.exists(App.DATASET_DIRECTORY):
            os.mkdir(App.DATASET_DIRECTORY)
        if not os.path.exists(App.CANDIDATES_DIR):
            os.mkdir(App.CANDIDATES_DIR)
        if not os.path.exists(App.KEYFRAMES_DIR):
            os.mkdir(App.KEYFRAMES_DIR)

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
        self.selected_file_id = None

        self.ml_model = None
        self.current_yolo_results = list()
        self.refresh_classes_lists()

        model = QFileSystemModel(self)
        self.form.candidatesList.setModel(model)
        self.form.candidatesList.setRootIndex(model.setRootPath(App.CANDIDATES_DIR))

        model = QFileSystemModel(self)
        self.form.keyframesList.setModel(model)
        self.form.keyframesList.setRootIndex(model.setRootPath(App.KEYFRAMES_DIR))

        self.form.listROI.clicked.connect(self.select_annotation_from_list)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_video)

        self.form.yoloThresholdSlider.valueChanged.connect(self.change_yolo_threshold)
        self.form.buttonStartVideo.clicked.connect(self.start_video)
        self.form.enableYOLO.clicked.connect(self.enable_yolo)
        self.form.selectDirImageButton.clicked.connect(self.set_image_browser_directory)
        self.form.classesList.currentRowChanged.connect(self.update_selected_class_editor)
        self.form.makeCandidateButton.pressed.connect(self.create_candidate)
        self.form.captureFrameButton.pressed.connect(self.create_candidate)
        self.form.eraseFrameButton.pressed.connect(self.erase_frame)
        self.form.candidatesList.selectionModel().selectionChanged.connect(self.candidate_image_select)
        self.form.keyframesList.selectionModel().selectionChanged.connect(self.keyframe_image_select)
        self.form.newClassButton.clicked.connect(self.create_new_class)
        self.form.deleteClassButton.clicked.connect(self.remove_class)
        self.form.deletePointButton.clicked.connect(self.remove_point)
        self.form.validateKFButton.clicked.connect(self.validate_keyframe)
        self.form.invalidateKFButton.clicked.connect(self.invalidate_keyframe)
        self.form.prepareDatasetButton.clicked.connect(self.prepare_training_files)

        self.window.show()

    def validate_keyframe(self):
        if not self.selected_file_id:
            return
        kfs = [s for s in os.listdir(App.CANDIDATES_DIR) if s.startswith(self.selected_file_id)]
        if len(kfs)==0:
            return
        os.rename(f"{App.CANDIDATES_DIR}/{kfs[0]}", f"{App.KEYFRAMES_DIR}/{kfs[0]}")

    def invalidate_keyframe(self):
        if not self.selected_file_id:
            return
        kfs = [s for s in os.listdir(App.KEYFRAMES_DIR) if s.startswith(self.selected_file_id)]
        if len(kfs)==0:
            return
        os.rename(f"{App.KEYFRAMES_DIR}/{kfs[0]}", f"{App.CANDIDATES_DIR}/{kfs[0]}")

    def erase_frame(self):
        if self.selected_file_id:
            self.request("DELETE FROM annotations WHERE file_id=?", [self.selected_file_id])
            dl = [s for s in os.listdir(App.CANDIDATES_DIR) if s.startswith(self.selected_file_id)]
            if len(dl)>0:
                os.remove(f"{App.CANDIDATES_DIR}/{dl[0]}")
            self.refresh_annotations_list()

    def refresh_classes_lists(self):
        self.form.classesList.clear()
        self.form.comboROIclass.clear()
        for cl in self.request("SELECT * FROM classes ORDER BY encoding;"):
            self.form.classesList.addItem(f"{cl[3]}\t{cl[1]}:  {cl[2]}")
            self.form.comboROIclass.addItem(f"{cl[3]}:{cl[1]}")

    @staticmethod
    def parse_class_Editor_line(s):
        arr = s.split("\t")
        encoding = int(arr[0])
        name = arr[1].split(":")[0]
        description = ":".join(arr[1].split(":")[1:])
        if len(arr)>2:
            description += "\t".join(arr[2:])
        return name, description, encoding

    def update_selected_class_editor(self):
        name, description, encoding = self.parse_class_Editor_line(self.form.classesList.currentItem().data(0))
        self.form.textClassName.setText(name)
        self.form.textClassDescription.setText(description)
        self.form.textClassEncoding.setText(str(encoding))

    def request(self, request, args=[]):
        cursor = self.db.cursor()
        cursor.execute(request, args)
        ret = list(cursor.fetchall())
        cursor.close()
        self.db.commit()
        return ret

    def remove_class(self):
        # TODO: check if no annotation uses this class
        name, description, encoding = self.parse_class_Editor_line(self.form.classesList.currentItem().data(0))
        self.request("DELETE FROM classes WHERE name=?", [name])
        self.refresh_classes_lists()

    def create_new_class(self):
        try:
            encoding = int(self.form.textClassEncoding.text())
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Can't create class")
            msg.setInformativeText('The encoding is not a valid numer')
            msg.setWindowTitle("Error")
            msg.exec_()
        name = self.form.textClassName.text()
        description = self.form.textClassDescription.text()
        self.request("INSERT INTO classes (name, description, encoding) VALUES(?,?,?)", [name, description, encoding])
        self.refresh_classes_lists()

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
        Image.fromarray(self.capture.current_np).save(fullpath)

    @staticmethod
    def get_uid_from_filename(s):
        return ".".join(s.split(".")[:-1]).split("_was")[0]

    def update_statusbar(self):
        self.window.statusBar().showMessage("Capture mode:"+str(self.capture_mode))

    def image_browser_select(self, index):
        self.current_image_browser_file=index.data()
        fullname = self.current_image_browser_dir + "/" + self.current_image_browser_file
        self.capture.open_file(fullname)

    def keyframe_image_select(self):
        filename_index = self.form.keyframesList.currentIndex()
        fullname = self.form.keyframesList.model().filePath(filename_index)
        filename = self.form.keyframesList.model().fileName(filename_index)
        self.capture.open_file(fullname)
        self.refresh_video()
        self.selected_file_id = self.get_uid_from_filename(filename)
        self.refresh_annotations_list()

    def candidate_image_select(self):
        filename_index = self.form.candidatesList.currentIndex()
        fullname = self.form.candidatesList.model().filePath(filename_index)
        filename = self.form.candidatesList.model().fileName(filename_index)
        self.capture.open_file(fullname)
        self.refresh_video()
        self.selected_file_id = self.get_uid_from_filename(filename)
        self.refresh_annotations_list()

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
            self.form.listROI.clear()
            self.form.listROI.addItems(self.current_yolo_results)
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
        self.form.zoomedView.setSceneRect(x - 10, y-10, 20,20)

    def refresh_annotations_list(self):
        annotations = self.request("SELECT id, type, x,y, class_id FROM annotations WHERE file_id=? ORDER BY id", [self.selected_file_id])
        for r in self.poi_lines:
            self.scene.removeItem(r)
        self.poi_lines.clear()
        w = self.background_image_item.boundingRect().width()
        h = self.background_image_item.boundingRect().height()
        i = 0
        selected_index = -1
        str_list = list()
        for aid, ty, x, y, cid in annotations:
            x = round(x*w)
            y = round(y*h)
            if ty == 1:
                anot_type = "POINT"
            else:
                anot_type = "UNKOWN"
            str_list.append(f"{anot_type} {x} {y} {1.0} {cid}")
            if self.scene_selection and self.scene_selection == aid:
                color = QColor(255, 0, 0)
                selected_index = i
            else:
                color = QColor(0, 0, 255)
            self.poi_lines.append(
                self.scene.addLine(x - 10, y, x + 10, y, color))
            self.poi_lines.append(
                self.scene.addLine(x, y - 10, x, y + 10, color))
            i += 1

        self.form.listROI.clear()
        self.form.listROI.addItems(str_list)
        self.form.listROI.setCurrentRow(selected_index)


    def zoomedView_onclick(self, event):
        p = self.form.zoomedView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        x/=self.background_image_item.boundingRect().width()
        y/=self.background_image_item.boundingRect().height()


        if self.form.selectPointButton.isChecked():
            annotations = self.request("SELECT id, x,y FROM annotations WHERE file_id=?",
                                       [self.selected_file_id])
            mindist=1e9
            ind = -1
            for ai, ax, ay in annotations:
                dist = (ax-x)**2 + (ay-y)**2
                if dist<mindist:
                    ind = ai
                    mindist = dist
            if ind > -1:
                self.scene_selection = ind
                self.refresh_annotations_list()

        if self.form.pointCreateButton.isChecked():
            obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
            self.request("INSERT INTO annotations (type, file_id, x, y, class_id) VALUES(1, ?,?,?,?)",
                         [self.selected_file_id, x, y, obj_class])
            self.refresh_annotations_list()

        if self.form.movePointButton.isChecked():
            if self.scene_selection:
                self.request("UPDATE annotations SET x=?, y=? WHERE id=?",
                             [x,y,self.scene_selection])
                self.refresh_annotations_list()

    def remove_point(self):
        if self.scene_selection:
            self.request("DELETE FROM annotations WHERE id=?",
                         [self.scene_selection])
            self.scene_selection = None
            self.refresh_annotations_list()

    def prepare_training_files(self):
        # Write YAML
        p = Path(App.DATASET_DIRECTORY)
        filename = p / "dataset.yaml"
        results = self.request("SELECT name, encoding FROM classes ORDER BY encoding;")
        classes = {e:c for c,e in results}
        s = ""
        s += f"""path: {p.resolve()} # dataset root dir
train: keyframes/ # train images relative to path
val: keyframes/ # validation images relative to path
test: 

nc: {len(classes)}
names: {str(list(classes.values()))}
"""
        yamlfile = open(filename, "w")
        yamlfile.write(s)
        yamlfile.close()

        # Write classes.txt
        # Write labels txt files
        if not os.path.exists(p/"labels"):
            os.mkdir(p/"labels")
        for fn in os.listdir(p / "keyframes"):
            label_file = open((p/"labels"/fn).with_suffix(".txt"), "w")
            annotations = self.request("SELECT x,y,class_id FROM annotations WHERE file_id=?", [str(Path(fn).with_suffix(""))])
            for x, y, cid in annotations:
                label_file.write(f"{list(classes.keys()).index(cid)} {x} {y} 0.03 0.03\n")
            fn.split()
        return

    def select_annotation_from_list(self):
        annotations = self.request("SELECT id FROM annotations WHERE file_id=? ORDER BY id",
                                   [self.selected_file_id])
        previous = self.scene_selection
        self.scene_selection = annotations[self.form.listROI.currentRow()][0]
        if previous != self.scene_selection:
            self.refresh_annotations_list()

app = App()
app.exec_()


