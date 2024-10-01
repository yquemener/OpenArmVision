from datetime import datetime

import torch
import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import QTimer, QRect, QRectF, QStringListModel, QAbstractListModel, QDir, QItemSelectionModel, Qt, \
    QModelIndex, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QKeySequence
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QFileDialog, QFileSystemModel, QListWidgetItem, \
    QErrorMessage, QMessageBox, QShortcut
import cv2
import sys
import numpy as np
from PIL import Image
import time
import sqlite3
import os
from pathlib import Path

import thirdparty.yolov5.train as yolov5_train


# Hack to make PyQt and cv2 load simultaneously
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(
    Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
)

# exp73: XY only, ComputeLoss
# exp99: XY only, Vector2dLoss
# exp111: XY bad, wh almost random, Vector2dLoss
# exp114: XY bof, wh almost random, latest, Vector2dLoss
# exp118: XY moyen, wh moyen, Vector2dLoss
# exp122: XY moyen, wh moyen, Vector2dLoss
# exp129: XY moyen+, wh moyen, latest, Vector2dLoss

# DONE: Connect the "refresh model" button!
# TODO: put the runs in the dataset path
# TODO: paths for yolov5 in a variable
# TODO: put training in a thread
# TODO: Add manual focus widget + 50/60 Hz compensation
# TODO: Erase labels files when regenerating the dataset
# TODO: drag for move and box drawing
# TODO: resize action
# TODO: float box positions
# TODO: multiple projects
# DONE: associate shape type to class


class VideoCaptureThread(QThread):
    new_frame = pyqtSignal(int)
    def __init__(self, index):
        super().__init__()
        self.source_type = Capture.SOURCE_TYPE_CAMERA
        self.video_capture = cv2.VideoCapture(index)
        self.running = True
        self.current_np = None
        self.frame_num = 0

    def run(self):
        while self.running:
            _, cv2_im = self.video_capture.read()
            self.current_np = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            self.frame_num += 1
            self.new_frame.emit(self.frame_num)
        self.video_capture.release()


class Capture(QObject):
    new_frame = pyqtSignal(int)
    SOURCE_TYPE_NONE = 0
    SOURCE_TYPE_CAMERA = 1
    SOURCE_TYPE_FILE = 2

    def __init__(self):
        super().__init__()
        self.source_type = Capture.SOURCE_TYPE_NONE
        self.video_capture = None
        self.file_path = None
        self.last_capture_time = None
        self.needs_refresh = False
        self.current_np = None
        self.video_thread = None

    def open_camera(self, index):
        if self.source_type == Capture.SOURCE_TYPE_CAMERA or self.video_thread is not None:
            self.video_thread.running = False
        self.source_type = Capture.SOURCE_TYPE_CAMERA
        self.video_thread = VideoCaptureThread(index)
        self.video_thread.start()
        self.video_thread.new_frame.connect(self.on_new_frame)

    def open_file(self, path):
        self.source_type = Capture.SOURCE_TYPE_FILE
        self.file_path = path
        if self.file_path is not None:
            self.current_np = np.array(Image.open(self.file_path))
            self.new_frame.emit(0)

    def on_new_frame(self, frame_num):
        if frame_num != self.video_thread.frame_num:
            # Our thread was delayed of more than one fra,e
            print(f"Skipping belated frame {frame_num}")
            return
        self.current_np = self.video_thread.current_np
        self.last_capture_time = time.time()
        self.new_frame.emit(frame_num)

    def stop_video(self):
        self.video_thread.running = False


class Annotation:
    type_int = {"POINT": 1, "VECTOR": 2, "BOX": 3}
    type_str = {1: "POINT", 2: "VECTOR", 3: "BOX"}

    def __init__(self, _type, x, y, x2, y2, class_id, file=None):
        if type(_type) is int:
            self.type = _type
        else:
            self.type = Annotation.type_int[_type]
        self.file_id = file
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id

    def sql_command(self, scale=(1,1)):
        if self.file_id is None:
            raise ValueError("file_id must not be None")
        x2 = self.x2
        y2 = self.y2
        if x2:
            x2/=scale[0]
        if y2:
            y2/=scale[1]
        return ("INSERT INTO annotations (type, file_id, x, y, x2,y2, class_id) VALUES(?,?,?,?,?,?,?)",
                [self.type, self.file_id, self.x/scale[0], self.y/scale[1], x2, y2, self.class_id])

    def __str__(self):
        return f"{Annotation.type_str[self.type]} {self.x} {self.y} {self.x2} {self.y2} {self.class_id}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def parse(s):
        args = s.split(" ")
        print(args)
        if len(args)<5:
            return None
        if args[0] == "POINT":
            if len(args) < 5:
                return
            return Annotation(1, float(args[1]), float(args[2]), None, None, int(float(args[-1])))
        elif args[0] == "BOX":
            if len(args) != 7:
                return
            return Annotation(3, float(args[1]), float(args[2]), float(args[3]), float(args[4]), int(float(args[-1])))
        elif args[0] == "VECTOR":
            if len(args) != 7:
                return
            return Annotation(2, float(args[1]), float(args[2]), float(args[3]), float(args[4]), int(float(args[-1])))


class App(QApplication):
    DATASET_DIRECTORY="vhelio_holes"

    def __init__(self):
        super().__init__([])

        self.db = sqlite3.connect(f"{App.DATASET_DIRECTORY}/visionUI.db")
        self.populate_db()
        self.capture = Capture()
        self.training_resume_weights = None

        if len(sys.argv)>1:
            App.DATASET_DIRECTORY = sys.argv[1]
        App.CANDIDATES_DIR = f"{App.DATASET_DIRECTORY}/candidates/"
        App.KEYFRAMES_DIR = f"{App.DATASET_DIRECTORY}/images/"
        if not os.path.exists(App.DATASET_DIRECTORY):
            os.mkdir(App.DATASET_DIRECTORY)
        if not os.path.exists(App.CANDIDATES_DIR):
            os.mkdir(App.CANDIDATES_DIR)
        if not os.path.exists(App.KEYFRAMES_DIR):
            os.mkdir(App.KEYFRAMES_DIR)

        Form, Window = uic.loadUiType("mainwindow.ui")
        self.window = Window()
        self.objects_classes = dict()
        self.form = Form()
        self.form.setupUi(self.window)
        self.form.splitter.setSizes([400, 200])
        self.form.splitter_video.setSizes([400, 200])

        self.form.imageView.mouseMoveEvent = self.imageView_onmove
        self.form.imageView.mousePressEvent = self.imageView_onclick
        self.form.zoomedView.mouseMoveEvent = self.zoomedView_onmove
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
        self.last_processed_frame = 0

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

        self.capture.new_frame.connect(self.refresh_video)
        self.capture.new_frame.connect(self.refresh_annotations_list)

        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.create_candidate)

        self.form.copyButton.clicked.connect(self.copy_annotations)
        self.form.pasteButton.clicked.connect(self.paste_annotations)
        self.form.stopButton.clicked.connect(self.capture.stop_video)
        self.form.changeClassButton.clicked.connect(self.change_selection_class)
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
        self.form.trainButton.clicked.connect(self.train_epochs)
        self.form.selectInferenceWeightsButton.clicked.connect(self.select_inference_model)
        self.form.selectTrainingWeightsButton.clicked.connect(self.select_training_model)
        self.form.refreshModelsButton.clicked.connect(self.refresh_models_list)
        self.form.autosaveButton.clicked.connect(self.autosave)

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("Del"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.remove_point)

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("M"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(lambda:self.form.movePointButton.setChecked(True))

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("S"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(lambda:self.form.selectPointButton.setChecked(True))

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("B"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(lambda:self.form.boxCreateButton.setChecked(True))

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("P"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(lambda:self.form.pointCreateButton.setChecked(True))

        def make_yolo_enabled(self):
            self.form.enableYOLO.setChecked(not self.form.enableYOLO.isChecked())
            self.enable_yolo()

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("E"))
        shortcut.setContext(Qt.ApplicationShortcut)

        shortcut.activated.connect(lambda: make_yolo_enabled(self))

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("Ctrl+c"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.copy_annotations)

        shortcut = QShortcut(self.window)
        shortcut.setKey(QKeySequence("Ctrl+v"))
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(self.paste_annotations)

        # Quick hack for VECTOR selection/creation
        self.vector_p1 = None
        self.vector_p2 = None
        self.vector_selection_state = 0

        color = QColor(128, 128, 196)
        self.target_lines = [
            self.scene.addLine(100, 0, 100, 1000, color),
            self.scene.addLine(0, 100, 1000, 100, color)]

        self.refresh_models_list()


        self.window.show()

    def autosave(self):
        if self.form.autosaveButton.isChecked():
            self.autosave_timer.start(1000)
        else:
            self.autosave_timer.stop()

    def refresh_models_list(self):
        self.form.weightsList.clear()
        self.form.weightsList.addItem("Pre-trained YOLOv5 small")
        self.form.weightsList.addItem("Pre-trained YOLOv5 big")
        list_to_sort=list()
        for exp in sorted(os.listdir("thirdparty/yolov5/runs/train")):
            path = f"thirdparty/yolov5/runs/train/{exp}/weights"
            if os.path.exists(path):
                for pt in sorted(os.listdir(path)):
                    if pt.endswith(".pt"):
                        fullname = f"{path}/{pt}"
                        list_to_sort.append((os.path.getmtime(fullname), fullname))
        for _,fn in reversed(sorted(list_to_sort)):
            self.form.weightsList.addItem(fn)
        path = f"/home/yves/Projects/active/HLA/OpenArmVision/models/"
        if os.path.exists(path):
            for pt in sorted(os.listdir(path)):
                if pt.endswith(".bin"):
                    self.form.weightsList.addItem(f"{path}/{pt}")

    def populate_db(self):
        ret = self.request("SELECT name FROM sqlite_master WHERE type='table' AND name='annotations';")
        if len(ret)==0:
            req = """CREATE TABLE "annotations" (
                "id"	INTEGER PRIMARY KEY AUTOINCREMENT,
                "file_id"	TEXT,
                "type"	INTEGER,
                "x"	NUMERIC,
                "y"	NUMERIC,
                "x2"	NUMERIC,
                "y2"	NUMERIC,
                "class_id"	INTEGER,
                "confidence"	NUMERIC NOT NULL DEFAULT 1.0
            )"""
            self.request(req)
        ret = self.request("SELECT name FROM sqlite_master WHERE type='table' AND name='classes';")
        if len(ret)==0:
            req = """CREATE TABLE "classes" (
                "id"	INTEGER PRIMARY KEY AUTOINCREMENT,
                "name"	TEXT,
                "type"	INTEGER,
                "description"	TEXT,
                "encoding"	INTEGER
            )"""
            self.request(req)

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
        self.objects_classes = dict()
        for cl in self.request("SELECT id, name, description, encoding, type FROM classes ORDER BY encoding;"):
            self.form.classesList.addItem(f"{cl[3]}\t{cl[1]}:  {cl[2]} ({Annotation.type_str[cl[4]]})")
            self.form.comboROIclass.addItem(f"{cl[3]}:{cl[1]}")
            self.objects_classes[cl[3]] = (cl[1], cl[2], cl[4])

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

        if self.form.enableYOLO.isChecked():
            w = self.background_image_item.boundingRect().width()
            h = self.background_image_item.boundingRect().height()
            for r in self.current_yolo_results:
                x,y,x2,y2,_,cid = r.split(" ")[0:6]
                self.request("INSERT INTO annotations (type, file_id, x, y, x2,y2, class_id) VALUES(2, ?,?,?,?)",
                             [uid, float(x)/w, float(y)/h, float(x2)/w, float(y2)/h, int(cid)])

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
        self.selected_file_id = self.get_uid_from_filename(filename)
        self.refresh_annotations_list()

    def candidate_image_select(self):
        filename_index = self.form.candidatesList.currentIndex()
        fullname = self.form.candidatesList.model().filePath(filename_index)
        filename = self.form.candidatesList.model().fileName(filename_index)
        self.capture.open_file(fullname)
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

    def refresh_video(self):
        img = self.capture.current_np
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888)
        self.background_image_item.setPixmap(QPixmap(qimg))

    def create_annotations_from_yolo(self):
        if self.capture.current_np is None:
            return
        self.ml_model.eval()
        with torch.no_grad():
            results = self.ml_model(self.capture.current_np).xyxy[0]
        results = results.detach().tolist()

        self.current_yolo_results.clear()
        for r in self.poi_lines:
            self.scene.removeItem(r)
        self.poi_lines.clear()
        for arr in results:
            x,y,x2,y2 = arr[0:4]
            w=x2-x
            h=y2-y
            color = QColor(255, 0, 0)
            if arr[5]==0.0:
                color = QColor(0,255,0)
                # w=5
                # h=5
            # print(objects_classes)
            obj_class = self.objects_classes[int(arr[5]) + 1]
            type_str = Annotation.type_str[obj_class[2]]
            self.current_yolo_results.append(f"{type_str} {x+w/2:.1f} {y+h/2:.1f} {w:.1f} {h:.1f} {arr[4]:.3f} {arr[5]+1}")
            if type_str=="BOX":
                self.poi_lines.append(self.scene.addRect(arr[0], arr[1], arr[2]-arr[0], arr[3]-arr[1],
                                                         color))
            if type_str=="POINT":
                cx=x+w/2
                cy=y+h/2
                self.poi_lines.append(
                    self.scene.addLine(cx - 10, cy, cx + 10, cy, color))
                self.poi_lines.append(
                    self.scene.addLine(cx, cy - 10, cx, cy + 10, color))

            self.form.listROI.clear()
        self.form.listROI.addItems(self.current_yolo_results)

    def change_yolo_threshold(self):
        value = self.form.yoloThresholdSlider.value() / 1000.0
        self.form.yoloThresholdLabel.setText(f"{value:.3f}")
        if self.ml_model is not None:
            self.ml_model.conf = value
            self.refresh_annotations_list()

    def enable_yolo(self):
        if self.form.enableYOLO.isChecked():
            if self.ml_model is None:
                self.ml_model = torch.hub.load('thirdparty/yolov5/', 'custom',
                                               path='thirdparty/yolov5/runs/train/exp167/weights/best.pt',
                                               source='local')
        self.refresh_annotations_list()

    def select_inference_model(self):
        model_path = self.form.weightsList.currentItem().text()
        if os.path.exists(model_path) and model_path.endswith(".pt"):
            self.ml_model = torch.hub.load('thirdparty/yolov5/', 'custom',
                                           path=model_path,
                                           source='local')
        elif os.path.exists(model_path) and model_path.endswith(".bin"):
            self.ml_model = torch.load(model_path)

    def select_training_model(self):
        model_path = self.form.weightsList.currentItem().text()
        if os.path.exists(model_path):
            self.training_resume_weights=model_path

    def imageView_onmove(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.target_lines[0].setLine(x, 0, x, 1000)
        self.target_lines[1].setLine(0, y, 1000, y)

    def imageView_onclick(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            p = self.form.imageView.mapToScene(event.pos())
            x = int(p.x())
            y = int(p.y())
            self.form.zoomedView.setSceneRect(x - 10, y-10, 20,20)
        else:
            p = self.form.imageView.mapToScene(event.pos())
            x = int(p.x())
            y = int(p.y())
            self.scene_onclick(x, y)

    def refresh_annotations_list(self):
        if self.form.enableYOLO.isChecked():
            self.create_annotations_from_yolo()
        else:
            self.create_annotations_from_db()

    def create_annotations_from_db(self):
        annotations = self.request("SELECT id, type, x,y, x2,y2, class_id FROM annotations WHERE file_id=? ORDER BY id", [self.selected_file_id])
        for r in self.poi_lines:
            self.scene.removeItem(r)
        self.poi_lines.clear()
        w = self.background_image_item.boundingRect().width()
        h = self.background_image_item.boundingRect().height()
        i = 0
        selected_index = -1
        str_list = list()
        for aid, ty, x, y, x2, y2, cid in annotations:
            x = round(x*w)
            y = round(y*h)
            if self.scene_selection and self.scene_selection == aid:
                color = QColor(255, 0, 0)
                selected_index = i
            else:
                color = QColor(0, 0, 255)
            if ty == 1:
                anot_type = "POINT"
                str_list.append(f"{anot_type} {x} {y} {1.0} {cid}")
                self.poi_lines.append(
                    self.scene.addLine(x - 10, y, x + 10, y, color))
                self.poi_lines.append(
                    self.scene.addLine(x, y - 10, x, y + 10, color))
            elif ty == 2:
                anot_type = "VECTOR"
                x2 = round(x2 * w)
                y2 = round(y2 * h)
                str_list.append(f"{anot_type} {x} {y} {x2} {y2} {1.0} {cid}")
                self.poi_lines.append(
                    self.scene.addLine(x, y, x2, y2, color))
                self.poi_lines.append(
                    self.scene.addEllipse(x-5, y-5, 10, 10, color))
            elif ty == 3:
                anot_type = "BOX"
                x2 = round(x2 * w)
                y2 = round(y2 * h)
                str_list.append(f"{anot_type} {x} {y} {x2} {y2} {1.0} {cid}")
                self.poi_lines.append(
                    self.scene.addRect(x-x2/2, y-y2/2, x2, y2, color))
            else:
                anot_type = "UNKOWN"
                str_list.append(f"{anot_type} {x} {y} {x2} {y2} {1.0} {cid}")
            i += 1

        self.form.listROI.clear()
        self.form.listROI.addItems(str_list)
        self.form.listROI.setCurrentRow(selected_index)

    def copy_annotations(self):
        for x in range(self.form.listROI.count()):
            print(self.form.listROI.item(x).text())
        self.form.enableYOLO.setChecked(False)
        self.annotations_clipboard = [Annotation.parse(self.form.listROI.item(x).text()) for x in range(self.form.listROI.count())]

    def paste_annotations(self):
        print(self.annotations_clipboard)
        for i in range(len(self.annotations_clipboard)):
            self.annotations_clipboard[i].file_id = self.selected_file_id
            cmd = self.annotations_clipboard[i].sql_command((self.background_image_item.boundingRect().width(),
                                                             self.background_image_item.boundingRect().height()))
            print(cmd)
            self.request(*cmd)
        self.refresh_annotations_list()


    def zoomedView_onclick(self, event):
        p = self.form.zoomedView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        self.scene_onclick(x, y)

    def zoomedView_onmove(self, event):
        p = self.form.zoomedView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        self.target_lines[0].setLine(x, 0, x, 1000)
        self.target_lines[1].setLine(0, y, 1000, y)

    def scene_onclick(self, x, y):
        x /= self.background_image_item.boundingRect().width()
        y /= self.background_image_item.boundingRect().height()

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

        # correct for a POINT but hastily hacking that to make it work with VECTOR by default
        if self.form.pointCreateButton.isChecked():
            obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
            self.request("INSERT INTO annotations (type, file_id, x, y, class_id) VALUES(1, ?,?,?,?)",
                         [self.selected_file_id, x, y, obj_class])
            self.refresh_annotations_list()

        # if self.form.pointCreateButton.isChecked():
        #     obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
        #     if self.vector_selection_state == 0:
        #         self.vector_p1 = (x,y)
        #         self.vector_selection_state = 1
        #     elif self.vector_selection_state == 1:
        #         self.vector_p2 = (x, y)
        #         self.request("INSERT INTO annotations (type, file_id, x, y, x2, y2, class_id) VALUES(2, ?,?,?,?,?,?)",
        #                      [self.selected_file_id, self.vector_p1[0], self.vector_p1[1], x, y, obj_class])
        #         self.refresh_annotations_list()
        #         self.vector_selection_state = 0

        if self.form.boxCreateButton.isChecked():
            obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
            if self.vector_selection_state == 0:
                self.vector_p1 = (x,y)
                self.vector_selection_state = 1
            elif self.vector_selection_state == 1:
                self.vector_p2 = (x, y)
                x1 = min(self.vector_p1[0], self.vector_p2[0])
                y1 = min(self.vector_p1[1], self.vector_p2[1])
                x2 = max(self.vector_p1[0], self.vector_p2[0])
                y2 = max(self.vector_p1[1], self.vector_p2[1])
                self.request("INSERT INTO annotations (type, file_id, x, y, x2, y2, class_id) VALUES(3, ?,?,?,?,?,?)",
                             [self.selected_file_id, (x2+x1)/2, (y2+y1)/2, x2-x1, y2-y1, obj_class])
                self.refresh_annotations_list()
                self.vector_selection_state = 0

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
train: images/ # train images relative to path
val: images/ # validation images relative to path
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
        for fn in os.listdir(App.KEYFRAMES_DIR):
            label_file = open((p/"labels"/fn).with_suffix(".txt"), "w")
            annotations = self.request("SELECT x,y,x2,y2,class_id FROM annotations WHERE file_id=?", [str(Path(fn).with_suffix(""))])
            for x, y, x2, y2, cid in annotations:
                if x2 and y2:
                    label_file.write(f"{list(classes.keys()).index(cid)} {x} {y} {x2} {y2}\n")
                else:
                    label_file.write(f"{list(classes.keys()).index(cid)} {x} {y} 0.05 0.05\n")
            fn.split()
        return

    def train_epochs(self):
        p = Path(App.DATASET_DIRECTORY)
        try:
            epochs = int(self.form.epochNumText.text())
        except ValueError:
            return
        weights = ""
        if self.training_resume_weights:
            weights = self.training_resume_weights
        yolov5_train.run(data=p / "dataset.yaml",
                         cfg="yolov5x.yaml",
                         weights=weights,
                         # evolve=0,
                         optimizer='AdamW',
                         freeze=list(range(10)),
                         batch_size=30, epochs=epochs, patience=500)

    def select_annotation_from_list(self):
        if self.form.enableYOLO.isChecked():
            return
        annotations = self.request("SELECT id FROM annotations WHERE file_id=? ORDER BY id",
                                   [self.selected_file_id])
        previous = self.scene_selection
        self.scene_selection = annotations[self.form.listROI.currentRow()][0]
        if previous != self.scene_selection:
            self.refresh_annotations_list()

    def change_selection_class(self):
        cenc = self.form.comboROIclass.currentText().split(":")[0]
        self.request("UPDATE annotations SET class_id=? WHERE id=?", [cenc,self.scene_selection])
        self.refresh_annotations_list()

app = App()
app.exec_()


