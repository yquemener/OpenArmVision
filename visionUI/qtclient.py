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

from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

import yolov5
import yolov5.train
import multi_obj
import torchvision

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

# TODO: Connect the "refresh model" button!
# TODO: put the runs in the dataset path
# TODO: paths for yolov5 in a variable
# TODO: put training in a thread
# TODO: Add manual focus widget + 50/60 Hz compensation
# TODO: Erase labels files when regenerating the dataset


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
    DATASET_DIRECTORY="screws_dataset"

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

        # Quick hack for VECTOR selection/creation
        self.vector_p1 = None
        self.vector_p2 = None
        self.vector_selection_state = 0
        self.refresh_models_list()

        self.window.show()

    def refresh_models_list(self):
        self.form.weightsList.clear()
        self.form.weightsList.addItem("Pre-trained YOLOv5 small")
        for exp in sorted(os.listdir("../yolov5/runs/train")):
            path = f"../yolov5/runs/train/{exp}/weights"
            if os.path.exists(path):
                for pt in sorted(os.listdir(path)):
                    if pt.endswith(".pt"):
                        self.form.weightsList.addItem(f"{path}/{pt}")
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
                "class_id"	INTEGER,
                "confidence"	NUMERIC NOT NULL DEFAULT 1.0
            )"""
            self.request(req)
        ret = self.request("SELECT name FROM sqlite_master WHERE type='table' AND name='classes';")
        if len(ret)==0:
            req = """CREATE TABLE "classes" (
                "id"	INTEGER PRIMARY KEY AUTOINCREMENT,
                "name"	TEXT,
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

        if self.form.enableYOLO.isChecked():
            w = self.background_image_item.boundingRect().width()
            h = self.background_image_item.boundingRect().height()
            for r in self.current_yolo_results:
                x,y,_,_,_,cid = r.split(" ")[0:6]
                self.request("INSERT INTO annotations (type, file_id, x, y, class_id) VALUES(1, ?,?,?,?)",
                             [uid, float(x)/w, float(y)/h, int(cid)])

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
            if self.form.modelType.currentText() == "YOLO":
                self.ml_model.eval()
                with torch.no_grad():
                    results = self.ml_model(self.capture.current_np).xyxy[0]
                results = results.detach().tolist()

            elif self.form.modelType.currentText() == "multi_obj":
                imgt = transforms.Resize((512, 512))(to_pil_image(self.capture.current_np))
                imgt = torch.Tensor(np.array(imgt)).unsqueeze(0).view(1,3,512,512).cuda()
                self.ml_model.eval()
                with torch.no_grad():
                    res = self.ml_model(imgt)
                thresh = self.form.yoloThresholdSlider.value()/1000.
                results = list()
                for i in range(res.shape[1]):
                    for j in range(res.shape[2]):
                        arr = res[0, i, j].detach().tolist()
                        if arr[0] < thresh:
                            continue
                        arr = arr[1:]
                        arr[0] = (arr[0] + j + 0.5) * 640 / res.shape[1]
                        arr[1] = (arr[1] + i + 0.5) * 480 / res.shape[2]
                        arr[2] = arr[0] + arr[2] * 640
                        arr[3] = arr[1] + arr[3] * 480
                        results.append(arr)


                # results = results[:, :, :, 1:][results[:,:,:,0]>0].view(-1,5)
                # results = results.detach().cpu().numpy()
                # results[:, 0:2] *= np.array((640,480))


            self.current_yolo_results.clear()
            for r in self.poi_lines:
                self.scene.removeItem(r)
            self.poi_lines.clear()
            for arr in results:
            # for r in results[:,:,:,1:][results[:,:,:,0]>0].view(-1,5):

                arr[2] = arr[0]+arr[2]*640
                arr[3] = arr[1]+arr[3]*480


            # for i in range(results.shape[1]):
            #     for j in range(results.shape[2]):
            #         arr = results[0, i, j].detach().tolist()
            #         if arr[0] < thresh:
            #             continue
            #         arr = arr[1:]
            #         print(arr)
            #         arr[0] = (arr[0] + j + 0.5) * 640 / results.shape[1]
            #         arr[1] = (arr[1] + i + 0.5) * 480 / results.shape[2]
            #         arr[2] = arr[0] + arr[2] * 640
            #         arr[3] = arr[1] + arr[3] * 480
            #         print(arr)
            #         print()
                self.current_yolo_results.append(f"{int(arr[0])} {int(arr[1])} {int(arr[2])} {int(arr[3])} {arr[4]:.3f}")
                # self.poi_lines.append(self.scene.addLine(r[0]-10, r[1], r[0]+10, r[1],
                #                                          QColor(255,0,0)))
                # self.poi_lines.append(self.scene.addLine(r[0], r[1]-10, r[0], r[1]+10,
                #                                          QColor(255, 0, 0)))

                self.poi_lines.append(self.scene.addLine(arr[0], arr[1], arr[2], arr[3],
                                                         QColor(255, 0, 0)))
                self.poi_lines.append(self.scene.addEllipse(arr[0]-5, arr[1]-5, 10,10,
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
                                           path='../yolov5/runs/train/exp44/weights/best.pt',
                                           source='local')
            # num_classes = 1
            # grid_size = 16
            # grid_size = 16
            # model = multi_obj.CustomYOLO(num_classes, grid_size=grid_size)
            # self.ml_model = torch.load("/home/yves/Projects/active/HLA/OpenArmVision/visionUI/model_colab_best.bin")

    def select_inference_model(self):
        model_path = self.form.weightsList.currentItem().text()
        if os.path.exists(model_path) and model_path.endswith(".pt"):
            self.ml_model = torch.hub.load('../yolov5/', 'custom',
                                           path=model_path,
                                           source='local')
        elif os.path.exists(model_path) and model_path.endswith(".bin"):
            self.ml_model = torch.load(model_path)

    def select_training_model(self):
        model_path = self.form.weightsList.currentItem().text()
        if os.path.exists(model_path):
            self.training_resume_weights=model_path

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
            x2 = round(x2*w)
            y2 = round(y2*h)
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
                str_list.append(f"{anot_type} {x} {y} {x2} {y2} {1.0} {cid}")
                self.poi_lines.append(
                    self.scene.addLine(x, y, x2, y2, color))
                self.poi_lines.append(
                    self.scene.addEllipse(x-5, y-5, 10, 10, color))
            else:
                anot_type = "UNKOWN"
                str_list.append(f"{anot_type} {x} {y} {x2} {y2} {1.0} {cid}")
            i += 1

        self.form.listROI.clear()
        self.form.listROI.addItems(str_list)
        self.form.listROI.setCurrentRow(selected_index)

    def zoomedView_onclick(self, event):
        p = self.form.zoomedView.mapToScene(event.pos())
        x = int(p.x())
        y = int(p.y())
        self.scene_onclick(x, y)

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
        # if self.form.pointCreateButton.isChecked():
        #     obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
        #     self.request("INSERT INTO annotations (type, file_id, x, y, class_id) VALUES(1, ?,?,?,?)",
        #                  [self.selected_file_id, x, y, obj_class])
        #     self.refresh_annotations_list()

        if self.form.pointCreateButton.isChecked():
            obj_class = int(self.form.comboROIclass.currentText().split(":")[0])
            if self.vector_selection_state == 0:
                self.vector_p1 = (x,y)
                self.vector_selection_state = 1
            elif self.vector_selection_state == 1:
                self.vector_p2 = (x, y)
                self.request("INSERT INTO annotations (type, file_id, x, y, x2, y2, class_id) VALUES(2, ?,?,?,?,?,?)",
                             [self.selected_file_id, self.vector_p1[0], self.vector_p1[1], x, y, obj_class])
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
                label_file.write(f"{list(classes.keys()).index(cid)} {x} {y} {x2-x} {y2-y}\n")
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
        yolov5.train.run(data=p / "dataset.yaml",
                         cfg="yolov5s.yaml",
                         weights=weights,
                         # evolve=0,
                         optimizer='AdamW',
                         freeze=list(range(10)),
                         batch_size=8, epochs=epochs, patience=50)

    def select_annotation_from_list(self):
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


