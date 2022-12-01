# -*- coding: utf-8 -*-
import faulthandler
import os
import traceback
import sys
from datetime import datetime

import cv2
import numpy as np
from numpy import random
from tqdm import tqdm

from Utils.folders import clear_directory
from Widgets.DetectedPerson.DetectedPersonWidget import DetectedPersonWidget, DetectedPerson

import torch
import wget

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QEvent
from PyQt5.QtWidgets import *


class Ui_MainWindow(QMainWindow):
    file_path = ''
    detected_persons = []

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1359, 712)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 0, 541, 45))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.TopUtilityBar = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.TopUtilityBar.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.TopUtilityBar.setContentsMargins(0, 0, 0, 0)
        self.TopUtilityBar.setObjectName("TopUtilityBar")
        self.NetChooser = QtWidgets.QVBoxLayout()
        self.NetChooser.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.NetChooser.setObjectName("NetChooser")
        self.NetLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.NetLabel.setFont(font)
        self.NetLabel.setObjectName("NetLabel")
        self.NetChooser.addWidget(self.NetLabel)
        self.NetComboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.NetComboBox.setObjectName("NetComboBox")
        self.NetChooser.addWidget(self.NetComboBox)
        self.TopUtilityBar.addLayout(self.NetChooser)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.TopUtilityBar.addItem(spacerItem)
        self.NetSizeChooser = QtWidgets.QVBoxLayout()
        self.NetSizeChooser.setObjectName("NetSizeChooser")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.NetSizeChooser.addWidget(self.label)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.NetSizeChooser.addWidget(self.comboBox)
        self.TopUtilityBar.addLayout(self.NetSizeChooser)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.TopUtilityBar.addItem(spacerItem1)
        self.ConfidenceChooser = QtWidgets.QVBoxLayout()
        self.ConfidenceChooser.setObjectName("ConfidenceChooser")
        self.ConfidenceLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.ConfidenceLabel.setText("Confidence threshold: 85")
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ConfidenceLabel.setFont(font)
        self.ConfidenceLabel.setObjectName("ConfidenceLabel")
        self.ConfidenceChooser.addWidget(self.ConfidenceLabel)
        self.ConfidenceSlider = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.ConfidenceSlider.setValue(85)
        self.ConfidenceSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ConfidenceSlider.setObjectName("ConfidenceSlider")

        self.ConfidenceSlider.valueChanged.connect(
            lambda: self.ConfidenceLabel.setText("Confidence threshold: " + str(self.ConfidenceSlider.value())))

        self.ConfidenceChooser.addWidget(self.ConfidenceSlider)
        self.TopUtilityBar.addLayout(self.ConfidenceChooser)
        self.FileExplorer = QtWidgets.QTreeView(self.centralwidget)
        self.FileExplorer.setGeometry(QtCore.QRect(10, 60, 191, 611))
        self.FileExplorer.setObjectName("FileExplorer")

        self.FileExplorer.doubleClicked.connect(self.select_file)

        # Creo l'oggetto del file explorer
        self.model = QtWidgets.QFileSystemModel()
        self.model.setRootPath("C:/home/user/Documents")
        self.FileExplorer.setModel(self.model)

        # Applico il filtro ai file che vedo, in modo da poter selezionare solamente delle immagini
        self.model.setNameFilters(["*.png", "*.jpg", "*.jpeg"])
        # Oltre ad applicare il filtro, faccio proprio sparire i file che non rispettano i file
        self.model.setNameFilterDisables(True)

        self.FileExplorer.setRootIndex(self.model.index("C:/home/user/Documents"))

        self.PhotoWidget = QtWidgets.QLabel(self.centralwidget)
        self.PhotoWidget.setGeometry(QtCore.QRect(210, 60, 945, 611))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PhotoWidget.sizePolicy().hasHeightForWidth())
        self.PhotoWidget.setSizePolicy(sizePolicy)
        self.PhotoWidget.setAlignment(QtCore.Qt.AlignCenter)
        self.PhotoWidget.setObjectName("PhotoWidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 40, 1341, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.InferenceButton = QtWidgets.QPushButton(self.centralwidget)
        self.InferenceButton.setGeometry(QtCore.QRect(560, 10, 91, 31))
        self.InferenceButton.setObjectName("InferenceButton")
        self.InferenceButton.clicked.connect(self.infer)
        self.listView = QtWidgets.QListWidget(self.centralwidget)
        self.listView.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.listView.alternatingRowColors()
        self.listView.setGeometry(QtCore.QRect(1160, 80, 191, 591))
        self.listView.setIconSize(QtCore.QSize(185, 185))
        self.listView.setItemAlignment(QtCore.Qt.AlignCenter)
        self.listView.setObjectName("listView")

        self.listView.itemClicked.connect(self.track)

        self.listView.installEventFilter(self)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(1160, 60, 181, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1359, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.NetComboBox.addItem("Yolov5")
        self.NetComboBox.addItem("Yolov6")
        self.NetComboBox.addItem("Yolov7")

        self.comboBox.addItem("Nano")
        self.comboBox.addItem("Small")
        self.comboBox.addItem("Medium")
        self.comboBox.addItem("Large")
        self.comboBox.setCurrentIndex(1)

    def eventFilter(self, source, event):
        # Filter the event that happen in the windows
        if event.type() == QEvent.ContextMenu and source is self.listView:
            # If the event is right click(context menu click) and comes from the listView, then show the context menu
            menu = QMenu(self)
            copy_action = menu.addAction("Copy feature vector")
            track_action = menu.addAction("Track in frames")

            # Get the action that the user clicked
            selected_action = menu.exec_(event.globalPos())
            # Find the widget/item the user selected
            selected_item = source.itemAt(event.pos())
            # Take the QModelIndex of the item thanks to selected_item, then extract its index by .row(), and then
            # select the right widget with the desired information from the list(there's a 1:1 relationship between
            # widgets and items in listView)
            detected_person = self.detected_persons[self.listView.indexFromItem(selected_item).row()]

            # Based on the action, do something
            if selected_action == copy_action:
                # TODO Copy feature
                print(f'Copy feature: {detected_person.getDetectedPerson().getCoord()}')
            elif selected_action == track_action:
                # TODO Track in frames
                print(f'Track: {detected_person.getDetectedPerson().getCoord()}')

            return True

        return super().eventFilter(source, event)

    def select_file(self, path):
        self.file_path = self.model.filePath(path)
        self.PhotoWidget.setPixmap(QtGui.QPixmap(self.file_path))
        self.listView.clear()

    def infer(self):

        print("168")

        if self.file_path == '' or self.file_path is None:
            msg = QMessageBox(MainWindow)
            msg.setText("Selezionare un frame o un video\nprima di effettuare l'inferenza.")
            msg.setWindowTitle("Attenzione")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()
            return

        print(
            "Valori inference: " + self.NetComboBox.currentText() + " " + self.comboBox.currentText() +
            " threshold: " + str(self.ConfidenceSlider.value()))

        if self.NetComboBox.currentText() == "Yolov5":
            self.yolov5_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())
        elif self.NetComboBox.currentText() == "Yolov6":
            self.yolov6_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())
        elif self.NetComboBox.currentText() == "Yolov7":
            self.yolov7_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())

    def track(self, item):
        print(self.detected_persons[self.listView.currentRow()].getDetectedPerson().getCoord())
        pass

    def yolov5_inference(self, path, model_size="small", conf_threshold=.75):

        import yolov5

        tmp_path = "./Inference/yolov5/runs/detect/exp/tmp"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        else:
            clear_directory(tmp_path)

        starting_time = datetime.now()

        # Imposto il nome corretto del peso che yolov5 andrà a scaricare
        weight_name = 'yolov5n.pt'

        if model_size.lower() == "small":
            weight_name = "yolov5s.pt"
        elif model_size.lower() == "medium":
            weight_name = "yolov5m.pt"
        elif model_size.lower() == "large":
            weight_name = "yolov5l.pt"

        # Path where store the weights
        weight_path = f"./Weights/yolov5_w/{weight_name}"

        # Check if the weight file is there; if is not, download it
        if not os.path.exists(weight_path):
            url = f'https://github.com/ultralytics/yolov5/releases/download/v6.2/{weight_name}'
            try:
                weight_path = wget.download(url, weight_path)
            except:
                msg = QMessageBox(MainWindow)
                msg.setText("C'è stato un problema nel download dei pesi,\n\
                                assicurati di essere connesso ad internet e riprova.")
                msg.setWindowTitle("Attenzione")
                msg.setDefaultButton(QMessageBox.Ok)
                msg.exec_()
                traceback.print_exc()
                return

        # Set up the model with custom parameters
        model = yolov5.load(weight_path, verbose=False)
        model.conf = (conf_threshold / 100)
        model.classes = [0]
        model.agnostic = False
        model.multi_label = False

        # Actual inference
        results = model(path)

        # Parse results
        predictions = results.pred[0]

        save_path = "./Inference/yolov5/runs/detect/exp"

        try:
            # Save the result in order to render it instead of the BBless image
            results.save(save_dir=save_path, exist_ok=True)
        except:
            msg = QMessageBox(MainWindow)
            msg.setText("Non è stato possibile eseguire la detection\n\
                            sull'immagine, riprova o cambia modello.")
            msg.setWindowTitle("Attenzione")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()
            traceback.print_exc()
            return

        # Load the new image with BB
        new_photo_path = save_path + "/" + self.file_path.split('/')[-1]
        self.PhotoWidget.setPixmap(QtGui.QPixmap(new_photo_path))

        boxes = predictions[:, :4]  # x1, y1, x2, y2

        try:

            # Before adding more item I clear the list, so there's no danger of duplicates
            self.listView.clear()
            # Before adding more detected persons I clear the list, so there's no danger of duplicates
            self.detected_persons.clear()

            # For each box, create an ItemWidget to add to the Widget List(right side list)
            i = 1
            for box in boxes:
                box = box.numpy()

                # Create a copy of the image, so I don't work on the original(I need it)
                person = Image.open(path).copy().crop(box)
                # Resize the image to have a maximum size but still keeping the same aspect_ratio
                person.thumbnail((111, 181))
                # IMPORTANT: save the image back, so it will not be destroyed after exiting this scope. If not saved,
                # the image would cause a segmentation fault error on scrolling the listWidget(took 3 days to figure
                # it out)
                person_img_save_path = f"{tmp_path}/{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.jpg"
                person.save(person_img_save_path)

                # Setup custom widget
                detectedPerson = DetectedPerson(box)
                personWidget = DetectedPersonWidget(detectedPerson)
                personWidget.setPersonImage(QtGui.QPixmap(person_img_save_path))
                personWidget.setLabelName(f'Person #{i}')
                personWidget.setCoord(box)
                i += 1
                self.detected_persons.append(personWidget)

                item = QListWidgetItem(self.listView)

                # Set custom widget size, so it shows properly
                item.setSizeHint(QSize(150, 240))
                self.listView.addItem(item)
                self.listView.setItemWidget(item, personWidget)

        except:
            traceback.print_exc()

        scores = predictions[:, 4]
        categories = predictions[:, 5]

        ending_time = datetime.now()
        print(ending_time - starting_time)

        return

    def yolov6_inference(self, path, model_size="small", conf_threshold=.75):

        sys.path.append('./Models/yolov6')

        from yolov6.core.inferer import Inferer
        from yolov6.utils.nms import non_max_suppression

        tmp_path = "./Inference/yolov6/runs/detect/exp/tmp"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        else:
            clear_directory(tmp_path)

        starting_time = datetime.now()

        # Set the corerct weight's name that the net will load
        # TODO probably to delete
        weight_name = 'yolov6n.pt'

        if model_size.lower() == "small":
            weight_name = "yolov6s.pt"
        elif model_size.lower() == "medium":
            weight_name = "yolov6m.pt"
        elif model_size.lower() == "large":
            weight_name = "yolov6l.pt"

        # Path where store the weights
        weight_path = f"./Weights/yolov6_w/{weight_name}"

        # Check if the weight file is there; if is not, download it
        if not os.path.exists(weight_path):
            url = f'https://github.com/meituan/YOLOv6/releases/download/0.2.1/{weight_name}'
            try:
                weight_path = wget.download(url, weight_path)
            except:
                msg = QMessageBox(MainWindow)
                msg.setText("C'è stato un problema nel download dei pesi, assicurati di essere connesso ad internet e "
                            "riprova.")
                msg.setWindowTitle("Attenzione")
                msg.setDefaultButton(QMessageBox.Ok)
                msg.exec_()
                traceback.print_exc()
                return

        # Set up the Inferer
        # Actually, yolov6 have a differente structure. It has an Inferer, which have internally the model
        # The inferer then works as a wrapper for the model, and use it privately. Given the fact that the inferer not
        # only setyp the model, but pre/post-process the images, i still need it(or its functions at least)
        inferer = Inferer(path, weight_path, 'cpu', "./Models/yolov6/data/coco.yaml", 640, False)
        # Extract the setupped model from the Inferer
        model = inferer.model
        inferer.img_size = inferer.check_img_size(inferer.img_size, s=inferer.stride)  # check image size

        try:
            # For each file loaded(yep, I can load a directory and perform inference on each imgs)
            for img_src, img_path, vid_cap in tqdm(inferer.files):
                # Preprocess the images
                preproc_img, preproc_img_src = Inferer.precess_image(img_src, inferer.img_size, model.stride, False)

                # Batch it if it's only one img
                if len(preproc_img.shape) == 3:
                    preproc_img = preproc_img[None]  # expand for batch dim

                # Make the predictions
                predictions = model(preproc_img)

                conf_threshold /= 100

                # Apply non-max-suppression
                detections = non_max_suppression(predictions, conf_threshold, 0.45, [0])[0]

                save_path = "./Inference/yolov6/runs/detect/exp"
                img_ori = img_src.copy()

                # The model takes fixed size img, so before feeding them we have to preprocess them. Of course, the
                # resulting BB are in preprocessed coordinates, so we rescale them back
                detections[:, :4] = inferer.rescale(preproc_img.shape[2:], detections[:, :4], img_src.shape).round()

                # For each detection, extract each and every information,to draw BB on a copy of the original img
                for *xyxy, conf, cls in reversed(detections):
                    class_num = int(cls)  # integer class
                    label = f'{inferer.class_names[class_num]} {conf:.2f}'

                    inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                               color=inferer.generate_colors(class_num, True))

                # Save the new image with BB
                img_src = np.asarray(img_ori)
                new_photo_path = save_path + "/" + self.file_path.split('/')[-1]
                cv2.imwrite(new_photo_path, img_src)

                # Load the image with BB
                self.PhotoWidget.setPixmap(QtGui.QPixmap(new_photo_path))

                boxes = detections[:, :4]  # x1, y1, x2, y2

                try:

                    # Before adding more item I clear the list, so there's no danger of duplicates
                    self.listView.clear()
                    # Before adding more detected persons I clear the list, so there's no danger of duplicates
                    self.detected_persons.clear()

                    # For each box, create an ItemWidget to add to the Widget List(right side list)
                    i = 1
                    for box in boxes:
                        box = box.numpy()

                        # Create a copy of the image, so I don't work on the original(I need it)
                        person = Image.open(path).copy().crop(box)
                        # Resize the image to have a maximum size but still keeping the same aspect_ratio
                        person.thumbnail((111, 181))
                        # IMPORTANT: save the image back, so it will not be destroyed after exiting this scope. If not saved,
                        # the image would cause a segmentation fault error on scrolling the listWidget(took 3 days to figure
                        # it out)
                        person_img_save_path = f"{tmp_path}/{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.jpg"
                        person.save(person_img_save_path)

                        # Setup custom widget
                        detectedPerson = DetectedPerson(box)
                        personWidget = DetectedPersonWidget(detectedPerson)
                        personWidget.setPersonImage(QtGui.QPixmap(person_img_save_path))
                        personWidget.setLabelName(f'Person #{i}')
                        personWidget.setCoord(box)
                        i += 1
                        self.detected_persons.append(personWidget)

                        item = QListWidgetItem(self.listView)

                        # Set custom widget size, so it shows properly
                        item.setSizeHint(QSize(150, 240))
                        self.listView.addItem(item)
                        self.listView.setItemWidget(item, personWidget)

                except:
                    traceback.print_exc()

        except:
            traceback.print_exc()

        ending_time = datetime.now()
        print(ending_time - starting_time)

        return

    def yolov7_inference(self, path, model_size="small", conf_threshold=.75):

        sys.path.append('.\Models\yolov7')

        from utils.general import check_img_size, non_max_suppression, scale_coords
        from models.experimental import attempt_load
        from utils.datasets import LoadImages
        from utils.plots import plot_one_box

        tmp_path = "./Inference/yolov7/runs/detect/exp/tmp"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        else:
            clear_directory(tmp_path)

        starting_time = datetime.now()

        # Set the corerct weight's name that the net will load
        # TODO probably to delete
        weight_name = 'yolov7-w6.pt'

        if model_size.lower() == "small":
            weight_name = "yolov7-e6.pt"
        elif model_size.lower() == "medium":
            weight_name = "yolov7-d6.pt"
        elif model_size.lower() == "large":
            weight_name = "yolov7-e6e.pt"

        # Path where store the weights
        weight_path = f"./Weights/yolov7_w/{weight_name}"

        # Check if the weight file is there; if is not, download it
        if not os.path.exists(weight_path):
            url = f'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{weight_name}'
            try:
                weight_path = wget.download(url, weight_path)
            except:
                msg = QMessageBox(MainWindow)
                msg.setText("C'è stato un problema nel download dei pesi, assicurati di essere connesso ad internet e "
                            "riprova.")
                msg.setWindowTitle("Attenzione")
                msg.setDefaultButton(QMessageBox.Ok)
                msg.exec_()
                traceback.print_exc()
                return

        # Load model
        model = attempt_load(weight_path, map_location='cpu')  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(640, s=stride)  # check img_size

        save_path = "./Inference/yolov7/runs/detect/exp"

        # Load image
        dataset = LoadImages(path, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to('cpu')
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]

            conf_threshold /= 100

            # Apply NMS
            pred = non_max_suppression(pred, conf_threshold, 0.45, classes=[0])

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Save the new image with BB
            img_src = np.asarray(im0)
            new_photo_path = save_path + "/" + self.file_path.split('/')[-1]
            cv2.imwrite(new_photo_path, img_src)

            # Load the image with BB
            self.PhotoWidget.setPixmap(QtGui.QPixmap(new_photo_path))

            boxes = pred[0][:, :4]  # x1, y1, x2, y2

            try:

                # Before adding more item I clear the list, so there's no danger of duplicates
                self.listView.clear()
                # Before adding more detected persons I clear the list, so there's no danger of duplicates
                self.detected_persons.clear()

                # For each box, create an ItemWidget to add to the Widget List(right side list)
                i = 1
                for box in boxes:
                    box = box.numpy()

                    # Create a copy of the image, so I don't work on the original(I need it)
                    person = Image.open(path).copy().crop(box)
                    # Resize the image to have a maximum size but still keeping the same aspect_ratio
                    person.thumbnail((111, 181))
                    # IMPORTANT: save the image back, so it will not be destroyed after exiting this scope. If not saved,
                    # the image would cause a segmentation fault error on scrolling the listWidget(took 3 days to figure
                    # it out)
                    person_img_save_path = f"{tmp_path}/{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.jpg"
                    person.save(person_img_save_path)

                    # Setup custom widget
                    detectedPerson = DetectedPerson(box)
                    personWidget = DetectedPersonWidget(detectedPerson)
                    personWidget.setPersonImage(QtGui.QPixmap(person_img_save_path))
                    personWidget.setLabelName(f'Person #{i}')
                    personWidget.setCoord(box)
                    i += 1
                    self.detected_persons.append(personWidget)

                    item = QListWidgetItem(self.listView)

                    # Set custom widget size, so it shows properly
                    item.setSizeHint(QSize(150, 240))
                    self.listView.addItem(item)
                    self.listView.setItemWidget(item, personWidget)

            except:
                traceback.print_exc()

        ending_time = datetime.now()
        print(ending_time - starting_time)

        return

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Framework"))
        self.NetLabel.setText(_translate("MainWindow", "Choose net"))
        self.label.setText(_translate("MainWindow", "Choose net size"))
        self.ConfidenceLabel.setText(_translate("MainWindow", "Confidence threshold: 85"))
        self.PhotoWidget.setText(_translate("MainWindow", "Seleziona un frame o un video"))
        self.InferenceButton.setText(_translate("MainWindow", "Inference"))
        self.label_2.setText(_translate("MainWindow", "Detected Persons"))


if __name__ == "__main__":
    faulthandler.enable()  # start @ the beginning

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
