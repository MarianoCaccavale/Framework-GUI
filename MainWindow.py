# -*- coding: utf-8 -*-
import faulthandler
import importlib
import os
import traceback
import sys
from datetime import datetime
import numpy as np

from Utils.folders import clear_directory
from Widgets.DetectedPerson.DetectedPersonWidget import DetectedPersonWidget, DetectedPerson

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QSize, QEvent
from PyQt5.QtWidgets import *

from Utils.plots import plot_box_and_label


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

        # self.NetComboBox.addItem("Yolov5")
        # self.NetComboBox.addItem("Yolov6")
        # self.NetComboBox.addItem("Yolov7")

        self.discover_models(self.NetComboBox)

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

        if self.file_path == '' or self.file_path is None:
            msg = QMessageBox(MainWindow)
            msg.setText("Selezionare un frame o un video\nprima di effettuare l'inferenza.")
            msg.setWindowTitle("Attenzione")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()
            return

        print(f"Quello che sta nella combo box: {self.NetComboBox.currentText()}")
        print(f"Quello che devo cercare nella cartella: ./Models/{self.NetComboBox.currentText()}.onnx")

        self.dynamic_infer(self.file_path, self.NetComboBox.currentText(), conf_threshold=self.ConfidenceSlider.value())

    def dynamic_infer(self, path, module_name, conf_threshold=45, IoU_threshold=50):

        starting_time = datetime.now()

        main_path = f"./Inference/{str(starting_time.time()).replace(':', '_').replace('.', '_')}_{module_name}_{conf_threshold}".strip()

        tmp_path = f"{main_path}/bounding_boxes"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        else:
            clear_directory(tmp_path)

        # Duplicate the image
        Image.open(path).save(f'{main_path}/original_image.png')

        # Dynamic import of the module
        module = importlib.import_module(f'Models.{module_name}.Inference')

        boxes = []
        labels = []
        confidence = []

        try:
            # Black-box inference
            boxes, confidence, labels = module.inference(path, max(conf_threshold / 100, 0.01),
                                                         max(IoU_threshold / 100, 0.01))
        except:
            traceback.print_exc()

        try:
            assert boxes.shape[1] == 4
            assert confidence.shape[1] == 1
            assert labels.shape[1] == 1
        except AssertionError as e:
            msg = QMessageBox(MainWindow)
            msg.setText("Formati degli output sbagliati,\n\
                                            assicurati di aver formatoto bene l'output e riprova.")
            msg.setWindowTitle("Attenzione")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()
            traceback.print_exc()
            return


        print(f"The result of the inference was: {boxes, labels, confidence}")

        try:
            # Before adding more item I clear the list, so there's no danger of duplicates
            self.listView.clear()
            # Before adding more detected persons I clear the list, so there's no danger of duplicates
            self.detected_persons.clear()
            # For each box, create an ItemWidget to add to the Widget List(right side list)
            i = 0
            with open(f'./{main_path}/features_vector.txt', 'w') as txt_file:

                # Open the original image to draw BB
                with Image.open(path).copy() as orig_image:
                    orig_image_array = np.asarray(orig_image)

                    for i, box in enumerate(boxes):
                        box = box.numpy()

                        # Draw BB
                        plot_box_and_label(orig_image_array, 2, box, f"Person {confidence[i][0]:.2f}")

                        i += 1
                        # Create a copy of the image, so I don't work on the original(I need it)
                        person = Image.open(path).copy().crop(box)
                        # Resize the image to have a maximum size but still keeping the same aspect_ratio
                        person.thumbnail((111, 181))
                        # IMPORTANT: save the image back, so it will not be destroyed after exiting this scope. If
                        # not saved, the image would cause a segmentation fault error on scrolling the listWidget(
                        # took 3 days to figure it out)
                        person_img_save_path = f"{tmp_path}/{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.jpg"
                        person.save(person_img_save_path)

                        # Write the file with the features
                        txt_file.write(f"{i}:{int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])},\n")

                        # Setup custom widget
                        detectedPerson = DetectedPerson(box)
                        personWidget = DetectedPersonWidget(detectedPerson)
                        personWidget.setPersonImage(QtGui.QPixmap(person_img_save_path))
                        personWidget.setLabelName(f'Person #{i}')
                        personWidget.setCoord(box)

                        self.detected_persons.append(personWidget)

                        item = QListWidgetItem(self.listView)

                        # Set custom widget size, so it shows properly
                        item.setSizeHint(QSize(150, 240))
                        self.listView.addItem(item)
                        self.listView.setItemWidget(item, personWidget)

                    # Save the complete BB image, so I don't lose its reference
                    Image.fromarray(orig_image_array).save(f'{main_path}/BB_image.png')
                    # Then reaload it in the central widget
                    self.PhotoWidget.setPixmap(QtGui.QPixmap(f'{main_path}/BB_image.png'))

        except:
            traceback.print_exc()

    def track(self, item):
        print(self.detected_persons[self.listView.currentRow()].getDetectedPerson().getCoord())
        pass

    def discover_models(self, combo_box):
        # Discover files in the Models directory
        for file in os.listdir("./Models"):
            # If they're directories...
            if os.path.isdir(f"./Models/{file}"):
                # ... and they have the "inference" module ...
                for module in os.listdir(f"./Models/{file}"):
                    if module.lower() == "inference.py" and os.path.isfile(f"./Models/{file}/{module}"):
                        # ... add the element to the net choose
                        combo_box.addItem(file)

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
