# -*- coding: utf-8 -*-
import faulthandler
import os
from datetime import datetime

import wget
import yolov5
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from Utils.folders import clear_directory

class Ui_MainWindow(object):
    file_path = ''

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
        #
        ## *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
        ## Solve the "crash when scroll" problem of the listWidget; idk why honestly... but it works(?)
        ## setting horizontal scroll mode
        # self.listView.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        ## resetting horizontal scroll mode
        # self.listView.resetHorizontalScrollMode()
        ## *^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^
        #
        self.listView.setGeometry(QtCore.QRect(1160, 80, 191, 591))
        self.listView.setIconSize(QtCore.QSize(185, 185))
        self.listView.setItemAlignment(QtCore.Qt.AlignCenter)
        self.listView.setObjectName("listView")
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

    def select_file(self, path):
        self.file_path = self.model.filePath(path)
        self.PhotoWidget.setPixmap(QtGui.QPixmap(self.file_path))

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
            "Valori inference: " + self.NetComboBox.currentText() + " " + self.comboBox.currentText() + " threshold: " + str(
                self.ConfidenceSlider.value()))

        if self.NetComboBox.currentText() == "Yolov5":
            self.yolov5_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())
        elif self.NetComboBox.currentText() == "Yolov6":
            self.yolov6_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())
        elif self.NetComboBox.currentText() == "Yolov7":
            self.yolov7_inference(self.file_path, model_size=self.comboBox.currentText(),
                                  conf_threshold=self.ConfidenceSlider.value())

    def yolov5_inference(self, path, model_size="small", conf_threshold=.75):

        tmp_path = "./Inference/yolov5/runs/detect/exp/tmp/"
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
                pass

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

        except Exception as e:
            msg = QMessageBox(MainWindow)
            msg.setText("Non è stato possibile eseguire la detection\n\
                            sull'immagine, riprova o cambia modello.")
            msg.setWindowTitle("Attenzione")
            msg.setDefaultButton(QMessageBox.Ok)
            msg.exec_()

            print(e)
            pass

        # Load the new image with BB
        new_photo_path = save_path + "/" + self.file_path.split('/')[-1]
        self.PhotoWidget.setPixmap(QtGui.QPixmap(new_photo_path))

        # Create a copy of the original photo to crop out the detected persons
        # in order to show them in the "detected persons list"
        # orig_img = Image.open(path)

        boxes = predictions[:, :4]  # x1, y1, x2, y2

        try:

            # Before adding more item I clear the list, so there's no danger of duplicates
            self.listView.clear()

            # For each box, create an ItemWidget to add to the Widget List(right side list)
            for box in boxes:
                box = box.numpy()

                person = Image.open(path).copy().crop(box)
                person_img_save_path = f"{save_path}/tmp/{int(box[0])}_{int(box[1])}_{int(box[2])}_{int(box[3])}.jpg"
                person.save(person_img_save_path)

                icon = QtGui.QIcon(QtGui.QPixmap(person_img_save_path))
                item = QListWidgetItem(icon, '')
                self.listView.addItem(item)

        except Exception as e:
            print(e)

        scores = predictions[:, 4]
        categories = predictions[:, 5]

        ending_time = datetime.now()
        print(ending_time - starting_time)

        pass

    def yolov6_inference(self, path, model_size="small", conf_threshold=.75):
        pass

    def yolov7_inference(self, path, model_size="small", conf_threshold=.75):
        pass

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

    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
