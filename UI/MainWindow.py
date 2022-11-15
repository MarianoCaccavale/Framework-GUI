# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QDir


class Ui_MainWindow(object):
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
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ConfidenceLabel.setFont(font)
        self.ConfidenceLabel.setObjectName("ConfidenceLabel")
        self.ConfidenceChooser.addWidget(self.ConfidenceLabel)
        self.ConfidenceSlider = QtWidgets.QSlider(self.horizontalLayoutWidget)
        self.ConfidenceSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ConfidenceSlider.setObjectName("ConfidenceSlider")
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
        self.PhotoWidget.setGeometry(QtCore.QRect(210, 60, 950, 611))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PhotoWidget.sizePolicy().hasHeightForWidth())
        self.PhotoWidget.setSizePolicy(sizePolicy)
        self.PhotoWidget.setScaledContents(True)
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
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(1160, 80, 191, 591))
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

    def select_file(self, path):
        file_path = self.model.filePath(path)
        self.PhotoWidget.setPixmap(QtGui.QPixmap(file_path))

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Framework"))
        self.NetLabel.setText(_translate("MainWindow", "Choose net"))
        self.label.setText(_translate("MainWindow", "Choose net size"))
        self.ConfidenceLabel.setText(_translate("MainWindow", "Confidence threshold: "))
        self.PhotoWidget.setText(_translate("MainWindow", "Seleziona un frame o un video"))
        self.InferenceButton.setText(_translate("MainWindow", "Inference"))
        self.label_2.setText(_translate("MainWindow", "Detected Persons"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())