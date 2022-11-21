from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget


class DetectedPersonWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.personsImage = QtWidgets.QLabel(self)
        self.personsImage.setGeometry(QtCore.QRect(30, 10, 111, 181))
        self.personsImage.setText("")
        self.personsImage.setObjectName("personsImage")
        self.labelName = QtWidgets.QLabel(self)
        self.labelName.setGeometry(QtCore.QRect(50, 210, 71, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelName.setFont(font)
        self.labelName.setText("")
        self.labelName.setObjectName("labelName")
        self.coords = QtWidgets.QLabel(self)
        self.coords.setGeometry(QtCore.QRect(10, 240, 151, 21))
        self.coords.setText("")
        self.coords.setObjectName("coords")

    def setPersonImage(self, new_image):
        self.personsImage.setPixmap(new_image)

    def setLabelName(self, new_name):
        self.labelName.setText(new_name)

    def setCoords(self, new_coords):
        self.coords.setText(new_coords)
