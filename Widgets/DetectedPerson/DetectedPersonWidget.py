from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from .DetectedPerson import DetectedPerson

class DetectedPersonWidget(QWidget):
    detectedPerson: DetectedPerson = None

    def __init__(self, detectedPerson, parent=None):
        QWidget.__init__(self, parent=parent)
        self.detectedPerson = detectedPerson
        self.personsImage = QtWidgets.QLabel(self)
        self.personsImage.setGeometry(QtCore.QRect(30, 10, 111, 181))
        self.personsImage.setText("")
        self.personsImage.setObjectName("personsImage")
        self.personsImage.setAlignment(QtCore.Qt.AlignCenter)
        self.labelName = QtWidgets.QLabel(self)
        self.labelName.setGeometry(QtCore.QRect(50, 210, 71, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.labelName.setFont(font)
        self.labelName.setText("")
        self.labelName.setObjectName("labelName")
        self.labelName.setAlignment(QtCore.Qt.AlignCenter)

    def setPersonImage(self, new_image):
        self.personsImage.setPixmap(new_image)

    def setLabelName(self, new_name):
        self.labelName.setText(new_name)

    def setCoord(self, coord):
        assert coord[0] is not None and coord[1] is not None and coord[2] is not None and coord[3] is not None
        self.detectedPerson.setCoord(coord)

    def getDetectedPerson(self) -> DetectedPerson:
        return self.detectedPerson

    def onClick(self):
        print(f'Clicked: {self.personsImage} - {self.labelName} - {self.coords}')
