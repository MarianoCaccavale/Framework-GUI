class DetectedPerson:
    _coord: (float, float, float, float) = (.0, .0, .0, .0)

    def __init__(self, coord=None):
        super().__init__()
        if coord is not None:
            self.coord = coord

    def setCoord(self, newCoord: (float, float, float, float)):
        self._coord = newCoord

    def getCoord(self):
        return self._coord
