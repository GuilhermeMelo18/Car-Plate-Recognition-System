import cv2
import numpy as np
import math

class PossibleChar:

    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

            #top-left coordinate
        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
            #width and height
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

            #Area
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

            #Centro
        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

            #Tamanho da Diagonal
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

            #Aspect Ratio
        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
    # end constructor

# end class








