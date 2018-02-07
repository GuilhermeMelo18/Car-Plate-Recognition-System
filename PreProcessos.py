import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

#Método para aplicação de Pré-Processos na Imagem GrayScale e Thresh
def preprocessos(imgOriginal):

    imgGrayscale = imagemGrayScale(imgOriginal)
    #plt.hist(imgGrayscale.ravel(), 256, [0, 256]);plt.show()

    #Aprimoramento de Contraste e Brilho da Imagem
    imgMaxContrastGrayscale = aprimorarContraste(imgGrayscale)

    #plt.hist(imgMaxContrastGrayscale.ravel(), 256, [0, 256]);plt.show()

    #Kernel 5x , 5y Gaussian Filter
    imgGaussian = cv2.GaussianBlur(imgMaxContrastGrayscale, (5, 5), 0)


    #Size Block 19 Thresh Adaptative
    imgThresh = cv2.adaptiveThreshold(imgGaussian, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    return imgGrayscale, imgThresh
# end function

#Método para converter Imagem para GrayScale
def imagemGrayScale(imgOriginal):

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

#Método Utilizado pra Aprimoramento do Contraste e Brilho da Imagem
def aprimorarContraste(imgGrayscale):

    # Rectangular Kernel 3x3
    elementoEstruturante = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    #difference between input image and Opening of the image
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, elementoEstruturante)
    #difference between the closing of the input image and input image
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, elementoEstruturante)

    #Aplicação Adição e Subtração para Aumento do Constraste da Imagem
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function










