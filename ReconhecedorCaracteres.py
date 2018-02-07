import cv2
import numpy as np
import math
import random
import EnumChar as en
import CategorizadorPlacas
import PreProcessos
import Caractere

#Algoritmo KNN OpenCV
algoritmoKNN = cv2.ml.KNearest_create()


#Método Para Detecção dos Caracteres nas Possíveis Placas Encontradas
def detectarCharsPlacas(listaPlacas,indicadorDescritorCor):

    intPlateCounter = 0
    imgContours = None
    contours = [] #Contornos encontrados nas Placas
    descritoresPlacas = [] #Descritores Placas
    descritoresPlacasORB = [] #Descritores Placas ORB


    for placa in listaPlacas:

        # Chamada dos Pré-Processos Necessários na Imagem GrayScale e Thresh
        placa.imgGrayscale, placa.imgThresh = PreProcessos.preprocessos(placa.imgPlate)

        placa.imgThresh = cv2.resize(placa.imgThresh, (0, 0), fx = 1.6, fy = 1.6) #Padroniza Tamanho da Placa

        thresholdValue, placa.imgThresh = cv2.threshold(placa.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        #Encontrar Prováveis Caracteres na Placa
        listaCharPlaca = reconhecerCaracteresPlaca(placa.imgGrayscale, placa.imgThresh)

        height, width, numChannels = placa.imgPlate.shape #Shape Placa

                # Encontrar  grupos de Caracteres com as mesmas Caracteristicas Dentro da Placa
        listaGruposMatchChar = encontrarListaMatchingChars(listaCharPlaca)


        if (len(listaGruposMatchChar) == 0):

            intPlateCounter = intPlateCounter + 1

            placa.strChars = ""
            continue
        # end if

        for i in range(0, len(listaGruposMatchChar)):
            listaGruposMatchChar[i].sort(key = lambda matchingChar: matchingChar.intCenterX)
            listaGruposMatchChar[i] = removerInnerOverlappingChars(listaGruposMatchChar[i])   # Remover overlapping chars internos
        # end for

            #Calcula a Maior Lista Disponível
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listaGruposMatchChar)):
            if len(listaGruposMatchChar[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listaGruposMatchChar[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # Pega a maior lista de Match Caracteres Encontrados
        maiorListaMatchChars = listaGruposMatchChar[intIndexOfLongestListOfChars]

            #Categorizar Caracteres nas Possíveis Placas Encontradas
        placa.strChars, listaDescritores, listaDescritoresOrb = categorizarCharsPlaca(placa.imgThresh, maiorListaMatchChars,indicadorDescritorCor)

        for list in listaDescritores:
            descritoresPlacas.append(list)    #Salva os Descritores da Possíveis Placas

        for l in listaDescritoresOrb:
            descritoresPlacasORB.append(l)

        intPlateCounter = intPlateCounter + 1

    # end loop

    return listaPlacas, descritoresPlacas, descritoresPlacasORB
# end function

#Método para reconhecer possíveis caracteres na Placa
def reconhecerCaracteresPlaca(imgGrayscale, imgThresh):
    listaPossiveisChars = []
    imgThreshCopy = imgThresh.copy()

        # Encontrar Contornos na Placas (Método Chain)
    imgContours, contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possibleChar = Caractere.PossibleChar(contour)  #Montar Caracteristicas dos Contornos

        if checkPossivelCaractere(possibleChar):              # Checar se é um possível Caractere
            listaPossiveisChars.append(possibleChar)
        # end if
    # end if

    return listaPossiveisChars
# end function

# Método para Checar Características de um Caractere
def checkPossivelCaractere(possibleChar):

    if (possibleChar.intBoundingRectArea > en.EnumChar.AREA_MIN_PIXEL and
        possibleChar.intBoundingRectWidth > en.EnumChar.LARG_MIN_PIXEL and possibleChar.intBoundingRectHeight > en.EnumChar.ALT_MIN_PIXEL and
        en.EnumChar.MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < en.EnumChar.MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

#Método para Match Caracteres (Retorna lista de Grupos de Match Caracteres)
def encontrarListaMatchingChars(listaPossiveisChars):

    listaGruposMatchChar = []

    for possibleChar in listaPossiveisChars:
        listaMatchChars = realizarMatchChars(possibleChar, listaPossiveisChars)        # Lista de Grupo de Caracteres

        listaMatchChars.append(possibleChar)

        if len(listaMatchChars) < en.EnumChar.MIN_NUMBER_OF_MATCHING_CHARS:     # Número Mínimo de Chars no Grupo
            continue

        # end if


        listaGruposMatchChar.append(listaMatchChars)  #Append Grupos na lista de Retorno

        listaRestanteChars = list(set(listaPossiveisChars) - set(listaMatchChars))

        listaRecursivaMatchChars = encontrarListaMatchingChars(listaRestanteChars) # Chama Método Recursivamente Para
                                                                                        # Restante dos Caracteres
        for recursivelistaMatchChars in listaRecursivaMatchChars:
            listaGruposMatchChar.append(recursivelistaMatchChars)
        # end for

        break

    # end for

    return listaGruposMatchChar
# end function

# Agrupar Possíveis Chars com Características Parecidas
def realizarMatchChars(possibleChar, listOfChars):

    listaMatchChars = []

    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar: #Se for Igual Pula os Comandos

            continue
        # end if

        rDistanciaEntreCaracteres = distanciaEntreCaracteres(possibleChar, possibleMatchingChar)

        rAnguloEntreCaracteres = anguloEntreCaracteres(possibleChar, possibleMatchingChar)

        changeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        changeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        changeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        # Check se os Caracteres Match
        if (rDistanciaEntreCaracteres < (possibleChar.fltDiagonalSize * en.EnumChar.MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            rAnguloEntreCaracteres < en.EnumChar.MAX_ANGLE_BETWEEN_CHARS and
            changeInArea < en.EnumChar.MAX_CHANGE_IN_AREA and
            changeInWidth < en.EnumChar.MAX_CHANGE_IN_WIDTH and
            changeInHeight < en.EnumChar.MAX_CHANGE_IN_HEIGHT):

            listaMatchChars.append(possibleMatchingChar)
        # end if
    # end for

    return listaMatchChars
# end function

# Calcula Distancia Euclidiana entre 2 Caracteres
def distanciaEntreCaracteres(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

#Calcula angulo entre dois Pontos (x,y)
def anguloEntreCaracteres(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)

    return fltAngleInDeg
# end function

#Método Separa todos Chars que se Sobrepõe, Removendo o Char mais interno (O menor)
def removerInnerOverlappingChars(listaMatchChars):
    listaOverCharLimpa = list(listaMatchChars) #lista com Overlappings Removidos

    for currentChar in listaMatchChars:
        for otherChar in listaMatchChars:
            if currentChar != otherChar:
                                                                            # verifica se centros são semelhantes
                if distanciaEntreCaracteres(currentChar, otherChar) < (currentChar.fltDiagonalSize * en.EnumChar.MIN_DIAG_SIZE_MULTIPLE_AWAY):

                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listaOverCharLimpa:
                            listaOverCharLimpa.remove(currentChar)         # remove menor char overlapping
                        # end if
                    else:
                        if otherChar in listaOverCharLimpa:
                            listaOverCharLimpa.remove(otherChar)
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listaOverCharLimpa
# end function

# Método para Reconhecimento dos Caracteres nas Possíveis Placas Usando KNN 
def categorizarCharsPlaca(imgThresh, listaMatchChars, indicadorDescritorCor):

    strChars = ""               # retorna a classificação dos caracteres COR

    strCharsOrb = ""            # retorna a classificação dos caracteres ORB

    strDescritor =""            # retorna os descritores dos caracteres COR

    strDescritorOrb= ""         # retorna os descritores dos caracteres ORB

    listaDescritores = list()       # retorna lista dos descritores dos caracteres COR

    listaDescritoresOrb = list() # Restorna lista dos descritores ORB dos Caracteres

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listaMatchChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)

    for currentChar in listaMatchChars:

        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, (0.0, 255.0, 0.0), 2)           # Contorna os Caracteres da Placa

        ################## Descritores CORES
                # Recorta Caractere da Imagem Thresh
        imgRecortada = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgRecortadaResized = cv2.resize(imgRecortada, (en.EnumChar.RESIZED_CHAR_IMAGE_WIDTH, en.EnumChar.RESIZED_CHAR_IMAGE_HEIGHT))


        imgRecortadaResized = imgRecortadaResized.reshape((1, en.EnumChar.RESIZED_CHAR_IMAGE_WIDTH * en.EnumChar.RESIZED_CHAR_IMAGE_HEIGHT))   # Coloca imagem dentro de 1D numpy array


        imgRecortadaResized = np.float32(imgRecortadaResized)               # converte array int para float
        retval, npaResults, neigh_resp, dists = algoritmoKNN.findNearest(imgRecortadaResized, k = 1)  #Categoriza Caractere com o KNN com K=1

        strCurrentChar = str(chr(int(npaResults[0][0])))

        strChars = strChars + strCurrentChar # String de Caracteres da Placa
        ##Descritor Cor
        for d in imgRecortadaResized:
            for s in d:
                strDescritor = strDescritor + str(" ") + str(s)
        ############################################### FIM Descritores CORES ######################



        ################## Retirada dos Descritores ORB

        imgRecortadaGrayScale = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight+2,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth+2]
        imgRecortadaResizedGrayScale = cv2.resize(imgRecortadaGrayScale, (200, 200))

        orb = cv2.ORB_create()
        kp, descritorOrb = orb.detectAndCompute(imgRecortadaResizedGrayScale, None)
        img2 = cv2.drawKeypoints(imgRecortadaResizedGrayScale, kp, None, color=(0, 255, 0), flags=0)


        #cv2.imshow('Caractere',imgRecortadaResizedGrayScale)
        #cv2.waitKey(0)

        #cv2.imshow('Caractere',img2)
        #cv2.waitKey(0)

        #continue se não achar descritor
        if(descritorOrb is None):
            continue

        descritorOrbResize = cv2.resize(descritorOrb, (en.EnumChar.RESIZED_CHAR_IMAGE_WIDTH, en.EnumChar.RESIZED_CHAR_IMAGE_HEIGHT))

        descritorOrbResize = descritorOrbResize.reshape((1, en.EnumChar.RESIZED_CHAR_IMAGE_WIDTH * en.EnumChar.RESIZED_CHAR_IMAGE_HEIGHT))   # Coloca imagem dentro de 1D numpy array

        descritorOrbResize = np.float32(descritorOrbResize)  # converte array int para float

        #print(descritorOrbResize)

        #classificacao = input('Entre com a Classificação do Caractere: ')

        ##Break Placa ORB
        #if(classificacao=="-"):
        #    break

        #strDescritorOrb = strDescritorOrb + str(classificacao)  # String de Caracteres da Placa

        ##Descritor ORB
        #for des in descritorOrbResize:
        #    for des2 in des:
        #        strDescritorOrb = strDescritorOrb + str(" ")+ str(des2)

        retvalOrb, npaResultsOrb, neigh_respOrb, distsOrb = algoritmoKNN.findNearest(descritorOrbResize,
                                                                         k=1)  # Categoriza Caractere com o KNN com K=1

        strCurrentCharOrb = str(chr(int(npaResultsOrb[0][0])))

        strCharsOrb = strCharsOrb + strCurrentCharOrb #Get Classificação ORB


        ########################### FIM Retirada dos Descritores ORB

        #print(str(strCurrentChar) + strDescritor)      #Acompanhamento dos Descritores
        listaDescritoresOrb.append(strDescritorOrb)
        listaDescritores.append(strDescritor)
        strDescritor = ""
        strDescritorOrb = ""

    #print(listaDescritoresOrb)
    #print(strCharsOrb)    #Acompanhamento dos Caracteres
    # end for

    if indicadorDescritorCor==True:
        strReturn = strChars
    else:
        strReturn = strCharsOrb

    return strReturn, listaDescritores, listaDescritoresOrb

# end function


#Método para Treinar o Algoritmo KNN com a Base Salva
def carregarBaseTreinoKNN(classificacoes, descritores):

    classificadores = np.loadtxt(classificacoes , np.float32)     # Ler Classificadores

    descritoresImagens = np.loadtxt(descritores, np.float32)

    classificadores = classificadores.reshape((classificadores.size, 1))    # Reshape Classificadores para Numpy Array 1D

    algoritmoKNN.setDefaultK(1)     # KNN K=1 Default

    algoritmoKNN.train(descritoresImagens, cv2.ml.ROW_SAMPLE, classificadores) # Treinamento KNN

    return True
# end function








