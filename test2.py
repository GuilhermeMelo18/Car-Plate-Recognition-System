import cv2
import numpy as np

listaplaca2 = ['ZOOMN65']
listaplaca1 = ['ZOOMN65']

acertos = 0
qtdCaracteres = 0

for counter in range(0, len(listaplaca1)):
    for c in listaplaca1[counter]:
        qtdCaracteres = qtdCaracteres + 1
        if (listaplaca2[counter].find(c) == 0):
            acertos = acertos + 1

        listaplaca2[counter] = listaplaca2[counter][1:]

porcentagemAcerto = (acertos * 100) / qtdCaracteres


print(str(np.float32("72")))