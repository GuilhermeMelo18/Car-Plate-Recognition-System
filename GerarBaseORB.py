import glob

import cv2
import numpy as np

import ReconhecedorCaracteres
import ReconhecedorPlacas



def main():


    knnTreinoOk = ReconhecedorCaracteres.carregarBaseTreinoKNN("categoriasORB.txt",
                                                               "descritoresORB.txt")  # Carrega Treino KNN

    for filename in glob.glob('imagensTreino/*.*'):    # Base Leitura das Imagens de Treino


        print("---------------------------------")
        print(filename)
        print("---------------------------------")
        imgNaoCategorizada = cv2.imread(filename)

        if imgNaoCategorizada is None:
            print("\n Leitura da Imagem Não Ok \n")
            # continue
        # end if

        listaPlacas = ReconhecedorPlacas.detectPlatesInScene(imgNaoCategorizada)  # reconhece placas

        listaPlacas, listaDescritores, listaDescritoresORB = ReconhecedorCaracteres.detectarCharsPlacas(listaPlacas,False)  # reconhece caracteres na placa

        if len(listaPlacas) == 0:
            print("\nLista de Placas Vazia\n")
            # continue
        else:

            listaPlacas.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            licPlate = listaPlacas[0]

            #cv2.imshow("imgPlate", licPlate.imgPlate) #Mostrar Placa
            # cv2.imshow("imgThresh", licPlate.imgThresh) # Mostrar Placa Thresh

            if len(licPlate.strChars) == 0:
                print("\n Placas sem Caracteres\n")
            #   continue
            # end if

            print("\nCaracteres Encontrados na Placa = " + licPlate.strChars + "\n")  # Caracteres Encontrados na Placa
            print("_________________________________________________________________")

            # cv2.waitKey(0)

            #Salvar Descritores
            avaliarDescritores(listaDescritoresORB)


#Método para Verificação da Taxa de Acertos dos Caracteres nas Placas
def avaliarDescritores(listaDescritoresORB):


    for descritor in listaDescritoresORB:
        if descritor[0]!="'":
            print(descritor)
            salvarDescritor(descritor)

#end function


def salvarDescritor(descritor):
    cat = open('categoriasORB.txt', 'a')  # Salvar Categorias em Arquivo

    cat.write(str(np.float32(ord(descritor[0]))))
    cat.write('\n')

    cat.close()

    des = open('descritoresORB.txt', 'a')  # Salvar Descritores em Arquivo

    des.write(descritor[1:len(descritor)])
    des.write('\n')

    des.close()
if __name__ == "__main__":
    main()
