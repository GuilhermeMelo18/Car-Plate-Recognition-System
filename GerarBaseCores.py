import cv2
import ReconhecedorCaracteres
import ReconhecedorPlacas
import numpy as np
import glob

#Método para Treinamento do KNN com as Imagens Definidas
#Algoritmo de Aprendizado KNN
def main():

    listaPlacasAvaliadas = []
    baseKNNOk = ReconhecedorCaracteres.carregarBaseTreinoKNN("categoriasBase.txt", "descritoresBase.txt")

    if baseKNNOk == False:
        return
    # end if

    for filename in glob.glob('imagensTreino/*.*'):    # Base Leitura das Imagens de Treino
        imagemBase = cv2.imread(filename)

        if imagemBase is None:
            print ("\n Leitura da Imagem Não Ok \n")
            continue
        # end if

        listaPlacasDetectadas = ReconhecedorPlacas.detectPlatesInScene(imagemBase)     # Lista de Possíveis Placas Detectadas

        if len(listaPlacasDetectadas) == 0:  # Se a lista de placas fo vazia
            print("\nLista de Placas Vazia\n")
            continue
        # end if

        listaPlacasDetectadas, listaDescritores, listaDescritoresOrb = ReconhecedorCaracteres.detectarCharsPlacas(listaPlacasDetectadas,True)    # Detectar Caracteres nas Possiveis Placas

        listaPlacasNaoOrdenado  =  list(listaPlacasDetectadas)

        if len(listaPlacasDetectadas) == 0:
            continue           #Se Placas não Foram Encontradas na Imagem
        else:

            listaPlacasDetectadas.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

            placa = listaPlacasDetectadas[0]  #Captura Placa na Imagem

            if len(placa.strChars) == 0:
                print ("\nPlaca sem Caracteres\n\n" )
                continue
            # end if

            print ("\nPlaca Para Treino = " + filename + "\n" )      # print Caracteres da Placa
            listaPlacasAvaliadas.append(placa.strChars)
            listaDescritoresPlaca = getDescritoresDasPlacas(listaPlacasNaoOrdenado, placa.strChars, listaDescritores)
            salvarBase(placa.strChars, listaDescritoresPlaca)
        print("\n-----------------------------------------------------------------------------\n")
    print(str("Qtd de Placas Para o KNN:")+str(len(listaPlacasAvaliadas)))
    print()
    return
# end main

#Método para Join dos Caracteres - Descritores
def getDescritoresDasPlacas(listaPlacas, strplaca, listaDescritores):

    strPlacas = ""

    caracteresPlaca = []

    for lp in listaPlacas:
        strPlacas = strPlacas + lp.strChars

    #print(strPlacas)  #acompanhar lista possiveis caracteres

    position = strPlacas.find(strplaca)

    finalPosition = position + (len(strplaca)-1)

    for list in listaDescritores:

        if(position <= listaDescritores.index(list) and finalPosition >= listaDescritores.index(list)):

            caracteresPlaca.append(list)

    return caracteresPlaca

#Método para Salvar Caracteres - Descritores
def salvarBase(caterorias, descritores):

    cat = open('categorias.txt', 'a')  # Salvar Categorias em Arquivo

    for c in caterorias:

        cat.write(str(np.float32(ord(c))))
        cat.write('\n')

    cat.close()

    des = open('descritores.txt', 'a')  # Salvar Descritores em Arquivo

    for d in descritores:

        des.write(d)
        des.write('\n')

    des.close()
#end function

if __name__ == "__main__":
    main()











