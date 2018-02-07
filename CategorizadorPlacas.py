import cv2
import ReconhecedorCaracteres
import ReconhecedorPlacas
import glob

def main():

    listaPlacasCategorizadasCores,listaFileNamesCores = categorizarPlacas("categoriasBase.txt","descritoresBase.txt",True)


    listaPlacasCategorizadasOrb,listaFileNamesOrb = categorizarPlacas("categoriasORB.txt","descritoresORB.txt",False)

    #print(listaFileNames)
    porcentagemAcertosCores = verificarTaxaAcerto(listaPlacasCategorizadasCores,listaFileNamesCores)

    porcentagemAcertosOrb = verificarTaxaAcerto(listaPlacasCategorizadasOrb, listaFileNamesOrb)

    print("--------------------------------------------------------------------------")
    print(str("Taxa de Acertos com Descritores de Cores: ")+ str(porcentagemAcertosCores) + str(" %")) #Imprime Taxa de Acertos de Caractes da Placas
    print("--------------------------------------------------------------------------")

    print("--------------------------------------------------------------------------")
    print(str("Taxa de Acertos com Desciritores ORB: ") + str(porcentagemAcertosOrb) + str(" %"))  # Imprime Taxa de Acertos de Caractes da Placas
    print("--------------------------------------------------------------------------")

    return
# end main

#Método para Verificação da Taxa de Acertos dos Caracteres nas Placas
def verificarTaxaAcerto(listaPlacas, listaPlacasBase):
    listaplaca1 = listaPlacas
    listaplaca2 = listaPlacasBase

    acertos = 0
    qtdCaracteres = 0

    for counter in range(0, len(listaplaca1)):
        for c in listaplaca1[counter]:
            qtdCaracteres = qtdCaracteres + 1
            if (listaplaca2[counter].find(c) == 0):
                acertos = acertos + 1

            listaplaca2[counter] = listaplaca2[counter][1:]

    porcentagemAcerto = (acertos * 100) / qtdCaracteres

    return porcentagemAcerto
#end function

#Método para categorizar Placas
def categorizarPlacas(categorias, descritores,indicadorDescritorCores):


    listaPlacasCategorizar = []
    listaFileNames = []
    knnTreinoOk = ReconhecedorCaracteres.carregarBaseTreinoKNN(categorias,
                                                               descritores)  # Carrega Treino KNN

    if knnTreinoOk == False:
        return
    # end if

    for filename in glob.glob('imagensCategorizar/*.*'):


        arquivoName1 = filename.split('\\')
        arquivoName2 = arquivoName1[1].split('.')

        #Ler Arquivo
        imgNaoCategorizada = cv2.imread(filename)

        if imgNaoCategorizada is None:
            print("\n Leitura da Imagem Não Ok \n")
            continue
        # end if

        listaPlacas = ReconhecedorPlacas.detectPlatesInScene(imgNaoCategorizada)           # reconhece placas

        listaPlacas, listaDescritores,listaDescritoresOrb = ReconhecedorCaracteres.detectarCharsPlacas(listaPlacas,indicadorDescritorCores)  # reconhece caracteres na placa

        if len(listaPlacas) == 0:
            print("\nLista de Placas Vazia\n")
            continue
        else:

            listaPlacas.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

            licPlate = listaPlacas[0]

            #cv2.imshow("imgPlate", licPlate.imgPlate) #Mostrar Placa
            #cv2.imshow("imgThresh", licPlate.imgThresh) # Mostrar Placa Thresh

            if len(licPlate.strChars) == 0:
                print ("\n Placas sem Caracteres\n" )
                continue
            # end if
            print("\nPlaca Não Categorizada: "+ arquivoName2[0])
            print ("\nCaracteres Encontrados na Placa = " + licPlate.strChars + "\n" )      # Caracteres Encontrados na Placa
            print ("__________________________________________________________________")

            listaPlacasCategorizar.append(licPlate.strChars) #salva placas
            listaFileNames.append(arquivoName2[0])
            # end if else

            #cv2.waitKey(0)
    return listaPlacasCategorizar, listaFileNames
#endfunction

if __name__ == "__main__":
    main()


















