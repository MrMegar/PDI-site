import cv2
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Uso: python Ex4.py <imagem_esteganografada>")
        return

    imagem_esteganografada_path = sys.argv[1]
    
    # Carregar a imagem esteganografada
    imagem_esteganografada = cv2.imread(imagem_esteganografada_path, cv2.IMREAD_COLOR)
    
    if imagem_esteganografada is None:
        print("A imagem não pôde ser carregada.")
        return

    # Obter as dimensões da imagem
    rows, cols, _ = imagem_esteganografada.shape

    # Criar uma imagem vazia para a imagem recuperada
    imagem_recuperada = np.zeros((rows, cols, 3), dtype=np.uint8)

    nbits = 3  # Número de bits a serem considerados

    # Recuperar a imagem escondida
    for i in range(rows):
        for j in range(cols):
            # Obter o valor do pixel da imagem esteganografada
            val_esteganografada = imagem_esteganografada[i, j]

            # Limpar os bits mais significativos e mover os bits menos significativos para os mais significativos
            val_recuperada = [
                (val_esteganografada[0] & ((1 << (8 - nbits)) - 1)),  # R
                (val_esteganografada[1] & ((1 << (8 - nbits)) - 1)),  # G
                (val_esteganografada[2] & ((1 << (8 - nbits)) - 1))   # B
            ]

            # Ajustar os bits recuperados
            imagem_recuperada[i, j] = [val_recuperada[0] << nbits, val_recuperada[1] << nbits, val_recuperada[2] << nbits]

    # Salvar a imagem recuperada
    cv2.imwrite("imagem_recuperada.png", imagem_recuperada)

    print("Imagem recuperada salva como 'imagem_recuperada.png'.")

if __name__ == "__main__":
    main()



#ReadMe

#Código para o programa funcionar bem:
#python Ex4.py (NomeDoArquivo.png)