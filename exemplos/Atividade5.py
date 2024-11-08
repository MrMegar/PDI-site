import cv2
import numpy as np
import sys

def main(image_path):
    # Carregar imagem em escala de cinza e verificar
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("A imagem não foi carregada corretamente.")
        return
    
    # Dimensões da imagem
    height, width = image.shape
    print(f"{width}x{height}")
    
    # Usar uma imagem de 16 bits para rotulação
    labeled_image = np.zeros((height, width), dtype=np.uint16)
    nobjects = 0  # Contador de objetos

    # Rotulação dos objetos
    for i in range(height):
        for j in range(width):
            if image[i, j] == 255 and labeled_image[i, j] == 0:  # Encontrou um objeto
                nobjects += 1
                # Rótulo para o objeto atual
                mask = np.zeros((height + 2, width + 2), dtype=np.uint8)  # Máscara para floodFill
                cv2.floodFill(image, mask, (j, i), nobjects, flags=8 | (255 << 8))
                labeled_image[image == nobjects] = nobjects  # Atribuir o rótulo ao objeto
    
    print(f"A figura tem {nobjects} objetos.")
    
    # Exibir e salvar imagem rotulada
    cv2.imshow("Imagem Rotulada", labeled_image)
    cv2.imwrite("labeling.png", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Receber o caminho da imagem como argumento de linha de comando
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python labeling.py <caminho_da_imagem>")
    else:
        main(sys.argv[1])
