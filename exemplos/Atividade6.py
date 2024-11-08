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
    
    # Criar uma cópia da imagem para rotulação
    labeled_image = np.zeros((height, width), dtype=np.int32)
    
    # Identificar e ignorar objetos conectados à borda
    border_mask = image.copy()
    cv2.floodFill(border_mask, None, (0, 0), 0)
    cv2.floodFill(border_mask, None, (width-1, 0), 0)
    cv2.floodFill(border_mask, None, (0, height-1), 0)
    cv2.floodFill(border_mask, None, (width-1, height-1), 0)
    
    # Contadores
    nobjects = 0
    objects_with_holes = 0
    objects_without_holes = 0

    # Rotular objetos internos (não conectados à borda)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if border_mask[i, j] == 255 and labeled_image[i, j] == 0:  # Encontrou um novo objeto
                nobjects += 1
                cv2.floodFill(labeled_image, None, (j, i), nobjects)
                
                # Detectar buracos dentro do objeto
                object_mask = np.zeros_like(image)
                cv2.floodFill(object_mask, None, (j, i), 255)
                
                # Inverter para detectar buracos
                inverted_object = cv2.bitwise_not(object_mask)
                hole_count = 0
                
                for x in range(1, height - 1):
                    for y in range(1, width - 1):
                        if inverted_object[x, y] == 255:
                            hole_count += 1
                            cv2.floodFill(inverted_object, None, (y, x), 0)
                
                # Classificar o objeto com ou sem buracos
                if hole_count > 0:
                    objects_with_holes += 1
                else:
                    objects_without_holes += 1

    print(f"Total de objetos internos: {nobjects}")
    print(f"Objetos com buracos: {objects_with_holes}")
    print(f"Objetos sem buracos: {objects_without_holes}")

    # Exibir e salvar imagem rotulada
    cv2.imshow("Imagem Rotulada", labeled_image.astype(np.uint8) * (255 // nobjects))
    cv2.imwrite("labeling_with_holes.png", labeled_image.astype(np.uint8) * (255 // nobjects))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Receber o caminho da imagem como argumento de linha de comando
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python labeling_with_holes.py <caminho_da_imagem>")
    else:
        main(sys.argv[1])
