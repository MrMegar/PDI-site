import cv2

# Função para trocar os quadrantes da imagem
def trocar_quadrantes(image):
    # Obtém as dimensões da imagem
    height, width = image.shape[:2]
    
    # Verifica se as dimensões são múltiplas de 2
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("A imagem deve ter dimensões múltiplas de 2.")

    # Define as regiões dos quadrantes
    h_half, w_half = height // 2, width // 2

    # Cria uma nova imagem para armazenar os quadrantes trocados
    result = image.copy()

    # Troca os quadrantes
    result[0:h_half, 0:w_half] = image[h_half:height, w_half:width]  # Quadrante inferior direito
    result[h_half:height, w_half:width] = image[0:h_half, 0:w_half]  # Quadrante superior esquerdo
    result[0:h_half, w_half:width] = image[h_half:height, 0:w_half]  # Quadrante inferior esquerdo
    result[h_half:height, 0:w_half] = image[0:h_half, w_half:width]  # Quadrante superior direito

    return result

# Carrega a imagem
image = cv2.imread("bolhas.png")
if image is None:
    print("Erro ao abrir a imagem 'bolhas.png'")
    exit()

# Mostra a imagem original
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# Troca os quadrantes
image_trocada = trocar_quadrantes(image)

# Exibe a imagem com os quadrantes trocados
cv2.imshow("Imagem com Quadrantes Trocados", image_trocada)
cv2.waitKey(0)
cv2.destroyAllWindows()