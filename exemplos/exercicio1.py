import cv2

# Função para criar o negativo de uma região da imagem
def apply_negative(image, p1, p2):
    # Obtém as coordenadas da região
    x1, y1 = p1
    x2, y2 = p2

    # Corrige a ordem das coordenadas, se necessário
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Aplica o negativo na região selecionada
    image[y_min:y_max, x_min:x_max] = 255 - image[y_min:y_max, x_min:x_max]

# Carrega a imagem em escala de cinza
image = cv2.imread("bolhas.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Erro ao abrir a imagem 'bolhas.png'")
    exit()

# Mostra a imagem original
cv2.imshow("Imagem Original", image)
cv2.waitKey(0)

# Solicita ao usuário as coordenadas dos pontos
x1, y1 = map(int, input("Digite as coordenadas de P1 (x1 y1): ").split())
x2, y2 = map(int, input("Digite as coordenadas de P2 (x2 y2): ").split())

# Aplica o negativo na região definida pelos pontos
apply_negative(image, (x1, y1), (x2, y2))

# Exibe a imagem com a região modificada
cv2.imshow("Imagem com Região Negativa", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
