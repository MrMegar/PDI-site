import cv2
import numpy as np

def apply_filter(frame, kernel_size):
    # Cria uma máscara de média com o tamanho especificado
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # Aplica a convolução usando o filtro de média
    filtered_frame = cv2.filter2D(frame, -1, kernel)
    return filtered_frame

def main():
    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Câmera indisponível")
        return

    # Configura a resolução da captura
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Largura =", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Altura =", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        # Captura o frame
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem")
            break

        # Converte para tons de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplica filtros de média com diferentes tamanhos de máscara
        filtered_3x3 = apply_filter(gray_frame, 3)
        filtered_11x11 = apply_filter(gray_frame, 11)
        filtered_21x21 = apply_filter(gray_frame, 21)

        # Exibe os resultados
        cv2.imshow("Original", gray_frame)
        cv2.imshow("Filtro 3x3", filtered_3x3)
        cv2.imshow("Filtro 11x11", filtered_11x11)
        cv2.imshow("Filtro 21x21", filtered_21x21)

        # Pressiona 'ESC' para sair
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Libera a captura e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
