import cv2
import numpy as np

def main():
    # Abre o vídeo
    video = cv2.VideoCapture("vasos720p.mp4")
    if not video.isOpened():
        print("Erro ao abrir o vídeo")
        return

    # Configura o vídeo para começar a partir do segundo 1 (aproximadamente 30 frames)
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps)  # Vai para o segundo 1
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Captura um único frame do vídeo para análise
    ret, frame = video.read()
    if not ret:
        print("Erro ao capturar o frame do vídeo")
        return
    
    # Converte o frame para tons de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica o filtro laplaciano de tamanho 3x3
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_32F, ksize=3)
    abs_laplacian = np.abs(laplacian)  # Valor absoluto do Laplaciano

    # Exibe a imagem Laplaciana para verificar o resultado da borda
    cv2.imshow("Laplaciano Absoluto", abs_laplacian / abs_laplacian.max())  # Normaliza para visualização

    # Define um limiar mais baixo para capturar mais áreas
    threshold = np.mean(abs_laplacian) * 0.5  # Usa metade da média como threshold
    print(f"Threshold adaptativo ajustado: {threshold}")

    # Cria uma máscara para as áreas mais nítidas
    sharp_mask = (abs_laplacian > threshold).astype(np.uint8) * 255

    # Exibe a máscara para verificar as áreas detectadas como nítidas
    cv2.imshow("Máscara de Nitidez", sharp_mask)

    # Cria uma imagem de saída para armazenar os pixels mais nítidos
    output_image = np.zeros_like(frame)

    # Copia os pixels coloridos das áreas mais nítidas para a imagem de saída
    output_image[sharp_mask == 255] = frame[sharp_mask == 255]

    # Exibe a imagem de saída
    cv2.imshow("Profundidade de Campo Corregida", output_image)
    cv2.waitKey(0)  # Pressione qualquer tecla para fechar
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
