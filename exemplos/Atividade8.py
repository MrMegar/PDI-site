import cv2
import numpy as np

def calcular_histograma(image):
    # Calcular histograma para a imagem em tons de cinza
    hist = cv2.calcHist([image], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def comparar_histogramas(hist1, hist2):
    # Comparar dois histogramas usando a correlação
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Câmera indisponível")
        return

    # Parâmetros de detecção de movimento
    limiar_diferenca = 0.8  # Limiar de correlação para detectar movimento
    histograma_anterior = None

    while True:
        # Captura o frame
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar a imagem.")
            break

        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcular o histograma atual
        histograma_atual = calcular_histograma(gray)

        # Se o histograma anterior existe, compará-lo com o atual
        if histograma_anterior is not None:
            diferenca = comparar_histogramas(histograma_anterior, histograma_atual)
            print(f"Diferença de Histograma: {diferenca}")

            # Verificar se a diferença entre os histogramas está abaixo do limiar
            if diferenca < limiar_diferenca:
                print("Movimento detectado! Alarme ativado.")
                cv2.putText(frame, "Movimento detectado!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Atualizar o histograma anterior para o próximo loop
        histograma_anterior = histograma_atual.copy()

        # Mostrar a imagem com o possível alerta
        cv2.imshow("Deteccao de Movimento", frame)

        # Pressione 'ESC' para sair
        if cv2.waitKey(30) == 27:
            break

    # Liberar a câmera e fechar janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
