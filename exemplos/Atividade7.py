import cv2
import numpy as np

def main():
    # Abrir a câmera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Câmera indisponível")
        return
    
    # Configurar a resolução da câmera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Largura = {width}")
    print(f"Altura = {height}")

    while True:
        # Capturar o frame
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar a imagem.")
            break

        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalizar o histograma da imagem em tons de cinza
        equalized = cv2.equalize
