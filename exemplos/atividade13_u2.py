import sys
import cv2
import numpy as np

def homomorphic_filter(
    image: np.ndarray, 
    d0: float = 30.0,  # Raio de corte (cutoff)
    rh: float = 2.0,   # Ganho para altas frequências (realçar detalhes)
    rl: float = 0.5,   # Ganho para baixas frequências (suprimir iluminação)
    c: float = 1.0     # Constante usada na forma do filtro
) -> np.ndarray:
    """
    Aplica filtro homomórfico para corrigir iluminação em uma imagem em tons de cinza.

    Parâmetros:
    -----------
    image : np.ndarray
        Imagem de entrada em tons de cinza (0-255).
    d0    : float
        Frequência de corte (raio) do filtro.
    rh    : float
        Ganho para altas frequências.
    rl    : float
        Ganho para baixas frequências.
    c     : float
        Parâmetro que controla a transição do filtro.

    Retorna:
    --------
    np.ndarray
        Imagem resultante (corrigida) em tons de cinza, normalizada (0-255).
    """

    # 1) Transformação logarítmica para passar de multiplicativo a aditivo
    #    Evita problemas de log(0) adicionando +1
    img_log = np.log1p(np.array(image, dtype="float"))

    # 2) Preparação para DFT (tamanho ótimo)
    rows, cols = img_log.shape
    dft_M = cv2.getOptimalDFTSize(rows)
    dft_N = cv2.getOptimalDFTSize(cols)

    # Cria uma nova imagem com padding
    padded = np.zeros((dft_M, dft_N), dtype=np.float32)
    padded[:rows, :cols] = img_log

    # 3) Calcula DFT: precisamos da parte real e imaginária
    complex_image = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)

    # 4) Centraliza o espectro: fftshift
    #    Em OpenCV, podemos separar os planos e usar np.fft.fftshift manualmente
    complex_image_shifted = np.fft.fftshift(complex_image, axes=[0, 1])

    # 5) Construção do filtro homomórfico
    #    Fórmula genérica: H(u,v) = (rh - rl) [1 - e^{-c*(D(u,v)^2 / d0^2)}] + rl
    #    onde D(u,v) é a distância ao centro da imagem no espaço de frequências
    crow, ccol = dft_M // 2, dft_N // 2
    # Cria duas matrizes para a parte real e imaginária do filtro
    H = np.zeros((dft_M, dft_N), dtype=np.float32)

    for i in range(dft_M):
        for j in range(dft_N):
            # Distância do ponto (i,j) ao centro (crow, ccol)
            d2 = (i - crow)**2 + (j - ccol)**2
            # Fórmula do homomórfico
            H[i, j] = (rh - rl) * (1 - np.exp(-c * (d2 / (d0**2)))) + rl

    # 6) Aplica o filtro à DFT
    #    complex_image_shifted tem 2 canais (Re, Im). Precisamos multiplicar H em cada canal
    planes = cv2.split(complex_image_shifted)
    # Multiplicamos cada canal do espectro pela matriz H
    planes[0] = planes[0] * H
    planes[1] = planes[1] * H
    # Junta de novo
    complex_image_filtered = cv2.merge(planes)

    # 7) Desfaz o shift: ifftshift
    complex_image_filtered = np.fft.ifftshift(complex_image_filtered, axes=[0, 1])

    # 8) IDFT para voltar ao domínio espacial
    inv_dft = cv2.idft(complex_image_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    # Recorta à região original
    inv_dft = inv_dft[:rows, :cols]

    # 9) Aplica a exponencial inversa (transformada contrária ao log)
    img_exp = np.expm1(inv_dft)  # exp(x) - 1

    # Ajusta valores para [0, 255]
    img_exp = np.clip(img_exp, 0, 255)
    img_exp = np.uint8(img_exp)

    return img_exp

def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} imagem_cinza.jpg [saida_opcional.jpg]")
        sys.exit(1)

    entrada = sys.argv[1]
    saida = sys.argv[2] if len(sys.argv) > 2 else "homomorphic_result.jpg"

    # Carrega a imagem em tons de cinza
    image = cv2.imread(entrada, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro ao ler a imagem:", entrada)
        sys.exit(1)

    # Ajuste os parâmetros do filtro conforme necessário
    # d0  -> Raio de corte
    # rh  -> Aumenta ganho de altas frequências (detalhes)
    # rl  -> Aumenta/Reduz ganho de baixas frequências (iluminação)
    # c   -> Ajuste fino da transição
    d0 = 30.0
    rh = 2.5
    rl = 0.5
    c = 1.0

    # Aplica o filtro homomórfico
    filtered = homomorphic_filter(image, d0=d0, rh=rh, rl=rl, c=c)

    # Mostra resultados
    cv2.imshow("Original", image)
    cv2.imshow("Filtrada (Homomorphic)", filtered)
    cv2.imwrite(saida, filtered)
    print(f"Imagem resultante salva em: {saida}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
