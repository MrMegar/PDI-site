import sys
import math
import cv2
import numpy as np

def hu_moments_log(img_bin):
    """Calcula os Momentos de Hu (transformados em escala log) de uma máscara binária."""
    m = cv2.moments(img_bin, binaryImage=True)
    hu = cv2.HuMoments(m)
    # Transforma em escala log
    for i in range(7):
        if hu[i][0] != 0:
            hu[i][0] = -1 * math.copysign(1.0, hu[i][0]) * math.log10(abs(hu[i][0]))
    return hu

def main():
    if len(sys.argv) < 3:
        print("Uso: python momentos_regioes.py pessoa.jpg multidao.jpg [opcional: scale=0.5]")
        sys.exit(1)

    # Carrega a imagem 'pessoa'
    pessoa_path = sys.argv[1]
    multidao_path = sys.argv[2]
    
    scale_factor = 1.0
    if len(sys.argv) == 4 and "scale=" in sys.argv[3]:
        try:
            scale_factor = float(sys.argv[3].split("=")[1])
        except:
            scale_factor = 1.0

    pessoa_gray = cv2.imread(pessoa_path, cv2.IMREAD_GRAYSCALE)
    multidao_gray = cv2.imread(multidao_path, cv2.IMREAD_GRAYSCALE)
    if pessoa_gray is None or multidao_gray is None:
        print("Erro ao carregar as imagens.")
        sys.exit(1)

    # Redimensionar se necessário para acelerar
    if scale_factor != 1.0:
        pessoa_gray = cv2.resize(pessoa_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        multidao_gray = cv2.resize(multidao_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Threshold para binarizar a 'pessoa'
    # (Dependendo da imagem, pode ser necessário ajustar / segmentar melhor)
    _, pessoa_bin = cv2.threshold(pessoa_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calcula Momentos de Hu para a 'pessoa'
    hu_pessoa = hu_moments_log(pessoa_bin)

    # Para encontrar a pessoa na imagem 'multidao', precisamos
    # extrair as componentes conectadas ou contornos.
    # Aqui, por simplicidade, vamos binarizar e encontrar contornos externos.
    _, multidao_bin = cv2.threshold(multidao_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(multidao_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    multidao_color = cv2.cvtColor(multidao_gray, cv2.COLOR_GRAY2BGR)

    melhor_distancia = float('inf')
    melhor_contorno = None

    # Calcula Momentos de Hu para cada contorno em 'multidao'
    for c in contours:
        mask_temp = np.zeros_like(multidao_bin)
        cv2.drawContours(mask_temp, [c], -1, 255, -1)  # preenche contorno
        
        hu_regiao = hu_moments_log(mask_temp)
        
        # Distância Euclidiana simples entre os vetores de Hu (log)
        dist = 0
        for i in range(7):
            dist += (hu_pessoa[i][0] - hu_regiao[i][0])**2
        dist = math.sqrt(dist)

        if dist < melhor_distancia:
            melhor_distancia = dist
            melhor_contorno = c

    # Desenha o melhor contorno encontrado em vermelho
    if melhor_contorno is not None:
        cv2.drawContours(multidao_color, [melhor_contorno], -1, (0,0,255), 2)
        print(f"Melhor distância: {melhor_distancia}")
    else:
        print("Não foi encontrado contorno similar.")

    cv2.imshow("Pessoa (bin)", pessoa_bin)
    cv2.imshow("Multidao (resultado)", multidao_color)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
