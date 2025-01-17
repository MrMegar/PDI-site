import sys
import cv2
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Uso: python kmeans_random.py entrada.jpg saida_prefixo")
        sys.exit(1)

    # Lê a imagem de entrada
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao carregar imagem")
        sys.exit(1)

    # Define número de clusters e número de rodadas (nRodadas=1)
    nClusters = 8
    nRodadas = 1
    saida_prefixo = sys.argv[2]

    # Prepara a matriz de amostras para o K-Means
    # (cada pixel é um ponto no espaço de cor [B,G,R])
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # Executa 10 rodadas "externas"
    # Em cada rodada, rodamos o K-Means (com nRodadas=1),
    # mas os centros iniciais são definidos aleatoriamente,
    # podendo levar a resultados diferentes
    for i in range(10):
        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
        compactness, rotulos, centros = cv2.kmeans(
            data,
            nClusters,
            None,            # Nenhum rótulo inicial
            criterios,
            nRodadas,        # nRodadas=1
            cv2.KMEANS_RANDOM_CENTERS
        )

        # Converte centros para inteiro (0-255)
        centros = np.uint8(centros)

        # Reconstrói a imagem com os centros (clusters)
        rotulada = centros[rotulos.flatten()]
        rotulada = rotulada.reshape((img.shape))

        nome_arquivo = f"{saida_prefixo}_{i}.jpg"
        cv2.imwrite(nome_arquivo, rotulada)
        print(f"Imagem salva: {nome_arquivo}")

if __name__ == '__main__':
    main()
