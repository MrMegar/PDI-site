import sys
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Uso: python morfologia.py <imagem>")
        sys.exit(1)

    imagem = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    if imagem is None:
        print("Erro ao abrir imagem:", sys.argv[1])
        sys.exit(1)

    estr = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosao = cv2.erode(imagem, estr)
    dilatacao = cv2.dilate(imagem, estr)
    abertura = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, estr)
    fechamento = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, estr)
    abertfecha = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, estr)
    saida = np.hstack([erosao, dilatacao, abertura, fechamento, abertfecha])

    cv2.imshow("morfologia", saida)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()