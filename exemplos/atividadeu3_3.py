import sys
import math
import cv2
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Uso: python momentos_contornos.py <imagem>")
        sys.exit(1)

    imagem = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print("Erro ao abrir imagem:", sys.argv[1])
        sys.exit(1)

    arq = open("momentos.txt", "w")

    cv2.threshold(imagem, 1, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV, imagem)
    contornos, hierarquia = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    n = 0

    for i, c in enumerate(contornos):
        if len(c) < 100:
            continue
        n += 1
        M = cv2.moments(c)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        hu = cv2.HuMoments(M)
        for j in range(7):
            if hu[j][0] != 0:
                hu[j][0] = -1 * math.copysign(1.0, hu[j][0]) * math.log10(abs(hu[j][0]))
        cv2.drawContours(imagem, [c], -1, (0, 0, 255) if hu[0][0] > 0 else (0, 255, 0), 2)
        cv2.putText(imagem, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 8)
        cv2.putText(imagem, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        arq.write(f"{i}, ")
        for j in range(7):
            arq.write(f"{hu[j][0]}, ")
        arq.write("\n")

    print("Numero de objetos:", n)
    arq.close()
    cv2.imwrite("contornos-rotulados.png", imagem)
    cv2.imshow("janela", imagem)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
