import sys
import cv2
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Uso: python contornos.py <entrada> <saida.svg> [approx]")
        sys.exit(1)

    input_image = sys.argv[1]
    output_svg = sys.argv[2]

    if len(sys.argv) == 4:
        approx_mode = sys.argv[3].upper()
    else:
        approx_mode = "NONE"

    image_gray = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        print("Erro ao abrir imagem:", input_image)
        sys.exit(1)

    _, img_bin = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if approx_mode == "NONE":
        chain_approx = cv2.CHAIN_APPROX_NONE
    else:
        chain_approx = cv2.CHAIN_APPROX_SIMPLE

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, chain_approx)
    image_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)

    height, width = image_gray.shape
    with open(output_svg, 'w') as f:
        f.write(f'<svg height="{height}" width="{width}" xmlns="http://www.w3.org/2000/svg">\n')
        for c in contours:
            c = c.reshape(-1, 2)
            x0, y0 = c[0]
            svg_path = f'<path d="M {x0} {y0} '
            for i in range(1, len(c)):
                x, y = c[i]
                svg_path += f'L{x} {y} '
            svg_path += 'Z" fill="#cccccc" stroke="black" stroke-width="1" />\n'
            f.write(svg_path)
            cv2.drawContours(image_bgr, [c], -1, (0, 0, 255), 1)
        f.write('</svg>\n')

    cv2.imshow("Contornos", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
