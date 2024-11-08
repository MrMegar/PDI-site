import matplotlib
matplotlib.use('Agg')  # Usar um backend que não precisa de Tcl/Tk
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Parâmetros da imagem
SIDE = 256
PERIODOS = 4  # Períodos da senóide
AMPLITUDE = 127

# Criação da imagem em branco
image = np.zeros((SIDE, SIDE), dtype=np.float32)

# Geração da senóide
for j in range(SIDE):
    image[:, j] = AMPLITUDE * np.sin(2 * np.pi * PERIODOS * j / SIDE) + (AMPLITUDE + 1)

# Salvar a imagem no formato YML
cv2.FileStorage("senoide-256.yml", cv2.FILE_STORAGE_WRITE).write("mat", image)
# Salvar a imagem em formato PNG
cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
image = image.astype(np.uint8)
cv2.imwrite("senoide-256.png", image)

# Ler a imagem do arquivo YML
fs = cv2.FileStorage("senoide-256.yml", cv2.FILE_STORAGE_READ)
image_yml = fs.getNode("mat").mat()
fs.release()

# Normalizar a imagem lida do YML
cv2.normalize(image_yml, image_yml, 0, 255, cv2.NORM_MINMAX)
image_yml = image_yml.astype(np.uint8)

# Comparar uma linha da imagem PNG e da imagem YML
linha = 128  # Linha a ser comparada
linha_png = image[linha, :]
linha_yml = image_yml[linha, :]

# Calcular a diferença entre as duas linhas
diferenca = linha_png.astype(np.int16) - linha_yml.astype(np.int16)

# Plotar a diferença
plt.plot(diferenca, color='red')
plt.title("Diferença entre as linhas da imagem PNG e YML")
plt.xlabel("Colunas")
plt.ylabel("Diferença de Intensidade")
plt.grid()
plt.savefig("diferenca_plot.png")  # Salva o gráfico como um arquivo
plt.close()  # Fecha a figura

# Exibir a imagem gerada
cv2.imshow("Imagem Senóide", image_yml)
cv2.waitKey(0)
cv2.destroyAllWindows()
