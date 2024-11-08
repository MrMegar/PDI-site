import cv2
import numpy as np

# Variáveis globais para os valores dos sliders
focus_height = 50
blur_strength = 15
focus_center = 50

# Carrega a imagem
image = cv2.imread("imagem.jpg")  # Substitua "imagem.jpg" pelo caminho correto da sua imagem

# Verifica se a imagem foi carregada com sucesso
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho do arquivo.")
else:
    # Função chamada ao mover os trackbars
    def apply_tiltshift(*args):
        global focus_height, blur_strength, focus_center

        # Copia a imagem original
        output = image.copy()
        rows, cols = output.shape[:2]

        # Define a região de foco com base na altura e posição central
        focus_y_center = int(focus_center / 100 * rows)
        focus_half_height = int(focus_height / 100 * rows / 2)
        top_focus = max(focus_y_center - focus_half_height, 0)
        bottom_focus = min(focus_y_center + focus_half_height, rows)

        # Aplica o efeito de desfoque nas regiões fora da área de foco
        blurred_image = cv2.GaussianBlur(image, (0, 0), blur_strength)

        # Combina as regiões focadas e desfocadas
        output[:top_focus, :] = blurred_image[:top_focus, :]
        output[bottom_focus:, :] = blurred_image[bottom_focus:, :]

        # Exibe a imagem final com tilt-shift
        cv2.imshow("Tilt-Shift Effect", output)

    # Cria a janela e os trackbars
    cv2.namedWindow("Tilt-Shift Effect")

    cv2.createTrackbar("Altura do Foco (%)", "Tilt-Shift Effect", focus_height, 100, lambda x: update_focus_height(x))
    cv2.createTrackbar("Força do Desfoque", "Tilt-Shift Effect", blur_strength, 50, lambda x: update_blur_strength(x))
    cv2.createTrackbar("Posição do Centro (%)", "Tilt-Shift Effect", focus_center, 100, lambda x: update_focus_center(x))

    def update_focus_height(val):
        global focus_height
        focus_height = val
        apply_tiltshift()

    def update_blur_strength(val):
        global blur_strength
        blur_strength = val
        apply_tiltshift()

    def update_focus_center(val):
        global focus_center
        focus_center = val
        apply_tiltshift()

    # Mostra a imagem inicial com os valores iniciais de tilt-shift
    apply_tiltshift()

    # Espera pela tecla 's' para salvar a imagem ou 'ESC' para sair
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Tecla ESC
            break
        elif key == ord('s'):  # Tecla 's' para salvar
            cv2.imwrite("tiltshift_output.jpg", output)
            print("Imagem salva como tiltshift_output.jpg")

    cv2.destroyAllWindows()
