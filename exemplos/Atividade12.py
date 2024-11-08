import cv2
import numpy as np

# Parâmetros do efeito tilt-shift
focus_height = 30  # Altura da região de foco em %
blur_strength = 15  # Intensidade do desfoque
focus_center = 50  # Posição vertical do centro do foco em %

# Configuração para descartar quadros (ajuste para criar efeito de stop motion)
frame_skip = 3  # Pular 3 quadros após processar um (ajuste conforme necessário)

def apply_tiltshift(frame):
    """
    Aplica o efeito tilt-shift em um único quadro.
    """
    rows, cols = frame.shape[:2]
    
    # Define a região de foco com base na altura e posição central
    focus_y_center = int(focus_center / 100 * rows)
    focus_half_height = int(focus_height / 100 * rows / 2)
    top_focus = max(focus_y_center - focus_half_height, 0)
    bottom_focus = min(focus_y_center + focus_half_height, rows)

    # Aplica o efeito de desfoque nas regiões fora da área de foco
    blurred_frame = cv2.GaussianBlur(frame, (0, 0), blur_strength)

    # Combina as regiões focadas e desfocadas
    output = blurred_frame.copy()
    output[top_focus:bottom_focus, :] = frame[top_focus:bottom_focus, :]

    return output

def process_video(input_path, output_path):
    # Carrega o vídeo de entrada
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print("Erro ao abrir o vídeo de entrada.")
        return

    # Obtém informações sobre o vídeo
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo (pode variar)

    # Cria o vídeo de saída
    out = cv2.VideoWriter(output_path, codec, fps / (frame_skip + 1), (width, height))

    frame_count = 0
    while True:
        # Lê o próximo quadro
        ret, frame = video.read()
        if not ret:
            break

        # Aplica o efeito tilt-shift em um quadro específico
        if frame_count % (frame_skip + 1) == 0:
            output_frame = apply_tiltshift(frame)
            out.write(output_frame)  # Grava o quadro processado no vídeo de saída

        frame_count += 1

    # Libera os objetos de vídeo
    video.release()
    out.release()
    print(f"Processamento concluído. Vídeo salvo em {output_path}")

# Caminho dos arquivos de entrada e saída
input_video_path = "video_input.mp4"  # Substitua pelo seu arquivo de vídeo
output_video_path = "tiltshift_output.mp4"

# Processa o vídeo com o efeito tilt-shift
process_video(input_video_path, output_video_path)
