import cv2
import numpy as np

# Background
def draw_background(width, height): 

    tela = np.full((height, width, 3), (0, 250, 255), dtype=np.uint8)

    return tela

def draw_blue_white_rectangles(image, white_size, blue_size):

    x1_quadrado = (image.shape[1] - white_size) // 2
    y1_quadrado = (image.shape[0] - white_size) // 2

    ponto_inicial_quadrado = (x1_quadrado, y1_quadrado)
    ponto_final_quadrado = (x1_quadrado + white_size, y1_quadrado + white_size)


    x1_azul = (image.shape[1] - blue_size) // 2
    x2_azul = x1_azul + blue_size


    ponto_inicial_azul = (x1_azul, y1_quadrado)
    ponto_final_azul = (x2_azul, y1_quadrado + white_size)

    cor_azul = (255, 0, 0)

    cv2.rectangle(image, ponto_inicial_azul, ponto_final_azul, cor_azul, -1)

    cor_branca = (255, 255, 255)
    cv2.rectangle(image, ponto_inicial_quadrado, ponto_final_quadrado, cor_branca, -1)

def draw_horizontal_vertical_lines(image):
    cor_vermelha = (0, 0, 255)
    espessura_linha = 5

    meio_x = image.shape[1] // 2
    ponto_inicial_vertical = (meio_x, 0)       # Ponto no topo da tela
    ponto_final_vertical = (meio_x, image.shape[0])    # Ponto na base da tela

    cv2.line(tela, ponto_inicial_vertical, ponto_final_vertical, cor_vermelha, espessura_linha)


    meio_y = image.shape[0] // 2
    ponto_inicial_horizontal = (0, meio_y)       # Ponto no canto esquerdo
    ponto_final_horizontal = (image.shape[1], meio_y)   # Ponto no canto direito

    cv2.line(tela, ponto_inicial_horizontal, ponto_final_horizontal, cor_vermelha, espessura_linha)

width = 5120
height = 2160

tela = draw_background(width, height)

white_size = 2080
blue_width = 5040

draw_blue_white_rectangles(tela, white_size, blue_width)

draw_horizontal_vertical_lines(tela)


# Save Image
nome_do_arquivo = f"quadrado_branco_com_fundo_azul_{width}_{height}.png" 
sucesso = cv2.imwrite(nome_do_arquivo, tela)


if sucesso:
    print(f"Imagem '{nome_do_arquivo}' salva com sucesso no diretório atual!")
else:
    print("Erro ao tentar salvar a imagem.")