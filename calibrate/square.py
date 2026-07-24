import cv2
import numpy as np

# Background
def draw_background(width, height): 
    background = np.full((height, width, 3), (0, 250, 255), dtype=np.uint8)
    return background

def draw_blue_white_rectangles(image, white_size, blue_size):
    x1_square = (image.shape[1] - white_size) // 2
    y1_square = (image.shape[0] - white_size) // 2

    square_start_point = (x1_square, y1_square)
    square_end_point = (x1_square + white_size, y1_square + white_size)

    x1_blue = (image.shape[1] - blue_size) // 2
    x2_blue = x1_blue + blue_size

    blue_start_point = (x1_blue, y1_square)
    blue_end_point = (x2_blue, y1_square + white_size)

    blue_color = (255, 0, 0)
    cv2.rectangle(image, blue_start_point, blue_end_point, blue_color, -1)

    white_color = (255, 255, 255)
    cv2.rectangle(image, square_start_point, square_end_point, white_color, -1)

def draw_horizontal_vertical_lines(image):
    red_color = (0, 0, 255)
    line_thickness = 5

    mid_x = image.shape[1] // 2
    vertical_start_point = (mid_x, 0)
    vertical_end_point = (mid_x, image.shape[0])

    cv2.line(image, vertical_start_point, vertical_end_point, red_color, line_thickness)

    mid_y = image.shape[0] // 2
    horizontal_start_point = (0, mid_y)
    horizontal_end_point = (image.shape[1], mid_y)

    cv2.line(image, horizontal_start_point, horizontal_end_point, red_color, line_thickness)

width = 5120
height = 2160

background = draw_background(width, height)

white_size = 2080
blue_width = 5040

draw_blue_white_rectangles(background, white_size, blue_width)

draw_horizontal_vertical_lines(background)

# Save Image
filename = f"white_square_with_blue_background_{width}_{height}.png"
success = cv2.imwrite(filename, background)

if success:
    print(f"Image '{filename}' saved successfully in current directory!")
else:
    print("Error saving image.")