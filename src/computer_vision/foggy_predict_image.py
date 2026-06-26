import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# SCRIPT_DIR = Path(__file__).resolve().parent
# if str(SCRIPT_DIR) not in sys.path:
#     sys.path.append(str(SCRIPT_DIR))

from utils.transformations import fog, magnitude_of_gradient, minimax_normalization

DEPTH_ANYTHING_DIR = ROOT_DIR / "Depth-Anything-V2"
if str(DEPTH_ANYTHING_DIR) not in sys.path:
    sys.path.append(str(DEPTH_ANYTHING_DIR))

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

#image_path = "/home/lucas/Documents/computer_vision/Depth-Anything-V2/assets/examples/demo11.jpg"
image_path = "/home/lucas/Documents/computer_vision/videos/original_image_plat.png"
encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available else 'cpu')

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
model = model.to(device).eval()

cmap = matplotlib.colormaps.get_cmap('Spectral')


raw_img = cv2.imread(image_path)
depth = model.infer_image(raw_img) # HxW depth map in meters in cuda

plt.imshow(depth, cmap=cmap)    
plt.show()

depth_u8 = minimax_normalization(depth)

print("Shape: ", depth_u8.shape)
print("Max: ", depth_u8.max(), "\nMin:", depth_u8.min())

cv2.imshow("Depth Map", depth_u8)
cv2.waitKey(0)
cv2.destroyWindow("Depth Map")

name = image_path.split('/')[-1]
cv2.imwrite(f"/home/lucas/Documents/computer_vision/videos/depth/{name}", depth_u8)

# Contrast
magnitude = magnitude_of_gradient(depth_u8)
magnitude_normalized = minimax_normalization(magnitude)

cv2.imshow("Magnitude", cv2.resize(magnitude_normalized, (1280,720)))
cv2.waitKey(0)
cv2.destroyAllWindows()

minimum_distance = 10

foggy_image = fog(image=raw_img, depth_map=depth, minimum_distance=minimum_distance)
foggy_image_clipped = np.clip(foggy_image,0,255).astype(np.uint8)

cv2.imshow("Foggy Image", cv2.resize(foggy_image_clipped, (1280, 720)))
cv2.waitKey(0)
cv2.destroyAllWindows()