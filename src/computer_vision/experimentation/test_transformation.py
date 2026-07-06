import cv2
from pathlib import Path
import sys
import numpy as np

COMPUTER_VISION_DIR = Path(__file__).resolve().parents[1]
print(COMPUTER_VISION_DIR)
if str(COMPUTER_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(COMPUTER_VISION_DIR))

from utils.transformations import salt_and_pepper

image = cv2.imread("/home/lucas/Documents/computer_vision/videos/original_image_plat.png")
if image is None:
    raise FileNotFoundError("Image not found")

image = cv2.resize(image, (1366, 768), interpolation=cv2.INTER_NEAREST)

image = salt_and_pepper(image, 0.03, 0.03)


cv2.imshow("Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()