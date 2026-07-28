import cv2
from pathlib import Path
import sys
import numpy as np

COMPUTER_VISION_DIR = Path(__file__).resolve().parents[1]
print(COMPUTER_VISION_DIR)
if str(COMPUTER_VISION_DIR) not in sys.path:
    sys.path.insert(0, str(COMPUTER_VISION_DIR))

from utils.transformations import salt_and_pepper_conv, gaussian_noise_conv, time_decaying_artifacts, crop, resize

image = cv2.imread("/home/lucas/Documents/computer_vision/videos/original_image_plat.png")
if image is None:
    raise FileNotFoundError("Image not found")

#image = crop(image, (5120,3840))
image = resize(image=image, width=5120, height=2160)

print(image.shape)

last_image, noise = time_decaying_artifacts(image=image,
                                    old_noise=None,
                                    event_probability=0.00001,
                                    decay_factor=0.9,
                                    kernel_size=11,
                                    kernel_factor=20,
                                    sigma=10
                                    )

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", last_image)
cv2.waitKey(100)

cv2.setWindowProperty("Test",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#image = salt_and_pepper_conv(image, 0.001, 0.001, 61)
#image = gaussian_noise_conv(image, 0, 60, 9)

while True:
    
    key = 0xFF & cv2.waitKey(0)

    if key == ord('d'):
        last_image, noise = time_decaying_artifacts(image=image,
                                                    old_noise=noise,
                                                    event_probability=0.1,
                                                    decay_factor=1.0,
                                                    kernel_size=121,
                                                    kernel_factor=100,
                                                    sigma=30)
        print("passou")
        pass
    elif key == ord('q'):
        break
    

cv2.destroyAllWindows()