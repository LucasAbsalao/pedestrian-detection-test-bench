import cv2
import numpy as np

'''
cv2.INTER_AREA  |  Shrinking  |  Minimizes distortion while downscaling.

cv2.INTER_LINEAR  |  General resizing  |  Balances speed and quality

cv2.INTER_CUBIC  |  Enlarging  |  Higher quality for upscaling

cv2.INTER_NEAREST  |  Fast resizing  |  Quick but lower quality
'''

img = cv2.imread("/home/lucas/Documents/computer_vision/videos/original_image_plat.png")

img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

img = cv2.resize(img, (100,50))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()