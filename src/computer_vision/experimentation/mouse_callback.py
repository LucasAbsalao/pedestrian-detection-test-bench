import cv2
import sys
import numpy as np
from pathlib import Path
from enum import Enum

COMP_VIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(COMP_VIS_DIR))

from utils.draw import write_lines

refPt = np.zeros((2,2), dtype=int)

class MarkState(Enum):
	NOT_MARKED = 0
	MARKING = 1
	MARKED = 2

mark = MarkState.NOT_MARKED

def mark_trapezes(event, x, y, flags, param):

	global refPt, mark, image

	if event == cv2.EVENT_LBUTTONDOWN and mark != MarkState.MARKING:
		refPt[0] = [x, y]
		refPt[1] = [-1,-1]
		if mark == MarkState.MARKED:
			print("resetting points")
			image = clone.copy()
		mark = MarkState.MARKING

	elif event == cv2.EVENT_LBUTTONDOWN and mark == MarkState.MARKING:

		refPt[1] = [x,y]
		print(f"trapeze created in points {refPt[0]} and {refPt[1]}")

		write_lines(image, refPt[0], refPt[1], width)
		cv2.imshow("image", image)

		mark = MarkState.MARKED


image = cv2.imread("/home/lucas/Documents/computer_vision/videos/original_image_plat.png")
image = cv2.resize(image, (1366, 768), cv2.INTER_LINEAR)
width = image.shape[1]
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", mark_trapezes)
# keep looping until the 'q' key is pressed
while True:

	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("r"):
		image = clone.copy()

	elif key == ord("c"):
		break

if mark == MarkState.MARKED:
	print("Saved points correctly")
	print(refPt)
else:
	print("Wasn't able to get both points")

cv2.destroyAllWindows()
