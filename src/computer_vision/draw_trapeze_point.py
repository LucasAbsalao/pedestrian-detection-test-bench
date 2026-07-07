import cv2
from pathlib import Path
import yaml
import sys
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from enum import Enum
import argparse

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from utils.draw import write_lines

DEFAULT_DATA_DIR = ROOT_DIR / 'data'

class MarkState(Enum):
	NOT_MARKED = 0
	MARKING = 1
	MARKED = 2

class TrapezoidMarker:

	def __init__(self, video_path : Path):
		self.cap = cv2.VideoCapture(str(video_path))
		if not self.cap.isOpened():
			raise FileNotFoundError("Could not open video file")
		
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		self.original_image = None
		self.display_image = None
		self.next_frame = True

		# State Variables
		self.state = MarkState.NOT_MARKED
		self.points = np.zeros((2,2), dtype=int)

	def mark_callback(self, event, x, y, flags, param): 

		if event == cv2.EVENT_LBUTTONDOWN and self.state != MarkState.MARKING:
			self.points[0] = [x, y]
			self.points[1] = [-1,-1]

			if self.state == MarkState.MARKED:
				print("resetting points...")
				self.display_image = self.original_image.copy()

			self.state = MarkState.MARKING

		elif event == cv2.EVENT_LBUTTONDOWN and self.state == MarkState.MARKING:
			# New point
			self.points[1] = [x,y]
			print(f"trapeze created in points {self.points[0]} and {self.points[1]}")

			write_lines(self.display_image, self.points[0], self.points[1], self.width)

			self.state = MarkState.MARKED

	def get_points(self) -> NDArray | None:
		if self.state == MarkState.MARKED:
			print("Saved points correctly:")
			print(self.points[0], " and ", self.points[1])
			return self.points
		else:
			print("Process aborted without capturing both points.")
			return None

	def run(self) -> NDArray | None:
		cv2.namedWindow("Video Frame")
		cv2.setMouseCallback("Video Frame", self.mark_callback)

		while True:
			if self.next_frame:
				succes, self.original_image = self.cap.read()
				if not succes:
					print("Video ended")
					break

				self.display_image = self.original_image.copy()
				self.next_frame = False

			cv2.imshow("Video Frame", self.display_image)
			key = cv2.waitKey(1) & 0xFF

			if key == ord('r'):
				self.display_image = self.original_image.copy()
				self.state = MarkState.NOT_MARKED
			if key == ord('p'):
				self.next_frame = True
			if key == ord('q'):
				print("Quitting run loop...")
				break

		cv2.destroyAllWindows()

		return self.get_points()

def parse_args(arg_list = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description  = "Script for establishing points to the ground truth."
	)
	parser.add_argument(
		"--name",
		type=str,
		default="points",
		help="Yaml file's name"
	)
	parser.add_argument(
		"--config",
		type=Path,
		default = DEFAULT_DATA_DIR,
		help="Path where yaml configuration file will be saved"
	)
	parser.add_argument(
		"--video",
		type=Path,
		required=True,
		help="Path to video"
	)

	return parser.parse_args(arg_list)

def mark_points(arg_list = None):
	args = parse_args(arg_list)

	video_path = args.video.resolve()

	try: 
		marker = TrapezoidMarker(video_path)
	except FileNotFoundError:
		print("File wasn't found. Exitting program...")
		sys.exit(1)

	trapezoid_points = marker.run()
	print("Trapezoid marking finished!!!")

	if trapezoid_points is not None:

		yaml_path = args.config / f'{args.name}.yaml'
		print(f"Saving points to {yaml_path}")

		if yaml_path.exists():
			# Extracting data from the actual yaml file
			with open(yaml_path, 'r') as yaml_file:
				data = yaml.safe_load(yaml_file) or {}

			data[str(video_path)] = trapezoid_points.flatten().tolist()
			# Saving new version of this yaml file
			with open(yaml_path, 'w') as yaml_file:
				yaml.dump(data, yaml_file, default_flow_style=False)

		else:
			data = {}
			data[str(video_path)] = trapezoid_points.flatten().tolist()
			with open(yaml_path, 'w') as yaml_file:
				yaml.dump(data, yaml_file, default_flow_style=False)





if __name__ == "__main__":
	mark_points()
