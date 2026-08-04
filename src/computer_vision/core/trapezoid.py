import cv2
from pathlib import Path
import yaml
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from enum import Enum
	
from utils.draw import write_lines_from_points
from.config import DATA_DIR

class MarkState(Enum):
	NOT_MARKED = 0
	MARKING = 1
	MARKED = 2

class TrapezoidMarker:
	def __init__(self, video_path : Path, config : Path = DATA_DIR, name : str = 'points'):
		self.video = video_path.resolve()
		
		self.cap = cv2.VideoCapture(str(self.video))
		if not self.cap.isOpened():
			raise FileExistsError("Could not open video file")
		
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		self.original_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
		self.display_image = self.original_image.copy()
		self.next_frame = True

		self.config = config
		self.name = name
	
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

			# Inverting if second point is lower than the first
			if self.points[0,1] < self.points[1,1]:
				self.points[[0,1]] = self.points[[1,0]]

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

			if self.state == MarkState.MARKED:
				write_lines_from_points(self.display_image, self.points[0], self.points[1], self.width)

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

	def save_file(self, points : NDArray, yaml_path : Path, video_path : Path):
		print(f"Saving points to {yaml_path}")

		if yaml_path.exists():
			# Extracting data from the actual yaml file
			with open(yaml_path, 'r') as yaml_file:
				data = yaml.safe_load(yaml_file) or {}

			data[str(video_path)] = points.flatten().tolist()
			# Saving new version of this yaml file
			with open(yaml_path, 'w') as yaml_file:
				yaml.dump(data, yaml_file, default_flow_style=False)

		else:
			data = {}
			data[str(video_path)] = points.flatten().tolist()
			with open(yaml_path, 'w') as yaml_file:
				yaml.dump(data, yaml_file, default_flow_style=False) 

	def mark_trapezoids(self):
		trapezoid_points = self.run()
		print("Trapezoid marking finished!!!")

		if trapezoid_points is not None:

			yaml_path = self.config / f'{self.name}.yaml'

			self.save_file(points=trapezoid_points, 
						yaml_path=yaml_path, 
						video_path=self.video)
		else:
			print("Process wasn't able to mark any points")
