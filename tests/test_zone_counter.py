import unittest
from pathlib import Path
import tempfile
import numpy as np
import cv2
import sys

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from scripts.zone_counter import parse_predictions, draw_bboxes


class TestZoneCounter(unittest.TestCase):
    def setUp(self):
        # Create a temporary prediction file
        self.temp_file = tempfile.NamedTuple = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
        self.temp_file.write("0\n")
        self.temp_file.write("3 100.5 150.0 200.5 250.0\n")
        self.temp_file.write("1 50.0 60.0 120.0 130.0\n")
        self.temp_file.write("1\n")
        self.temp_file.write("2 10.0 20.0 30.0 40.0\n")
        self.temp_file.close()
        self.txt_path = Path(self.temp_file.name)

    def tearDown(self):
        self.txt_path.unlink()

    def test_parse_predictions(self):
        predictions = parse_predictions(self.txt_path)
        self.assertEqual(len(predictions), 2)
        self.assertIn(0, predictions)
        self.assertIn(1, predictions)
        
        # Check frame 0
        self.assertEqual(len(predictions[0]), 2)
        self.assertEqual(predictions[0][0][0], 3)
        self.assertEqual(predictions[0][0][1], (100.5, 150.0, 200.5, 250.0))
        self.assertEqual(predictions[0][1][0], 1)
        self.assertEqual(predictions[0][1][1], (50.0, 60.0, 120.0, 130.0))
        
        # Check frame 1
        self.assertEqual(len(predictions[1]), 1)
        self.assertEqual(predictions[1][0][0], 2)
        self.assertEqual(predictions[1][0][1], (10.0, 20.0, 30.0, 40.0))

    def test_draw_bboxes(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_data = [(3, (100.0, 150.0, 200.0, 250.0))]
        result_img = draw_bboxes(img, frame_data)
        self.assertEqual(result_img.shape, (480, 640, 3))


if __name__ == '__main__':
    unittest.main()