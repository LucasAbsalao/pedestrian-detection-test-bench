import unittest
import sys
from pathlib import Path
import numpy as np

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from src.computer_vision.benchmark.stats import Stats


class TestStats(unittest.TestCase):
    def test_get_detection_duration(self):
        """Test get_detection_duration with binary closing to merge small gaps."""
        # Create ground truth vector of 200 elements (binary: 1=detection, 0=no detection)
        ground_truth = np.zeros(200, dtype=int)
        
        # Set detection zones: 30-55, 57, 100-150
        ground_truth[30:56] = 1  # indices 30 to 55 inclusive
        ground_truth[57] = 1      # single frame at 57
        ground_truth[100:151] = 1 # indices 100 to 150 inclusive
        
        stats = Stats(general=True)
        
        # Use closing_se_size=3 to close the single-frame gap at index 56
        # Structure of size 3 will dilate by 1 on each side, closing gaps of size 1
        timestamps = stats.get_detection_duration_in_frames(ground_truth, closing_se_size=3)
        
        # Expected: gap at 56 is closed (30-55 and 57 merge), 
        # but gap between 57 and 100 (42 frames) remains open
        expected = [[30, 57], [100, 150]]
        self.assertEqual(timestamps, expected)

    def test_get_detection_duration_no_closing(self):
        """Test without closing (closing_se_size=1) - gaps should remain."""
        ground_truth = np.zeros(200, dtype=int)
        ground_truth[30:56] = 1
        ground_truth[57:59] = 1
        ground_truth[100:151] = 1
        
        stats = Stats(general=True)
        
        # closing_se_size=1 means no closing (structure of size 1)
        timestamps = stats.get_detection_duration_in_frames(ground_truth, closing_se_size=1)
        
        # All three segments should be separate
        expected = [[30, 55], [57, 58], [100, 150]]
        self.assertEqual(timestamps, expected)

    def test_get_detection_duration_larger_closing(self):
        """Test with larger closing that merges both gaps."""
        ground_truth = np.zeros(200, dtype=int)
        ground_truth[30:56] = 1
        ground_truth[57] = 1
        ground_truth[100:151] = 1
        
        stats = Stats(general=True)
        
        # closing_se_size=50 would close both gaps (gap of 1 and gap of 42)
        timestamps = stats.get_detection_duration_in_frames(ground_truth, closing_se_size=50)
        
        # Everything merges into one segment
        expected = [[30, 150]]
        self.assertEqual(timestamps, expected)

    def test_get_detection_duration_empty(self):
        """Test with no detections."""
        ground_truth = np.zeros(200, dtype=int)  # All no-detection
        
        stats = Stats(general=True)
        timestamps = stats.get_detection_duration_in_frames(ground_truth, closing_se_size=3)
        
        self.assertEqual(timestamps, [])

    def test_get_detection_duration_full(self):
        """Test with continuous detection - use closing_se_size=1 to avoid edge effects."""
        ground_truth = np.ones(200, dtype=int)  # All frames have detection
    
        stats = Stats(general=True)
        timestamps = stats.get_detection_duration_in_frames(ground_truth, closing_se_size=1)
    
        self.assertEqual(timestamps, [[0, 199]])


if __name__ == '__main__':
    unittest.main()