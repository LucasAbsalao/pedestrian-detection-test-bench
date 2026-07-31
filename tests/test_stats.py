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

    def test_get_frame_delay_basic(self):
        """Audio detection starts a few frames after video detection start."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[33:56] = 1

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [3])

    def test_get_frame_delay_zero_delay(self):
        """Audio detection at the exact same frame as video detection start."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[30:56] = 1

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [0])

    def test_get_frame_delay_no_audio_detection(self):
        """No audio detection at all -> delay equals window for each detection."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [10])

    def test_get_frame_delay_multiple_detections(self):
        """Two separate video detections, each with its own audio delay."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_video[100:151] = 1
        b_audio[35] = 1
        b_audio[102] = 1

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [5, 2])

    def test_get_frame_delay_audio_after_window(self):
        """Audio detection present but beyond the window -> delay equals window."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[45] = 1

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        # window=10, search range is [30, 40); audio at 45 is not found
        self.assertEqual(delays, [10])

    def test_get_frame_delay_window_boundary_exclusive(self):
        """Audio at index start+window is NOT detected (range is exclusive)."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[40] = 1  # 30 + window(10) = 40, excluded

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [10])

    def test_get_frame_delay_last_index_in_window(self):
        """Audio at the last index inside the window -> delay = window - 1."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[39] = 1  # inside [30, 40)

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [9])

    def test_get_frame_delay_audio_before_video_start(self):
        """Audio before video detection start is not in the window -> delay = window."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_audio[25] = 1  # before start frame 30

        stats = Stats(general=True)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [10])

    def test_get_frame_delay_window_clipped_at_video_end(self):
        """Window beyond the end of the array is clipped to array length."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[195:200] = 1
        b_audio[197] = 1

        stats = Stats(general=True)
        # window=10 but only 5 frames remain: [195, 200)
        delays = stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)

        self.assertEqual(delays, [2])

    def test_get_frame_delay_with_closing(self):
        """Video detection gap merged by closing; delay measured from merged start."""
        n = 200
        b_video = np.zeros(n, dtype=int)
        b_audio = np.zeros(n, dtype=int)
        b_video[30:56] = 1
        b_video[57] = 1  # 1-frame gap at index 56, closed with size 3
        b_audio[40] = 1

        stats = Stats(general=True)
        # closing_se_size=3 merges into [30, 57]; audio at 40 -> delay 10
        delays = stats.get_frame_delay(b_video, b_audio, window=15, closing_se_size=3)

        self.assertEqual(delays, [10])

    def test_get_frame_delay_mismatched_lengths(self):
        """Different length vectors should raise RuntimeError."""
        b_video = np.zeros(100, dtype=int)
        b_audio = np.zeros(50, dtype=int)

        stats = Stats(general=True)
        with self.assertRaises(RuntimeError):
            stats.get_frame_delay(b_video, b_audio, window=10, closing_se_size=1)


if __name__ == '__main__':
    unittest.main()