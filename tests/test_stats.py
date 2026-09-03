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

    def _build_seconds_delay_inputs(self, n_frames=40, n_audio=200, audio_duration=40.0,
                                    audio_start_timestamp=0.0, frame_seconds=None):
        """Helper: build standard inputs for get_seconds_delay."""
        if frame_seconds is None:
            frame_seconds = np.arange(n_frames, dtype=float)
        b_video = np.zeros(n_frames, dtype=int)
        audio_detection = np.zeros(n_audio, dtype=int)
        return b_video, frame_seconds, audio_detection, audio_start_timestamp, audio_duration

    def test_get_first_idx_after_time_zero(self):
        """Time 0 maps to index 0."""
        stats = Stats(general=True)
        self.assertEqual(stats.get_first_idx_after_time(0.0, 0.2), 0)

    def test_get_first_idx_after_time_exact_multiple(self):
        """Exact multiples of the sampling period map to the exact index."""
        stats = Stats(general=True)
        self.assertEqual(stats.get_first_idx_after_time(2.0, 0.2), 10)

    def test_get_first_idx_after_time_rounds_up(self):
        """Non-multiple times round up to the next sample index."""
        stats = Stats(general=True)
        self.assertEqual(stats.get_first_idx_after_time(2.1, 0.2), 11)

    def test_get_first_idx_after_time_slightly_above(self):
        """A tiny value rounds up to index 1."""
        stats = Stats(general=True)
        self.assertEqual(stats.get_first_idx_after_time(0.01, 0.2), 1)

    def test_get_seconds_delay_zero_delay(self):
        """Audio detection exactly at the video detection start -> delay 0."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        # start_frame=10 -> limit_inf_seconds=10.0, sampling period 40/200=0.2
        # limit_inf_audio_idx = ceil(10.0/0.2) = 50
        audio_detection[50] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [0.0])

    def test_get_seconds_delay_positive_delay(self):
        """Audio detection a few samples after video start -> positive delay."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        # index 52 -> delay = 52*0.2 - 10.0 = 0.4
        audio_detection[52] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [0.4])

    def test_get_seconds_delay_no_audio_detection(self):
        """No audio detection in the window -> sentinel max_seconds_delay."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        # window=5 -> sentinel = (frame_seconds[-1]-frame_seconds[0])/len * window
        #             = (39 - 0)/39 * 5 = 5.0
        max_seconds_delay = (frame_seconds[-1] - frame_seconds[0]) / (len(frame_seconds)-1) * 5
        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [max_seconds_delay])

    def test_get_seconds_delay_audio_after_window(self):
        """Audio detection beyond the search window -> sentinel delay."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        # window=5 -> search range [50, 75); index 80 is outside
        audio_detection[80] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [5.0])

    def test_get_seconds_delay_window_boundary_exclusive(self):
        """Audio at the upper window boundary is NOT detected (range exclusive)."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        # search range is [50, 75); index 75 is excluded
        audio_detection[75] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [5.0])

    def test_get_seconds_delay_multiple_detections(self):
        """Two video segments, each producing its own delay."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:20] = 1
        b_video[30:40] = 1
        # First: [10,19] -> inf idx 50, sup idx 75; delay = 52*0.2 - 10 = 0.4
        # Second: [30,39] -> inf idx 150, sup idx 175; delay = 152*0.2 - 30 = 0.4
        audio_detection[52] = 1
        audio_detection[152] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [0.4, 0.4])

    def test_get_seconds_delay_window_clipped_at_video_end(self):
        """Window exceeding the video length is clipped to the last frame."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[35:40] = 1
        # start=35, window=10 -> limit_sup_frame = min(45, 39) = 39
        # inf idx = ceil(35/0.2) = 175, sup idx = ceil(39/0.2) = 195
        # delay = 177*0.2 - 35 = 0.4
        audio_detection[177] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=10, closing_se_size=1)

        np.testing.assert_allclose(delays, [0.4])

    def test_get_seconds_delay_with_closing(self):
        """Closing merges a small gap; delay measured from merged start."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs()
        b_video[10:15] = 1
        b_video[16:20] = 1  # 1-frame gap at index 15, closed with size 3
        # closing merges into [10,19]; delay = 52*0.2 - 10 = 0.4
        audio_detection[52] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=3)

        np.testing.assert_allclose(delays, [0.4])

    def test_get_seconds_delay_respects_audio_start_timestamp(self):
        """Audio_start_timestamp shifts the audio sample indices."""
        b_video, frame_seconds, audio_detection, audio_start, audio_duration = \
            self._build_seconds_delay_inputs(audio_start_timestamp=2.0)
        b_video[10:20] = 1
        # start=10 -> inf idx = ceil((10-2)/0.2) = 40
        # delay = 2 + 42*0.2 - 10 = 0.4
        audio_detection[42] = 1

        stats = Stats(general=True)
        delays = stats.get_seconds_delay(b_video_detection=b_video, frame_seconds=frame_seconds,
                                         audio_detection=audio_detection,
                                         audio_start_timestamp=audio_start,
                                         audio_duration=audio_duration,
                                         window=5, closing_se_size=1)

        np.testing.assert_allclose(delays, [0.4])

    def test_get_seconds_delay_user_scenario(self):
        """Audio detection starts after some seconds (user's original scenario)."""
        b_video = np.zeros(40, dtype=int)
        b_video[27:35] = 1

        frame_seconds = np.arange(1, 41, 1)

        audio_detection = np.zeros(200, dtype=int)
        audio_detection[136:140] = 1
        audio_duration = 40
        audio_start_time = 0.9

        stats = Stats(general=True)
        second_delays = stats.get_seconds_delay(b_video_detection=b_video,
                                                frame_seconds=frame_seconds,
                                                audio_start_timestamp=audio_start_time,
                                                audio_detection=audio_detection,
                                                audio_duration=audio_duration,
                                                window=10,
                                                closing_se_size=3)

        expected_delays = [0.1]

        self.assertEqual(len(second_delays), len(expected_delays))
        np.testing.assert_allclose(second_delays, expected_delays)

    def test_calculate_latency_recall_no_detection_returns_none(self):
        """No detection in the video -> no time_stamps -> None."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        predictions = np.zeros(n, dtype=int)

        stats = Stats(general=True)
        la_recall = stats.calculate_latency_recall(ground_truth=ground_truth,
                                                   predictions=predictions,
                                                   closing_structure_size=10)

        self.assertIsNone(la_recall)

    def test_calculate_latency_recall_no_detection_with_false_alarms_returns_none(self):
        """No GT detection but spurious predictions still -> None (sum_gt == 0)."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        predictions = np.zeros(n, dtype=int)
        predictions[50:60] = 1  # false alarms only

        stats = Stats(general=True)
        la_recall = stats.calculate_latency_recall(ground_truth=ground_truth,
                                                   predictions=predictions,
                                                   closing_structure_size=10)

        self.assertIsNone(la_recall)

    def test_calculate_latency_recall_perfect_detection_returns_one(self):
        """Perfect predictions (== ground truth) -> latency recall of 1.0."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1  # detection present

        stats = Stats(general=True)
        la_recall = stats.calculate_latency_recall(ground_truth=ground_truth,
                                                   predictions=ground_truth,
                                                   closing_structure_size=10)

        self.assertIsNotNone(la_recall)
        np.testing.assert_allclose(la_recall, 1.0)

    def test_calculate_latency_recall_no_detection_non_general_returns_none(self):
        """general=False always returns None regardless of detections."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1

        stats = Stats(general=False)
        la_recall = stats.calculate_latency_recall(ground_truth=ground_truth,
                                                   predictions=ground_truth,
                                                   closing_structure_size=10)

        self.assertIsNone(la_recall)

    def test_calculate_is_detected_no_ground_truth(self):
        """No ground truth detections -> (None, None)."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        predictions = np.zeros(n, dtype=int)

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (None, None))

    def test_calculate_is_detected_all_segments_detected(self):
        """Every GT segment has at least one prediction inside it."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1
        ground_truth[120:150] = 1
        predictions = ground_truth.copy()

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (2, 2))

    def test_calculate_is_detected_partial_detection(self):
        """One GT segment detected, the other missed."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1
        ground_truth[120:150] = 1
        predictions = np.zeros(n, dtype=int)
        predictions[35:40] = 1  # only in the first segment

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (1, 2))

    def test_calculate_is_detected_no_segments_detected(self):
        """GT segments exist but no predictions overlap any of them."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1
        ground_truth[120:150] = 1
        predictions = np.zeros(n, dtype=int)

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (0, 2))

    def test_calculate_is_detected_false_alarms_do_not_count(self):
        """Predictions outside GT segments are ignored."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1
        predictions = np.zeros(n, dtype=int)
        predictions[80:90] = 1  # false alarm, outside the segment

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (0, 1))

    def test_calculate_is_detected_single_frame_segment(self):
        """Single-frame GT segment -> empty slice, never detected (current behavior)."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[5] = 1
        predictions = np.zeros(n, dtype=int)
        predictions[5] = 1  # prediction at the exact same frame

        stats = Stats(general=True)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        # predictions[5:5] is empty -> not counted as detected
        self.assertEqual(result, (1, 1))

    def test_calculate_is_detected_non_general(self):
        """general=False -> (None, None) even with detections present."""
        n = 200
        ground_truth = np.zeros(n, dtype=int)
        ground_truth[30:60] = 1
        predictions = ground_truth.copy()

        stats = Stats(general=False)
        result = stats.calculate_is_detected(ground_truth=ground_truth,
                                             predictions=predictions,
                                             closing_structure_size=10)

        self.assertEqual(result, (None, None))

if __name__ == '__main__':
    unittest.main()