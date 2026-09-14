import unittest
import sys
from pathlib import Path
import numpy as np

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]
if str(ROOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(ROOT_DIRECTORY))

from src.computer_vision.core.audio_handler import AudioHandler


class TestNormalizedCrossCorrelation(unittest.TestCase):
    """Unit tests for AudioHandler.normalized_cross_correlation.

    The function computes zero-normalized cross-correlation between a 2D
    (frequency x time) `data` array and a 2D `kernel`. Output is bounded in
    [-1, 1]: 1.0 = perfect match, -1.0 = inverted match, 0.0 = no correlation
    (or zero-variance patch/kernel).
    """

    def _make_handler(self):
        # Bypass __init__ to avoid opening a microphone stream.
        return AudioHandler.__new__(AudioHandler)

    def test_output_length_matches_data(self):
        ah = self._make_handler()
        data = np.random.default_rng(0).random((4, 20))
        kernel = data[:, 5:9]
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertEqual(ncc.shape[0], data.shape[1])

    def test_self_match_returns_one_at_expected_index(self):
        ah = self._make_handler()
        data = np.random.default_rng(1).random((4, 20))
        kernel = data[:, 5:9].copy()  # K=4, int(K/2)=2 -> peak at 5 + 2 = 7
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertAlmostEqual(float(ncc[7]), 1.0, places=6)
        self.assertEqual(int(np.argmax(ncc)), 7)

    def test_negative_match_returns_minus_one(self):
        ah = self._make_handler()
        data = np.random.default_rng(2).random((4, 20))
        kernel = -data[:, 5:9].copy()
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertAlmostEqual(float(ncc[7]), -1.0, places=6)

    def test_scale_and_offset_invariance(self):
        ah = self._make_handler()
        base = np.random.default_rng(3).random((4, 20))
        kernel = base[:, 5:9].copy()
        # Positive scaling + constant offset should not change the match score.
        scaled = base * 3.0 + 7.5
        ncc = ah.normalized_cross_correlation(scaled, kernel)
        self.assertAlmostEqual(float(ncc[7]), 1.0, places=6)

    def test_values_are_bounded(self):
        ah = self._make_handler()
        data = np.random.default_rng(5).random((6, 30))
        kernel = data[:, 10:16]
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertTrue(np.all(ncc <= 1.0 + 1e-12))
        self.assertTrue(np.all(ncc >= -1.0 - 1e-12))

    def test_zero_variance_patch_returns_zero(self):
        ah = self._make_handler()
        data = np.zeros((4, 20))
        kernel = np.ones((4, 4))
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertTrue(np.all(ncc == 0.0))

    def test_zero_variance_kernel_returns_zero(self):
        ah = self._make_handler()
        data = np.random.default_rng(4).random((4, 20))
        kernel = np.full((4, 4), 5.0)  # constant kernel -> zero variance
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertTrue(np.all(ncc == 0.0))

    def test_uncorrelated_signal_gives_low_scores(self):
        ah = self._make_handler()
        rng = np.random.default_rng(6)
        data = rng.random((4, 20))
        kernel = rng.random((4, 4))  # independent random kernel
        ncc = ah.normalized_cross_correlation(data, kernel)
        self.assertTrue(float(ncc.max()) < 0.9)


if __name__ == '__main__':
    unittest.main()
