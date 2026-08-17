from __future__ import annotations

import math
import unittest

import numpy as np

from prime_lab.sieve import geometric_blocks, segmented_prime_mask, simple_sieve, von_mangoldt_segment
from prime_lab.singular_series import pair_singular_series
from prime_lab.statistics import autocorrelation_lags, direct_autocorrelation


class SieveTests(unittest.TestCase):
    def test_simple_sieve(self) -> None:
        self.assertEqual(simple_sieve(30).tolist(), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    def test_segmented_sieve(self) -> None:
        base = simple_sieve(20)
        mask = segmented_prime_mask(90, 121, base)
        found = (90 + np.flatnonzero(mask)).tolist()
        self.assertEqual(found, [97, 101, 103, 107, 109, 113])

    def test_von_mangoldt_includes_prime_powers(self) -> None:
        low, high = 2, 30
        base = simple_sieve(math.isqrt(high - 1))
        mask = segmented_prime_mask(low, high, base)
        weights = von_mangoldt_segment(low, high, mask, base)
        self.assertAlmostEqual(weights[8 - low], math.log(2))
        self.assertAlmostEqual(weights[9 - low], math.log(3))
        self.assertEqual(weights[12 - low], 0.0)
        self.assertAlmostEqual(weights[29 - low], math.log(29))

    def test_blocks_cover_without_overlap(self) -> None:
        blocks = list(geometric_blocks(100, 1_000, 1.5))
        self.assertEqual(blocks[0][0], 100)
        self.assertEqual(blocks[-1][1], 1_000)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(blocks, blocks[1:])))


class StatisticsTests(unittest.TestCase):
    def test_fft_autocorrelation_matches_direct(self) -> None:
        values = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.float64)
        expected = direct_autocorrelation(values, 4)
        observed = autocorrelation_lags(values, 4, workers=1)
        np.testing.assert_allclose(observed, expected, atol=1e-10)

    def test_prime_pair_count_small_interval(self) -> None:
        low, high = 100, 10_000
        mask = segmented_prime_mask(low, high)
        observed = np.rint(autocorrelation_lags(mask.astype(np.float64), 30, workers=1)).astype(int)
        for gap in [2, 6, 10, 30]:
            direct = int(np.count_nonzero(mask[:-gap] & mask[gap:]))
            self.assertEqual(observed[gap], direct)


class SingularSeriesTests(unittest.TestCase):
    def test_expected_relative_factors(self) -> None:
        offsets = np.array([2, 4, 6, 10, 14, 30])
        series = pair_singular_series(offsets)
        ratios = series / series[0]
        expected = np.array([1.0, 1.0, 2.0, 4.0 / 3.0, 6.0 / 5.0, 8.0 / 3.0])
        np.testing.assert_allclose(ratios, expected, rtol=1e-14)

    def test_odd_offset_is_locally_blocked(self) -> None:
        series = pair_singular_series(np.array([1, 2, 3]))
        self.assertEqual(series[0], 0.0)
        self.assertGreater(series[1], 0.0)
        self.assertEqual(series[2], 0.0)


if __name__ == "__main__":
    unittest.main()
