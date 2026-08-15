from __future__ import annotations

import unittest

import numpy as np

from mgi_dssm.preprocessing import (
    capacity_soh,
    fit_train_minmax,
    isolated_sigma_interpolate,
)


class PreprocessingTests(unittest.TestCase):
    def test_repairs_only_isolated_sigma_candidate(self) -> None:
        values = np.linspace(2.0, 1.8, 31)
        values[15] = 3.0
        result = isolated_sigma_interpolate(values, window=21)
        self.assertTrue(result.sigma_candidate[15])
        self.assertTrue(result.isolated[15])
        self.assertTrue(result.repaired[15])
        self.assertTrue(np.isclose(result.values[15], (values[14] + values[16]) / 2.0))

    def test_retains_adjacent_candidate_run(self) -> None:
        values = np.linspace(2.0, 1.8, 31)
        values[14:16] = 3.0
        result = isolated_sigma_interpolate(values, window=21)
        self.assertTrue(result.sigma_candidate[14:16].all())
        self.assertFalse(result.isolated[14:16].any())
        self.assertTrue(np.array_equal(result.values[14:16], values[14:16]))

    def test_preserves_endpoints_and_interpolates_internal_missing(self) -> None:
        values = np.linspace(2.0, 1.8, 31)
        values[0] = 4.0
        values[10] = np.nan
        result = isolated_sigma_interpolate(values, window=21)
        self.assertEqual(result.values[0], 4.0)
        self.assertTrue(result.repaired[10])
        self.assertTrue(np.isfinite(result.values[10]))

    def test_train_minmax_and_capacity_soh(self) -> None:
        values = np.asarray([[1.0, 4.0], [3.0, 8.0], [2.0, 6.0]])
        offset, scale = fit_train_minmax(values)
        self.assertTrue(np.allclose(offset, [1.0, 4.0]))
        self.assertTrue(np.allclose(scale, [2.0, 4.0]))
        self.assertTrue(
            np.allclose(capacity_soh(np.asarray([2.0, 1.4]), 2.0), [1.0, 0.7])
        )


if __name__ == "__main__":
    unittest.main()
