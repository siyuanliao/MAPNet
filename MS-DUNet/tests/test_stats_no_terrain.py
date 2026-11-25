"""Check finite-only training statistics without any bottom_depth_index file."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ms_dunet import data


class StatsNoTerrainTests(unittest.TestCase):
    def test_no_bottom_file_required_and_all_depths_are_counted(self):
        self.assertNotIn("bottom_depth_index_r2.npy", data.EXPECTED)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            arr_ssp = np.tile(np.linspace(1500, 1549, 50, dtype=np.float32), (4, 1))
            arr_ssp[:, 40:] = np.nan
            arr_bathy = np.stack((np.full((61, 61), 1000, dtype=np.float32),
                                  np.full((61, 61), 1200, dtype=np.float32)))
            arr_tl = np.full((4, 9, 200, 250), 6500, dtype=np.int16)
            arr_tl[:, :, 199, :] = 9000  # finite 'below-bottom' values count
            arr_tl[0, 0, 0, 0] = -32768  # explicit invalid sentinel
            np.save(root / "ssp.npy", arr_ssp)
            np.save(root / "ssp_valid_count.npy", np.array([40] * 4, dtype=np.int16))
            np.save(root / "bathymetry.npy", arr_bathy)
            np.save(root / "ssp_depth_m.npy", np.linspace(0, 4900, 50))
            np.save(root / "tl.npy", arr_tl)
            with patch.object(data, "N_SOURCES", 2), \
                 patch.dict(data.SPLIT_SLICES, {"train": (0, 4)}), \
                 patch.object(data, "check_files", return_value=root):
                stats = data.compute_train_stats(root, tl_invalid_raw=-32768,
                                                 chunk_size=2, verbose=False)
            expected_count = 4 * 9 * 200 * 250 - 1
            self.assertEqual(stats["tl_finite_points"], expected_count)
            self.assertEqual(stats["mask_policy"], "finite_only_v1")
            expected_mean = (float(arr_tl.astype(np.float64).sum()) - (-32768))
            expected_mean /= 100.0 * expected_count
            self.assertAlmostEqual(stats["tl_mean"], expected_mean, places=5)
            self.assertFalse((root / "bottom_depth_index.npy").exists())


if __name__ == "__main__":
    unittest.main()
