"""TL normalization counts all finite grid values including very deep cells."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from aglc_net import data

class TestFiniteStats(unittest.TestCase):
    def test_stats_use_entire_finite_tl_grid_without_bottom_file(self):
        self.assertNotIn('bottom_depth_index_r2.npy', data.EXPECTED)
        ssp = np.full((2, 50), 1500.0, dtype=np.float32)
        ssp[1, :30] += 2
        ssp[:, 45:] = np.nan
        counts = np.full((2,), 45, dtype=np.int16)
        bathy = np.arange(2 * 61 * 61, dtype=np.float32).reshape(2, 61, 61)
        tl = np.full((2, 9, 200, 250), 7000, dtype=np.int16)
        tl[1] += 100
        tl[:, :, 199, :] = 9000  # 'seabed-below' sentinel-like finite cells
        lookups = {
            'ssp_r2.npy': ssp,
            'ssp_valid_count_r2.npy': counts,
            'bathymetry_r2.npy': bathy,
            'inputs_r2.npy': tl,
        }
        with patch.object(data, 'check_files', side_effect=lambda x: x), \
             patch.object(data, 'resolve_file', side_effect=lambda root, name: name), \
             patch.object(data.np, 'load', side_effect=lambda filename, **kwargs: lookups[filename]), \
             patch.dict(data.SPLIT_SLICES, {'train': (0, 2)}):
            stats = data.compute_train_stats(Path('.'), chunk_size=1, verbose=False)
        self.assertEqual(stats['tl_valid_points'], tl.size)
        self.assertEqual(stats['tl_statistics_region'], 'all_grid_points')
        self.assertAlmostEqual(stats['tl_mean'], (tl / 100).mean(), places=5)
        self.assertAlmostEqual(stats['tl_train_max_db'], 90.0, places=5)
        self.assertEqual(stats['loss_policy'], 'full_grid_mse_no_mask_v1')

    def test_tl_sentinel_is_replaced_but_not_excluded(self):
        raw = np.full((2,9,200,250), 7000, dtype=np.int16)
        raw[0,0,199,0] = -32768
        ssp = np.full((2,50),1500.,np.float32)
        ssp[1] += 1.
        bathy = np.arange(2*61*61,dtype=np.float32).reshape(2,61,61)
        counts = np.full(2,50,dtype=np.int16)
        files = {'ssp_r2.npy':ssp,'ssp_valid_count_r2.npy':counts,
                 'bathymetry_r2.npy':bathy,'inputs_r2.npy':raw}
        with patch.object(data,'check_files',side_effect=lambda p:p), \
             patch.object(data,'resolve_file',side_effect=lambda r,k:k), \
             patch.object(data.np,'load',side_effect=lambda k,**kw:files[k]), \
             patch.dict(data.SPLIT_SLICES,{'train':(0,2)}):
            stats=data.compute_train_stats(Path('.'),tl_invalid_raw=-32768,
                    invalid_target_fill_db=80.,chunk_size=1,verbose=False)
        self.assertEqual(stats['tl_valid_points'], raw.size)
        self.assertAlmostEqual(stats['tl_mean'],(70*(raw.size-1)+80)/raw.size,places=6)

if __name__ == '__main__': unittest.main()
