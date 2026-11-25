import tempfile
import unittest
from pathlib import Path
import numpy as np
from rc_cnn.data import resolve_data_files, SPLIT_SLICES

class DataTests(unittest.TestCase):
    def test_year_splits(self):
        self.assertEqual(SPLIT_SLICES['train'], (0, 15972))
        self.assertEqual(SPLIT_SLICES['val'], (15972, 17424))
        self.assertEqual(SPLIT_SLICES['test'], (17424, 18876))
        for i in (0, 121, 1452, 15972, 18875):
            self.assertEqual(i % 121, (i // 121 * 121 + i % 121) % 121)
    def test_plain_and_r2_filename_resolution(self):
        names = ['inputs.npy', 'bathymetry.npy', 'ssp.npy',
                 'ssp_depth_m.npy', 'ssp_valid_count.npy']
        with tempfile.TemporaryDirectory() as tmp:
            for n in names: (Path(tmp) / n).touch()
            fs = resolve_data_files(tmp)
            self.assertEqual(fs['tl'].name, 'inputs.npy')
            self.assertEqual(fs['ssp'].name, 'ssp.npy')
            (Path(tmp) / 'ssp_r2.npy').touch()
            self.assertEqual(resolve_data_files(tmp)['ssp'].name, 'ssp_r2.npy')

if __name__ == '__main__':
    unittest.main()
