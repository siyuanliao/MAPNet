import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aglc_net import reference

class TestRefBank(unittest.TestCase):
    def test_picks_first_train_month_exclusively(self):
        # Stub arrays: 122 entries are sufficient to prove it never touches ID121.
        raw = np.arange(122,dtype=np.int16).reshape(122,1,1,1)
        raw = np.broadcast_to(raw, (122,9,200,250))
        ssp = np.zeros((122,50), np.float32)
        ssp[121] = 9000  # should not appear in reference bank
        bathy = np.zeros((122,61,61), np.float32)
        lookup = {'inputs_r2.npy':raw, 'ssp_r2.npy':ssp, 'bathymetry_r2.npy':bathy}
        def fake_resolve(root, filename):
            return filename
        def fake_load(filename, **kwargs):
            return lookup[filename]
        real_load = np.load
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(reference,'check_files', side_effect=lambda root:root), \
             patch.object(reference,'resolve_file',side_effect=fake_resolve), \
             patch.object(np,'load',side_effect=fake_load):
            # mocking np.load also impacts np.savez? save works independently.
            path = Path(tmp)/'ref.npz'
            reference.create_reference_bank(tmp, path)
            with real_load(path, allow_pickle=False) as bank:
                self.assertTrue(np.array_equal(bank['sample_ids'],np.arange(121)))
                self.assertEqual(int(bank['tl_raw'][120,0,0,0]),120)
                self.assertEqual(bank['tl_raw'].shape,(121,9,200,250))
                self.assertFalse((bank['ssp'] == 9000).any())
                self.assertNotIn('bottom', bank.files)

if __name__ == '__main__':
    unittest.main()
