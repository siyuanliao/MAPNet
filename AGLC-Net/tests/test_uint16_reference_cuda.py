"""Regression test for the uint16 CUDA advanced-indexing failure seen on HPC."""
import unittest
import numpy as np
import torch
from aglc_net.model import AGLCNet


class TestUInt16Reference(unittest.TestCase):
    @staticmethod
    def make_model():
        raw = np.full((121, 9, 200, 250), 36000, dtype=np.uint16)
        raw[1, 0, 0, 0] = 65535
        ssp = np.full((121, 50), 1500, dtype=np.float32)
        bathy = np.zeros((121, 61, 61), dtype=np.float32)
        bank = {"tl_raw": raw, "ssp": ssp, "bathy": bathy}
        stats = {"tl_mean": 0., "tl_std": 1., "ssp_mean": 1500.,
                 "ssp_std": 10., "bathy_mean": 1000., "bathy_std": 100.}
        model = AGLCNet(bank, stats, widths=(8, 16, 24, 32, 40),
                                   latent_dim=32, bathy_dim=16, hidden=64,
                                   angle_dim=8)
        return model

    def test_lossless_uint16_conversion_cpu(self):
        model = self.make_model()
        self.assertEqual(model.ref_raw.dtype, torch.int32)
        ref = model.reference_batch(torch.tensor([1]), 0, 1)
        self.assertEqual(ref.shape, (1, 1, 200, 250))
        self.assertAlmostEqual(float(ref[0, 0, 0, 0]), 655.35, places=2)
        self.assertAlmostEqual(float(ref[0, 0, 0, 1]), 360., places=2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_uint16_reference_advanced_indexing_cuda(self):
        model = self.make_model().cuda()
        ref = model.reference_batch(torch.tensor([1, 0], device="cuda"), 0, 3)
        self.assertEqual(ref.shape, (6, 1, 200, 250))
        self.assertTrue(torch.isfinite(ref).all().item())
        self.assertAlmostEqual(float(ref[0, 0, 0, 0].item()), 655.35, places=2)


if __name__ == "__main__":
    unittest.main()
