"""A3消融专用测试：严格只删除M3/CAI，保留其余模块边界。"""
import unittest
import torch
from map_net_a3.model import MAPNetA3, PAPER_MODULES, count_parameters


class A3AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model():
        return MAPNetA3(
            torch.linspace(0, 5000, 50),
            ssp_mean=1500, ssp_std=30,
            bathy_mean=2500, bathy_std=1000,
            tl_mean=75, tl_std=20,
        )

    def test_only_cai_removed_and_all_original_interfaces_exist(self):
        model = self.model()
        self.assertEqual(PAPER_MODULES["M3_CAI"], ())
        self.assertEqual(count_parameters(model), 20_061_426)
        for key in ("M1_EFE", "M2_DPF", "M4_MTR"):
            for attr in PAPER_MODULES[key]:
                self.assertTrue(hasattr(model, attr), f"missing {key}: {attr}")
        self.assertEqual(len(PAPER_MODULES["M2_DPF"]), 3)
        self.assertEqual(len(PAPER_MODULES["M4_MTR"]), 8)

    def test_model_forward_equals_explicit_m1_m2_m3_identity_m4(self):
        torch.manual_seed(123)
        model = self.model().eval()
        ssp = torch.full((1, 50), 1500.0)
        ssp[:, 43:] = float('nan')
        bathy = torch.linspace(2000, 3000, 61 * 61).view(1, 61, 61)
        with torch.no_grad():
            features_ssp, features_bathy, terrain_global = model._module1_encode_environment(ssp, bathy)
            path, bottom_ray, context = model._module2_fuse_directional_paths(
                features_ssp, features_bathy, terrain_global, bathy
            )
            self.assertEqual(path.shape, (1, 9, 128, 250))
            self.assertEqual(bottom_ray.shape, (1, 9, 250))
            self.assertEqual(context.shape, (1, 256))
            self.assertIs(model._module3_interact_azimuths(path), path)
            manual = model._module4_reconstruct_tl(path, bottom_ray, context)
            actual = model(ssp, bathy)
        self.assertEqual(actual.shape, (1, 9, 200, 250))
        self.assertTrue(torch.equal(actual, manual))
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
