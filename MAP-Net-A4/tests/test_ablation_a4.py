"""A4/w/o MTR独立验证：仅将M4替换为共享MLP，保留全部输入输出接口。"""
import unittest
import torch
from torch import nn
from map_net_a4.model import MAPNetA4, MODEL_ARCHITECTURE, PAPER_MODULES, count_parameters


class A4AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def model(gradient_checkpointing=False):
        return MAPNetA4(
            torch.linspace(0, 5000, 50),
            ssp_mean=1500, ssp_std=30,
            bathy_mean=2500, bathy_std=1000,
            tl_mean=75, tl_std=20,
            gradient_checkpointing=gradient_checkpointing,
        )

    def test_exact_ablation_boundary_and_parameters(self):
        m = self.model()
        self.assertEqual(MODEL_ARCHITECTURE, "MAP-Net-A4")
        self.assertEqual(PAPER_MODULES["M4_MTR"], ("tl_mlp",))
        self.assertEqual(count_parameters(m), 19_011_467)
        self.assertEqual(sum(p.numel() for p in m.tl_mlp.parameters()), 14_039_888)
        self.assertEqual(
            sum(sum(p.numel() for p in getattr(m, name).parameters())
                for name in PAPER_MODULES["M1_EFE"]), 4_365_505)
        self.assertEqual(
            sum(sum(p.numel() for p in getattr(m, name).parameters())
                for name in PAPER_MODULES["M2_DPF"]), 523_384)
        self.assertEqual(
            sum(sum(p.numel() for p in getattr(m, name).parameters())
                for name in PAPER_MODULES["M3_CAI"]), 82_690)
        for name in ("start", "base", "up1", "up2", "up3", "tail", "path_skip", "context_proj"):
            self.assertFalse(hasattr(m, name), name)
        self.assertFalse(any(isinstance(x, (nn.Conv1d, nn.Conv2d)) for x in m.tl_mlp.modules()))
        self.assertEqual(m.tl_mlp[0].in_features, 4386)
        self.assertEqual(m.tl_mlp[-1].out_features, 50_000)

    def test_module_boundaries_forward_and_finite(self):
        torch.manual_seed(123)
        m = self.model().eval()
        ssp = torch.full((1, 50), 1500.)
        ssp[:, 42:] = float('nan')
        bathy = torch.linspace(1800, 3000, 61 * 61).reshape(1, 61, 61)
        with torch.no_grad():
            s, features, terrain_global = m._module1_encode_environment(ssp, bathy)
            path, bottom_ray, context = m._module2_fuse_directional_paths(
                s, features, terrain_global, bathy)
            enriched = m._module3_interact_azimuths(path)
            expected = m._module4_reconstruct_tl(enriched, bottom_ray, context)
            actual = m(ssp, bathy)
            db = m.predict_db(ssp, bathy)
        self.assertEqual(tuple(path.shape), (1, 9, 128, 250))
        self.assertEqual(tuple(bottom_ray.shape), (1, 9, 250))
        self.assertEqual(tuple(context.shape), (1, 256))
        self.assertEqual(tuple(actual.shape), (1, 9, 200, 250))
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.allclose(db, actual * 20 + 75, atol=1e-4))

    def test_decoder_uses_both_bypasses(self):
        m = self.model().eval()
        path = torch.randn(1, 9, 128, 250)
        bottom = torch.ones(1, 9, 250) * 2500
        ctx = torch.randn(1, 256)
        with torch.no_grad():
            base = m._module4_reconstruct_tl(path, bottom, ctx)
            changed_bottom = m._module4_reconstruct_tl(path, bottom + 400, ctx)
            changed_context = m._module4_reconstruct_tl(path, bottom, ctx + 1)
        self.assertGreater((base - changed_bottom).abs().max().item(), 1e-8)
        self.assertGreater((base - changed_context).abs().max().item(), 1e-8)


if __name__ == "__main__":
    unittest.main()
