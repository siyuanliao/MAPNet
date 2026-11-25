"""A1（MLP替代EFE）网络前向、反向及模块边界测试。"""
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from map_net_a1.model import (
    AngularLatentAttention, MAPNetA1, MODEL_ARCHITECTURE,
    MODEL_CONFIG, PAPER_MODULES, build_model, count_parameters,
)
from map_net_a1.metrics import water_valid_mask, masked_error_sums


class MAPNetA1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.kwargs = dict(
            ssp_mean=1500.0, ssp_std=30.0,
            bathy_mean=2500.0, bathy_std=1000.0,
            tl_mean=75.0, tl_std=20.0,
        )
        cls.depth = torch.linspace(0, 5000, 50)

    def build(self, *, checkpointing=False):
        return MAPNetA1(
            self.depth, **self.kwargs,
            gradient_checkpointing=checkpointing,
        )

    def test_a1_architecture_and_parameter_budget(self):
        model = self.build()
        self.assertEqual(MODEL_ARCHITECTURE, "MAP-Net-A1")
        self.assertEqual(MODEL_CONFIG["base_blocks"], 9)
        self.assertEqual(set(PAPER_MODULES), {"M1_EFE_MLP", "M2_DPF", "M3_CAI", "M4_MTR"})
        self.assertEqual(count_parameters(model), 19_415_267)
        self.assertLess(count_parameters(model), 20_144_116)
        self.assertEqual(sum(p.numel() for p in model.ssp.parameters()), 348_224)
        self.assertEqual(sum(p.numel() for p in model.bathy.parameters()), 3_288_432)
        for module in (model.ssp, model.bathy):
            self.assertTrue(any(isinstance(x, torch.nn.Linear) for x in module.modules()))
            self.assertFalse(any(isinstance(x, (torch.nn.Conv1d, torch.nn.Conv2d)) for x in module.modules()))
        # 除M1以外M2、M3、M4的参数量和基线一致。
        for label, expected in (("M2_DPF", 523_384), ("M3_CAI", 82_690), ("M4_MTR", 15_172_537)):
            size = sum(sum(p.numel() for p in getattr(model, sub).parameters())
                       for sub in PAPER_MODULES[label])
            self.assertEqual(size, expected, label)
        self.assertNotIn("profile", inspect.signature(MAPNetA1).parameters)
        self.assertNotIn("profile", inspect.signature(build_model).parameters)

    def test_a1_module1_interface_and_nan_handling(self):
        model = self.build().eval()
        ssp = torch.full((2, 50), 1500.0)
        ssp[0, 25:] = float("nan")
        ssp[1, 30:] = float("nan")
        bathy = torch.full((2, 61, 61), 2500.0)
        bathy[0, 12:20, 14:23] = 2200.0
        with torch.no_grad():
            s, features, g = model._module1_encode_environment(ssp, bathy)
        self.assertEqual(tuple(s.shape), (2, 192))
        self.assertEqual([tuple(f.shape) for f in features],
                         [(2,48,61,61), (2,96,31,31), (2,192,16,16)])
        self.assertEqual(tuple(g.shape), (2,192))
        self.assertTrue(torch.isfinite(s).all().item())
        self.assertTrue(torch.isfinite(g).all().item())
        self.assertTrue(all(torch.isfinite(f).all().item() for f in features))

    def test_mla_uses_compressed_q_kv_and_periodic_rotary_encoding(self):
        attention = AngularLatentAttention().train()
        self.assertEqual(attention.q_down.out_features, 64)
        self.assertEqual(attention.kv_down.out_features, 64)
        self.assertEqual(attention.k_rotary.out_features, 16)
        self.assertTrue(torch.allclose(attention.rope_cos[0], torch.ones(8), atol=1e-6))

        full_turn = 2 * torch.pi * torch.arange(1, 9, dtype=torch.float32)
        self.assertTrue(torch.allclose(full_turn.cos(), attention.rope_cos[0], atol=1e-5))
        x = torch.randn(2, 9, 128, 13, requires_grad=True)
        y = attention(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for key in ("q_down", "kv_down", "q_rotary", "k_rotary", "out"):
            self.assertIsNotNone(getattr(attention, key).weight.grad, key)
            self.assertTrue(torch.isfinite(getattr(attention, key).weight.grad).all(), key)

    def test_mla_mixes_directions(self):
        attention = AngularLatentAttention().eval()
        x = torch.randn(1, 9, 128, 5)
        changed = x.clone()
        changed[:, 3, ::2, :] += 10.0
        with torch.no_grad():
            a = attention(x)
            b = attention(changed)

        self.assertGreater((a[:, 1] - b[:, 1]).abs().max().item(), 1e-8)

    def test_forward_and_db_conversion(self):
        model = self.build().eval()
        ssp = torch.full((1, 50), 1500.0)
        ssp[:, 41:] = float("nan")
        bathy = torch.full((1, 61, 61), 2500.0)
        with torch.no_grad():
            normalized = model(ssp, bathy)
            physical = normalized * model.tl_std + model.tl_mean
        self.assertEqual(tuple(normalized.shape), (1, 9, 200, 250))
        self.assertTrue(torch.isfinite(physical).all().item())
        self.assertTrue(torch.allclose(physical, normalized * 20 + 75, atol=1e-4))

    def test_backward_mask_optimizer_step_and_checkpointing(self):
        torch.manual_seed(5)
        model = self.build(checkpointing=True).train()
        ssp = torch.full((1, 50), 1510.0)
        ssp[:, 44:] = float("nan")
        bathy = torch.full((1, 61, 61), 2500.0)
        target = torch.zeros((1, 9, 200, 250))
        target[:, :, 130:, :] = float("nan")
        bottom = torch.full((121, 9, 250), 120, dtype=torch.int16)
        mask = water_valid_mask(target, torch.tensor([121]), bottom)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        prediction = model(ssp, bathy)
        self.assertTrue(torch.isfinite(prediction).all().item())
        sq, _, n = masked_error_sums(prediction, target, mask)
        self.assertEqual(int(n), 9 * 121 * 250)
        (sq / n).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all().item(), name)

        # M1的SSP和地形MLP均实际参与输出和反向传播，不是无效挂接层。
        self.assertIsNotNone(model.ssp.net[0].weight.grad)
        self.assertIsNotNone(model.bathy.global_mlp[1].weight.grad)
        self.assertIsNotNone(model.bathy.feature_heads[0].weight.grad)
        self.assertIsNotNone(model.angular_mla.q_down.weight.grad)
        self.assertIsNotNone(model.angular_mla.out.weight.grad)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        self.assertTrue(torch.isfinite(norm).item())
        optimizer.step()
        with torch.no_grad():
            model.eval()
            self.assertTrue(torch.isfinite(model(ssp, bathy)).all().item())

    def test_build_model_and_strict_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.save(Path(tmp) / "ssp_depth_m_r2.npy", np.linspace(0, 5000, 50, dtype="float32"))
            a = build_model(tmp, self.kwargs)
            b = build_model(tmp, self.kwargs)
            self.assertEqual(count_parameters(a), 19_415_267)
            b.load_state_dict(a.state_dict(), strict=True)

            bad = {k: v for k, v in a.state_dict().items() if not k.startswith("angular_mla.")}
            with self.assertRaises(RuntimeError):
                b.load_state_dict(bad, strict=True)


if __name__ == "__main__":
    unittest.main()
