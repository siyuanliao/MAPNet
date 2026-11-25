"""A3/w/o CAI：前向、反向、结构与checkpoint测试。"""
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from map_net_a3.model import (
    MAPNetA3, MODEL_ARCHITECTURE,
    MODEL_CONFIG, build_model, count_parameters,
)
from map_net_a3.metrics import water_valid_mask, masked_error_sums


class A3ModelTests(unittest.TestCase):
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
        return MAPNetA3(
            self.depth, **self.kwargs,
            gradient_checkpointing=checkpointing,
        )

    def test_a3_architecture_parameter_budget_and_no_cai_weights(self):
        self.assertEqual(MODEL_ARCHITECTURE, "MAP-Net-A3")
        self.assertEqual(MODEL_CONFIG["base_blocks"], 9)
        model = self.build()
        self.assertEqual(count_parameters(model), 20_061_426)
        self.assertFalse(hasattr(model, "angular"))
        self.assertFalse(hasattr(model, "angular_mla"))
        self.assertFalse(any("angular" in k for k in model.state_dict()))
        self.assertNotIn("profile", inspect.signature(MAPNetA3).parameters)
        self.assertNotIn("profile", inspect.signature(build_model).parameters)

    def test_m3_identity_preserves_same_tensor_without_operations(self):
        model = self.build().eval()
        path = torch.randn(2, 9, 128, 250, requires_grad=True)
        forwarded = model._module3_interact_azimuths(path)
        self.assertIs(forwarded, path)
        forwarded.square().sum().backward()
        self.assertTrue(torch.isfinite(path.grad).all())

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

        # 无M3可学习层；梯度仍应通过恒等旁路抵达上游M2/DPF。
        self.assertIsNotNone(model.path.proj0.weight.grad)
        self.assertIsNotNone(model.global_context[0].weight.grad)
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
            self.assertEqual(count_parameters(a), 20_061_426)
            b.load_state_dict(a.state_dict(), strict=True)

            bad = {k: v for k, v in a.state_dict().items() if k != "path.proj0.weight"}
            with self.assertRaises(RuntimeError):
                b.load_state_dict(bad, strict=True)


if __name__ == "__main__":
    unittest.main()
