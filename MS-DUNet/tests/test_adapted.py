"""Unit/smoke tests. No real ocean dataset is needed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ms_dunet.model import (
    MSDUNet, AdaptiveMultiScaleDilated,
    MODEL_CONFIG, REFERENCE_PARAMETERS, count_parameters, build_model,
)
from ms_dunet.metrics import finite_target_mask, finite_error_sums
from ms_dunet.data import resolve_file, TLDataset
from train import train_one_epoch


class AdaptedDUNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.depth = torch.linspace(0, 5000, 50)
        cls.stats = dict(ssp_mean=1500.0, ssp_std=30.0,
                         bathy_mean=2500.0, bathy_std=1000.0,
                         tl_mean=75.0, tl_std=20.0)

    def model(self, *, gradient_checkpointing=False, angle_chunk=3):
        return MSDUNet(
            self.depth, **self.stats,
            gradient_checkpointing=gradient_checkpointing,
            angle_chunk=angle_chunk)

    def inputs(self):
        ssp = torch.linspace(1510, 1480, 50)[None, :].clone()
        ssp[:, 30:] = float('nan')
        # Gradients in lat/lon ensure 9 bearings see different environments.
        y = torch.linspace(-1, 1, 61)[:, None]
        x = torch.linspace(-1, 1, 61)[None, :]
        bathy = (2800 + 300 * y + 250 * x)[None].float()
        return ssp, bathy

    def test_parameter_fairness_and_paper_module(self):
        model = self.model()
        params = count_parameters(model)
        self.assertEqual(params, 20_109_645)
        self.assertLess(abs(params / REFERENCE_PARAMETERS - 1), .01)
        self.assertEqual(len(model.encoders), 5)
        self.assertEqual(len(model.decoders), 4)
        self.assertEqual(MODEL_CONFIG['encoder_channels'], (64,128,256,448,984))
        for decoder in model.decoders:
            self.assertIsInstance(decoder.dilated_fuse, AdaptiveMultiScaleDilated)
            self.assertEqual(decoder.dilated_fuse.dilations, (1,2,3))

    def test_ssp_nan_and_directional_rasterization(self):
        model = self.model()
        ssp, bathy = self.inputs()
        prep = model.adapter.prepare(ssp, bathy)
        raster = model.adapter.rays(prep, 0, 9)
        self.assertEqual(tuple(raster.shape), (9, 32, 50, 63))
        self.assertTrue(torch.isfinite(raster).all())
        # channel 24 = normalized ray bottom; 0deg north and 80deg east
        self.assertGreater(raster[0, 24, 10, -1].detach().item(), raster[4, 24, 10, -1].detach().item() - .2)
        self.assertGreater(raster[0, 24, 10, -1].detach().item(), raster[5, 24, 10, -1].detach().item())
        # channel 26 is SSP valid mask; deepest model row is beyond valid SSP.
        self.assertEqual(raster[0, 26, -1, -1].detach().item(), 0)
        self.assertEqual(raster[0, 26, 0, 0].detach().item(), 1)

    def test_output_and_angle_chunk_invariance_eval(self):
        model = self.model(angle_chunk=9).eval()
        ssp, bathy = self.inputs()
        with torch.no_grad():
            a = model(ssp, bathy)
            model.angle_chunk = 3
            b = model(ssp, bathy)
            physical = model.predict_db(ssp, bathy)
        self.assertEqual(tuple(a.shape), (1, 9, 200, 250))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.allclose(a, b, atol=2e-4, rtol=2e-4))
        self.assertTrue(torch.allclose(physical, b * 20 + 75, atol=2e-4, rtol=2e-4))
        self.assertGreater(float((b[0, 0] - b[0, 5]).abs().max()), 0)

    def test_finite_only_loss_includes_deep_cells(self):
        pred = torch.zeros(1, 9, 200, 250, requires_grad=True)
        truth = torch.ones_like(pred).detach()
        truth[:, :, 12, 5] = float('nan')
        truth[:, :, 199, 249] = float('inf')
        valid = finite_target_mask(truth)
        expected_count = 9 * 200 * 250 - 18
        self.assertEqual(int(valid.sum()), expected_count)
        self.assertTrue(bool(valid[0, 0, 198, 240]))  # Deep finite TL is included
        sq1, _, n1 = finite_error_sums(pred, truth, valid)
        self.assertEqual(int(n1), expected_count)
        sq1.backward()
        self.assertNotEqual(float(pred.grad[0, 0, 198, 240]), 0.0)
        self.assertEqual(float(pred.grad[0, 0, 12, 5]), 0.0)

    def test_backward_and_training_step(self):
        model = self.model(gradient_checkpointing=True)
        ssp, bathy = self.inputs()
        target = torch.full((1,9,200,250), 73.0)
        target[0, 0, 161, 30] = float('nan')
        ids = torch.tensor([121], dtype=torch.long)
        ds = TensorDataset(ssp,bathy,target,ids)
        initial_head = model.head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        args = SimpleNamespace(grad_accum=1, amp='off', grad_clip=1.0)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        result = train_one_epoch(model, DataLoader(ds, batch_size=1), optimizer,
                                 scaler, self.stats, args, torch.device('cpu'), 1)
        self.assertTrue(np.isfinite(result['train_rmse_db']))
        self.assertEqual(result['valid_points'], 9*200*250-1)
        self.assertGreater((model.head.weight.detach() - initial_head).abs().sum().item(), 0)

    def test_checkpoint_does_not_double_update_batchnorm(self):
        with_ckpt = self.model(gradient_checkpointing=True, angle_chunk=3).train()
        without_ckpt = self.model(gradient_checkpointing=False, angle_chunk=3).train()
        without_ckpt.load_state_dict(with_ckpt.state_dict())
        ssp, bathy = self.inputs()
        a = with_ckpt(ssp, bathy)
        b = without_ckpt(ssp, bathy)
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))
        (a.square().mean()).backward()
        (b.square().mean()).backward()
        a_bn = with_ckpt.encoders[0][1]
        b_bn = without_ckpt.encoders[0][1]
        self.assertTrue(torch.allclose(a_bn.running_mean, b_bn.running_mean, atol=1e-6))
        self.assertTrue(torch.allclose(a_bn.running_var, b_bn.running_var, atol=1e-6))
        self.assertEqual(int(a_bn.num_batches_tracked), int(b_bn.num_batches_tracked))

    def test_dataset_generic_files_and_integer_tl_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('ssp.npy', 'ssp_valid_count.npy', 'bathymetry.npy', 'tl.npy'):
                (root / name).touch()
            ssp = np.full((1, 50), 1502, dtype=np.float32)
            counts = np.array([40], dtype=np.int16)
            bathy = np.full((121, 61, 61), 2700, dtype=np.float32)
            tl = np.full((1, 9, 200, 250), 7543, dtype=np.int16)
            fake = {'ssp.npy': ssp, 'ssp_valid_count.npy': counts,
                    'bathymetry.npy': bathy, 'tl.npy': tl}
            def fake_load(path, **kwargs):
                return fake[Path(path).name]
            with patch('ms_dunet.data.np.load', side_effect=fake_load):
                ds = TLDataset(root, 'train', check=False)
                s, b, y, idx = ds[0]
            self.assertEqual(idx.item(), 0)
            self.assertTrue(torch.isnan(s[40:]).all())
            self.assertEqual(tuple(b.shape), (61,61))
            self.assertAlmostEqual(float(y[0,0,0]), 75.43, places=4)

    def test_file_names_and_strict_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            np.save(td / 'ssp_depth_m.npy', self.depth.numpy())
            self.assertEqual(resolve_file(td, 'ssp_depth_m_r2.npy').name, 'ssp_depth_m.npy')
            a = build_model(td, self.stats)
            b = build_model(td, self.stats)
            b.load_state_dict(a.state_dict(), strict=True)
            self.assertEqual(count_parameters(a), 20_109_645)


if __name__ == '__main__':
    unittest.main()
