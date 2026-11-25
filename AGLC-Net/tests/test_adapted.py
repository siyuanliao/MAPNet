import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aglc_net.model import (
    TLCodec, AGLCNet, count_parameters, BathymetryContext,
    SSPPerturbationMapper, LatentCorrectionModule, REFERENCE_PARAMETERS,
)
from aglc_net.metrics import full_grid_error_sums, FullGridMetrics
from train import train_epoch


def synthetic_bank():
    # 121 sources; unique exact bathymetry key for each.
    bathy = np.broadcast_to(500 + np.arange(121, dtype=np.float32)[:, None, None] * 2.0,
                            (121, 61, 61)).copy()
    ssp = np.full((121, 50), 1500.0, dtype=np.float32)
    ssp[:, 42:] = np.nan
    raw = np.full((121, 9, 200, 250), 7100, dtype=np.int16)
    return dict(bathy=bathy, ssp=ssp, tl_raw=raw)


STATS = dict(ssp_mean=1500.0, ssp_std=10.0, bathy_mean=620.0,
             bathy_std=70.0, tl_mean=70.0, tl_std=12.0, tl_invalid_raw=None)


class TestParameterMatching(unittest.TestCase):
    def test_full_network_total_params_very_close(self):
        total = sum(map(count_parameters, [TLCodec(), BathymetryContext(64),
                       SSPPerturbationMapper(), LatentCorrectionModule()]))
        print('FULL MODEL params:', total, 'vs', REFERENCE_PARAMETERS)
        self.assertEqual(total, 20_130_731)
        self.assertLess(abs(total / REFERENCE_PARAMETERS-1), .10)


class TestSmallModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        cls.bank = synthetic_bank()
        cls.kwargs = dict(widths=(8, 16, 24, 32, 40), latent_dim=32,
                          bathy_dim=16, hidden=64, angle_dim=8)

    def test_reference_source_and_numerical_validity(self):
        model = AGLCNet(self.bank, STATS, **self.kwargs)
        bathy = torch.from_numpy(self.bank['bathy'][[7, 18]])
        src = model.source_index(bathy)
        self.assertEqual(src.tolist(), [7, 18])
        ref = model.reference_batch(src[:1], 0, 2)
        self.assertEqual(tuple(ref.shape), (2, 1, 200, 250))
        self.assertTrue(torch.isfinite(ref).all())

    def test_unknown_source_rejected(self):
        model = AGLCNet(self.bank, STATS, **self.kwargs)
        changed = torch.from_numpy(self.bank['bathy'][[5]].copy())
        changed[0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, 'Unseen bathymetry'):
            model.source_index(changed)

    def test_autoencoder_backward(self):
        codec = TLCodec(widths=self.kwargs['widths'], latent_dim=32)
        inp = torch.randn(1, 1, 200, 250)
        output = codec(inp)
        self.assertEqual(tuple(output.shape), (1, 1, 200, 250))
        output.mean().backward()
        self.assertTrue(torch.isfinite(codec.to_latent.weight.grad).all())

    def test_forward_two_inputs_and_ssp_nan(self):
        model = AGLCNet(self.bank, STATS, **self.kwargs)
        ssp = torch.from_numpy(self.bank['ssp'][[7]].copy())
        ssp[0, :42] += 0.5
        bathy = torch.from_numpy(self.bank['bathy'][[7]].copy())
        out = model(ssp, bathy, angle_chunk=3)
        self.assertEqual(tuple(out.shape), (1, 9, 200, 250))
        self.assertTrue(torch.isfinite(out).all())

    def test_correction_module_training_grad(self):
        model = AGLCNet(self.bank, STATS, **self.kwargs)
        ssp = torch.from_numpy(self.bank['ssp'][[9]].copy())
        ssp[:, :42] += 1.0
        bathy = torch.from_numpy(self.bank['bathy'][[9]].copy())
        out = model(ssp, bathy, angle_start=0, angle_end=1)
        out.mean().backward()
        self.assertIsNotNone(model.corrector.beta.grad)
        self.assertIsNotNone(model.mapper.net[0].weight.grad)
        self.assertIsNotNone(model.bathy_encoder.feature[0].weight.grad)
        self.assertIsNotNone(model.codec.from_latent.weight.grad)
        self.assertIsNone(model.codec.encoder_blocks[0].conv[0].weight.grad)

    def test_full_grid_includes_all_cells_and_gradients(self):
        target = torch.ones(1, 9, 200, 250)
        # This deep cell would often be under the physical bottom: INCLUDED.
        target[:, :, 130, 20] = 2.
        pred = torch.zeros_like(target, requires_grad=True)
        sq, ab, n = full_grid_error_sums(pred, target)
        self.assertEqual(n, 9*200*250)
        self.assertEqual(int(sq.item()), n+9*3)
        sq.backward()
        self.assertEqual(float(pred.grad[0, 0, 130, 20]), -4.)
        self.assertEqual(float(pred.grad[0, 0, 3, 4]), -2.)

    def test_full_grid_rejects_nonfinite_target(self):
        target = torch.ones(1, 9, 200, 250)
        target[0, 0, 3, 4] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            full_grid_error_sums(torch.zeros_like(target), target)

    def test_full_grid_metrics_deep_cells(self):
        target = torch.full((1, 9, 200, 250), 75.)
        pred = target.clone()
        pred[:,:,199,:] += 2.
        metrics = FullGridMetrics('cpu')
        metrics.add(pred,target)
        out = metrics.result()['overall']
        self.assertEqual(out['valid_points'], 9*200*250)
        self.assertGreater(out['rmse_db'], 0.)
        self.assertEqual(metrics.result()['per_angle']['0deg']['valid_points'],200*250)

    def test_pretrain_one_step_full_grid_optimizer(self):
        class ToyAutoencoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.factor = torch.nn.Parameter(torch.tensor(.7))
            def forward(self, inp):
                return inp * self.factor
        toy = ToyAutoencoder()
        opt = torch.optim.SGD(toy.parameters(), lr=.1)
        args = SimpleNamespace(stage='pretrain',grad_accum=1,angle_chunk=3,
                               amp='off',grad_clip=1.)
        target = torch.full((1,9,200,250), 75.)
        dat = [(torch.zeros(1,50),torch.zeros(1,61,61),target,torch.tensor([0]))]
        scaler = torch.amp.GradScaler('cuda',enabled=False)
        prev = toy.factor.item()
        res = train_epoch(toy,dat,opt,scaler,STATS,args,torch.device('cpu'),1)
        self.assertGreater(toy.factor.item(),prev)
        self.assertEqual(res['valid_points'], 9*200*250)


if __name__ == '__main__':
    unittest.main()
