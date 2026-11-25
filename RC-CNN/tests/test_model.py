import math
import unittest
import torch
from rc_cnn.model import (
    RCCNN, BathymetryRayFeatureGrid,
    count_parameters, REFERENCE_PARAMETERS, PARAMETER_TOLERANCE,
    MODEL_ARCHITECTURE,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def build(self, ckpt=False):
        return RCCNN(torch.linspace(0, 5000, 50),
                       ssp_mean=1500, ssp_std=35, bathy_mean=2500,
                       bathy_std=500, tl_mean=76, tl_std=19,
                       gradient_checkpointing=ckpt)

    def test_parameter_budget_and_architecture(self):
        m = self.build()
        n = count_parameters(m)
        self.assertEqual(n, 19_951_553)
        self.assertLess(abs(n - REFERENCE_PARAMETERS) / REFERENCE_PARAMETERS, PARAMETER_TOLERANCE)
        self.assertEqual(MODEL_ARCHITECTURE, "RC-CNN")
        self.assertEqual(m.encoder.fc[-1].out_features, 64)
        self.assertEqual(len(m.encoder.blocks), 4)
        self.assertEqual(len(m.decoder.deconvs), 4)
        self.assertFalse(any("attention" in name.lower() for name, _ in m.named_modules()))

    def test_continuous_bathy_features_not_binary(self):
        r=BathymetryRayFeatureGrid(bathy_mean=2000,bathy_std=500)
        bathy=torch.full((1,61,61),2200.0)
        inp=r(bathy)
        self.assertEqual(tuple(inp.shape),(1,9,3,200,250))
        self.assertTrue(torch.allclose(inp[0,:,0],torch.full((9,200,250),.4),atol=1e-5))
        self.assertAlmostEqual(float(inp[0,0,1,0,0]),0.0)
        self.assertAlmostEqual(float(inp[0,0,1,-1,0]),1.0)
        # Check a small terrain perturbation persists without hard thresholding.
        changed=bathy.clone();changed+=7.0
        inp2=r(changed)
        self.assertGreater(float((inp2[:, :, 0]-inp[:, :, 0]).abs().mean()), .01)

    def test_forward_nan_handling_and_db_conversion(self):
        m = self.build().eval()
        s = torch.full((1, 50), 1500.0)
        s[0, 38:] = float("nan")
        bathy = torch.full((1, 61, 61), 2200.0)
        with torch.no_grad():
            x = m(s, bathy)
            y = m.predict_db(s, bathy)
        self.assertEqual(tuple(x.shape), (1, 9, 200, 250))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(torch.allclose(y, x * 19 + 76, atol=1e-5))

    def test_backward_dense_loss_and_checkpointed_fc(self):
        from rc_cnn.metrics import dense_error_sums
        m = self.build(ckpt=True).train()
        s = torch.full((1, 50), 1510.0)
        s[:, 40:] = float("nan")
        bathy = torch.full((1, 61, 61), 2250.0)
        target = torch.zeros((1, 9, 200, 250))
        target[:, :, 100:] = 3.0
        pred = m(s, bathy)
        sq, _, n = dense_error_sums(pred, target)
        self.assertEqual(int(n), 9 * 200 * 250)
        (sq / n).backward()
        self.assertTrue(torch.isfinite(m.encoder.fc[0].weight.grad).all())
        self.assertTrue(torch.isfinite(m.decoder.fc[2].weight.grad).all())
        self.assertTrue(torch.isfinite(m.ssp_encoder.net[0].weight.grad).all())
        self.assertTrue(torch.isfinite(m.encoder.blocks[0][0].weight.grad).all())

    def test_nan_input_exception(self):
        m = self.build().eval()
        s = torch.full((1, 50), float("nan"))
        bathy = torch.full((1, 61, 61), 1500.0)
        with self.assertRaises(ValueError):
            m(s, bathy)


if __name__ == "__main__":
    unittest.main()
