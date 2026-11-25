"""Tests for strict FULL-GRID training/evaluation; no seabed mask allowed."""
import math
import unittest
import torch
from dd_punet.losses import fullgrid_error_sums
from dd_punet.metrics import FullGridMetrics


class FullGridLossTests(unittest.TestCase):
    def test_every_cell_including_seabed_contributes(self):
        prediction=torch.zeros((1,9,200,250),requires_grad=True)
        target=torch.zeros_like(prediction)
        target[...,199,249]=10  # arbitrary deepest/farthest cell
        sse,sae,n=fullgrid_error_sums(prediction,target)
        self.assertEqual(n, 9*200*250)
        self.assertAlmostEqual(float(sse.detach()), 900.0)
        self.assertAlmostEqual(float(sae.detach()), 90.0)
        sse.backward()
        self.assertEqual(float(prediction.grad[...,199,249].min()),-20.)

    def test_nonfinite_is_rejected_not_silently_filtered(self):
        p=torch.zeros((1,1,200,250))
        t=p.clone();t[...,0,0]=float('nan')
        with self.assertRaises(ValueError):
            fullgrid_error_sums(p,t)
        with self.assertRaises(FloatingPointError):
            fullgrid_error_sums(torch.full_like(p,float('inf')),p)

    def test_full_grid_metrics_grouped_and_per_point(self):
        met=FullGridMetrics('cpu')
        target=torch.zeros((1,9,200,250))
        pred=target.clone()
        pred[0,0,199,249]=10
        met.add(pred,target)
        out=met.result()
        self.assertEqual(out['overall']['valid_points'],450000)
        self.assertAlmostEqual(out['overall']['mae_db'],10/450000)
        self.assertAlmostEqual(out['overall']['rmse_db'],math.sqrt(100/450000))
        self.assertEqual(out['per_angle']['0deg']['valid_points'],50000)
        self.assertEqual(out['per_angle']['40deg']['rmse_db'],0)
        self.assertGreater(out['per_depth_m']['3000-4000']['rmse_db'],0)
        self.assertGreater(out['per_range_km']['40-50']['rmse_db'],0)

    def test_metrics_reject_invalid_labels(self):
        met=FullGridMetrics('cpu')
        t=torch.zeros((1,9,200,250));t[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):
            met.add(torch.zeros_like(t),t)


if __name__=='__main__':
    unittest.main()
