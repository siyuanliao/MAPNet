import unittest
import torch
from rc_cnn.metrics import dense_error_sums,DenseMetrics

class DenseTests(unittest.TestCase):
    def test_every_pixel_contributes_to_loss_and_gradient(self):
        # Construct one arbitrary 'below-seabed' cell; still participates.
        pred=torch.zeros((1,9,200,250),requires_grad=True)
        target=torch.zeros_like(pred)
        target=target.detach();target[0,3,199,249]=5.0
        sq,ab,n=dense_error_sums(pred,target)
        self.assertEqual(n,450000)
        self.assertAlmostEqual(float(sq.detach()),25.0)
        self.assertAlmostEqual(float(ab),5.0)
        sq.backward()
        self.assertAlmostEqual(float(pred.grad[0,3,199,249]),-10.0)

    def test_metrics_all_pixels(self):
        m=DenseMetrics('cpu')
        t=torch.zeros((1,9,200,250))
        p=t.clone();p[0,0,0,0]=4.0
        m.add(p,t)
        result=m.result()
        self.assertEqual(result['overall']['evaluated_points'],450000)
        self.assertAlmostEqual(result['overall']['rmse_db'],4.0/(450000**.5))

    def test_reject_nan_target_instead_of_masking(self):
        t=torch.zeros((1,9,200,250));t[0,0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            dense_error_sums(torch.zeros_like(t),t)
