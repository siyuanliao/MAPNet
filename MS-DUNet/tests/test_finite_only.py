"""No ocean-bottom masking is used; finite TL remains measurable at any depth."""
import unittest
import torch
from ms_dunet.metrics import finite_target_mask, finite_error_sums, TLMetrics


class FiniteOnlyTests(unittest.TestCase):
    def test_finite_truth_includes_all_depths(self):
        truth = torch.ones(2, 9, 200, 250)
        truth[0, 0, 0, 0] = float("nan")
        truth[1, 0, 1, 1] = float("inf")
        finite = finite_target_mask(truth)
        self.assertEqual(tuple(finite.shape), (2, 9, 200, 250))
        self.assertEqual(int(finite.sum()), 2 * 9 * 200 * 250 - 2)
        self.assertTrue(bool(finite[0, 0, 199, 0]))
        self.assertTrue(bool(finite[1, 8, 199, 249]))
        pred = torch.zeros_like(truth, requires_grad=True)
        sq, ab, count = finite_error_sums(pred, truth, finite)
        self.assertEqual(float(sq.detach()), float(count))
        self.assertEqual(float(ab), float(count))
        sq.backward()
        self.assertNotEqual(float(pred.grad[0, 0, 199, 0]), 0.0)
        self.assertEqual(float(pred.grad[0, 0, 0, 0]), 0.0)

    def test_nonfinite_prediction_at_finite_truth_is_error(self):
        truth = torch.zeros((1, 9, 200, 250))
        pred = torch.zeros_like(truth)
        pred[0, 0, 199, 249] = float("nan")
        metrics = TLMetrics("cpu")
        metrics.add(pred, truth, finite_target_mask(truth))
        with self.assertRaises(FloatingPointError):
            metrics.result()

    def test_global_dB_metrics_and_nonfinite_truth(self):
        truth = torch.full((1, 9, 200, 250), float("nan"))
        truth[0, 0, 199, :2] = 0.0
        pred = torch.zeros_like(truth)
        pred[0, 0, 199, :2] = torch.tensor([3.0, 4.0])
        metrics = TLMetrics("cpu")
        metrics.add(pred, truth, finite_target_mask(truth))
        result = metrics.result()["overall"]
        self.assertEqual(result["valid_points"], 2)
        self.assertAlmostEqual(result["mae_db"], 3.5)
        self.assertAlmostEqual(result["rmse_db"], 12.5**0.5)


if __name__ == "__main__":
    unittest.main()
