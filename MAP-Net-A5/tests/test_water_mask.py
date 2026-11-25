import unittest
import torch
from map_net_a5.metrics import water_valid_mask, masked_error_sums, MaskedMetrics


class WaterMaskTests(unittest.TestCase):
    def test_source_id_and_bottom_are_used_exactly(self):
        truth = torch.ones(2, 9, 200, 250)

        ids = torch.tensor([121, 243])
        bottom = torch.full((121, 9, 250), -1, dtype=torch.int16)
        bottom[0] = 3
        bottom[1] = 10
        bottom[1, 2, 4] = 1
        truth[0, 0, 0, 0] = float("nan")
        truth[1, 0, 1, 1] = float("inf")
        valid = water_valid_mask(truth, ids, bottom)
        self.assertEqual(tuple(valid.shape), (2, 9, 200, 250))
        self.assertFalse(valid[0, 0, 0, 0].item())
        self.assertTrue(valid[0, 1, 3, 0].item())
        self.assertFalse(valid[0, 1, 4, 0].item())
        self.assertTrue(valid[1, 2, 1, 4].item())
        self.assertFalse(valid[1, 2, 2, 4].item())
        self.assertFalse(valid[1, 0, 1, 1].item())

        pred = torch.zeros_like(truth, requires_grad=True)
        sq, ab, count = masked_error_sums(pred, truth, valid)
        self.assertTrue(torch.isfinite(sq).item())
        self.assertEqual(float(sq.item()), float(count.item()))
        self.assertEqual(float(ab.item()), float(count.item()))
        sq.backward()
        self.assertEqual(pred.grad[0, 1, 4, 0].item(), 0.0)
        self.assertEqual(pred.grad[1, 2, 2, 4].item(), 0.0)
        self.assertEqual(pred.grad[0, 0, 0, 0].item(), 0.0)
        self.assertNotEqual(pred.grad[0, 1, 3, 0].item(), 0.0)

    def test_metrics_reject_nonfinite_valid_prediction(self):
        mask = torch.zeros((1, 9, 200, 250), dtype=torch.bool)
        mask[0, 0, 0, 0] = True
        pred = torch.zeros_like(mask, dtype=torch.float32)
        pred[0, 0, 0, 0] = float("nan")
        metrics = MaskedMetrics("cpu")
        metrics.add(pred, torch.zeros_like(pred), mask)
        with self.assertRaises(FloatingPointError):
            metrics.result()

    def test_metrics_db_are_accumulated_globally(self):
        mask = torch.zeros((1, 9, 200, 250), dtype=torch.bool)
        mask[0, 0, 0, :2] = True
        truth = torch.zeros((1, 9, 200, 250))
        pred = torch.zeros_like(truth)
        pred[0, 0, 0, :2] = torch.tensor([3.0, 4.0])
        metrics = MaskedMetrics("cpu")
        metrics.add(pred, truth, mask)
        result = metrics.result()["overall"]
        self.assertEqual(result["valid_points"], 2)
        self.assertAlmostEqual(result["mae_db"], 3.5)
        self.assertAlmostEqual(result["rmse_db"], (12.5) ** .5)


if __name__ == "__main__":
    unittest.main()
