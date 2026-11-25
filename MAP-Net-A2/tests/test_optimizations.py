"""优化后训练归一化、误差分组与权重保存测试。"""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from map_net_a2.metrics import MaskedMetrics
from train import parse_args, train_one_epoch, update_best_checkpoint


class ToyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        ssp = torch.zeros(50)
        bathy = torch.zeros(61, 61)
        tl = torch.full((9, 200, 250), float(index))
        return ssp, bathy, tl, torch.tensor(index)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.level = nn.Parameter(torch.tensor(0.1))

    def forward(self, ssp, bathy):
        return self.level.expand(ssp.shape[0], 9, 200, 250)


class OptimizationTests(unittest.TestCase):
    def test_default_training_batch(self):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.grad_accum, 1)

    def test_exact_masked_gradient_for_accumulation(self):
        bottom = torch.full((121, 9, 250), -1, dtype=torch.int16)
        bottom[0] = 9
        bottom[1] = 19
        stats = {"tl_mean": 0.0, "tl_std": 1.0}
        expected = 0.1 - 0.1 * 2 * (0.1 - 20 / 30)
        for batch_size, accumulate in ((2, 1), (1, 2)):
            model = ToyModel()
            loader = DataLoader(ToyDataset(), batch_size=batch_size)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            args = SimpleNamespace(grad_accum=accumulate, amp="off", grad_clip=0)
            result = train_one_epoch(
                model, loader, opt, scaler, bottom, stats, args, torch.device("cpu"), 1
            )
            self.assertAlmostEqual(model.level.item(), expected, places=5)
            self.assertEqual(result["valid_points"], 9 * 250 * 30)

    def test_distance_and_depth_bins(self):
        torch.manual_seed(12)
        truth = torch.randn(2, 9, 200, 250)
        prediction = truth + torch.randn_like(truth) * 2.0
        mask = torch.rand_like(truth) > 0.75
        metrics = MaskedMetrics("cpu")
        metrics.add(prediction, truth, mask)
        result = metrics.result()
        for k in range(5):
            part = mask[..., 50*k:50*(k+1)]
            err = (prediction - truth)[..., 50*k:50*(k+1)][part]
            row = result["per_range_km"][f"{k*10}-{(k+1)*10}"]
            self.assertEqual(row["valid_points"], part.sum().item())
            self.assertAlmostEqual(row["rmse_db"], math.sqrt(err.double().square().mean().item()), places=5)
        for k in range(4):
            part = mask[:, :, 50*k:50*(k+1), :]
            err = (prediction - truth)[:, :, 50*k:50*(k+1), :][part]
            row = result["per_depth_m"][f"{k*1000}-{(k+1)*1000}"]
            self.assertEqual(row["valid_points"], part.sum().item())
            self.assertAlmostEqual(row["mae_db"], err.double().abs().mean().item(), places=5)

    def test_best_checkpoint_is_independent_of_future_last(self):
        with tempfile.TemporaryDirectory() as temp:
            last = Path(temp) / "last.pt"
            best = Path(temp) / "best.pt"
            last.write_bytes(b"initial")
            update_best_checkpoint(last, best)
            self.assertEqual(best.read_bytes(), b"initial")
            tmp = Path(temp) / "last.pt.tmp"
            tmp.write_bytes(b"newer")
            tmp.replace(last)
            self.assertEqual(best.read_bytes(), b"initial")
            self.assertEqual(last.read_bytes(), b"newer")


if __name__ == "__main__":
    unittest.main()
