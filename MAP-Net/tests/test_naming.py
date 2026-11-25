"""MAP-Net 网络命名、默认路径及Checkpoint元信息的一致性测试。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from map_net import MAPNet, MODEL_ARCHITECTURE, MODEL_FULL_NAME
from map_net.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT
from train import parse_args, save_checkpoint
from evaluate import CHECKPOINT as EVAL_CHECKPOINT, OUTPUT_DIR as EVAL_OUTPUT
from plot_predicted_tl_9slices import CHECKPOINT as PLOT_CHECKPOINT, OUTPUT_DIR as PLOT_OUTPUT


class MAPNetNamingTests(unittest.TestCase):
    def test_model_identifiers(self):
        self.assertEqual(MAPNet.__name__, "MAPNet")
        self.assertEqual(MODEL_ARCHITECTURE, "MAP-Net")
        self.assertEqual(MODEL_FULL_NAME, "Multi-Azimuth Path-Aware Network")

    def test_default_paths_and_scripts(self):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
        self.assertEqual(args.data_dir, DEFAULT_DATA_DIR)
        self.assertEqual(args.output_dir, DEFAULT_OUTPUT_DIR)
        self.assertEqual(DEFAULT_OUTPUT_DIR.name, "map_net")
        self.assertEqual(DEFAULT_CHECKPOINT, EVAL_CHECKPOINT)
        self.assertEqual(DEFAULT_CHECKPOINT, PLOT_CHECKPOINT)
        self.assertEqual(EVAL_OUTPUT, DEFAULT_OUTPUT_DIR / "evaluation")
        self.assertEqual(PLOT_OUTPUT, DEFAULT_OUTPUT_DIR / "single_sample_tl_slices")
        scripts = DEFAULT_OUTPUT_DIR.parent.parent
        self.assertTrue((scripts / "run-map-net-1gpu.sh").is_file())
        self.assertTrue((scripts / "run-map-net-2gpu.sh").is_file())

    def test_saved_checkpoint_has_new_name(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(
                path, epoch=0, model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, stats={"tl_mean": 75, "tl_std": 20},
                best_rmse=1.0, bad_epochs=0,
                args=SimpleNamespace(epochs=1), val_metrics={},
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(state["model_name"], MODEL_ARCHITECTURE)
        self.assertEqual(state["model_full_name"], MODEL_FULL_NAME)


if __name__ == "__main__":
    unittest.main()
