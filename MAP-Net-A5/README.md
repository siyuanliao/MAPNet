# MAP-Net-A5 (w/o water mask)

The MAP-Net architecture and normalization statistics are unchanged. Full-grid MSE replaces masked MSE during training only; validation and final test metrics retain the water-column mask.

Default parameter count: 20,144,116. Model: `map_net_a5/model.py` (`MAPNetA5`). See [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup and data

Use Python 3.10+ and an appropriate PyTorch installation. Run commands from this model directory. Place the six `_r2.npy` files in the shared `../dataset/` directory (see [dataset specification](../README.md#dataset)).

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Training

Training objective: Full-grid MSE (normalized TL). Validation and checkpoint selection use water-column RMSE in dB. The original training configuration is retained. Normalization continues to use training-set water-column statistics; `--stats-from` can load the matching statistics file. Adjust the Slurm environment to the local cluster.

```bash
python train.py --epochs 150 --batch-size 4 --data-dir ../dataset
sbatch run-map-net-a5-2gpu.sh
```

## Test evaluation

The evaluator exports per-sample MAE/RMSE in dB using the common water-column mask. JSON aggregate errors are grid-point-weighted, whereas manuscript sample means are computed from the per-sample arrays.

```bash
python evaluate.py --split test --full-split
python evaluate.py --split test --subset --sample-ids 17424
```

The default checkpoint is `runs/map_net_a5/best.pt`. For existing state-dictionary weights, use `--checkpoint /path/to/best.pt` or copy the matching checkpoint into this directory. The historical model identity is accepted for compatibility. A5 checkpoints must contain the original full-grid training-objective metadata to distinguish them from full MAP-Net checkpoints. Only load trusted checkpoints.
