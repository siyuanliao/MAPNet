# MAP-Net-A1 (w/o EFE)

MLP-based SSP and bathymetry encoders replace EFE; fixed pooling and pointwise linear layers supply the three spatial scales for DPF.

Default parameter count: 19,415,267. Model: `map_net_a1/model.py` (`MAPNetA1`). See [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup and data

Use Python 3.10+ and an appropriate PyTorch installation. Run commands from this model directory. Place the six `_r2.npy` files in the shared `../dataset/` directory (see [dataset specification](../README.md#dataset)).

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Training

Training objective: Water-column masked MSE (normalized TL). Validation and checkpoint selection use water-column RMSE in dB. The original training configuration is retained. Adjust the Slurm environment to the local cluster.

```bash
python train.py --epochs 150 --batch-size 4 --data-dir ../dataset
sbatch run-map-net-a1-2gpu.sh
```

## Test evaluation

The evaluator exports per-sample MAE/RMSE in dB using the common water-column mask. JSON aggregate errors are grid-point-weighted, whereas manuscript sample means are computed from the per-sample arrays.

```bash
python evaluate.py --split test --full-split
python evaluate.py --split test --subset --sample-ids 17424
```

The default checkpoint is `runs/map_net_a1/best.pt`. For existing state-dictionary weights, use `--checkpoint /path/to/best.pt` or copy the matching checkpoint into this directory. The historical model identity is accepted for compatibility. Only load trusted checkpoints.
