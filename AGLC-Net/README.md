# AGLC-Net

TL autoencoder pretraining followed by latent-space correction, using a training-only historical TL/SSP reference bank for 121 source locations. The reference bank is required at inference; arbitrary unseen source locations are not supported. Parameter count includes the frozen encoder but excludes the reference bank.

Default parameter count: 20,130,731. Model implementation: `aglc_net/model.py` (`AGLCNet`).

This is a three-dimensional task adaptation, not an exact reproduction of the original method.

Huo et al. (2026), [Latent-space correction for transmission loss prediction in deep water under sound speed profile perturbations](https://doi.org/10.1121/10.0047248).

## Setup and data

Use Python 3.10+ and a PyTorch installation appropriate for the available CPU/CUDA environment.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See the [shared data specification](../README.md#dataset). Run commands from this model directory. Training does not require the bottom-depth-index file. The standalone `evaluate.py` does require it for the common water-column test protocol.

## Training

Training objective: Full-grid MSE (normalized TL). Validation and checkpoint selection use Full-grid RMSE in dB. The default configuration is defined in `train.py`; the Slurm launcher specifies the two-GPU configuration. Adapt the environment setup in the launcher to the local cluster.

```bash
python train.py --stage pretrain --epochs 50 --batch-size 4 --data-dir ../dataset
python train.py --stage correct --epochs 150 --batch-size 4 --decoder-freeze-epochs 20 --data-dir ../dataset
# Two GPUs (Slurm):
sbatch run-aglc-net-2gpu.sh
```

The Stage II `best.pt` must be used with the matching `reference_bank.npz`. The evaluator checks the reference-bank SHA-256 hash. The supplied two-stage launcher uses 50 pretraining epochs and 150 correction epochs; the decoder is frozen for the first 20 correction epochs.

## Test evaluation

All five standalone evaluators calculate dB MAE/RMSE within the common water-column mask and export sample-level errors. The per-sample arrays can be averaged for the statistics reported in the manuscript; the aggregate metrics in JSON are grid-point-weighted and need not equal the sample mean.

```bash
python evaluate.py --split test --full-split
python evaluate.py --split test --subset --sample-ids 17424
```

The default checkpoint is `runs/aglc_net/best.pt`. To evaluate an existing compatible checkpoint, specify `--checkpoint /path/to/best.pt`. Only load checkpoints from trusted sources. For AGLC-Net, use `--reference-bank` if the matching reference bank is not located beside the checkpoint.
