# MS-DUNet

Five-level U-Net with adaptive multi-scale dilated convolution (dilation rates 1, 2, 3). Environmental features are mapped to a depth–range grid independently for each bearing.

Default parameter count: 20,109,645. Model implementation: `ms_dunet/model.py` (`MSDUNet`).

This is a three-dimensional task adaptation, not an exact reproduction of the original method.

Sun et al. (2025), [End-to-end underwater acoustic transmission loss prediction with adaptive multi-scale dilated network](https://doi.org/10.1121/10.0034857).

## Setup and data

Use Python 3.10+ and a PyTorch installation appropriate for the available CPU/CUDA environment.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See the [shared data specification](../README.md#dataset). Run commands from this model directory. Training does not require the bottom-depth-index file. The standalone `evaluate.py` does require it for the common water-column test protocol.

## Training

Training objective: Full-grid MSE for the finite TL dataset (normalized TL). Validation and checkpoint selection use Full-grid RMSE for the finite TL dataset in dB. The default configuration is defined in `train.py`; the Slurm launcher specifies the two-GPU configuration. Adapt the environment setup in the launcher to the local cluster.

```bash
python train.py --epochs 150 --batch-size 4 --data-dir ../dataset
# Two GPUs (Slurm):
sbatch run-ms-dunet-2gpu.sh
```


## Test evaluation

All five standalone evaluators calculate dB MAE/RMSE within the common water-column mask and export sample-level errors. The per-sample arrays can be averaged for the statistics reported in the manuscript; the aggregate metrics in JSON are grid-point-weighted and need not equal the sample mean.

```bash
python evaluate.py --split test --full-split
python evaluate.py --split test --subset --sample-ids 17424
```

The default checkpoint is `runs/ms_dunet/best.pt`. To evaluate an existing compatible checkpoint, specify `--checkpoint /path/to/best.pt`. Only load checkpoints from trusted sources. For AGLC-Net, use `--reference-bank` if the matching reference bank is not located beside the checkpoint.

## Visualization

```bash
# Use the NPZ exports from evaluate.py for visualization.
```

Plotting scripts with configuration blocks can also be run from Spyder. The existing plotting coordinate conventions are retained; use the physical output-grid coordinates of the corresponding dataset when interpreting exported plots.
