# MAP-Net

Multi-Azimuth Path-Aware Network. The model uses environmental feature encoding (EFE), directional path-aware fusion (DPF), cross-azimuth interaction (CAI), and multi-scale TL reconstruction (MTR).

Default parameter count: 20,144,116. Model implementation: `map_net/model.py` (`MAPNet`).

The four architectural stages are described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup and data

Use Python 3.10+ and a PyTorch installation appropriate for the available CPU/CUDA environment.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See the [shared data specification](../README.md#dataset). Run commands from this model directory. Training and evaluation require `bottom_depth_index_r2.npy`.

## Training

Training objective: Water-column masked MSE (normalized TL). Validation and checkpoint selection use Water-column RMSE in dB. The default configuration is defined in `train.py`; the Slurm launcher specifies the two-GPU configuration. Adapt the environment setup in the launcher to the local cluster.

```bash
python train.py --epochs 150 --batch-size 4 --data-dir ../dataset
# Two GPUs (Slurm):
sbatch run-map-net-2gpu.sh
```


## Test evaluation

All five standalone evaluators calculate dB MAE/RMSE within the common water-column mask and export sample-level errors. The per-sample arrays can be averaged for the statistics reported in the manuscript; the aggregate metrics in JSON are grid-point-weighted and need not equal the sample mean.

```bash
python evaluate.py --split test --full-split
python evaluate.py --split test --subset --sample-ids 17424
```

The default checkpoint is `runs/map_net/best.pt`. To evaluate an existing compatible checkpoint, specify `--checkpoint /path/to/best.pt`. Only load checkpoints from trusted sources. For AGLC-Net, use `--reference-bank` if the matching reference bank is not located beside the checkpoint.

## Visualization

```bash
python plot_predicted_tl_9slices.py
python evaluate_single_3d_MAP_Net.py --sample-id 17424
python plot_MAP_Net_TL_range_curves.py
python benchmark_MAP_Net.py --help
```

Plotting scripts with configuration blocks can also be run from Spyder. The existing plotting coordinate conventions are retained; use the physical output-grid coordinates of the corresponding dataset when interpreting exported plots.
