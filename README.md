# MAP-Net and underwater transmission-loss prediction baselines

PyTorch implementations for multi-azimuth underwater acoustic transmission loss (TL) prediction. The four baseline networks are adaptations to the common three-dimensional prediction task, rather than exact reproductions of the cited papers.

| Model | Python package | Training objective | Final test evaluation |
|---|---|---|---|
| [MAP-Net](MAP-Net/) | `map_net` | Water-column masked MSE | Water-column MAE/RMSE |
| [DD-PUNet](DD-PUNet/) | `dd_punet` | Full-grid MSE | Water-column MAE/RMSE |
| [AGLC-Net](AGLC-Net/) | `aglc_net` | Full-grid MSE (two stages) | Water-column MAE/RMSE |
| [MS-DUNet](MS-DUNet/) | `ms_dunet` | Full-grid MSE on finite TL data | Water-column MAE/RMSE |
| [RC-CNN](RC-CNN/) | `rc_cnn` | Full-grid MSE | Water-column MAE/RMSE |

## Ablation experiments

Standalone PyTorch implementations of the five ablation variants used in the MAP-Net study. Each experiment uses the same chronological data split and final water-column evaluation.

| Model | Removed or replaced component | Trainable parameters | Training objective |
|---|---|---:|---|
| [MAP-Net-A1](MAP-Net-A1/) | EFE | 19,415,267 | Water-column masked MSE |
| [MAP-Net-A2](MAP-Net-A2/) | DPF | 19,752,572 | Water-column masked MSE |
| [MAP-Net-A3](MAP-Net-A3/) | CAI | 20,061,426 | Water-column masked MSE |
| [MAP-Net-A4](MAP-Net-A4/) | MTR | 19,011,467 | Water-column masked MSE |
| [MAP-Net-A5](MAP-Net-A5/) | water mask | 20,144,116 | Full-grid MSE |

The five ablation projects are standalone and use the same dataset directory. A5 changes the training objective only. See individual READMEs for training and evaluation.

## Dataset

Place the NumPy files in `dataset/` alongside the ten model directories. Use the exact `_r2.npy` filenames for compatibility with all ten projects.

| File | Shape | Description |
|---|---|---|
| `ssp_r2.npy` | `[18876, 50]` | Source SSP (m/s) |
| `ssp_depth_m_r2.npy` | `[50]` | SSP depths (m) |
| `ssp_valid_count_r2.npy` | `[18876]` | Observed SSP layer counts |
| `bathymetry_r2.npy` | `[18876, 61, 61]` | Local water depth (m), positive downward |
| `inputs_r2.npy` | `[18876, 9, 200, 250]` | Integer TL in hundredths of a dB |
| `bottom_depth_index_r2.npy` | `[121, 9, 250]` | Inclusive bottom depth indices for the water-column mask |

Sample order is longitude, latitude, month, then year (longitude varies fastest). Source-location index is `sample_id % 121`. The nine azimuths are 0°, 40°, ..., 320°, clockwise from north. Training, validation, and test sets contain 15,972, 1,452, and 1,452 samples, respectively. TL is converted to dB by dividing the stored integer by 100. SSP may have missing deeper layers; the study's TL values are finite.

## Run a model

From the corresponding model directory, install dependencies, run the unit tests, train the model, and run `evaluate.py`. See each model README for the commands and options. AGLC-Net requires its two-stage training procedure and matching historical reference bank.

Training/validation metrics for the baselines cover the full grid of the study dataset. Use the standalone evaluators and their common water-column metrics for cross-model test comparison. Each evaluator exports aligned sample IDs and per-sample errors. Model checkpoints, datasets and run results are not included in this source archive.

## Existing checkpoints

Model parameter names and shapes are retained; the renamed code expects state-dictionary checkpoints from the corresponding architecture and training objective. Select an existing file using `--checkpoint`, or move the matching run directory into the new default `runs/<package>/` directory. For AGLC-Net, preserve the reference bank, normalization metadata, and stage checkpoints. Serialized complete Python model objects are not covered by this compatibility guarantee.

## Citation and license

Please cite the accompanying MAP-Net manuscript and the original baseline publications when using these implementations. Publication metadata and a repository license should be provided by the repository owner before public release.
