# MAP-Net: architecture

Multi-Azimuth Path-Aware Network. The model uses environmental feature encoding (EFE), directional path-aware fusion (DPF), cross-azimuth interaction (CAI), and multi-scale TL reconstruction (MTR).

The network predicts `[B, 9, 200, 250]` normalized TL from SSP `[B, 50]` and bathymetry `[B, 61, 61]`. `predict_db()` converts the output to dB.

Core model: `map_net/model.py`.

## Functional stages

| Stage | Implementation | Output |
|---|---|---|
| EFE | `SSPEncoder`, `BathyEncoder` | SSP and multi-scale terrain features |
| DPF | `DirectionalPathEncoder`, `global_context`, `film` | Directional path features `[B,9,128,250]`, context and central bottom ray |
| CAI | `AngularInteraction`, `AngularLatentAttention` | Enhanced directional path features `[B,9,128,250]` |
| MTR | `context_proj`, `path_skip`, `start`, `base`, `up1`–`up3`, `tail` | Normalized TL `[B,9,200,250]` |

DPF samples three parallel paths for each azimuth. CAI combines periodic local convolution and multi-head latent attention with physical-angle rotary encoding. MTR shares its decoder across azimuths and uses context and bottom-ray geometry. Water-column supervision is implemented separately in `metrics.py` and `train.py`.
