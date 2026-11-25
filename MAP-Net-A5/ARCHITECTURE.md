# MAP-Net-A5 architecture

The MAP-Net architecture and normalization statistics are unchanged. Full-grid MSE replaces masked MSE during training only; validation and final test metrics retain the water-column mask.

| Stage | Implementation |
|---|---|
| EFE | SSP and bathymetry CNN encoders |
| DPF | Directional path encoder and FiLM |
| CAI | Periodic angular convolution and latent attention |
| MTR | Geometry-conditioned multiscale decoder |

Inputs: SSP `[B,50]`, bathymetry `[B,61,61]`. Output: normalized TL `[B,9,200,250]`; `predict_db()` converts to dB.

Trainable parameters: 20,144,116. Training objective: Full-grid MSE. Validation and evaluation use the water-column mask.
