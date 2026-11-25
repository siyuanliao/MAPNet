# MAP-Net-A4 architecture

MTR is replaced by a shared per-azimuth MLP. Radial path features and bottom depths are pooled to 32 positions, concatenated with context and angle features, and mapped to a 200 by 250 TL section.

| Stage | Implementation |
|---|---|
| EFE | SSP and bathymetry CNN encoders |
| DPF | Directional path encoder and FiLM |
| CAI | Periodic angular convolution and latent attention |
| MTR | Shared per-azimuth MLP |

Inputs: SSP `[B,50]`, bathymetry `[B,61,61]`. Output: normalized TL `[B,9,200,250]`; `predict_db()` converts to dB.

Trainable parameters: 19,011,467. Training objective: Water-column masked MSE. Validation and evaluation use the water-column mask.
