# MAP-Net-A2 architecture

DPF is replaced by a minimal global-context and coordinate adapter. Center-ray bathymetry sampling supplies the decoder geometry; no learned directional terrain sampling or FiLM is used.

| Stage | Implementation |
|---|---|
| EFE | SSP and bathymetry CNN encoders |
| DPF | Minimal context/coordinate adapter and center-ray depth sampling |
| CAI | Periodic angular convolution and latent attention |
| MTR | Geometry-conditioned multiscale decoder |

Inputs: SSP `[B,50]`, bathymetry `[B,61,61]`. Output: normalized TL `[B,9,200,250]`; `predict_db()` converts to dB.

Trainable parameters: 19,752,572. Training objective: Water-column masked MSE. Validation and evaluation use the water-column mask.
