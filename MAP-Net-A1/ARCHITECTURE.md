# MAP-Net-A1 architecture

MLP-based SSP and bathymetry encoders replace EFE; fixed pooling and pointwise linear layers supply the three spatial scales for DPF.

| Stage | Implementation |
|---|---|
| EFE | SSP and bathymetry MLP encoders |
| DPF | Directional path encoder and FiLM |
| CAI | Periodic angular convolution and latent attention |
| MTR | Geometry-conditioned multiscale decoder |

Inputs: SSP `[B,50]`, bathymetry `[B,61,61]`. Output: normalized TL `[B,9,200,250]`; `predict_db()` converts to dB.

Trainable parameters: 19,415,267. Training objective: Water-column masked MSE. Validation and evaluation use the water-column mask.
