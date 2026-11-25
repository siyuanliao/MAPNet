# MAP-Net-A3 architecture

CAI is bypassed: the DPF output passes directly to MTR without angular convolution or latent attention.

| Stage | Implementation |
|---|---|
| EFE | SSP and bathymetry CNN encoders |
| DPF | Directional path encoder and FiLM |
| CAI | Identity bypass |
| MTR | Geometry-conditioned multiscale decoder |

Inputs: SSP `[B,50]`, bathymetry `[B,61,61]`. Output: normalized TL `[B,9,200,250]`; `predict_db()` converts to dB.

Trainable parameters: 20,061,426. Training objective: Water-column masked MSE. Validation and evaluation use the water-column mask.
