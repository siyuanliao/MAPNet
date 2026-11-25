# DD-PUNet: architecture

Shared U-Net encoder, gated decoupled attention, and separate structure/detail decoders. The sum of decoder outputs is supervised. The SSP and bathymetry are rasterized along each propagation bearing.

The network predicts `[B, 9, 200, 250]` normalized TL from SSP `[B, 50]` and bathymetry `[B, 61, 61]`. `predict_db()` converts the output to dB.

Core model: `dd_punet/model.py`.

## Reference

Sun et al. (2026), [Dual-decoder neural network based for end-to-end prediction of acoustic transmission loss in deep-sea environments](https://doi.org/10.1121/10.0043242).
