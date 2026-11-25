# MS-DUNet: architecture

Five-level U-Net with adaptive multi-scale dilated convolution (dilation rates 1, 2, 3). Environmental features are mapped to a depth–range grid independently for each bearing.

The network predicts `[B, 9, 200, 250]` normalized TL from SSP `[B, 50]` and bathymetry `[B, 61, 61]`. `predict_db()` converts the output to dB.

Core model: `ms_dunet/model.py`.

## Reference

Sun et al. (2025), [End-to-end underwater acoustic transmission loss prediction with adaptive multi-scale dilated network](https://doi.org/10.1121/10.0034857).
