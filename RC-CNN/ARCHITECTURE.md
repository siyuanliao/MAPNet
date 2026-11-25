# RC-CNN: architecture

Four-stage convolutional encoder, 64-dimensional bottleneck, SSP conditioning, and fully connected expansion followed by a convolutional decoder. This adaptation does not implement replay-based continual learning.

The network predicts `[B, 9, 200, 250]` normalized TL from SSP `[B, 50]` and bathymetry `[B, 61, 61]`. `predict_db()` converts the output to dB.

Core model: `rc_cnn/model.py`.

## Reference

Deo et al. (2025), [Predicting transmission loss in underwater acoustics using continual learning with range-dependent conditional convolutional neural networks](https://doi.org/10.1121/10.0036773).
