# AGLC-Net: architecture

TL autoencoder pretraining followed by latent-space correction, using a training-only historical TL/SSP reference bank for 121 source locations. The reference bank is required at inference; arbitrary unseen source locations are not supported. Parameter count includes the frozen encoder but excludes the reference bank.

The network predicts `[B, 9, 200, 250]` normalized TL from SSP `[B, 50]` and bathymetry `[B, 61, 61]`. `predict_db()` converts the output to dB.

Core model: `aglc_net/model.py`.

## Reference

Huo et al. (2026), [Latent-space correction for transmission loss prediction in deep water under sound speed profile perturbations](https://doi.org/10.1121/10.0047248).
