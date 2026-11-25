#!/usr/bin/env python3
"""Report model dimensions and param count; does not require real data."""
import torch
from rc_cnn.model import (
    RCCNN, count_parameters, REFERENCE_PARAMETERS, PARAMETER_TOLERANCE,
)

torch.set_num_threads(min(torch.get_num_threads(), 2))
model = RCCNN(
    torch.linspace(0, 5000, 50),
    ssp_mean=1500.0, ssp_std=30.0,
    bathy_mean=2500.0, bathy_std=1000.0,
    tl_mean=75.0, tl_std=20.0,
)
model.eval()
n = count_parameters(model)
print(f"RC-CNN: {n:,} trainable parameters")
print(f"MAP-Net: {REFERENCE_PARAMETERS:,} trainable parameters")
print(f"Difference: {n - REFERENCE_PARAMETERS:,} / "
      f"{(n / REFERENCE_PARAMETERS - 1) * 100:.4f}%")
assert abs(n / REFERENCE_PARAMETERS - 1) <= PARAMETER_TOLERANCE
with torch.no_grad():
    ssp = torch.full((1, 50), 1500.0)
    ssp[0, 40:] = float('nan')
    bathy = torch.full((1, 61, 61), 2300.0)
    y = model(ssp, bathy)
print(f"Input: SSP={tuple(ssp.shape)}, bathy={tuple(bathy.shape)}")
print(f"Output normalized TL={tuple(y.shape)}, finite={bool(torch.isfinite(y).all())}")
