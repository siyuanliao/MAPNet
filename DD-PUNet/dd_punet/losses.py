"""Unmasked full-grid TL objective for DD-PUNet.

The structure/detail decoders remain part of the *network*, but training
compares their SUM to the true normalized TL at every grid cell. No masks,
weighting or ignored voxels are used.
"""
from __future__ import annotations
import torch


def fullgrid_error_sums(pred_norm, target_norm):
    """Return (SSE with gradient, absolute-error sum detached, cell count).

    Both tensors must represent the same physical grid. The scalar objective
    is SSE/cell_count, i.e. normalized full-grid MSE.
    """
    if pred_norm.shape != target_norm.shape or pred_norm.ndim != 4:
        raise ValueError('Expected matching [B,A,200,250] tensors')
    if not bool(torch.isfinite(target_norm).all()):
        raise ValueError('Unmasked training requires finite target TL everywhere')
    if not bool(torch.isfinite(pred_norm).all()):
        raise FloatingPointError('Network prediction contains NaN/Inf')
    error = pred_norm.float() - target_norm.float()
    return error.square().sum(), error.detach().abs().sum(), target_norm.numel()
