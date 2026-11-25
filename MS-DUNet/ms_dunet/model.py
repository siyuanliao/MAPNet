"""MS-DUNet: environment-adapted U-Net baseline for 9-azimuth TL.

Paper backbone: 5-level double-3x3-conv U-Net + encoder/decoder skip
connections + adaptive multi-scale dilated modules (dilation 1/2/3).
Adaptation (NOT in the original paper): map bathymetry and SSP onto
per-azimuth depth-range environment images and reconstruct 200x250 TL.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager, nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_ARCHITECTURE = "MS-DUNet"
# Compatible identifier in checkpoints saved before the public release.
CHECKPOINT_MODEL_NAMES = (MODEL_ARCHITECTURE, "MultiScale-DUNet-TL")
REFERENCE_PARAMETERS = 20_144_116
PARAMETER_TOLERANCE = 0.10
# Widths tuned for a comparable *trainable parameter count*, not performance.
MODEL_CONFIG = {
    "encoder_channels": (64, 128, 256, 448, 984),
    "bathy_features": 24,
    "input_grid": (50, 63),
    "output_grid": (200, 250),
    "angle_chunk": 3,
}


class DoubleConv(nn.Sequential):
    """Paper-style: 3x3 Conv-BN-ReLU repeated twice."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class AdaptiveMultiScaleDilated(nn.Module):
    """Three branches d={1,2,3}; residual per branch; learned weighted sum.

    This preserves the mechanism in Fig.3 of Sun et al. (JASA 2025).
    Softmax-constrained branch weights are a stable implementation choice.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        branch_width = max(16, out_channels // 4)
        self.dilations = (1, 2, 3)
        self.projections = nn.ModuleList()
        self.dilated = nn.ModuleList()
        for d in self.dilations:
            self.projections.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_width, 1, bias=False),
                nn.BatchNorm2d(branch_width),
                nn.ReLU(inplace=True),
            ))
            self.dilated.append(nn.Sequential(
                nn.Conv2d(branch_width, branch_width, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_width),
                nn.ReLU(inplace=True),
            ))
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.fuse = nn.Conv2d(branch_width, out_channels, 1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        weights = torch.softmax(self.branch_logits, dim=0)
        result = None
        for i in range(3):
            y = self.projections[i](x)
            y = y + self.dilated[i](y)
            weighted = weights[i] * y
            result = weighted if result is None else result + weighted
        return self.activation(self.fuse_bn(self.fuse(result)))


class UNetUp(nn.Module):
    """Transposed upsampling -> encoder skip concat -> adapted dilated block."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.dilated_fuse = AdaptiveMultiScaleDilated(
            out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.dilated_fuse(torch.cat((x, skip), dim=1))


class FineUp(nn.Module):
    """Two economical final resolution stages, outside the 5-level U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.layers(x)


class EnvironmentRasterizer(nn.Module):
    """Map [B,50] and [B,61,61] to a 2D depth-range input per azimuth.

    An intentionally small environmental adapter replaces the paper's source
    depth/frequency sparse embedding; the core predictor remains a U-Net.
    """
    def __init__(self, ssp_depth_m, *, ssp_mean, ssp_std, bathy_mean, bathy_std,
                 z_max_m=4000.0, r_max_m=50000.0, bathy_features=24,
                 grid_size=(50, 63)):
        super().__init__()
        depth = torch.as_tensor(ssp_depth_m, dtype=torch.float32).flatten()
        if depth.numel() != 50 or not bool(torch.isfinite(depth).all()):
            raise ValueError("ssp_depth_m should contain 50 finite depths")
        if not bool((depth[1:] > depth[:-1]).all()):
            raise ValueError("ssp_depth_m must be strictly increasing")
        if ssp_std <= 0 or bathy_std <= 0:
            raise ValueError("input std must be positive")
        self.register_buffer("ssp_depth_m", depth)
        self.register_buffer("ssp_mean", torch.tensor(float(ssp_mean)))
        self.register_buffer("ssp_std", torch.tensor(float(ssp_std)))
        self.register_buffer("bathy_mean", torch.tensor(float(bathy_mean)))
        self.register_buffer("bathy_std", torch.tensor(float(bathy_std)))
        z_count, r_count = grid_size
        target_z = torch.linspace(0, float(z_max_m), z_count)
        idx_hi = torch.searchsorted(depth.contiguous(), target_z.contiguous()).clamp(1, 49)
        idx_lo = idx_hi - 1
        ratio = ((target_z - depth[idx_lo]) /
                 (depth[idx_hi] - depth[idx_lo]).clamp_min(1e-6)).clamp(0, 1)
        self.register_buffer("idx_lo", idx_lo)
        self.register_buffer("idx_hi", idx_hi)
        self.register_buffer("interp_ratio", ratio)
        self.register_buffer("target_z", target_z)
        self.register_buffer("z_coord", torch.linspace(0, 1, z_count))
        self.register_buffer("r_coord", torch.linspace(0, 1, r_count))
        angle = torch.arange(9, dtype=torch.float32) * (40 * torch.pi / 180.0)
        self.register_buffer("angles", angle)
        radius = torch.linspace(0, float(r_max_m), r_count)
        east = radius[None, :] * angle.sin()[:, None]
        north = radius[None, :] * angle.cos()[:, None]
        # F.grid_sample: x=east (columns), y=north (rows).
        grid = torch.stack((east / 60000.0, north / 60000.0), dim=-1)
        if not bool((grid.abs() <= 1).all()):
            raise ValueError("azimuth rays leave the +/-60 km bathymetry patch")
        self.register_buffer("ray_grid", grid)
        self.bathy_context = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, bathy_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(bathy_features), nn.ReLU(inplace=True),
            nn.Conv2d(bathy_features, bathy_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(bathy_features), nn.ReLU(inplace=True),
        )
        # [bathy CNN features, seafloor depth, SSP value, SSP validity,
        #  z, r, bottom-minus-z, sin(angle), cos(angle)]
        self.channels = bathy_features + 8
        self.grid_size = (z_count, r_count)

    def _ssp_on_grid(self, ssp):
        finite = torch.isfinite(ssp)
        if not bool(finite[:, 0].all()):
            raise ValueError("each SSP must have a finite surface layer")
        # Bottom truncation: hold the last valid value without treating it as
        # an observed water-column value at deeper coordinates.
        idx = torch.arange(50, device=ssp.device).view(1, 50).expand_as(ssp)
        prev = torch.where(finite, idx, -1).cummax(dim=1).values.clamp_min(0)
        profile = torch.where(finite, ssp, 0.0).gather(1, prev)
        v0, v1 = profile[:, self.idx_lo], profile[:, self.idx_hi]
        speed = v0 + (v1 - v0) * self.interp_ratio[None, :]
        norm_speed = (speed - self.ssp_mean) / self.ssp_std
        last_idx = torch.where(finite, idx, -1).max(dim=1).values
        last_z = self.ssp_depth_m[last_idx]
        is_valid_depth = self.target_z[None, :] <= last_z[:, None]
        return norm_speed, is_valid_depth.to(dtype=ssp.dtype)

    def prepare(self, ssp, bathy):
        """Compute shared environmental features only once for all 9 angles."""
        if ssp.ndim != 2 or ssp.shape[-1] != 50:
            raise ValueError("ssp must be [B,50]")
        if bathy.ndim != 3 or bathy.shape[-2:] != (61, 61):
            raise ValueError("bathy must be [B,61,61]")
        if ssp.shape[0] != bathy.shape[0]:
            raise ValueError("input batch sizes differ")
        if not bool(torch.isfinite(bathy).all()):
            raise ValueError("bathy must be finite")
        profile, valid = self._ssp_on_grid(ssp)
        terrain = (bathy[:, None] - self.bathy_mean) / self.bathy_std
        terrain_features = self.bathy_context(terrain)
        return profile, valid, terrain_features, bathy

    def rays(self, prepared, angle_start, angle_end):
        profile, valid, terrain_features, bathy = prepared
        batch = bathy.shape[0]
        angles = angle_end - angle_start
        z_count, r_count = self.grid_size
        # Grid shape [B, angle_count, range_count, xy]
        grid = self.ray_grid[angle_start:angle_end].unsqueeze(0).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(terrain_features, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        bottom = F.grid_sample(bathy[:, None], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        sampled = sampled.permute(0, 2, 1, 3).reshape(
            batch * angles, sampled.shape[1], 1, r_count).expand(-1, -1, z_count, -1)
        bottom = bottom.permute(0, 2, 1, 3).reshape(batch * angles, 1, 1, r_count)
        # Use the 0–4000 m and 0–50000 m coordinate convention.
        bottom_norm = (bottom / 4000.0).expand(-1, 1, z_count, -1)
        zp = self.z_coord.view(1, 1, z_count, 1)
        rp = self.r_coord.view(1, 1, 1, r_count)
        z = zp.expand(batch * angles, 1, z_count, r_count)
        r = rp.expand(batch * angles, 1, z_count, r_count)
        s = profile[:, None, :].expand(batch, angles, z_count).reshape(
            batch * angles, 1, z_count, 1).expand(-1, 1, -1, r_count)
        v = valid[:, None, :].expand(batch, angles, z_count).reshape(
            batch * angles, 1, z_count, 1).expand(-1, 1, -1, r_count)
        ang = self.angles[angle_start:angle_end]
        sn = ang.sin().view(1, angles, 1, 1, 1).expand(batch, -1, 1, z_count, r_count).reshape(
            batch * angles, 1, z_count, r_count)
        cs = ang.cos().view(1, angles, 1, 1, 1).expand(batch, -1, 1, z_count, r_count).reshape(
            batch * angles, 1, z_count, r_count)
        return torch.cat((sampled, bottom_norm, s, v, z, r, bottom_norm - z, sn, cs), dim=1)


@contextmanager
def _bn_without_running_stats_update(module):
    """Keep BN's identical autograd graph during checkpoint replay.

    Setting momentum=0 keeps the same training-batch normalization and saved
    tensors, but prevents a second update of running mean/variance. Restore
    num_batches_tracked afterward as replay still increments it.
    """
    layers = [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]
    originals = [(m.momentum, m.num_batches_tracked.clone()
                  if m.num_batches_tracked is not None else None) for m in layers]
    try:
        for m in layers:
            m.momentum = 0.0
        yield
    finally:
        for m, (momentum, count) in zip(layers, originals):
            m.momentum = momentum
            if count is not None:
                m.num_batches_tracked.copy_(count)


class MSDUNet(nn.Module):
    """Input: [B,50], [B,61,61]. Output: normalized [B,9,200,250]."""
    def __init__(self, ssp_depth_m, *, ssp_mean, ssp_std, bathy_mean, bathy_std,
                 tl_mean, tl_std, gradient_checkpointing=False, angle_chunk=None):
        super().__init__()
        if tl_std <= 0:
            raise ValueError("tl_std must be positive")
        self.register_buffer("tl_mean", torch.tensor(float(tl_mean)))
        self.register_buffer("tl_std", torch.tensor(float(tl_std)))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.angle_chunk = MODEL_CONFIG["angle_chunk"] if angle_chunk is None else int(angle_chunk)
        if not 1 <= self.angle_chunk <= 9:
            raise ValueError("angle_chunk must be within [1,9]")
        self.adapter = EnvironmentRasterizer(
            ssp_depth_m, ssp_mean=ssp_mean, ssp_std=ssp_std,
            bathy_mean=bathy_mean, bathy_std=bathy_std,
            grid_size=MODEL_CONFIG["input_grid"],
            bathy_features=MODEL_CONFIG["bathy_features"],
        )
        c0, c1, c2, c3, c4 = MODEL_CONFIG["encoder_channels"]
        self.encoders = nn.ModuleList([
            DoubleConv(self.adapter.channels, c0),
            DoubleConv(c0, c1), DoubleConv(c1, c2),
            DoubleConv(c2, c3), DoubleConv(c3, c4),
        ])
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.decoders = nn.ModuleList([
            UNetUp(c4, c3, c3), UNetUp(c3, c2, c2),
            UNetUp(c2, c1, c1), UNetUp(c1, c0, c0),
        ])
        self.fine1 = FineUp(c0, 32)
        self.fine2 = FineUp(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def _block(self, mod, *inputs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                mod, *inputs, use_reentrant=False,
                context_fn=lambda: (nullcontext(), _bn_without_running_stats_update(mod)),
            )
        return mod(*inputs)

    def _single_chunk(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = self._block(enc, x)
            skips.append(x)
            if i < len(self.encoders) - 1:
                x = self.pool(x)
        for stage, skip in zip(self.decoders, skips[-2::-1]):
            x = self._block(stage, x, skip)
        x = self._block(self.fine1, x, (100, 125))
        x = self._block(self.fine2, x, MODEL_CONFIG["output_grid"])
        return self.head(x)

    def forward(self, ssp, bathy):
        batch = ssp.shape[0]
        prepared = self.adapter.prepare(ssp, bathy)
        chunks = []
        for a0 in range(0, 9, self.angle_chunk):
            a1 = min(9, a0 + self.angle_chunk)
            raster = self.adapter.rays(prepared, a0, a1)
            prediction = self._single_chunk(raster)
            chunks.append(prediction.reshape(batch, a1 - a0, 200, 250))
        return torch.cat(chunks, dim=1)

    def predict_db(self, ssp, bathy):
        return self.forward(ssp, bathy).float() * self.tl_std + self.tl_mean


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(data_dir, stats, *, gradient_checkpointing=False, angle_chunk=None):
    from .data import resolve_file
    depth = np.load(resolve_file(data_dir, "ssp_depth_m_r2.npy"), allow_pickle=False)
    return MSDUNet(
        depth, ssp_mean=stats["ssp_mean"], ssp_std=stats["ssp_std"],
        bathy_mean=stats["bathy_mean"], bathy_std=stats["bathy_std"],
        tl_mean=stats["tl_mean"], tl_std=stats["tl_std"],
        gradient_checkpointing=gradient_checkpointing, angle_chunk=angle_chunk,
    )


if __name__ == "__main__":
    model = MSDUNet(
        torch.linspace(0, 5000, 50), ssp_mean=1500, ssp_std=30,
        bathy_mean=2500, bathy_std=1000, tl_mean=75, tl_std=20)
    actual = count_parameters(model)
    print(f"{MODEL_ARCHITECTURE}: {actual:,} parameters; target={REFERENCE_PARAMETERS:,}; "
          f"relative_diff={100*(actual/REFERENCE_PARAMETERS-1):.3f}%")
