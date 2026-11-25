"""Paper-inspired DualDecoder-PUNet adapted to 9 azimuth TL maps.

Sun et al., JASA 2026 (DOI: 10.1121/10.0043242): shared U-Net encoder,
GDAM channel attention + 5x5/3x3 branches, and TWO independently parameterized
structure/detail U-Net decoders. The original paper does not condition on a
changing bathymetry tensor; ray rasterization is the minimum input adaptation.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ARCHITECTURE = "DD-PUNet"
# Compatible identifier in checkpoints saved before the public release.
CHECKPOINT_MODEL_NAMES = (MODEL_ARCHITECTURE, "DualDecoder-PUNet-3D-Comparable")
REFERENCE_PARAMETERS = 20_144_116
PARAMETER_TOLERANCE = 0.10
N_ANGLES, N_DEPTHS, N_RANGES = 9, 200, 250


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def double_conv(cin: int, cout: int) -> nn.Module:
    # GroupNorm avoids dependence on the chosen (possibly small) angle chunk.
    # The source article does not give norm details for every double-conv stage.
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout), nn.ReLU(inplace=True),
    )


class InputRasterizer(nn.Module):
    """Create per-azimuth range-depth image from SSP and local bathymetry.

    Feature channels are (1) normalized vertically-interpolated SSP,
    (2) depth-valid SSP coverage, (3) seabed occupancy, (4) normalized
    distance to local seafloor, (5) normalized horizontal range.
    This is NON-LEARNED input adaptation and is not the authoritative loss mask.
    """
    def __init__(self, depth_m, *, max_range_m=50_000.0,
                 max_depth_m=4_000.0, bathy_half_width_m=60_000.0):
        super().__init__()
        depth = torch.as_tensor(depth_m, dtype=torch.float32).flatten()
        if depth.shape != (50,) or not bool(torch.isfinite(depth).all()) or not bool((depth[1:] > depth[:-1]).all()):
            raise ValueError("ssp_depth_m must contain 50 finite increasing depths")
        if max_range_m <= 0 or max_range_m > bathy_half_width_m or max_depth_m <= 0:
            raise ValueError("max_range_m must be within bathymetry coverage, max_depth_m > 0")
        self.register_buffer("ssp_depth_m", depth)
        z = torch.linspace(0., float(max_depth_m), N_DEPTHS)
        r = torch.linspace(0., float(max_range_m), N_RANGES)
        self.register_buffer("receiver_depth_m", z)
        self.register_buffer("receiver_range_m", r)
        theta = torch.arange(N_ANGLES).float() * (40.0 * math.pi / 180.0)
        east = torch.sin(theta)[:, None] * r[None, :]
        north = torch.cos(theta)[:, None] * r[None, :]
        # bathymetry axis 1 is south -> north, axis 2 west -> east.
        grid = torch.stack((east / bathy_half_width_m, north / bathy_half_width_m), -1)
        if not bool((grid.abs() <= 1.000001).all()):
            raise ValueError("Ray extends beyond 61x61 bathymetry map")
        self.register_buffer("ray_grid", grid)  # [9,250,2]
        self.max_depth_m = float(max_depth_m)
        self.max_range_m = float(max_range_m)

    def _ssp_depth_columns(self, ssp, ssp_mean, ssp_std):
        """Linear interpolation on the true 50 physical depth coordinates.

        Internal NaNs are forward-filled only for interpolation; invalid
        layers are marked in the accompanying validity channel. Deeper than
        the final observed depth, hold the last valid speed and set valid=0.
        """
        b = ssp.shape[0]
        good = torch.isfinite(ssp)
        if not bool(good.any(1).all()):
            raise ValueError("SSP must have at least one valid depth per sample")
        positions = torch.arange(50, device=ssp.device)[None, :].expand(b, -1)
        # Forward fill NaN safely. Leading invalid samples default to first valid.
        left = torch.where(good, positions, torch.zeros_like(positions)).cummax(dim=1).values
        first = torch.argmax(good.int(), dim=1)
        left = torch.where(positions < first[:, None], first[:, None], left)
        filled = torch.gather(torch.nan_to_num(ssp.float(), nan=0., posinf=0., neginf=0.), 1, left)
        norm = (filled - ssp_mean) / ssp_std
        z = self.receiver_depth_m
        hi = torch.searchsorted(self.ssp_depth_m.contiguous(), z.contiguous()).clamp(1, 49)
        lo = hi - 1
        alpha = ((z - self.ssp_depth_m[lo]) /
                 (self.ssp_depth_m[hi] - self.ssp_depth_m[lo])).clamp(0, 1)
        interpolated = norm[:, lo] + (norm[:, hi] - norm[:, lo]) * alpha[None, :]
        valid_last = good.int().sum(dim=1).clamp(1, 50) - 1
        # We assume the ssp_valid_count policy: finite observations occupy a
        # prefix, and all deeper layers are NaN (as in the source dataset).
        last_z = self.ssp_depth_m[valid_last]
        coverage = (z[None, :] <= last_z[:, None]).float()
        return interpolated, coverage

    def forward(self, ssp, bathy, *, angle_indices=None, ssp_mean, ssp_std):
        if ssp.ndim != 2 or ssp.shape[1] != 50 or bathy.ndim != 3 or bathy.shape[1:] != (61, 61):
            raise ValueError("SSP: [B,50]; bathymetry: [B,61,61]")
        if ssp.shape[0] != bathy.shape[0] or not bool(torch.isfinite(bathy).all()):
            raise ValueError("SSP and bathymetry batch sizes must match; bathymetry must be finite")
        if angle_indices is None:
            indices = list(range(N_ANGLES))
        else:
            indices = list(map(int, angle_indices))
            if not indices or any(a < 0 or a >= N_ANGLES for a in indices):
                raise ValueError("angle_indices must be a nonempty subset of 0..8")
        B, A = ssp.shape[0], len(indices)
        grids = self.ray_grid[indices].unsqueeze(0).expand(B, -1, -1, -1)
        # A treated as grid_sample's output height; y axis is north-positive.
        path = F.grid_sample(bathy.float().unsqueeze(1), grids, mode="bilinear",
                             padding_mode="border", align_corners=True)[:, 0]
        z = self.receiver_depth_m.reshape(1, 1, N_DEPTHS, 1)
        bottom = path[:, :, None, :]
        occupancy = (z > bottom).float()
        signed_depth = ((bottom - z) / self.max_depth_m).clamp(-2, 2)
        ssp_z, coverage = self._ssp_depth_columns(ssp, ssp_mean, ssp_std)
        speed = ssp_z[:, None, :, None].expand(-1, A, -1, N_RANGES)
        cov = coverage[:, None, :, None].expand_as(speed)
        radial = (self.receiver_range_m / self.max_range_m).reshape(1, 1, 1, N_RANGES)
        radial = radial.expand(B, A, N_DEPTHS, N_RANGES)
        feats = torch.stack((speed, cov, occupancy, signed_depth, radial), 2)
        return feats.reshape(B*A, 5, N_DEPTHS, N_RANGES)


class UNetEncoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        cs = [width * 2**k for k in range(5)]
        self.blocks = nn.ModuleList([
            double_conv(5, cs[0]),
            *[double_conv(cs[i-1], cs[i]) for i in range(1, 5)]
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            if i:
                x = self.pool(x)
            x = block(x)
            if i < 4:
                skips.append(x)
        return x, skips


class GDAM(nn.Module):
    """Paper Fig.3: squeeze-excitation then wide/narrow convolutions and gates."""
    def __init__(self, c, reduction=16):
        super().__init__()
        mid = max(8, c // reduction)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1), nn.Sigmoid()
        )
        self.structure = nn.Sequential(nn.Conv2d(c, c, 5, padding=2, bias=False),
                                       nn.GroupNorm(min(8, c), c), nn.ReLU(inplace=True))
        self.detail = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False),
                                    nn.GroupNorm(min(8, c), c), nn.ReLU(inplace=True))
        # Separate trainable scalar gates, constrained to (0,1) by sigmoid.
        self.g_structure = nn.Parameter(torch.tensor(0.0))
        self.g_detail = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = x * self.att(x)
        return (torch.sigmoid(self.g_structure) * self.structure(x),
                torch.sigmoid(self.g_detail) * self.detail(x))


class UpBlock(nn.Module):
    def __init__(self, cin, skip_c, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.conv = double_conv(cout + skip_c, cout)

    def forward(self, x, skip):
        x = self.up(x)
        # Shape alignment for 200x250 after four 2x pools: 12x15 -> 25x31
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        c = [width * 2**k for k in range(5)]
        self.up = nn.ModuleList([
            UpBlock(c[4], c[3], c[3]),
            UpBlock(c[3], c[2], c[2]),
            UpBlock(c[2], c[1], c[1]),
            UpBlock(c[1], c[0], c[0]),
        ])
        self.out = nn.Conv2d(c[0], 1, kernel_size=1)

    def forward(self, x, skips):
        for block, skip in zip(self.up, reversed(skips)):
            x = block(x, skip)
        return self.out(x)


class DDPUNet(nn.Module):
    """Single-shot per-angle output, with shared parameters across all nine angles.

    Inputs: ssp [B,50], bathymetry [B,61,61]
    Outputs (normalized): total, structure, detail each [B,A,200,250]
    A=9 by default, or A=len(angle_indices) for low-VRAM training.
    """
    def __init__(self, depth_m, *, ssp_mean, ssp_std, bathy_mean, bathy_std,
                 tl_mean, tl_std, width=32, max_range_m=50_000., max_depth_m=4_000.):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("width must be divisible by 8 and >=8")
        for name, value in (("ssp_std", ssp_std), ("bathy_std", bathy_std), ("tl_std", tl_std)):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for k, v in dict(ssp_mean=ssp_mean, ssp_std=ssp_std,
                         bathy_mean=bathy_mean, bathy_std=bathy_std,
                         tl_mean=tl_mean, tl_std=tl_std).items():
            self.register_buffer(k, torch.tensor(float(v), dtype=torch.float32))
        self.rasterizer = InputRasterizer(depth_m, max_range_m=max_range_m,
                                          max_depth_m=max_depth_m)
        self.encoder = UNetEncoder(width)
        self.gdam = GDAM(width * 16)
        self.structure_decoder = UNetDecoder(width)
        self.detail_decoder = UNetDecoder(width)
        self.width = width
        self.max_range_m = max_range_m
        self.max_depth_m = max_depth_m

    def forward_components(self, ssp, bathymetry, *, angle_indices=None):
        count = N_ANGLES if angle_indices is None else len(angle_indices)
        image = self.rasterizer(ssp, bathymetry, angle_indices=angle_indices,
                                ssp_mean=self.ssp_mean, ssp_std=self.ssp_std)
        h, skips = self.encoder(image)
        h_structure, h_detail = self.gdam(h)
        structure = self.structure_decoder(h_structure, skips)
        detail = self.detail_decoder(h_detail, skips)
        b = ssp.shape[0]
        shape = (b, count, N_DEPTHS, N_RANGES)
        structure = structure.reshape(shape)
        detail = detail.reshape(shape)
        return structure + detail, structure, detail

    def forward(self, ssp, bathymetry, *, angle_indices=None, return_components=False):
        components = self.forward_components(ssp, bathymetry, angle_indices=angle_indices)
        return components if return_components else components[0]

    def predict_db(self, ssp, bathymetry, *, angle_indices=None):
        return self.forward(ssp, bathymetry, angle_indices=angle_indices).float() * self.tl_std + self.tl_mean


def build_model(data_dir, stats, *, width=32, range_max_m=50_000., depth_max_m=4_000.):
    from pathlib import Path
    root = Path(data_dir)
    depth_file = next((root/name for name in ("ssp_depth_m_r2.npy", "ssp_depth_m.npy")
                       if (root/name).is_file()), None)
    if depth_file is None:
        raise FileNotFoundError(f"ssp_depth_m.npy or ssp_depth_m_r2.npy missing in {root}")
    depth = np.load(depth_file, allow_pickle=False)
    return DDPUNet(
        depth, ssp_mean=stats["ssp_mean"], ssp_std=stats["ssp_std"],
        bathy_mean=stats["bathy_mean"], bathy_std=stats["bathy_std"],
        tl_mean=stats["tl_mean"], tl_std=stats["tl_std"], width=width,
        max_range_m=range_max_m, max_depth_m=depth_max_m,
    )
