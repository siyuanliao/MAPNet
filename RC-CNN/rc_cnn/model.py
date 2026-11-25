"""RC-CNN: Deo et al. (JASA, 2025) inspired baseline.

Core structure retained: four CNN encoder stages -> 64-D latent vector ->
FC expansion -> transpose-conv 2-D TL decoder. Bathymetry is represented
by continuous along-ray depths; no binary seabed grids are constructed. Necessary adaptations: one shared network per 9 ray
azimuths; source SSP encoded separately and fused at the 64-D bottleneck.
No learned azimuth interaction, FiLM, U-Net skip paths, or attention.

Coordinates: bathy rows SOUTH -> NORTH, cols WEST -> EAST, center source.
Angles in degrees from north clockwise: 0, 40, ..., 320.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODEL_ARCHITECTURE = "RC-CNN"
# Compatible identifier in checkpoints saved before the public release.
CHECKPOINT_MODEL_NAMES = (MODEL_ARCHITECTURE, "RC-CNN-3D-Comparable-NoMask")
REFERENCE_PARAMETERS = 20_144_116
PARAMETER_TOLERANCE = 0.10
MODEL_CONFIG = {
    "latent_dim": 64,
    "fc_width": 6144,
    "cnn_channels": (16, 32, 64, 96),
    "angles_deg": tuple(range(0, 360, 40)),
    "range_max_m": 50_000.0,
    "depth_max_m": 4_000.0,
    "bathy_half_width_m": 60_000.0,
    "n_ranges": 250,
    "n_depths": 200,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SSPConditionEncoder(nn.Module):
    """Synthesize a 64-D SSP descriptor from normalized speed/depth/validity.

    For missing layers, speed and depth features are zero and validity=0.
    Missing values are *never* fed into the network as NaN.
    """
    def __init__(self, depth_m: torch.Tensor):
        super().__init__()
        depth = torch.as_tensor(depth_m, dtype=torch.float32).flatten()
        if depth.numel() != 50 or not torch.isfinite(depth).all():
            raise ValueError("ssp_depth_m must be 50 finite depth coordinates")
        if not torch.all(depth[1:] > depth[:-1]):
            raise ValueError("ssp_depth_m must be strictly increasing")
        self.register_buffer("depth_norm", depth / depth[-1].clamp_min(1.0))
        self.net = nn.Sequential(
            nn.Linear(3 * 50, 128), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64), nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, ssp: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        if ssp.ndim != 2 or ssp.shape[-1] != 50:
            raise ValueError("SSP must have shape [B, 50]")
        valid = torch.isfinite(ssp)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every SSP sample must contain >=1 finite layer")
        normal = torch.where(valid, (ssp.float() - mean) / std, 0.0)
        depth = self.depth_norm.unsqueeze(0).expand_as(normal)
        features = torch.cat((normal, depth * valid.float(), valid.float()), dim=-1)
        return self.net(features)


class BathymetryRayFeatureGrid(nn.Module):
    """Continuous bathymetric features aligned to 9 azimuthal TL planes.

    No occupancy, seabed, or water validity grid is generated. The three
    numerical channels are: continuous sampled depth (z_b), receiver-depth
    coordinate z, and receiver-range coordinate r. All channels are float32.
    Interpolation keeps the native bathymetry information instead of
    thresholding it into a binary map.
    """
    def __init__(self, *, max_range_m=50_000.0, max_depth_m=4_000.0,
                 bathy_half_width_m=60_000.0,
                 bathy_mean=2500.0, bathy_std=500.0):
        super().__init__()
        if not (0 < max_range_m <= bathy_half_width_m):
            raise ValueError("max_range_m must be in (0, bathy_half_width_m]")
        if max_depth_m <= 0 or bathy_std <= 0:
            raise ValueError("max_depth_m and bathy_std must be positive")
        theta = torch.arange(9, dtype=torch.float32) * (40.0 * math.pi / 180.0)
        radius = torch.linspace(0.0, max_range_m, 250)
        east = theta.sin()[:, None] * radius[None, :]
        north = theta.cos()[:, None] * radius[None, :]
        grid = torch.stack((east / bathy_half_width_m,
                            north / bathy_half_width_m), dim=-1)
        if not bool((grid.abs() <= 1.000001).all()):
            raise ValueError("Sampling grid exceeds supplied bathymetry extent")
        self.register_buffer("grid", grid.unsqueeze(0))  # [1,9,250,2]
        self.register_buffer("receiver_depth", torch.linspace(0, 1, 200).view(1,1,1,200,1))
        self.register_buffer("receiver_range", torch.linspace(0, 1, 250).view(1,1,1,1,250))
        self.register_buffer("bathy_mean", torch.tensor(float(bathy_mean)))
        self.register_buffer("bathy_std", torch.tensor(float(bathy_std)))

    def forward(self, bathy: torch.Tensor) -> torch.Tensor:
        if bathy.ndim != 3 or bathy.shape[-2:] != (61, 61):
            raise ValueError("Bathymetry must have shape [B,61,61]")
        if not bool(torch.isfinite(bathy).all()):
            raise ValueError("Bathymetry must be finite")
        batch = bathy.shape[0]
        path_depth = F.grid_sample(
            bathy.float().unsqueeze(1), self.grid.expand(batch, -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=True,
        )[:, 0]  # [B,9,250], continuous meters
        path_channel = ((path_depth - self.bathy_mean) / self.bathy_std)[:, :, None, None, :].expand(-1,-1,1,200,-1)
        z_channel = self.receiver_depth.expand(batch,9,-1,-1,250)
        r_channel = self.receiver_range.expand(batch,9,-1,200,-1)
        return torch.cat((path_channel,z_channel,r_channel),dim=2)  # [B,9,3,200,250]


class ConvEncoder(nn.Module):
    """Four Conv2d+BN+LeakyReLU stages and four max-pools, adapted to 200x250."""
    def __init__(self):
        super().__init__()
        channels = (3, 16, 32, 64, 96)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, 2),
            )
            for i in range(4)
        ])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        w = MODEL_CONFIG["fc_width"]
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, w), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(w, 64),
        )

    def forward(self, x, *, checkpoint_fc=False):
        for block in self.blocks:
            x = block(x)
        x = self.flatten(self.pool(x))
        # Checkpoint only the BN-free fully connected bottleneck, so BatchNorm
        # running statistics are never updated twice by recomputation.
        if checkpoint_fc and self.training and torch.is_grad_enabled():
            return checkpoint(self.fc, x, use_reentrant=False)
        return self.fc(x)


class ConvDecoder(nn.Module):
    """Mirror dense expansion followed by four transpose-convolutions."""
    def __init__(self):
        super().__init__()
        w = MODEL_CONFIG["fc_width"]
        self.fc = nn.Sequential(
            nn.Linear(64, w), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(w, 96 * 4 * 4), nn.LeakyReLU(0.1, inplace=True),
        )
        channels = (96, 96, 64, 32, 16)
        self.deconvs = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1], 3, padding=1,
                                   bias=False),
                nn.BatchNorm2d(channels[i + 1]),
                nn.LeakyReLU(0.1, inplace=True),
            )
            for i in range(4)
        ])
        self.out = nn.Conv2d(16, 1, 3, padding=1)
        self.sizes = ((12, 15), (25, 31), (50, 62), (100, 125))

    def forward(self, z, *, checkpoint_fc=False):
        if checkpoint_fc and self.training and torch.is_grad_enabled():
            x = checkpoint(self.fc, z, use_reentrant=False)
        else:
            x = self.fc(z)
        x = x.reshape(-1, 96, 4, 4)
        for size, layer in zip(self.sizes, self.deconvs):
            x = layer(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        x = F.interpolate(x, size=(200, 250), mode="bilinear", align_corners=False)
        return self.out(x)


class RCCNN(nn.Module):
    """Batched baseline, shared weights across azimuths, no attention.

    Args data shape: SSP [B,50]; Bathy [B,61,61]
    Output: normalized TL [B,9,200,250]
    """
    def __init__(self, depth_m, *, ssp_mean, ssp_std, bathy_mean, bathy_std,
                 tl_mean, tl_std, gradient_checkpointing=False,
                 range_max_m=50_000.0, depth_max_m=4_000.0):
        super().__init__()
        for name, v in {"ssp_std": ssp_std, "bathy_std": bathy_std, "tl_std": tl_std}.items():
            if not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        # Training stats are buffers for strict reproducibility; data binned as original.
        for name, value in {
            "ssp_mean": ssp_mean, "ssp_std": ssp_std,
            "bathy_mean": bathy_mean, "bathy_std": bathy_std,
            "tl_mean": tl_mean, "tl_std": tl_std,
        }.items():
            self.register_buffer(name, torch.tensor(float(value), dtype=torch.float32))
        self.terrain_features = BathymetryRayFeatureGrid(
            max_range_m=range_max_m, max_depth_m=depth_max_m,
            bathy_mean=bathy_mean, bathy_std=bathy_std)
        self.ssp_encoder = SSPConditionEncoder(depth_m)
        self.encoder = ConvEncoder()
        self.fusion = nn.Sequential(nn.Linear(64 + 64, 64), nn.LeakyReLU(0.1, inplace=True))
        self.decoder = ConvDecoder()
        self.gradient_checkpointing = gradient_checkpointing

    def _encode_decode(self, terrain_features, ssp_features):
        latent_geom = self.encoder(terrain_features, checkpoint_fc=self.gradient_checkpointing)
        latent = self.fusion(torch.cat((latent_geom, ssp_features), dim=1))
        return self.decoder(latent, checkpoint_fc=self.gradient_checkpointing)

    def forward(self, ssp: torch.Tensor, bathymetry: torch.Tensor):
        if ssp.shape[0] != bathymetry.shape[0]:
            raise ValueError("SSP and Bathymetry batch sizes must match")
        b = int(ssp.shape[0])
        geometry = self.terrain_features(bathymetry).reshape(b * 9, 3, 200, 250)
        # Each direction receives the same SSP, but different terrain.
        ssp_feature = self.ssp_encoder(ssp, self.ssp_mean, self.ssp_std)
        ssp_feature = ssp_feature[:, None, :].expand(-1, 9, -1).reshape(b * 9, 64)
        out = self._encode_decode(geometry, ssp_feature)
        return out.reshape(b, 9, 200, 250)

    def predict_db(self, ssp, bathymetry):
        return self(ssp, bathymetry).float() * self.tl_std + self.tl_mean


def build_model(data_dir, stats, *, gradient_checkpointing=False,
                range_max_m=50_000.0, depth_max_m=4_000.0):
    from .data import resolve_data_files
    files = resolve_data_files(data_dir)
    depth = np.load(files["ssp_depth_m"], allow_pickle=False)
    return RCCNN(
        depth,
        ssp_mean=stats["ssp_mean"], ssp_std=stats["ssp_std"],
        bathy_mean=stats["bathy_mean"], bathy_std=stats["bathy_std"],
        tl_mean=stats["tl_mean"], tl_std=stats["tl_std"],
        gradient_checkpointing=gradient_checkpointing,
        range_max_m=range_max_m, depth_max_m=depth_max_m,
    )
