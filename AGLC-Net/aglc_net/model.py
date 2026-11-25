"""Training-only-reference adaptation of Huo et al. (JASA 2026).

Paper ingredients: TL autoencoder E/D, SSP perturbation mapper G, learned
scalar fusion, SE-style latent channel correction, two-stage training.

Task adaptations: 9 shared-weight azimuthal TL maps; training-only bank
of reference TL/SSP for 121 locations; local bathymetry resolves the source
and contributes a small learned conditioning vector. Crucially, an actual
historical/reference TL bank is extra prior data: parameter matching DOES
NOT make the information budget identical to a two-input direct predictor.
"""
from __future__ import annotations

import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_ARCHITECTURE = "AGLC-Net"
# Compatible identifier in checkpoints saved before the public release.
CHECKPOINT_MODEL_NAMES = (MODEL_ARCHITECTURE, "LatentCorrection-TL-ReferenceBank")
REFERENCE_PARAMETERS = 20_144_116
PARAMETER_TOLERANCE = 0.10
# Determined by parameter-match unit test. All widths can be edited, but
# altering them invalidates existing pretrained weights.
MODEL_CONFIG = {
    "widths": (64, 128, 256, 416, 576),
    "latent_dim": 256,
    "bathy_dim": 64,
    "mapper_hidden": 512,
    "angle_embed_dim": 16,
    "input_grid": (50, 63),
    "output_grid": (200, 250),
}


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(16, ch_out), ch_out), nn.SiLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(16, ch_out), ch_out), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.conv = DoubleConv(input_channels + skip_channels, output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip), dim=1))


class TLCodec(nn.Module):
    """Asymmetric U-Net-like TL encoder/decoder shared across 9 directions.

    Bottleneck GAP+linear creates a d-dimensional latent vector. Decoder
    starts from a projected latent grid and uses TL-reference encoder skips.
    In Stage I, encoder skips come from target TL (autoencoder).
    In Stage II, encoder skips come exclusively from historical reference TL.
    The second encoder channel is a constant 1 so no TL-label validity mask
    or water mask is used, without changing any trainable layer shapes.
    """
    def __init__(self, widths=(64, 128, 256, 416, 576), latent_dim=256):
        super().__init__()
        w0, w1, w2, w3, w4 = widths
        self.widths = tuple(widths)
        self.latent_dim = latent_dim
        self.encoder_blocks = nn.ModuleList([
            DoubleConv(2, w0), DoubleConv(w0, w1), DoubleConv(w1, w2),
            DoubleConv(w2, w3), DoubleConv(w3, w4),
        ])
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.to_latent = nn.Linear(w4, latent_dim)
        # 50x63 -> 25x32 -> 13x16 -> 7x8 -> 4x4.
        self.from_latent = nn.Linear(latent_dim, w4 * 4 * 4)
        self.up3 = UpBlock(w4, w3, w3)
        self.up2 = UpBlock(w3, w2, w2)
        self.up1 = UpBlock(w2, w1, w1)
        self.up0 = UpBlock(w1, w0, w0)
        self.fine1 = DoubleConv(w0, w0)
        self.fine2 = DoubleConv(w0, 24)
        self.out = nn.Conv2d(24, 1, kernel_size=1)

    def encode(self, tl_norm):
        """TL [N,1,200,250] -> latent [N,d], skip maps; no masking."""
        if tl_norm.ndim != 4 or tl_norm.shape[1:] != (1, 200, 250):
            raise ValueError('TL autoencoder expects [N,1,200,250]')
        x = F.interpolate(torch.cat((tl_norm, torch.ones_like(tl_norm)), dim=1), size=(50, 63),
                          mode="bilinear", align_corners=False)
        skips = []
        for i, block in enumerate(self.encoder_blocks):
            if i:
                x = self.pool(x)
            x = block(x)
            skips.append(x)
        z = self.to_latent(x.mean(dim=(-2, -1)))
        return z, tuple(skips)

    def decode(self, z, skips):
        x = self.from_latent(z).reshape(z.shape[0], self.widths[-1], 4, 4)
        x = self.up3(x, skips[3])
        x = self.up2(x, skips[2])
        x = self.up1(x, skips[1])
        x = self.up0(x, skips[0])
        x = F.interpolate(x, size=(100, 125), mode="bilinear", align_corners=False)
        x = self.fine1(x)
        x = F.interpolate(x, size=(200, 250), mode="bilinear", align_corners=False)
        return self.out(self.fine2(x))

    def forward(self, tl_norm):
        latent, skips = self.encode(tl_norm)
        return self.decode(latent, skips)


class BathymetryContext(nn.Module):
    """Bathymetry informs correction in addition to identifying reference TL."""
    def __init__(self, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, output_dim), nn.SiLU(),
        )

    def forward(self, bathy_norm):
        return self.feature(bathy_norm[:, None])


class SSPPerturbationMapper(nn.Module):
    """MLP maps SSP perturbation + validity + bathymetry + azimuth to d."""
    def __init__(self, bathy_dim=64, angle_dim=16, latent_dim=256, hidden=512):
        super().__init__()
        self.angle_emb = nn.Embedding(9, angle_dim)
        self.net = nn.Sequential(
            nn.Linear(50 * 2 + bathy_dim + angle_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
        )

    def forward(self, delta, valid, bathy_features, azimuths):
        n, a = delta.shape[0], azimuths.numel()
        common = torch.cat((delta, valid, bathy_features), dim=1)
        common = common[:, None, :].expand(n, a, -1)
        ang = self.angle_emb(azimuths).unsqueeze(0).expand(n, -1, -1)
        return self.net(torch.cat((common, ang), dim=-1)).reshape(n * a, -1)


class LatentCorrectionModule(nn.Module):
    """Huo et al. Eqs. (5)-(8), scalar weighted fusion + SE channel gates."""
    def __init__(self, latent_dim=256, reduction=4):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.channel_attention = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim // reduction, 8)), nn.ReLU(),
            nn.Linear(max(latent_dim // reduction, 8), latent_dim), nn.Sigmoid(),
        )

    def forward(self, z_ref, delta_z):
        z = F.relu(self.alpha * z_ref + self.beta * delta_z)
        return z * self.channel_attention(z)


class AGLCNet(nn.Module):
    """External forward inputs: only SSP[B,50] and bathy[B,61,61].

    Internal buffers hold TRAINING-ONLY reference TL (additional data prior).
    `persistent=False` excludes huge bank arrays from .pt; `reference_bank.npz`
    MUST travel with checkpoint for inference.
    """
    def __init__(self, bank: dict, stats: dict, widths=None, latent_dim=256,
                 bathy_dim=64, hidden=512, angle_dim=16,
                 invalid_raw=None, invalid_target_fill_db=None, bathy_match_atol=0.02):
        super().__init__()
        if widths is None:
            widths = MODEL_CONFIG["widths"]
        if stats["tl_std"] <= 0 or stats["ssp_std"] <= 0 or stats["bathy_std"] <= 0:
            raise ValueError("training standard deviations must be positive")
        self.codec = TLCodec(widths=widths, latent_dim=latent_dim)
        self.bathy_encoder = BathymetryContext(bathy_dim)
        self.mapper = SSPPerturbationMapper(bathy_dim, angle_dim, latent_dim, hidden)
        self.corrector = LatentCorrectionModule(latent_dim)
        for name in ("tl_mean", "tl_std", "ssp_mean", "ssp_std", "bathy_mean", "bathy_std"):
            self.register_buffer(name, torch.tensor(float(stats[name]), dtype=torch.float32))
        if (invalid_raw is None) != (invalid_target_fill_db is None):
            raise ValueError('Reference TL integer sentinel requires an explicit finite dB replacement')
        if invalid_target_fill_db is not None and not math.isfinite(invalid_target_fill_db):
            raise ValueError('Reference TL replacement must be finite')
        self.invalid_raw = invalid_raw
        self.invalid_target_fill_db = invalid_target_fill_db
        self.bathy_match_atol = float(bathy_match_atol)
        # CUDA advanced indexing does not support torch.uint16 on many PyTorch versions.
        # Convert once at initialization, BEFORE moving the bank to GPU.
        # int32 is lossless for both uint16 and int16 raw TL values; unlike int16,
        # it also preserves unsigned samples above 32767.
        self.register_buffer("ref_raw", torch.as_tensor(np.array(bank["tl_raw"], copy=True),
                                                        dtype=torch.int32), persistent=False)
        self.register_buffer("ref_ssp", torch.as_tensor(np.array(bank["ssp"], dtype=np.float32, copy=True)), persistent=False)
        self.register_buffer("ref_bathy", torch.as_tensor(np.array(bank["bathy"], dtype=np.float32, copy=True)), persistent=False)
        if tuple(self.ref_raw.shape) != (121, 9, 200, 250):
            raise ValueError("reference tl_raw must have shape [121,9,200,250]")
        if tuple(self.ref_ssp.shape) != (121, 50) or tuple(self.ref_bathy.shape) != (121, 61, 61):
            raise ValueError("invalid reference SSP/bathymetry shapes")

    @torch.no_grad()
    def source_index(self, bathy):
        """Identify one of the same 121 source locations via training-only geometry.

        Index resolution is exact in the specified dataset; reject novel source
        locations, for which a 121-source historical reference is not defined.
        """
        if bathy.ndim != 3 or bathy.shape[-2:] != (61, 61):
            raise ValueError("bathymetry must be [B,61,61]")
        if not torch.isfinite(bathy).all():
            raise ValueError("bathymetry contains NaN/Inf")
        ref = self.ref_bathy.flatten(1).float()
        inp = bathy.flatten(1).float()
        # Low-resolution fingerprint first; check selected full map precisely.
        keys = torch.arange(0, ref.shape[1], 17, device=ref.device)
        d = torch.cdist(inp[:, keys], ref[:, keys])
        source_idx = d.argmin(dim=1)
        max_difference = (inp - ref[source_idx]).abs().amax(dim=1)
        # Fingerprint ties can occur over flat bathymetry; resolve them with
        # the entire 61x61 map before rejecting a known source.
        unresolved = torch.where(max_difference > self.bathy_match_atol)[0]
        if unresolved.numel():
            exact_d = torch.cdist(inp[unresolved], ref)
            source_idx[unresolved] = exact_d.argmin(dim=1)
            max_difference = (inp - ref[source_idx]).abs().amax(dim=1)
        if (max_difference > self.bathy_match_atol).any():
            raise ValueError("Unseen bathymetry/source: reference bank only supports 121 trained source positions")
        return source_idx

    def reference_batch(self, source_idx, angle_start, angle_end):
        """Full-grid reference TL from the first TRAIN year, including every cell."""
        raw = self.ref_raw[source_idx, angle_start:angle_end]
        values = raw.float() * 0.01
        if self.invalid_raw is not None:
            values = torch.where(raw == self.invalid_raw, self.invalid_target_fill_db, values)
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Reference TL contains non-finite values')
        norm = (values - self.tl_mean) / self.tl_std
        n, a = norm.shape[:2]
        return norm.reshape(n * a, 1, 200, 250)

    def _conditioning(self, ssp, bathy, source_idx):
        if ssp.ndim != 2 or ssp.shape[1] != 50 or ssp.shape[0] != bathy.shape[0]:
            raise ValueError("SSP must be [B,50] and match bathymetry batch")
        ref_ssp = self.ref_ssp[source_idx]
        valid = torch.isfinite(ssp) & torch.isfinite(ref_ssp)
        delta = torch.where(valid, (ssp - ref_ssp) / self.ssp_std, 0.0)
        valid = valid.float()
        b = (bathy[:, None] - self.bathy_mean) / self.bathy_std
        features = self.bathy_encoder(b.squeeze(1))
        return delta, valid, features

    def forward(self, ssp, bathy, *, angle_chunk=9, angle_start=0, angle_end=9):
        if angle_chunk < 1 or angle_chunk > 9 or not (0 <= angle_start < angle_end <= 9):
            raise ValueError("invalid azimuth slice/chunk")
        src = self.source_index(bathy)
        delta, valid, bathy_features = self._conditioning(ssp, bathy, src)
        outputs = []
        for start in range(angle_start, angle_end, angle_chunk):
            end = min(angle_end, start + angle_chunk)
            rnorm = self.reference_batch(src, start, end)
            # Stage II: reference TL encoder is frozen; its outputs are constants.
            with torch.no_grad():
                z_ref, skips = self.codec.encode(rnorm)
            azimuths = torch.arange(start, end, device=ssp.device)
            delta_z = self.mapper(delta, valid, bathy_features, azimuths)
            z_hat = self.corrector(z_ref, delta_z)
            out = self.codec.decode(z_hat, skips)
            outputs.append(out.reshape(ssp.shape[0], end-start, 200, 250))
        return torch.cat(outputs, dim=1)

    def predict_db(self, ssp, bathy, *, angle_chunk=9):
        return self.forward(ssp, bathy, angle_chunk=angle_chunk) * self.tl_std + self.tl_mean


def build_model(reference_bank_path, stats, **kwargs):
    """Build after separately generating reference_bank.npz from year-1 month-1."""
    with np.load(reference_bank_path, allow_pickle=False) as f:
        bank = {k: f[k] for k in ("tl_raw", "ssp", "bathy")}
        if "sample_ids" in f and not np.array_equal(f["sample_ids"], np.arange(121)):
            raise ValueError("Reference bank must contain training sample IDs 0..120")
        if "tl_invalid_raw" in f:
            bank_invalid = int(f["tl_invalid_raw"][()])
            expected = -999999 if stats.get("tl_invalid_raw") is None else int(stats["tl_invalid_raw"])
            if bank_invalid != expected:
                raise ValueError("Reference bank TL invalid sentinel is inconsistent with stats")
    return AGLCNet(bank, stats,
                              invalid_raw=stats.get("tl_invalid_raw"),
                              invalid_target_fill_db=stats.get("invalid_target_fill_db"), **kwargs)


if __name__ == "__main__":
    print("Import build_model(path_to_reference_bank, training_stats) to create the model.")
