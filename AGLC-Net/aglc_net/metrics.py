"""Full-grid TL loss and globally aggregated dB metrics.

Every element of [B,9,200,250] participates. No terrain, numerical-validity,
or other target mask is used by any objective or metric. Any non-finite input
must be corrected explicitly in the data pipeline, or triggers an error.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist

LOSS_POLICY = 'full_grid_mse_no_mask_v1'


def full_grid_error_sums(pred, target):
    """Differentiable SSE and detached SAE over *all* elements, with hard checks."""
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] not in range(1,10) or pred.shape[2:] != (200, 250):
        raise ValueError(f'Expected matching [B,A,200,250] with A in 1..9; got {pred.shape}, {target.shape}')
    if not bool(torch.isfinite(target).all()):
        raise ValueError('Non-finite TL target; the full-grid objective cannot discard cells')
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError('Non-finite TL prediction')
    diff = pred.float() - target.float()
    sse = diff.square().sum()
    sae = diff.detach().abs().sum()
    return sse, sae, target.numel()


class FullGridMetrics:
    """Grouped metrics; valid_points counts all cells."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.sse = torch.zeros((), device=device, dtype=torch.float64)
        self.sae = torch.zeros((), device=device, dtype=torch.float64)
        self.n = torch.zeros((), device=device, dtype=torch.float64)
        self.angle_sse = torch.zeros(9, device=device, dtype=torch.float64)
        self.angle_sae = torch.zeros(9, device=device, dtype=torch.float64)
        self.angle_n = torch.zeros(9, device=device, dtype=torch.float64)
        self.range_sse = torch.zeros(5, device=device, dtype=torch.float64)
        self.range_sae = torch.zeros(5, device=device, dtype=torch.float64)
        self.range_n = torch.zeros(5, device=device, dtype=torch.float64)
        self.depth_sse = torch.zeros(4, device=device, dtype=torch.float64)
        self.depth_sae = torch.zeros(4, device=device, dtype=torch.float64)
        self.depth_n = torch.zeros(4, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, pred_db, target_db):
        if pred_db.shape != target_db.shape or pred_db.ndim != 4 or pred_db.shape[1:] != (9, 200, 250):
            raise ValueError('Expected matching TL tensors [B,9,200,250]')
        if not bool(torch.isfinite(target_db).all()):
            raise ValueError('Non-finite TL label: full-grid evaluation requires every value')
        if not bool(torch.isfinite(pred_db).all()):
            raise FloatingPointError('Non-finite TL prediction during evaluation')
        delta = pred_db.double() - target_db.double()
        se, ae = delta.square(), delta.abs()
        self.sse += se.sum()
        self.sae += ae.sum()
        self.n += target_db.numel()
        self.angle_sse += se.sum(dim=(0, 2, 3))
        self.angle_sae += ae.sum(dim=(0, 2, 3))
        self.angle_n += target_db.shape[0] * 200 * 250
        self.range_sse += se.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_sae += ae.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_n += target_db.shape[0] * 9 * 200 * 50
        self.depth_sse += se.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_sae += ae.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_n += target_db.shape[0] * 9 * 50 * 250

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            names = ('sse', 'sae', 'n', 'angle_sse', 'angle_sae', 'angle_n',
                     'range_sse', 'range_sae', 'range_n', 'depth_sse', 'depth_sae', 'depth_n')
            packed = torch.cat([getattr(self, key).flatten() for key in names])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            start = 0
            for key in names:
                value = getattr(self, key)
                stop = start + value.numel()
                value.copy_(packed[start:stop].reshape(value.shape))
                start = stop

    def result(self):
        def one(sse, sae, n):
            count = int(n.item())
            return {'mae_db': float(sae.item() / count) if count else None,
                    'rmse_db': math.sqrt(max(0., float(sse.item() / count))) if count else None,
                    'valid_points': count}
        return {
            'overall': one(self.sse, self.sae, self.n),
            'per_angle': {f'{k*40}deg': one(self.angle_sse[k], self.angle_sae[k], self.angle_n[k]) for k in range(9)},
            'per_range_km': {f'{10*k}-{10*(k+1)}': one(self.range_sse[k], self.range_sae[k], self.range_n[k]) for k in range(5)},
            'per_depth_m': {f'{1000*k}-{1000*(k+1)}': one(self.depth_sse[k], self.depth_sae[k], self.depth_n[k]) for k in range(4)},
        }
