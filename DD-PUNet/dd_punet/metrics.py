"""Full-grid TL metrics, aggregated over ALL [B,9,200,250] cells.

Neither a bathymetry/bottom mask nor finite-value filtering is performed:
non-finite targets or predictions are errors, not points to be silently removed.
Grouped metrics are reported by azimuth, range, and depth.
"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist


class FullGridMetrics:
    def __init__(self, device):
        self.device = torch.device(device)
        self.sse = torch.zeros((), dtype=torch.float64, device=device)
        self.sae = torch.zeros((), dtype=torch.float64, device=device)
        self.n = torch.zeros((), dtype=torch.float64, device=device)
        self.angle_sse = torch.zeros(9, dtype=torch.float64, device=device)
        self.angle_sae = torch.zeros(9, dtype=torch.float64, device=device)
        self.angle_n = torch.zeros(9, dtype=torch.float64, device=device)
        self.range_sse = torch.zeros(5, dtype=torch.float64, device=device)
        self.range_sae = torch.zeros(5, dtype=torch.float64, device=device)
        self.range_n = torch.zeros(5, dtype=torch.float64, device=device)
        self.depth_sse = torch.zeros(4, dtype=torch.float64, device=device)
        self.depth_sae = torch.zeros(4, dtype=torch.float64, device=device)
        self.depth_n = torch.zeros(4, dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, pred_db, target_db):
        if pred_db.shape != target_db.shape or pred_db.ndim != 4 or pred_db.shape[1:] != (9, 200, 250):
            raise ValueError('Expected matching full TL tensors [B,9,200,250]')
        if not bool(torch.isfinite(target_db).all()):
            raise ValueError('Full-grid metric requires finite TL labels at every point')
        if not bool(torch.isfinite(pred_db).all()):
            raise FloatingPointError('Full-grid metric received non-finite model predictions')
        err = pred_db.double() - target_db.double()
        abs_err = err.abs()
        sq_err = err.square()
        self.sse += sq_err.sum()
        self.sae += abs_err.sum()
        self.n += target_db.numel()
        self.angle_sse += sq_err.sum(dim=(0, 2, 3))
        self.angle_sae += abs_err.sum(dim=(0, 2, 3))
        self.angle_n += pred_db.shape[0] * 200 * 250
        self.range_sse += sq_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_sae += abs_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_n += pred_db.shape[0] * 9 * 200 * 50
        self.depth_sse += sq_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_sae += abs_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_n += pred_db.shape[0] * 9 * 50 * 250

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            names = ('sse', 'sae', 'n', 'angle_sse', 'angle_sae', 'angle_n',
                     'range_sse', 'range_sae', 'range_n', 'depth_sse', 'depth_sae', 'depth_n')
            packed = torch.cat([getattr(self, k).view(-1) for k in names])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            offset = 0
            for k in names:
                t = getattr(self, k)
                num = t.numel()
                t.copy_(packed[offset:offset+num].reshape_as(t))
                offset += num

    def result(self):
        def one(sse, sae, n):
            number = int(n.item())
            if not number:
                return {'mae_db': None, 'rmse_db': None, 'valid_points': 0}
            return {'mae_db': float(sae.item()/number),
                    'rmse_db': math.sqrt(max(0.0, float(sse.item()/number))),
                    'valid_points': number}

        return {
            'overall': one(self.sse, self.sae, self.n),
            'per_angle': {f'{k*40}deg': one(self.angle_sse[k], self.angle_sae[k], self.angle_n[k]) for k in range(9)},
            'per_range_km': {f'{10*k}-{10*(k+1)}': one(self.range_sse[k], self.range_sae[k], self.range_n[k]) for k in range(5)},
            'per_depth_m': {f'{1000*k}-{1000*(k+1)}': one(self.depth_sse[k], self.depth_sae[k], self.depth_n[k]) for k in range(4)},
        }
