"""有效水体区域的损失与评价指标。"""
from __future__ import annotations
import math
import torch
import torch.distributed as dist


def water_valid_mask(target, sample_ids, bottom_idx):
    """按原始样本编号生成掩码，并排除真实TL的非有限值。"""
    if target.ndim != 4 or target.shape[1:] != (9, 200, 250):
        raise ValueError(f"TL shape must be [B,9,200,250], got {tuple(target.shape)}")
    # 注意：sample_ids为18876条数据的原始样本编号，而不是当前batch内的索引。
    src = sample_ids.to(device=target.device, dtype=torch.long) % 121
    h = bottom_idx[src].to(torch.long)
    z = torch.arange(200, device=target.device).view(1, 1, 200, 1)


    return (z <= h.unsqueeze(2)) & torch.isfinite(target)


def masked_error_sums(pred, target, valid):
    """计算有效水体网格点上的误差和与点数。"""
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError("pred, target, valid shapes must be identical")
    # 必须先用where过滤，再作差；否则0*NaN仍为NaN。
    delta = torch.where(valid, pred.float(), 0.0) - torch.where(valid, target.float(), 0.0)
    with torch.no_grad():
        abs_sum = delta.detach().abs().sum()
    return delta.square().sum(), abs_sum, valid.sum()


class MaskedMetrics:
    """按所有有效水体网格点聚合误差，避免对每个Batch的RMSE求平均。"""

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
        self.nonfinite = torch.zeros((), device=device, dtype=torch.int64)

    @torch.no_grad()
    def add(self, pred_db, target_db, valid):
        # 水体外及真实TL非有限点不纳入误差，分组边界沿用原始物理网格。
        self.nonfinite += (valid & ~torch.isfinite(pred_db)).sum()
        err = torch.where(valid, pred_db.float(), 0.0) - torch.where(valid, target_db.float(), 0.0)
        abs_err = err.abs().double()
        sq_err = err.square().double()
        self.sse += sq_err.sum()
        self.sae += abs_err.sum()
        self.n += valid.sum().double()
        self.angle_sse += sq_err.sum(dim=(0, 2, 3))
        self.angle_sae += abs_err.sum(dim=(0, 2, 3))
        self.angle_n += valid.sum(dim=(0, 2, 3)).double()

        self.range_sse += sq_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_sae += abs_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_n += valid.reshape(-1, 5, 50).sum(dim=(0, 2)).double()
        self.depth_sse += sq_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_sae += abs_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_n += valid.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3)).double()

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            # 统一规约，减少分布式验证时的通信次数。
            names = (
                "sse", "sae", "n", "angle_sse", "angle_sae", "angle_n",
                "range_sse", "range_sae", "range_n", "depth_sse", "depth_sae", "depth_n",
            )
            shape = [getattr(self, name).shape for name in names]
            packed = torch.cat([getattr(self, name).reshape(-1) for name in names])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            idx = 0
            for name, size in zip(names, shape):
                length = math.prod(size)
                getattr(self, name).copy_(packed[idx:idx + length].reshape(size))
                idx += length
            dist.all_reduce(self.nonfinite, op=dist.ReduceOp.SUM)

    def result(self):
        if int(self.nonfinite.item()) > 0:
            raise FloatingPointError("有效水体区域的预测TL含NaN/Inf，无法计算评价指标")

        def one(sse, sae, n):
            n = float(n.item())
            if n == 0:
                return {"mae_db": None, "rmse_db": None, "valid_points": 0}
            return {
                "mae_db": float(sae.item() / n),
                "rmse_db": math.sqrt(max(0.0, float(sse.item() / n))),
                "valid_points": int(n),
            }

        return {
            "overall": one(self.sse, self.sae, self.n),
            "per_angle": {f"{k*40}deg": one(self.angle_sse[k], self.angle_sae[k], self.angle_n[k]) for k in range(9)},
            "per_range_km": {f"{10*k}-{10*(k+1)}": one(self.range_sse[k], self.range_sae[k], self.range_n[k]) for k in range(5)},
            "per_depth_m": {f"{1000*k}-{1000*(k+1)}": one(self.depth_sse[k], self.depth_sae[k], self.depth_n[k]) for k in range(4)},
        }
