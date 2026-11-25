"""Unmasked full-grid loss components and dB metrics."""
from __future__ import annotations
import math
import torch
import torch.distributed as dist


def dense_error_sums(pred, target):
    """Sum squared/absolute errors over EVERY grid cell; differentiable SSE."""
    if pred.shape != target.shape or pred.ndim != 4 or tuple(pred.shape[1:]) != (9,200,250):
        raise ValueError("Both arrays must have shape [B,9,200,250]")
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("Target TL contains NaN/Inf; unmasked loss cannot accept it")
    delta = pred.float() - target.float()
    return delta.square().sum(), delta.detach().abs().sum(), target.numel()


class DenseMetrics:
    """Aggregate MAE/RMSE over all 9*200*250 cells; no spatial masking."""
    def __init__(self,device):
        self.device=torch.device(device)
        self.sse=torch.zeros((),device=device,dtype=torch.float64)
        self.sae=torch.zeros((),device=device,dtype=torch.float64)
        self.n=torch.zeros((),device=device,dtype=torch.float64)
        self.angle_sse=torch.zeros(9,device=device,dtype=torch.float64)
        self.angle_sae=torch.zeros(9,device=device,dtype=torch.float64)
        self.angle_n=torch.zeros(9,device=device,dtype=torch.float64)
        self.range_sse=torch.zeros(5,device=device,dtype=torch.float64)
        self.range_sae=torch.zeros(5,device=device,dtype=torch.float64)
        self.range_n=torch.zeros(5,device=device,dtype=torch.float64)
        self.depth_sse=torch.zeros(4,device=device,dtype=torch.float64)
        self.depth_sae=torch.zeros(4,device=device,dtype=torch.float64)
        self.depth_n=torch.zeros(4,device=device,dtype=torch.float64)

    @torch.no_grad()
    def add(self,pred_db,target_db):
        if pred_db.shape!=target_db.shape or tuple(target_db.shape[1:])!=(9,200,250):
            raise ValueError("Expected [B,9,200,250] tensors")
        if not torch.isfinite(pred_db).all() or not torch.isfinite(target_db).all():
            raise FloatingPointError("All TL cells must be finite")
        err=pred_db.float()-target_db.float()
        a=err.abs().double()
        e=err.square().double()
        self.sse+=e.sum();self.sae+=a.sum();self.n+=err.numel()
        self.angle_sse+=e.sum((0,2,3));self.angle_sae+=a.sum((0,2,3))
        self.angle_n+=err.shape[0]*200*250
        self.range_sse+=e.reshape(-1,5,50).sum((0,2))
        self.range_sae+=a.reshape(-1,5,50).sum((0,2))
        self.range_n+=err.shape[0]*9*200*50
        self.depth_sse+=e.reshape(-1,4,50,250).sum((0,2,3))
        self.depth_sae+=a.reshape(-1,4,50,250).sum((0,2,3))
        self.depth_n+=err.shape[0]*9*50*250

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            names=('sse','sae','n','angle_sse','angle_sae','angle_n',
                   'range_sse','range_sae','range_n','depth_sse','depth_sae','depth_n')
            shapes=[getattr(self,name).shape for name in names]
            flat=torch.cat([getattr(self,name).reshape(-1) for name in names])
            dist.all_reduce(flat,op=dist.ReduceOp.SUM)
            i=0
            for name,shape in zip(names,shapes):
                length=math.prod(shape)
                getattr(self,name).copy_(flat[i:i+length].reshape(shape));i+=length

    def result(self):
        def to_result(sse,sae,n):
            n=float(n.item())
            return {'mae_db':float(sae.item()/n) if n else None,
                    'rmse_db':math.sqrt(max(float(sse.item()/n),0.0)) if n else None,
                    'evaluated_points':int(n)}
        return {
            'overall':to_result(self.sse,self.sae,self.n),
            'per_angle':{f'{i*40}deg':to_result(self.angle_sse[i],self.angle_sae[i],self.angle_n[i]) for i in range(9)},
            'per_range_km':{f'{i*10}-{(i+1)*10}':to_result(self.range_sse[i],self.range_sae[i],self.range_n[i]) for i in range(5)},
            'per_depth_m':{f'{i*1000}-{(i+1)*1000}':to_result(self.depth_sse[i],self.depth_sae[i],self.depth_n[i]) for i in range(4)},
        }
