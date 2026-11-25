"""Run explicitly: torchrun --standalone --nproc_per_node=2 tests/ddp_smoke.py"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from train import train_one_epoch

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.p=nn.Parameter(torch.tensor(0.1))
    def forward(self, ssp, bathy):
        return self.p.expand(ssp.shape[0], 9, 200, 250)

def main():
    torch.set_num_threads(2)
    dist.init_process_group('gloo')
    rank=dist.get_rank()
    world=dist.get_world_size()
    assert world==2
    ssp=torch.zeros(2,50)
    bathy=torch.zeros(2,61,61)
    truth=torch.stack([torch.zeros(9,200,250), torch.ones(9,200,250)])
    truth[1,0,199,249]=float('nan')
    ids=torch.arange(2)
    dataset=TensorDataset(ssp,bathy,truth,ids)
    sampler=DistributedSampler(dataset, num_replicas=2,rank=rank,shuffle=False)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler)
    model=DDP(Toy())
    opt=torch.optim.SGD(model.parameters(),lr=0.1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    args=SimpleNamespace(grad_accum=1,amp='off',grad_clip=0)
    result=train_one_epoch(model,loader,opt,scaler,{'tl_mean':0., 'tl_std':1.},args,torch.device('cpu'),world)
    if rank==0:
        # weighted average target slightly below 0.5 due to one NaN
        count=2*9*200*250-1
        mean_target=(9*200*250-1)/count
        expected=.1 - .1*2*(.1-mean_target)
        assert abs(float(model.module.p.detach())-expected)<1e-5,(model.module.p,expected)
        assert result['valid_points']==count
        print('CPU DDP finite-only smoke passed, parameters=',float(model.module.p.detach()), 'valid_points=',count)
    dist.destroy_process_group()

if __name__=='__main__': main()
