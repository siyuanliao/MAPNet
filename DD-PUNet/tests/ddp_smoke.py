import os,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from dd_punet.data import DistributedEvalSampler
from train import train_one_epoch,validate

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.bias=nn.Parameter(torch.tensor(0.))
    def forward(self,ssp,bathy,angle_indices=None):
        a=9 if angle_indices is None else len(angle_indices)
        return self.bias.expand(ssp.shape[0],a,200,250)
    def predict_db(self,ssp,bathy,angle_indices=None):
        return self.forward(ssp,bathy,angle_indices=angle_indices)*20.+90.

if __name__=='__main__':
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank=dist.get_rank();world=dist.get_world_size()
    ds=TensorDataset(torch.ones(3,50),torch.ones(3,61,61),
        torch.tensor([95.,105.,115.]).view(3,1,1,1).expand(3,9,200,250).clone(),torch.arange(3))
    train_dl=DataLoader(ds,batch_size=1,sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=False))
    val_dl=DataLoader(ds,batch_size=1,sampler=DistributedEvalSampler(ds,rank,world))
    model=DistributedDataParallel(Toy(),broadcast_buffers=False)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    args=SimpleNamespace(amp='off',angle_chunk=3,grad_accum=2,grad_clip=1.0)
    stats={'tl_mean':90.,'tl_std':20.}
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    tr=train_one_epoch(model,train_dl,opt,scaler,stats,args,torch.device('cpu'),world)
    val=validate(model.module,val_dl,stats,args,torch.device('cpu'))
    params=[torch.zeros_like(model.module.bias.detach()) for _ in range(world)]
    dist.all_gather(params,model.module.bias.detach())
    assert torch.allclose(params[0],params[1]),params
    assert tr['grid_points']==world*2*9*200*250,tr
    assert val['overall']['valid_points']==3*9*200*250,val
    if rank==0:print('DDP Gloo: training/validation, no_sync and global grid-count OK',tr,val['overall'],flush=True)
    dist.destroy_process_group()
