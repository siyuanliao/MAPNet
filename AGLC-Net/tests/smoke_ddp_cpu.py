"""Manual smoke: torchrun --standalone --nproc_per_node=2 tests/smoke_ddp_cpu.py"""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from train import train_epoch
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor=torch.nn.Parameter(torch.tensor(.5))
    def forward(self,inp):
        return inp*self.factor
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
torch.set_num_threads(1)
dist.init_process_group('gloo')
net=DDP(Tiny());opt=torch.optim.SGD(net.parameters(),lr=.1)
args=SimpleNamespace(stage='pretrain',grad_accum=2,angle_chunk=3,amp='off',grad_clip=1.0)
loader=[]
for i in range(2):
    target=torch.full((1,9,200,250),72+rank+i,dtype=torch.float32)
    loader.append((torch.zeros(1,50),torch.zeros(1,61,61),target,torch.tensor([rank*2+i])))
stats={'tl_mean':65.,'tl_std':12.}
scaler=torch.amp.GradScaler('cuda',enabled=False)
out=train_epoch(net,loader,opt,scaler,stats,args,torch.device('cpu'),world)
print('RANK',rank,'factor',net.module.factor.item(),'rmse',out['rmse_db'], flush=True)
assert 0.5 < net.module.factor.item() < 1.0
dist.destroy_process_group()
