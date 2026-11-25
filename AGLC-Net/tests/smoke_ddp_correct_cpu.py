"""Manual correction-stage DDP smoke: torchrun --standalone --nproc_per_node=2 tests/smoke_ddp_correct_cpu.py"""
from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from test_adapted import synthetic_bank,STATS
from aglc_net.model import AGLCNet
from train import train_epoch

rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
torch.set_num_threads(1)
dist.init_process_group('gloo')
model=AGLCNet(synthetic_bank(),STATS,widths=(8,16,24,32,40),latent_dim=32,
                         bathy_dim=16,hidden=64,angle_dim=8)
for p in model.codec.encoder_blocks.parameters():p.requires_grad_(False)
for p in model.codec.to_latent.parameters():p.requires_grad_(False)
net=DDP(model,broadcast_buffers=False)
opt=torch.optim.SGD([p for p in net.parameters() if p.requires_grad],lr=.01)
args=SimpleNamespace(stage='correct',grad_accum=2,angle_chunk=3,amp='off',grad_clip=1.0)
loader=[]
for i in range(2):
    src=rank*2+i
    ssp=torch.full((1,50),1500.25+rank*.1)
    ssp[:,42:]=float('nan')
    bathy=torch.full((1,61,61),500+src*2.,dtype=torch.float32)
    target=torch.full((1,9,200,250),71+rank+i,dtype=torch.float32)
    loader.append((ssp,bathy,target,torch.tensor([src])))
scaler=torch.amp.GradScaler('cuda',enabled=False)
res=train_epoch(net,loader,opt,scaler,STATS,args,torch.device('cpu'),world)
print('rank',rank,'rmse',res['rmse_db'],'beta',net.module.corrector.beta.item(),flush=True)
assert net.module.corrector.beta.grad is None  # cleared after step
assert res['valid_points']>0
dist.destroy_process_group()
