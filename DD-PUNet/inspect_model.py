#!/usr/bin/env python3
"""No dataset needed: parameter counts and small-angle shape smoke test."""
import argparse
import torch
from dd_punet.model import DDPUNet,count_parameters,REFERENCE_PARAMETERS

p=argparse.ArgumentParser()
p.add_argument('--run-forward',action='store_true',help='Run one azimuth forward pass on CPU')
a=p.parse_args()
torch.set_num_threads(min(torch.get_num_threads(),4))
model=DDPUNet(
    torch.linspace(0,5000,50),ssp_mean=1500.,ssp_std=30.,
    bathy_mean=2300.,bathy_std=600.,tl_mean=90.,tl_std=20.,width=32)
n=count_parameters(model)
print(f'DD-PUNet params: {n:,}')
print(f'MAP-Net params: {REFERENCE_PARAMETERS:,}')
print(f'Difference: {(n/REFERENCE_PARAMETERS-1)*100:+.3f}%')
assert abs(n/REFERENCE_PARAMETERS-1)<=.10
if a.run_forward:
    ssp=torch.full((1,50),1500.);ssp[:,40:]=float('nan')
    bathy=torch.full((1,61,61),2200.)
    model.eval()
    with torch.no_grad():out=model(ssp,bathy,angle_indices=[0])
    print('One-angle output:',tuple(out.shape),'finite:',bool(torch.isfinite(out).all()))
print('Full 9-angle forward (default): [B,9,200,250].')
