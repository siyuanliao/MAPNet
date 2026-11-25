#!/usr/bin/env python3
"""Dataset inspector: no seabed-index or water-region selection."""
import argparse
import numpy as np
from pathlib import Path
from rc_cnn.data import EXPECTED,check_files,resolve_data_files
from rc_cnn.paths import DEFAULT_DATA_DIR
p=argparse.ArgumentParser()
p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
p.add_argument('--sample-id',type=int,default=0)
a=p.parse_args()
check_files(a.data_dir)
files=resolve_data_files(a.data_dir)
if not 0<=a.sample_id<18876:raise ValueError('sample-id outside [0,18875]')
print('Five required files:')
for key,shape in EXPECTED.items():
    x=np.load(files[key],mmap_mode='r',allow_pickle=False)
    print(key,files[key].name,x.shape,x.dtype)
i=a.sample_id
ssp=np.load(files['ssp'],mmap_mode='r',allow_pickle=False)[i]
count=np.load(files['ssp_valid_count'],mmap_mode='r',allow_pickle=False)[i]
tl=np.load(files['tl'],mmap_mode='r',allow_pickle=False)[i]
print('sample_id',i,'year',i//1452+1,'source_id',i%121,'month', (i//121)%12+1)
print('SSP finite levels',np.isfinite(ssp).sum(),'valid_count',int(count))
print('All-grid TL integers: min/max',int(tl.min()),int(tl.max()),'shape',tl.shape)
print('No seabed-index file, no terrain occupancy grid, no loss mask.')
