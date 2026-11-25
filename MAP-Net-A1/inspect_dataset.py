#!/usr/bin/env python3
"""检查输入数据维度与有效性。"""
import argparse
import numpy as np
from pathlib import Path
from map_net_a1.data import EXPECTED, check_files
from map_net_a1.paths import DEFAULT_DATA_DIR

p = argparse.ArgumentParser()
p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
p.add_argument('--sample-id', type=int, default=0)
p.add_argument('--tl-invalid-raw', type=int, default=None, help='原始整数TL哨兵值（如有）')
a = p.parse_args()

check_files(a.data_dir)
if not 0 <= a.sample_id < 18876:
    raise ValueError('sample-id must be in [0,18875]')
print('全部六个文件的形状检查通过：')
for name, shape in EXPECTED.items():
    x = np.load(a.data_dir / name, mmap_mode='r', allow_pickle=False)
    print(f'  {name:26s} {str(x.shape):28s} {x.dtype}')
n=a.sample_id
ssp=np.load(a.data_dir/'ssp_r2.npy',mmap_mode='r',allow_pickle=False)
cnt=np.load(a.data_dir/'ssp_valid_count_r2.npy',mmap_mode='r',allow_pickle=False)
h=np.load(a.data_dir/'bottom_depth_index_r2.npy',mmap_mode='r',allow_pickle=False)[n%121]
tl=np.load(a.data_dir/'inputs_r2.npy',mmap_mode='r',allow_pickle=False)[n]
water=np.arange(200)[None,:,None] <= h[:,None,:]
valid=water & np.isfinite(tl)
if a.tl_invalid_raw is not None:
    valid &= (tl != a.tl_invalid_raw)
print(f'\nsample_id={n}, year_index={n//1452 + 1}, source_idx={n%121}, month_index={(n//121)%12 + 1}')
print('SSP finite levels:', int(np.isfinite(ssp[n]).sum()), 'valid_count:',int(cnt[n]))
print('Water cells:', int(water.sum()), 'of',water.size)
print('TL raw valid-water min/max:',int(tl[valid].min()) if valid.any() else None,
      int(tl[valid].max()) if valid.any() else None)
print('提示：如果TL整数中使用特殊缺失值，请在训练时传入 --tl-invalid-raw 整数值。')
