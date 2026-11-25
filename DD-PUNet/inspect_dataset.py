#!/usr/bin/env python3
"""检查输入数据维度与有效性。"""
import argparse
import numpy as np
from pathlib import Path
from dd_punet.data import EXPECTED, check_files, resolve_data_files
from dd_punet.paths import DEFAULT_DATA_DIR

p = argparse.ArgumentParser()
p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
p.add_argument('--sample-id', type=int, default=0)
p.add_argument('--tl-invalid-raw', type=int, default=None, help='原始整数TL哨兵值（如有）')
a = p.parse_args()

check_files(a.data_dir)
files = resolve_data_files(a.data_dir)
if not 0 <= a.sample_id < 18876:
    raise ValueError('sample-id must be in [0,18875]')
print('全部五个必需文件的形状检查通过（不读取bottom_depth_index）：')
for name, shape in EXPECTED.items():
    x = np.load(files[name], mmap_mode='r', allow_pickle=False)
    print(f'  {str(files[name].name):26s} {str(x.shape):28s} {x.dtype}')
n=a.sample_id
ssp=np.load(files["ssp"],mmap_mode='r',allow_pickle=False)
cnt=np.load(files["ssp_valid_count"],mmap_mode='r',allow_pickle=False)
tl=np.load(files["tl"],mmap_mode='r',allow_pickle=False)[n]
print(f'\nsample_id={n}, year_index={n//1452 + 1}, source_idx={n%121}, month_index={(n//121)%12 + 1}')
print('SSP finite levels:', int(np.isfinite(ssp[n]).sum()), 'valid_count:',int(cnt[n]))
print('全网格TL点数:',tl.size)
print('TL raw full-grid min/max:', int(tl.min()), int(tl.max()))
if a.tl_invalid_raw is not None:
    print('TL sentinel count:',int((tl==a.tl_invalid_raw).sum()))
print('提示：如果TL整数中使用特殊缺失值，请在训练时同时传入 --tl-invalid-raw 和 --invalid-target-fill-db。')
