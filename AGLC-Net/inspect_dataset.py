#!/usr/bin/env python3
"""Inspect input tensors and ALL TL grid cells without constructing a loss mask."""
import argparse
from pathlib import Path
import numpy as np
from aglc_net.data import EXPECTED, check_files, resolve_file, validate_tl_fill
from aglc_net.paths import DEFAULT_DATA_DIR


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR)
    p.add_argument('--sample-id',type=int,default=0)
    p.add_argument('--tl-invalid-raw',type=int,default=None)
    p.add_argument('--tl-invalid-fill-db',type=float,default=None)
    args=p.parse_args()
    validate_tl_fill(args.tl_invalid_raw,args.tl_invalid_fill_db)
    check_files(args.data_dir)
    if not 0 <= args.sample_id < 18876:
        raise ValueError('sample ID outside [0,18875]')
    print('Validated input files:')
    for name, expected in EXPECTED.items():
        path=resolve_file(args.data_dir,name)
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        print(f'  {path.name:27s} shape={a.shape} dtype={a.dtype}')
    i=args.sample_id
    ssp=np.load(resolve_file(args.data_dir,'ssp_r2.npy'),mmap_mode='r',allow_pickle=False)[i]
    valid_count=np.load(resolve_file(args.data_dir,'ssp_valid_count_r2.npy'),mmap_mode='r',allow_pickle=False)[i]
    raw=np.load(resolve_file(args.data_dir,'inputs_r2.npy'),mmap_mode='r',allow_pickle=False)[i]
    tl=raw.astype(np.float32)*.01
    sentinel_count=0
    if args.tl_invalid_raw is not None:
        sentinel_count=int(np.count_nonzero(raw==args.tl_invalid_raw))
        tl[raw==args.tl_invalid_raw]=args.tl_invalid_fill_db
    if not np.isfinite(tl).all():
        raise ValueError('TL contains non-finite targets; full-grid training cannot discard grid points')
    print(f'ID {i}: source={i%121}, month={(i//121)%12+1}, year={i//1452+1}')
    print(f'SSP finite layers={np.isfinite(ssp).sum()}, valid_count={valid_count}')
    print(f'ALL TL grid cells={tl.size}, sentinel replacements={sentinel_count}')
    print(f'full-grid TL dB range: {tl.min():.2f} to {tl.max():.2f}')
    print(f'Reference source TRAIN sample_id={i%121}, year=1 month=1')


if __name__=='__main__':main()
