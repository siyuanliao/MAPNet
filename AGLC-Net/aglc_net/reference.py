"""Immutable reference TL/SSP bank constructed ONLY from training labels."""
from pathlib import Path
import numpy as np
from .data import check_files, resolve_file


def create_reference_bank(data_dir, bank_path, tl_invalid_raw=None):
    data_dir = check_files(data_dir)
    bank_path = Path(bank_path)
    if bank_path.exists():
        return bank_path
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.arange(121, dtype=np.int64)  # First year, first month; all 121 sources.
    tl = np.load(resolve_file(data_dir, "inputs_r2.npy"), mmap_mode="r", allow_pickle=False)
    if tl.dtype.kind not in 'iu':
        raise ValueError("TL must be integer /100 data")
    ssp = np.load(resolve_file(data_dir, "ssp_r2.npy"), mmap_mode="r", allow_pickle=False)
    bathy = np.load(resolve_file(data_dir, "bathymetry_r2.npy"), mmap_mode="r", allow_pickle=False)
    # Reference fields are from training labels ONLY; validation/test never read.
    data = {
        'sample_ids': ids,
        'tl_raw': np.array(tl[ids], copy=True),
        'ssp': np.array(ssp[ids], dtype=np.float32, copy=True),
        'bathy': np.array(bathy[ids], dtype=np.float32, copy=True),
        'tl_invalid_raw': np.array(-999999 if tl_invalid_raw is None else tl_invalid_raw, dtype=np.int64),
    }
    temp = bank_path.with_name(bank_path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez(f, **data)
    temp.replace(bank_path)
    return bank_path
