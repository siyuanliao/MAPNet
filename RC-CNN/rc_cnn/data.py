"""按年份读取数据，延迟读取大型TL文件。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

N_SOURCES = 121
SAMPLES_PER_YEAR = 11 * 11 * 12
N_SAMPLES = 13 * SAMPLES_PER_YEAR
SPLIT_SLICES = {
    "train": (0, 11 * SAMPLES_PER_YEAR),
    "val": (11 * SAMPLES_PER_YEAR, 12 * SAMPLES_PER_YEAR),
    "test": (12 * SAMPLES_PER_YEAR, 13 * SAMPLES_PER_YEAR),
}

# Logical names and expected shapes. r2 and non-r2 layouts are both supported.
EXPECTED = {
    "tl": (N_SAMPLES, 9, 200, 250),
    "bathymetry": (N_SAMPLES, 61, 61),
    "ssp": (N_SAMPLES, 50),
    "ssp_depth_m": (50,),
    "ssp_valid_count": (N_SAMPLES,),
}
FILE_CANDIDATES = {
    "tl": ("inputs_r2.npy", "inputs.npy", "tl.npy", "TL.npy"),
    "bathymetry": ("bathymetry_r2.npy", "bathymetry.npy"),
    "ssp": ("ssp_r2.npy", "ssp.npy"),
    "ssp_depth_m": ("ssp_depth_m_r2.npy", "ssp_depth_m.npy"),
    "ssp_valid_count": ("ssp_valid_count_r2.npy", "ssp_valid_count.npy"),
}


def resolve_data_files(root):
    """Return paths to the five input/label .npy arrays.

    No seabed-index file is read. Raises if one of the files cannot be found. If multiple are present,
    priority is the _r2 layout.
    """
    root = Path(root)
    result = {}
    for key, candidates in FILE_CANDIDATES.items():
        found = next((root / name for name in candidates if (root / name).is_file()), None)
        if found is None:
            raise FileNotFoundError(f"{key}: expected one of {candidates} in {root}")
        result[key] = found
    return result


def check_files(root):
    files = resolve_data_files(root)
    for key, shape in EXPECTED.items():
        filename = files[key]
        arr = np.load(filename, mmap_mode="r", allow_pickle=False)
        if arr.shape != shape:
            raise ValueError(f"{filename}: expected shape {shape}, found {arr.shape}")
        if key == "tl" and arr.dtype.kind not in "iu":
            raise TypeError(f"TL file must be integer encoded in 0.01 dB: {arr.dtype}")
        if key in ("bathymetry", "ssp") and arr.dtype.kind != "f":
            raise TypeError(f"{filename}: expected float input, got {arr.dtype}")
    counts = np.load(files["ssp_valid_count"], mmap_mode="r", allow_pickle=False)
    if counts.min() < 1 or counts.max() > 50:
        raise ValueError("ssp_valid_count must lie in [1,50]")
    depth = np.load(files["ssp_depth_m"], allow_pickle=False)
    if not np.isfinite(depth).all() or not (np.diff(depth) > 0).all():
        raise ValueError("ssp_depth_m must contain 50 finite strictly increasing depths")
    return Path(root)


def _mean_std(values, label):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError(f"没有有效的 {label} 数据")
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError(f"{label} 标准差无效或太小: {std}")
    return mean, std


def compute_train_stats(root, *, chunk_size=16, verbose=True):
    """All targets participate in normalization; no grid-position filtering.

    Computes standardization parameters ONLY on years 1-11. The full
    rectangular TL target, including below-seafloor values as stored in the
    file, contributes to mean/std and to subsequent MSE.
    """
    root = check_files(root)
    files = resolve_data_files(root)
    end_train = SPLIT_SLICES["train"][1]
    ssp = np.load(files["ssp"], mmap_mode="r", allow_pickle=False)
    counts = np.load(files["ssp_valid_count"], mmap_mode="r", allow_pickle=False)
    sv = np.asarray(ssp[:end_train], dtype=np.float32)
    valid_ssp = (np.arange(50)[None,:] < counts[:end_train,None]) & np.isfinite(sv)
    if not valid_ssp.any(axis=1).all():
        raise ValueError("There are SSP samples without any usable levels")
    ssp_mean, ssp_std = _mean_std(sv[valid_ssp], "SSP")
    del sv, valid_ssp

    bathy = np.load(files["bathymetry"], mmap_mode="r", allow_pickle=False)
    unique = np.asarray(bathy[:N_SOURCES], dtype=np.float64)
    if not np.isfinite(unique).all():
        raise ValueError("Bathymetry has NaN/Inf values")
    bathy_mean,bathy_std = _mean_std(unique.ravel(), "Bathymetry")
    del unique

    tl = np.load(files["tl"], mmap_mode="r", allow_pickle=False)
    total_n, total_sum, total_sq_sum = 0, 0.0, 0.0
    min_db, max_db = float('inf'), float('-inf')
    for start in range(0,end_train,chunk_size):
        stop = min(start+chunk_size,end_train)
        raw = np.asarray(tl[start:stop], dtype=np.float64)
        raw *= .01
        if not np.isfinite(raw).all():
            raise ValueError("TL contains invalid values; preprocess before unmasked training")
        total_n += raw.size
        total_sum += float(raw.sum(dtype=np.float64))
        total_sq_sum += float(np.square(raw).sum(dtype=np.float64))
        min_db = min(min_db,float(raw.min()))
        max_db = max(max_db,float(raw.max()))
        if verbose and ((start//chunk_size+1)%150==0 or stop==end_train):
            print(f"[Full-grid TL stats] {stop}/{end_train}, pixels={total_n:,}",flush=True)
    tl_mean=total_sum/total_n
    tl_std=float(np.sqrt(max(total_sq_sum/total_n-tl_mean**2,0)))
    if tl_std<1e-6:
        raise ValueError("Full-grid TL standard deviation too small")
    return {
        'ssp_mean':ssp_mean,'ssp_std':ssp_std,
        'bathy_mean':bathy_mean,'bathy_std':bathy_std,
        'tl_mean':float(tl_mean),'tl_std':tl_std,
        'tl_train_points':int(total_n),
        'tl_train_min_db':min_db,'tl_train_max_db':max_db,
        'train_samples':end_train,
        'normalization_scope':'full_grid',
        'description':'Years 1-11 only; all 9x200x250 TL values per sample are included',
    }


def save_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class TLDataset(Dataset):
    """按需读取目标TL，返回原始样本编号及物理量。"""
    def __init__(self, root, split, *, check=True):
        if split not in SPLIT_SLICES:
            raise ValueError(f"未知数据划分: {split}")
        self.root = Path(root)
        self.files = resolve_data_files(self.root)
        if check:
            check_files(self.root)
        self.start, self.stop = SPLIT_SLICES[split]
        self.ssp = np.asarray(np.load(self.files["ssp"], allow_pickle=False), dtype=np.float32)
        self.valid_counts = np.asarray(np.load(self.files["ssp_valid_count"], allow_pickle=False))

        b = np.load(self.files["bathymetry"], mmap_mode="r", allow_pickle=False)
        self.unique_bathy = np.array(b[:N_SOURCES], dtype=np.float32, copy=True)
        if not np.isfinite(self.unique_bathy).all():
            raise ValueError("Bathymetry包含NaN/Inf")
        self._tl = None  # 每个DataLoader worker延迟打开自己的TL内存映射，避免复制大文件。

    def __len__(self):
        return self.stop - self.start

    def _get_tl(self):
        if self._tl is None:
            self._tl = np.load(self.files["tl"], mmap_mode="r", allow_pickle=False)
        return self._tl

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tl"] = None
        return state

    def __getitem__(self, local_index):
        index = self.start + int(local_index)
        ssp = self.ssp[index].copy()
        count = int(self.valid_counts[index])
        ssp[count:] = np.nan
        if not np.isfinite(ssp).any():
            raise ValueError(f"样本{index}的SSP没有有效层")

        bathy = self.unique_bathy[index % N_SOURCES].copy()
        raw = self._get_tl()[index]
        target = np.array(raw, dtype=np.float32, copy=True)
        target *= 0.01
        return (
            torch.from_numpy(ssp), torch.from_numpy(bathy),
            torch.from_numpy(target), torch.tensor(index, dtype=torch.long)
        )


class DistributedEvalSampler(Sampler):
    """按进程切分验证数据，避免重复填充样本。"""
    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))
