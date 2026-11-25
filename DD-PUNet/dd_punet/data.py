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
    """Return paths to the five required input/target .npy arrays.

    Raises if one of the five files cannot be found. If multiple are present,
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


def compute_train_stats(root, *, tl_invalid_raw=None, invalid_target_fill_db=None, chunk_size=16, verbose=True):
    """只用前11年统计量；TL统计全网格，不读取海底掩码。"""
    root = check_files(root)
    if chunk_size < 1:
        raise ValueError('stats-chunk-size must be >= 1')
    if (tl_invalid_raw is None) != (invalid_target_fill_db is None):
        raise ValueError('Specify both integer sentinel and a finite fill value, or neither')
    if invalid_target_fill_db is not None and not np.isfinite(invalid_target_fill_db):
        raise ValueError('invalid_target_fill_db must be finite')
    files = resolve_data_files(root)
    # 严格按年份划分，所有均值/标准差只来自前11年，避免数据泄漏。
    end_train = SPLIT_SLICES["train"][1]
    ssp = np.load(files["ssp"], mmap_mode="r", allow_pickle=False)
    counts = np.load(files["ssp_valid_count"], mmap_mode="r", allow_pickle=False)
    sv = np.asarray(ssp[:end_train], dtype=np.float32)
    valid = (np.arange(50)[None, :] < counts[:end_train, None]) & np.isfinite(sv)
    if not valid.any(axis=1).all():
        raise ValueError("训练集存在没有有效深度点的SSP")
    ssp_mean, ssp_std = _mean_std(sv[valid], "SSP")
    del sv, valid


    bathy = np.load(files["bathymetry"], mmap_mode="r", allow_pickle=False)
    unique_bathy = np.asarray(bathy[:N_SOURCES], dtype=np.float64)
    if not np.isfinite(unique_bathy).all():
        raise ValueError("海底地形包含NaN/Inf，请先检查数据")
    bathy_mean, bathy_std = _mean_std(unique_bathy.ravel(), "Bathymetry")
    del unique_bathy

    tl = np.load(files["tl"], mmap_mode="r", allow_pickle=False)
    total_n = 0
    total_sum, total_sq_sum = 0.0, 0.0
    min_db, max_db = float("inf"), float("-inf")
    for start in range(0, end_train, chunk_size):
        stop = min(end_train, start + chunk_size)
        # 对完整矩形网格统计标准化，绝不按水体掩码筛选。
        raw = np.asarray(tl[start:stop])
        values = raw.astype(np.float64) / 100.0
        if tl_invalid_raw is not None:
            # Count replaced cells too: the normalization must match the
            # EXACT FULL GRID that the Dataset will hand to the objective.
            values[raw == tl_invalid_raw] = invalid_target_fill_db
        values = values.reshape(-1)
        total_n += values.size
        total_sum += values.sum(dtype=np.float64)
        total_sq_sum += np.square(values).sum(dtype=np.float64)
        if values.size:
            min_db = min(min_db, float(values.min()))
            max_db = max(max_db, float(values.max()))
        if verbose and ((start // chunk_size + 1) % 150 == 0 or stop == end_train):
            print(f"[统计训练集TL] {stop}/{end_train}，全网格点={total_n:,}", flush=True)
    if total_n == 0:
        raise ValueError("训练集TL网格为空")
    tl_mean = total_sum / total_n
    variance = max(total_sq_sum / total_n - tl_mean**2, 0.0)
    tl_std = float(np.sqrt(variance))
    if tl_std < 1e-6:
        raise ValueError("训练集TL标准差太小，检查TL值及哨兵值")
    return {
        "ssp_mean": ssp_mean, "ssp_std": ssp_std,
        "bathy_mean": bathy_mean, "bathy_std": bathy_std,
        "tl_mean": float(tl_mean), "tl_std": tl_std,
        "tl_valid_points": int(total_n),
        "tl_train_min_db": min_db, "tl_train_max_db": max_db,
        "tl_invalid_raw": tl_invalid_raw,
        "tl_scope": "full_grid",
        "invalid_target_fill_db": invalid_target_fill_db,
        "train_samples": end_train,
        "description": "Stats from first 11 years only; ALL grid cells counted. Optional sentinel is replaced by the specified fixed dB value, not ignored.",
    }


def save_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class TLDataset(Dataset):
    """按需读取目标TL，返回原始样本编号及物理量。"""
    def __init__(self, root, split, *, tl_invalid_raw=None, invalid_target_fill_db=None, check=True):
        if split not in SPLIT_SLICES:
            raise ValueError(f"未知数据划分: {split}")
        self.root = Path(root)
        self.files = resolve_data_files(self.root)
        if check:
            check_files(self.root)
        self.start, self.stop = SPLIT_SLICES[split]
        self.tl_invalid_raw = tl_invalid_raw
        self.invalid_target_fill_db = invalid_target_fill_db
        if tl_invalid_raw is not None and (invalid_target_fill_db is None or
                                          not np.isfinite(invalid_target_fill_db)):
            raise ValueError("指定整数哨兵值时，必须同时指定有限的 invalid_target_fill_db；不会用mask忽略这些点")
        self.ssp = np.load(self.files["ssp"], mmap_mode="r", allow_pickle=False)
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
        if not np.isfinite(ssp[:count]).all():
            raise ValueError(f'样本{index}的前{count}个SSP有效深度层包含NaN/Inf')
        if not np.isfinite(ssp).any():
            raise ValueError(f"样本{index}的SSP没有有效层")

        bathy = self.unique_bathy[index % N_SOURCES].copy()
        raw = self._get_tl()[index]
        target = np.array(raw, dtype=np.float32, copy=True)
        target *= 0.01
        if self.tl_invalid_raw is not None:
            target[np.asarray(raw) == self.tl_invalid_raw] = self.invalid_target_fill_db
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
