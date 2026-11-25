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

EXPECTED = {
    "inputs_r2.npy": (N_SAMPLES, 9, 200, 250),
    "bathymetry_r2.npy": (N_SAMPLES, 61, 61),
    "ssp_r2.npy": (N_SAMPLES, 50),
    "bottom_depth_index_r2.npy": (N_SOURCES, 9, 250),
    "ssp_depth_m_r2.npy": (50,),
    "ssp_valid_count_r2.npy": (N_SAMPLES,),
}


def check_files(root):
    root = Path(root)
    for name, shape in EXPECTED.items():
        filename = root / name
        if not filename.is_file():
            raise FileNotFoundError(f"缺少文件: {filename}")
        arr = np.load(filename, mmap_mode="r", allow_pickle=False)
        if arr.shape != shape:
            raise ValueError(f"{filename}: 期待 {shape}, 实际 {arr.shape}")
        if name == "inputs_r2.npy" and arr.dtype.kind not in "iu":
            raise TypeError(f"TL文件应为整数类型, 实际 {arr.dtype}")
    h = np.load(root / "bottom_depth_index_r2.npy", mmap_mode="r", allow_pickle=False)
    if int(h.min()) < -1 or int(h.max()) > 199:
        raise ValueError("bottom_depth_index 必须是0基索引，取值应在[-1,199]")
    counts = np.load(root / "ssp_valid_count_r2.npy", mmap_mode="r", allow_pickle=False)
    if counts.min() < 1 or counts.max() > 50:
        raise ValueError("ssp_valid_count 应处于[1,50]，每条SSP需至少1个有效值")
    depth = np.load(root / "ssp_depth_m_r2.npy", allow_pickle=False)
    if not np.isfinite(depth).all() or not (np.diff(depth) > 0).all():
        raise ValueError("ssp_depth_m 需要50个有限、严格递增的深度坐标")
    return root


def _mean_std(values, label):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        raise ValueError(f"没有有效的 {label} 数据")
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std < 1e-6:
        raise ValueError(f"{label} 标准差无效或太小: {std}")
    return mean, std


def compute_train_stats(root, *, tl_invalid_raw=None, chunk_size=16, verbose=True):
    """只使用前11年样本统计标准化参数，TL仅计入有效水体点。"""
    root = check_files(root)
    # 严格按年份划分，所有均值/标准差只来自前11年，避免数据泄漏。
    end_train = SPLIT_SLICES["train"][1]
    ssp = np.load(root / "ssp_r2.npy", mmap_mode="r", allow_pickle=False)
    counts = np.load(root / "ssp_valid_count_r2.npy", mmap_mode="r", allow_pickle=False)
    sv = np.asarray(ssp[:end_train], dtype=np.float32)
    valid = (np.arange(50)[None, :] < counts[:end_train, None]) & np.isfinite(sv)
    if not valid.any(axis=1).all():
        raise ValueError("训练集存在没有有效深度点的SSP")
    ssp_mean, ssp_std = _mean_std(sv[valid], "SSP")
    del sv, valid


    bathy = np.load(root / "bathymetry_r2.npy", mmap_mode="r", allow_pickle=False)
    unique_bathy = np.asarray(bathy[:N_SOURCES], dtype=np.float64)
    if not np.isfinite(unique_bathy).all():
        raise ValueError("海底地形包含NaN/Inf，请先检查数据")
    bathy_mean, bathy_std = _mean_std(unique_bathy.ravel(), "Bathymetry")
    del unique_bathy

    h = np.load(root / "bottom_depth_index_r2.npy", mmap_mode="r", allow_pickle=False)
    tl = np.load(root / "inputs_r2.npy", mmap_mode="r", allow_pickle=False)
    z = np.arange(200, dtype=np.int16)[None, None, :, None]
    total_n = 0
    total_sum, total_sq_sum = 0.0, 0.0
    min_db, max_db = float("inf"), float("-inf")
    for start in range(0, end_train, chunk_size):
        stop = min(end_train, start + chunk_size)
        # TL文件约16 GiB，按块内存映射读取，不能一次性加载到内存。
        raw = np.asarray(tl[start:stop])

        src = np.arange(start, stop) % N_SOURCES
        water = z <= h[src, :, None, :]
        ok = water & np.isfinite(raw)
        if tl_invalid_raw is not None:
            ok &= raw != tl_invalid_raw
        values = raw[ok].astype(np.float64) / 100.0
        total_n += values.size
        total_sum += values.sum(dtype=np.float64)
        total_sq_sum += np.square(values).sum(dtype=np.float64)
        if values.size:
            min_db = min(min_db, float(values.min()))
            max_db = max(max_db, float(values.max()))
        if verbose and ((start // chunk_size + 1) % 150 == 0 or stop == end_train):
            print(f"[统计训练集TL] {stop}/{end_train}，有效水体网格={total_n:,}", flush=True)
    if total_n == 0:
        raise ValueError("训练集TL中没有有效水体网格点")
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
        "train_samples": end_train,
        "description": "Stats from first 11 years only; TL uses valid water and finite values",
    }


def save_stats(stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class TLDataset(Dataset):
    """按需读取目标TL，返回原始样本编号及物理量。"""
    def __init__(self, root, split, *, tl_invalid_raw=None, check=True):
        if split not in SPLIT_SLICES:
            raise ValueError(f"未知数据划分: {split}")
        self.root = Path(root)
        if check:
            check_files(self.root)
        self.start, self.stop = SPLIT_SLICES[split]
        self.tl_invalid_raw = tl_invalid_raw
        self.ssp = np.asarray(np.load(self.root / "ssp_r2.npy", allow_pickle=False), dtype=np.float32)
        self.valid_counts = np.asarray(np.load(self.root / "ssp_valid_count_r2.npy", allow_pickle=False))

        b = np.load(self.root / "bathymetry_r2.npy", mmap_mode="r", allow_pickle=False)
        self.unique_bathy = np.array(b[:N_SOURCES], dtype=np.float32, copy=True)
        if not np.isfinite(self.unique_bathy).all():
            raise ValueError("Bathymetry包含NaN/Inf")
        self._tl = None  # 每个DataLoader worker延迟打开自己的TL内存映射，避免复制大文件。

    def __len__(self):
        return self.stop - self.start

    def _get_tl(self):
        if self._tl is None:
            self._tl = np.load(self.root / "inputs_r2.npy", mmap_mode="r", allow_pickle=False)
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
        if self.tl_invalid_raw is not None:
            target[np.asarray(raw) == self.tl_invalid_raw] = np.nan
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
