"""MAP-Net inference benchmark. Run from the MAP-Net project directory."""
import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from map_net.data import SPLIT_SLICES, N_SOURCES
from map_net.model import build_model
from map_net.paths import DEFAULT_DATA_DIR, DEFAULT_CHECKPOINT
from evaluate_MAP_Net import CrossPlatformPickle


@torch.inference_mode()
def benchmark(sample_counts, batch_size=4, repeats=3, warmup=5,
              data_dir=DEFAULT_DATA_DIR, checkpoint=DEFAULT_CHECKPOINT,
              device="auto", amp="auto", split="test"):
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if device == "auto" else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cpu":
        amp = "off"
    elif amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    lo, hi = ((0, SPLIT_SLICES["test"][1]) if split == "all"
              else SPLIT_SLICES[split])
    if not sample_counts or min(sample_counts) < 1 or max(sample_counts) > hi - lo:
        raise ValueError(f"Sample counts must be in [1, {hi-lo}] for split={split}")
    if min(batch_size, repeats, warmup) < 1:
        raise ValueError("batch_size, repeats and warmup must be positive")

    # Load your own trusted training checkpoint, including normalization stats.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False,
                      pickle_module=CrossPlatformPickle)
    model = build_model(data_dir, ckpt["stats"])
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    del ckpt

    # Load real environmental inputs once. No reference TL is needed.
    root = Path(data_dir)
    n_max = max(sample_counts)
    ids = np.arange(lo, lo + n_max)
    ssp = np.array(np.load(root / "ssp_r2.npy", mmap_mode="r")[ids],
                   dtype=np.float32, copy=True)
    counts = np.load(root / "ssp_valid_count_r2.npy", mmap_mode="r")[ids]
    ssp[np.arange(50)[None, :] >= counts[:, None]] = np.nan
    if not np.isfinite(ssp).any(axis=1).all():
        raise ValueError("An SSP has no valid depth levels")
    bathy = np.array(np.load(root / "bathymetry_r2.npy", mmap_mode="r")
                     [ids % N_SOURCES], dtype=np.float32, copy=True)
    ssp, bathy = torch.from_numpy(ssp).to(device), torch.from_numpy(bathy).to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def infer(start, stop):
        # Same prediction interface as evaluate_MAP_Net.py; output is in dB.
        return model.predict_db(ssp[start:stop], bathy[start:stop])

    hardware = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else f"CPU ({torch.get_num_threads()} PyTorch threads)")
    print(f"Device: {hardware} | Precision: {amp if amp != 'off' else 'fp32'} | "
          f"Batch size: {batch_size} | Split: {split} | Repeats: {repeats}")
    print(f"{'Samples':>8} {'Total(s)':>12} {'SD(s)':>12} "
          f"{'ms/sample':>12} {'samples/s':>12}")
    results = []
    # Use a separate autocast context per batch, as in the evaluation script.
    for n in sample_counts:
        # Warm up both full and final partial batch shapes, outside the timer.
        for size in sorted({min(n, batch_size), n % batch_size or min(n, batch_size)}):
            for _ in range(warmup):
                with (torch.autocast("cuda", dtype=dtype) if amp != "off" else nullcontext()):
                    infer(0, size)
        elapsed = []
        for _ in range(repeats):
            sync()
            t0 = time.perf_counter()
            for start in range(0, n, batch_size):
                with (torch.autocast("cuda", dtype=dtype) if amp != "off" else nullcontext()):
                    infer(start, min(start + batch_size, n))
            sync()
            elapsed.append(time.perf_counter() - t0)
        mean, std = float(np.mean(elapsed)), float(np.std(elapsed))
        print(f"{n:8d} {mean:12.6f} {std:12.6f} {mean/n*1000:12.3f} {n/mean:12.2f}")
        results.append((n, mean, std, mean/n*1000, n/mean))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, nargs="+", default=[1, 10, 100, 1000])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    args = p.parse_args()
    benchmark(args.samples, args.batch_size, args.repeats, args.warmup,
              args.data_dir, args.checkpoint, args.device, args.amp, args.split)
