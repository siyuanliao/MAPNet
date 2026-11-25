#!/usr/bin/env python3
"""Train MAP-Net-A4 using the year-based split and optional distributed mixed precision."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import inspect
import math
import shutil
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from map_net_a4.data import (
    TLDataset, DistributedEvalSampler, check_files,
    compute_train_stats, save_stats,
)
from map_net_a4.metrics import MaskedMetrics, masked_error_sums, water_valid_mask
from map_net_a4.model import build_model, count_parameters, MODEL_ARCHITECTURE, checkpoint_matches
from map_net_a4.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="A4 (w/o MTR / MLP decoder) ablation training")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="共享数据目录，默认是当前工程的同级dataset目录")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="权重/日志等输出目录，默认位于本网络工程runs/map_net_a4")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="反向传播时重算解码特征以节省显存")
    p.add_argument("--batch-size", type=int, default=4, help="每张GPU每次处理的样本数")
    p.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数，默认每批更新")
    p.add_argument("--workers", type=int, default=4, help="每个GPU进程的数据加载进程数")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tl-invalid-raw", type=int, default=None,
                   help="TL整数文件中表示无效值的编码（除100之前）")
    p.add_argument("--stats-chunk-size", type=int, default=16)
    p.add_argument("--stats-from", type=Path, default=None,
                   help="复用相同数据集的训练集标准化参数，避免重复扫描TL")
    p.add_argument("--resume", type=Path, default=None,
                   help="从当前网络保存的last.pt恢复训练")
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1:
        kwargs = {"backend": "nccl" if device.type == "cuda" else "gloo"}
        if device.type == "cuda" and "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return rank, world, local_rank, device


def seed_everything(seed, rank, deterministic=False):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def allreduce_(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def loader_for(dataset, *, training, args, rank, world, device):
    if world > 1:
        sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank,
                                      shuffle=True, drop_last=False, seed=args.seed)
                   if training else DistributedEvalSampler(dataset, rank, world))
    else:
        sampler = None
    config = dict(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=training and sampler is None, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )
    if args.workers:
        config["prefetch_factor"] = 2
    return DataLoader(**config), sampler


def amp_context(device, amp_name):
    if device.type != "cuda" or amp_name == "off":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_name == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_one_epoch(model, loader, optimizer, scaler, bottom, stats, args, device, world):
    """在标准化TL上训练；跨GPU按有效水体网格点数归一化。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    number_batches = len(loader)
    sum_scale = 1_000_000.0
    group_local_count = torch.zeros((), device=device, dtype=torch.float64)
    for step, (ssp, bathy, target_db, sample_id) in enumerate(loader):
        ssp = ssp.to(device, non_blocking=True)
        bathy = bathy.to(device, non_blocking=True)
        target_db = target_db.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        is_step = ((step + 1) % args.grad_accum == 0) or ((step + 1) == number_batches)
        sync_context = (model.no_sync() if world > 1 and not is_step
                        else contextlib.nullcontext())

        with sync_context:
            with amp_context(device, args.amp):
                pred_norm = model(ssp, bathy)
            # 仅用前11年训练集统计量标准化目标；无效点随后会被掩码完全排除。
            target_norm = (target_db - stats["tl_mean"]) / stats["tl_std"]

            valid = water_valid_mask(target_db, sample_id, bottom)
            sq, ab, local_count = masked_error_sums(pred_norm, target_norm, valid)
            # 仅在参数更新时同步有效点数，避免每个微批进行一次NCCL通信。
            group_local_count += local_count.detach()
            loss = sq / sum_scale
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        with torch.no_grad():
            sums[0] += sq.detach().double()
            sums[1] += ab.detach().double()
            sums[2] += local_count.detach().double()
        if is_step:
            global_count = group_local_count.clone()
            allreduce_(global_count)
            if global_count.item() == 0:
                raise ValueError("当前更新窗口没有有效水体网格点")
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            # DDP会对梯度取进程平均，此处按全局有效水体点数精确还原损失梯度。
            factor = world * sum_scale / global_count
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if gradients:
                torch._foreach_mul_(gradients, factor)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip if args.grad_clip > 0 else float("inf"),
                error_if_nonfinite=not scaler.is_enabled(),
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_local_count.zero_()
    allreduce_(sums)
    count = sums[2].item()
    return {
        "loss_masked_mse_normalized": float(sums[0].item() / count),
        "train_mae_db": float(sums[1].item() / count * stats["tl_std"]),
        "train_rmse_db": float(math.sqrt(sums[0].item()/count) * stats["tl_std"]),
        "valid_points": int(count),
    }


@torch.no_grad()
def validate(model_unwrapped, loader, bottom, device, amp):
    model_unwrapped.eval()
    metrics = MaskedMetrics(device)
    for ssp, bathy, target_db, sample_id in loader:
        ssp = ssp.to(device, non_blocking=True)
        bathy = bathy.to(device, non_blocking=True)
        target_db = target_db.to(device, non_blocking=True)
        sample_id = sample_id.to(device, non_blocking=True)
        with amp_context(device, amp):
            pred_db = model_unwrapped.predict_db(ssp, bathy)
        valid = water_valid_mask(target_db, sample_id, bottom)
        metrics.add(pred_db, target_db, valid)
    metrics.sync()
    return metrics.result()


def append_history(path, row):
    path = Path(path)
    existed = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not existed:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, *, epoch, model, optimizer, scheduler, scaler, stats,
                    best_rmse, bad_epochs, args, val_metrics):
    model_unwrapped = model.module if isinstance(model, DDP) else model
    state = {
        "epoch": epoch, "model": model_unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "stats": stats,
        "best_val_rmse_db": best_rmse, "bad_epochs": bad_epochs,
        "args": vars(args), "validation_metrics_db": val_metrics,
        "model_name": MODEL_ARCHITECTURE,
        "model_parameters": count_parameters(model_unwrapped),
        "year_split": "years 1-11 / 12 / 13",
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def update_best_checkpoint(last_file, best_file):
    """优先使用硬链接保存最佳权重，避免重复序列化完整优化器状态。"""
    temporary = best_file.with_name(best_file.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(last_file, temporary)
    except OSError:
        shutil.copyfile(last_file, temporary)
    os.replace(temporary, best_file)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        raise ValueError("batch-size/grad-accum/epochs 必须均大于0")
    rank, world, local_rank, device = setup_distributed()
    try:
        seed_everything(args.seed, rank, args.deterministic)
        if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前GPU不支持BF16；请将--amp设为fp16或off")
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"设备={device}, GPU进程数={world}, 共享数据目录={args.data_dir}, 输出目录={args.output_dir}", flush=True)
            check_files(args.data_dir)
            stats_file = args.output_dir / "stats.json"
            if stats_file.exists():
                stats = json.loads(stats_file.read_text(encoding="utf-8"))
                if stats.get("tl_invalid_raw") != args.tl_invalid_raw:
                    raise ValueError("已有stats.json的tl-invalid-raw配置不同，请使用新输出目录")
                print("使用已有训练集统计量:", stats_file, flush=True)
            elif args.stats_from is not None:


                stats = json.loads(args.stats_from.read_text(encoding="utf-8"))
                if stats.get("tl_invalid_raw") != args.tl_invalid_raw:
                    raise ValueError("--stats-from 中tl-invalid-raw与本次配置不同")
                if stats.get("train_samples") != 15972:
                    raise ValueError("--stats-from 不是前11年训练样本的统计量")
                for key in ("ssp_mean", "ssp_std", "bathy_mean", "bathy_std", "tl_mean", "tl_std"):
                    if key not in stats or not math.isfinite(float(stats[key])):
                        raise ValueError(f"--stats-from 含无效统计量: {key}")
                save_stats(stats, stats_file)
                print("复制既有训练集统计量:", args.stats_from, flush=True)
            else:
                print("开始扫描前11年的有效水体TL，计算训练集标准化统计量...", flush=True)
                stats = compute_train_stats(
                    args.data_dir, tl_invalid_raw=args.tl_invalid_raw,
                    chunk_size=args.stats_chunk_size,
                )
                save_stats(stats, stats_file)
            print("标准化统计量:",json.dumps(stats,ensure_ascii=False), flush=True)
        barrier()
        stats = json.loads((args.output_dir / "stats.json").read_text(encoding="utf-8"))
        if stats.get("tl_invalid_raw") != args.tl_invalid_raw:
            raise ValueError("tl-invalid-raw 与统计量不一致")
        train_set = TLDataset(args.data_dir, "train", tl_invalid_raw=args.tl_invalid_raw, check=False)
        val_set = TLDataset(args.data_dir, "val", tl_invalid_raw=args.tl_invalid_raw, check=False)
        train_loader, train_sampler = loader_for(
            train_set, training=True, args=args, rank=rank, world=world, device=device)
        val_loader, _ = loader_for(
            val_set, training=False, args=args, rank=rank, world=world, device=device)
        bottom = torch.from_numpy(np.array(np.load(
            args.data_dir / "bottom_depth_index_r2.npy", allow_pickle=False), copy=True)).to(device)
        model = build_model(
            args.data_dir, stats,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        param_n = count_parameters(model)
        if rank == 0:
            print(f"{MODEL_ARCHITECTURE} 可训练参数量: {param_n:,}", flush=True)
            print(f"激活检查点={args.gradient_checkpointing}, AMP={args.amp}, "
                  f"每GPU微批={args.batch_size}, 累积={args.grad_accum}", flush=True)
        optimizer_options = {"lr": args.lr, "weight_decay": args.weight_decay}
        if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
            optimizer_options["fused"] = True
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
        start_epoch, best_rmse, bad_epochs = 0, math.inf, 0
        if args.resume is not None:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            # Validate checkpoint identity before restoring training state.
            if not checkpoint_matches(state):
                raise ValueError(
                    f"--resume的模型身份为{state.get('model_name')!r}，"
                    f"当前A4要求{MODEL_ARCHITECTURE!r}。"
                )
            if state["stats"] != stats:
                raise ValueError("resume中训练集统计量与本次训练不一致")

            saved_count = state.get("model_parameters")
            if saved_count is not None and int(saved_count) != param_n:
                raise ValueError(
                    f"Checkpoint参数量={saved_count}，当前网络参数量={param_n}；"
                    "只能续训相同结构的A4消融模型。"
                )
            try:
                model.load_state_dict(state["model"], strict=True)
            except RuntimeError as exc:
                raise RuntimeError("续训权重与当前A4/MLP解码器结构不兼容。") from exc
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
            start_epoch = int(state["epoch"]) + 1
            best_rmse = float(state["best_val_rmse_db"])
            bad_epochs = int(state["bad_epochs"])
            if rank == 0:
                print(f"从epoch={start_epoch + 1}恢复训练", flush=True)
        if rank == 0:
            print(f"AdamW融合实现={bool(optimizer.param_groups[0].get('fused', False))}", flush=True)
        if world > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                        broadcast_buffers=False)
        unwrapped = model.module if isinstance(model, DDP) else model
        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            start = time.time()
            train_result = train_one_epoch(
                model, train_loader, optimizer, scaler, bottom, stats, args, device, world)
            val_result = validate(unwrapped, val_loader, bottom, device, args.amp)
            val_rmse = val_result["overall"]["rmse_db"]
            if val_rmse is None:
                raise ValueError("验证集没有有效水体数据")
            improved = val_rmse < best_rmse - 1e-8
            best_rmse = min(best_rmse, val_rmse)
            bad_epochs = 0 if improved else bad_epochs + 1
            lr_current = optimizer.param_groups[0]["lr"]
            scheduler.step()
            if rank == 0:
                row = {
                    "epoch": epoch + 1,
                    "lr": lr_current,
                    "train_mse_normalized": train_result["loss_masked_mse_normalized"],
                    "train_mae_db": train_result["train_mae_db"],
                    "train_rmse_db": train_result["train_rmse_db"],
                    "val_mae_db": val_result["overall"]["mae_db"],
                    "val_rmse_db": val_rmse,
                    "seconds": round(time.time()-start, 2),
                }
                append_history(args.output_dir / "history.csv", row)
                save_checkpoint(
                    args.output_dir / "last.pt", epoch=epoch, model=model,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    stats=stats, best_rmse=best_rmse, bad_epochs=bad_epochs,
                    args=args, val_metrics=val_result)
                if improved:
                    update_best_checkpoint(
                        args.output_dir / "last.pt", args.output_dir / "best.pt"
                    )
                print(
                    f"Epoch {epoch+1:03d}/{args.epochs} | "
                    f"train RMSE {train_result['train_rmse_db']:.4f} dB | "
                    f"val MAE {val_result['overall']['mae_db']:.4f} dB | "
                    f"val RMSE {val_rmse:.4f} dB | "
                    f"best {best_rmse:.4f} dB | {row['seconds']}s",
                    flush=True,
                )
                (args.output_dir / "last_val_metrics.json").write_text(
                    json.dumps(val_result, indent=2, ensure_ascii=False), encoding="utf-8")
            barrier()
            if bad_epochs >= args.patience:
                if rank == 0:
                    print(f"验证集RMSE连续{args.patience}轮未改善，Early Stopping。", flush=True)
                break
        if rank == 0:
            print(f"训练结束。最佳验证集RMSE={best_rmse:.4f} dB，模型位于 {args.output_dir/'best.pt'}")
            print("测试集不在训练过程使用。运行 evaluate.py --split test 完成最终独立评估。")
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
