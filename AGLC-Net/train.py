#!/usr/bin/env python3
"""Two-stage DDP training for Huo-style latent correction adapted to 3D TL.

Stage I : train TL autoencoder E+D on all TRAIN years only.
Stage II: freeze E; use first TRAIN year/month reference TL; train G/U and
          decoder D (decoder learning rate=0 for first 20 correction epochs).

IMPORTANT: reference bank is an additional prior learned from training labels.
It makes this a reference-assisted baseline, NOT a strictly equal-information
comparison with direct SSP+bathymetry -> TL prediction.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from aglc_net.data import (
    TLDataset, DistributedEvalSampler, check_files, compute_train_stats,
    save_stats, validate_tl_fill,
)
from aglc_net.metrics import FullGridMetrics, full_grid_error_sums, LOSS_POLICY
from aglc_net.model import (
    TLCodec, build_model, count_parameters, MODEL_ARCHITECTURE,
    REFERENCE_PARAMETERS, PARAMETER_TOLERANCE,
)
from aglc_net.reference import create_reference_bank
from aglc_net.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Latent-space correction baseline: pretrain or correct")
    p.add_argument("--stage", choices=("pretrain", "correct"), required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=None, help="Default: 100 pretrain / 150 correct")
    p.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--angle-chunk", type=int, default=3, help="Angles per forward/backward pass (1..9)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tl-invalid-raw", type=int, default=None,
                   help="Optional integer TL sentinel; requires --tl-invalid-fill-db")
    p.add_argument("--tl-invalid-fill-db", type=float, default=None,
                   help="Finite dB value replacing every sentinel; ALL replaced cells count in loss")
    p.add_argument("--stats-chunk-size", type=int, default=16)
    p.add_argument("--stats-from", type=Path, default=None,
                   help="Reuse ONLY full-grid stats with tl_statistics_region=all_grid_points")
    p.add_argument("--reference-bank", type=Path, default=None,
                   help="Optional bank location; defaults to OUTPUT_DIR/reference_bank.npz")
    p.add_argument("--pretrained", type=Path, default=None,
                   help="Stage-II source Stage-I pretrain_best.pt")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume the SAME stage from its *_last.pt")
    p.add_argument("--decoder-freeze-epochs", type=int, default=20)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--depth-max-m", type=float, default=4000., help="Assumed maximum receiver depth in plots/metric bins")
    p.add_argument("--range-max-m", type=float, default=50000., help="Assumed maximum range in plots/metric bins")
    return p.parse_args()


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world > 1:
            torch.set_num_threads(max(1, torch.get_num_threads() // world))
        device = torch.device("cpu")
    if world > 1:
        kwargs = {"backend": "nccl" if device.type == "cuda" else "gloo"}
        if device.type == "cuda" and "device_id" in inspect.signature(dist.init_process_group).parameters:
            kwargs["device_id"] = device
        dist.init_process_group(**kwargs)
    return rank, world, local_rank, device


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def sum_all_(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def seed_everything(seed, rank, deterministic):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def amp_context(device, amp):
    if device.type != "cuda" or amp == "off":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def data_loader(ds, training, args, rank, world, device):
    sampler = None
    if world > 1:
        if training:
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                         shuffle=True, drop_last=False, seed=args.seed)
        else:
            sampler = DistributedEvalSampler(ds, rank, world)
    opts = dict(dataset=ds, batch_size=args.batch_size, sampler=sampler,
                shuffle=training and sampler is None,
                num_workers=args.workers, drop_last=False,
                persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    if args.workers:
        opts["prefetch_factor"] = 2
    return DataLoader(**opts), sampler


def check_stats_compatible(stats, args):
    required = ('ssp_mean', 'ssp_std', 'bathy_mean', 'bathy_std', 'tl_mean', 'tl_std')
    if stats.get('train_samples') != 15972 or stats.get('tl_statistics_region') != 'all_grid_points':
        raise ValueError('stats.json必须基于前11年完整TL网格，不能复用MAP-Net或旧版AGLC-Net的统计量')
    if stats.get('loss_policy') != LOSS_POLICY:
        raise ValueError('旧版有限值筛选统计量不能复用，请换一个输出目录重新统计并训练')
    if stats.get('tl_invalid_raw') != args.tl_invalid_raw or stats.get('invalid_target_fill_db') != args.tl_invalid_fill_db:
        raise ValueError('TL整数哨兵/替代值与统计量不一致')
    for key in required:
        if key not in stats or not math.isfinite(float(stats[key])):
            raise ValueError(f'统计量缺少或包含无效值: {key}')
    if any(stats[k] <= 0 for k in ('ssp_std', 'bathy_std', 'tl_std')):
        raise ValueError('标准差必须为正')


def train_stats(args, rank):
    stat_path = args.output_dir / 'stats.json'
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        check_files(args.data_dir)
        if stat_path.is_file():
            stats = json.loads(stat_path.read_text(encoding='utf-8'))
            check_stats_compatible(stats, args)
            print('使用已有训练集统计量:', stat_path, flush=True)
        elif args.stats_from is not None:
            stats = json.loads(args.stats_from.read_text(encoding='utf-8'))
            check_stats_compatible(stats, args)
            save_stats(stats, stat_path)
            print('复用全网格统计量:', args.stats_from, flush=True)
        else:
            print('扫描前11年所有TL网格点的标准化统计量，不筛选任何网格...', flush=True)
            stats = compute_train_stats(args.data_dir, tl_invalid_raw=args.tl_invalid_raw,
                                        invalid_target_fill_db=args.tl_invalid_fill_db,
                                        chunk_size=args.stats_chunk_size)
            check_stats_compatible(stats, args)
            save_stats(stats, stat_path)
        print('标准化统计量:', json.dumps(stats, ensure_ascii=False), flush=True)
    barrier()
    stats = json.loads(stat_path.read_text(encoding='utf-8'))
    check_stats_compatible(stats, args)
    return stats

def decoder_params(codec):
    modules = (codec.from_latent, codec.up3, codec.up2, codec.up1, codec.up0,
               codec.fine1, codec.fine2, codec.out)
    return [p for module in modules for p in module.parameters()]


def set_phase_model(stage, args, stats, bank_path, device):
    if stage == "pretrain":
        model = TLCodec().to(device)
        parts = [{"params": list(model.parameters()), "lr": args.lr}]
        param_total = count_parameters(model)
    else:
        model = build_model(bank_path, stats).to(device)
        # Encoder remains frozen throughout Stage II, as specified in paper.
        for p in model.codec.encoder_blocks.parameters():
            p.requires_grad_(False)
        for p in model.codec.to_latent.parameters():
            p.requires_grad_(False)
        dec = decoder_params(model.codec)
        other = [*model.bathy_encoder.parameters(), *model.mapper.parameters(),
                 *model.corrector.parameters()]
        parts = [{"params": dec, "lr": args.lr}, {"params": other, "lr": args.lr}]
        param_total = count_parameters(model)
        relative = abs(param_total / REFERENCE_PARAMETERS - 1)
        if relative > PARAMETER_TOLERANCE:
            raise ValueError(f"network count {param_total} outside +/-10% of {REFERENCE_PARAMETERS}")
    return model, parts, param_total


def optimizer_step(model, optimizer, scaler, world, group_count, grad_clip):
    """DDP averages local gradients; normalize using GLOBAL grid point count."""
    global_count = sum_all_(group_count.clone())
    if global_count.item() <= 0:
        raise ValueError('Empty optimization group')
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    # Each backward uses SSE / 1e6; multiply by world*1e6/total grid cells.
    factor = world * 1_000_000.0 / global_count.item()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads:
        torch._foreach_mul_(grads, factor)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, loader, optimizer, scaler, stats, args, device, world):
    model.train()
    model.zero_grad(set_to_none=True)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    group_count = torch.zeros((), dtype=torch.float64, device=device)
    for step, (ssp, bathy, target, sample_ids) in enumerate(loader):
        ssp, bathy, target = (t.to(device, non_blocking=True) for t in (ssp, bathy, target))
        if not bool(torch.isfinite(target).all()):
            raise ValueError('TL训练标签含非有限值，全网格MSE禁止忽略点')
        target_norm = (target - stats['tl_mean']) / stats['tl_std']
        group_count += target.numel()
        last_update = ((step+1) % args.grad_accum == 0 or step+1 == len(loader))
        for start in range(0, 9, args.angle_chunk):
            stop = min(start+args.angle_chunk, 9)
            last_angle = stop == 9
            nosync = (model.no_sync() if world > 1 and not (last_update and last_angle)
                      else contextlib.nullcontext())
            with nosync:
                with amp_context(device, args.amp):
                    if args.stage == 'pretrain':
                        n, a = target.shape[0], stop-start
                        inp = target_norm[:,start:stop].reshape(n*a,1,200,250)
                        pred = model(inp).reshape(n,a,200,250)
                    else:
                        pred = model(ssp, bathy, angle_start=start, angle_end=stop,
                                     angle_chunk=args.angle_chunk)
                sq, ab, count = full_grid_error_sums(pred, target_norm[:,start:stop])
                loss = sq / 1_000_000.0
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            with torch.no_grad():
                totals[0] += sq.detach().double()
                totals[1] += ab.double()
                totals[2] += count
        if last_update:
            optimizer_step(model, optimizer, scaler, world, group_count, args.grad_clip)
            group_count.zero_()
    sum_all_(totals)
    n = int(totals[2].item())
    if n == 0:
        raise ValueError('Training epoch has no TL grid cells')
    return {'loss_mse_normalized': totals[0].item()/n,
            'mae_db': totals[1].item()/n*stats['tl_std'],
            'rmse_db': math.sqrt(totals[0].item()/n)*stats['tl_std'],
            'valid_points': n}


@torch.no_grad()
def validate(model, loader, stats, args, device):
    model.eval()
    metrics = FullGridMetrics(device)
    for ssp, bathy, target, sid in loader:
        ssp, bathy, target = (t.to(device, non_blocking=True) for t in (ssp, bathy, target))
        if not bool(torch.isfinite(target).all()):
            raise ValueError('TL验证标签含非有限值，全网格评价禁止忽略点')
        with amp_context(device, args.amp):
            if args.stage == 'correct':
                pred = model.predict_db(ssp, bathy, angle_chunk=args.angle_chunk)
            else:
                target_norm = (target - stats['tl_mean']) / stats['tl_std']
                pieces = []
                for start in range(0, 9, args.angle_chunk):
                    stop = min(start+args.angle_chunk, 9)
                    n, a = target.shape[0], stop-start
                    inp = target_norm[:,start:stop].reshape(n*a,1,200,250)
                    pieces.append(model(inp).reshape(n,a,200,250))
                pred = torch.cat(pieces,dim=1)*stats['tl_std']+stats['tl_mean']
        metrics.add(pred, target)
    metrics.sync()
    return metrics.result()

def save_checkpoint(path, model, optimizer, stats, epoch, best, bad, args, param_count,
                    metrics, scaler, reference_hash=None):
    unwrapped = model.module if isinstance(model, DDP) else model
    state = {"model": unwrapped.state_dict(), "optimizer": optimizer.state_dict(),
             "stage": args.stage, "epoch": epoch, "best_val_rmse_db": best,
             "bad_epochs": bad, "stats": stats, "args": vars(args),
             "parameter_count": param_count, "model_parameters": param_count,
             "architecture": MODEL_ARCHITECTURE, "model_name": MODEL_ARCHITECTURE,
             "validation": metrics, "val_metrics": metrics, "scaler": scaler.state_dict(),
             "reference_sha256": reference_hash,
             "extra_reference_tl_prior": args.stage == "correct",
             "loss_policy": LOSS_POLICY}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def clone_best(last_path, best_path):
    temp = best_path.with_name(best_path.name + ".tmp")
    temp.unlink(missing_ok=True)
    try:
        os.link(last_path, temp)
    except OSError:
        shutil.copyfile(last_path, temp)
    os.replace(temp, best_path)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    if args.epochs is None:
        args.epochs = 100 if args.stage == 'pretrain' else 150
    if not (1 <= args.angle_chunk <= 9) or any(x < 1 for x in (args.epochs, args.batch_size, args.grad_accum, args.stats_chunk_size)):
        raise ValueError('epochs/batch-size/grad-accum/stats-chunk-size >= 1, angle-chunk in [1,9]')
    if args.workers < 0 or args.patience < 1 or args.decoder_freeze_epochs < 0:
        raise ValueError('workers >= 0, patience >= 1, decoder-freeze-epochs >= 0')
    if args.depth_max_m <= 0 or args.range_max_m <= 0:
        raise ValueError('depth-max-m and range-max-m must be positive')
    validate_tl_fill(args.tl_invalid_raw, args.tl_invalid_fill_db)
    rank, world, local_rank, device = setup_distributed()
    try:
        seed_everything(args.seed, rank, args.deterministic)
        if device.type == "cuda" and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16")
        stats = train_stats(args, rank)
        bank_path = args.reference_bank or (args.output_dir / "reference_bank.npz")
        reference_hash = None
        if args.stage == "correct":
            if rank == 0:
                create_reference_bank(args.data_dir, bank_path, args.tl_invalid_raw)
                reference_hash = file_sha256(bank_path)
            barrier()
            reference_hash = file_sha256(bank_path)
        train_set = TLDataset(args.data_dir, 'train', tl_invalid_raw=args.tl_invalid_raw,
                              invalid_target_fill_db=args.tl_invalid_fill_db, check=False)
        val_set = TLDataset(args.data_dir, 'val', tl_invalid_raw=args.tl_invalid_raw,
                            invalid_target_fill_db=args.tl_invalid_fill_db, check=False)
        train_loader, sampler = data_loader(train_set, True, args, rank, world, device)
        val_loader, _ = data_loader(val_set, False, args, rank, world, device)
        model, groups, param_count = set_phase_model(args.stage, args, stats, bank_path, device)
        if rank == 0:
            print(f'{MODEL_ARCHITECTURE} [{args.stage}] 网络总参数量: {param_count:,} '
                  f'(MAP-Net={REFERENCE_PARAMETERS:,}, 差异={(param_count/REFERENCE_PARAMETERS-1)*100:+.4f}%)', flush=True)
            print(f'设备={device}, GPU进程数={world}, 共享数据目录={args.data_dir}, 输出目录={args.output_dir}', flush=True)
            print(f'AMP={args.amp}, 每GPU微批={args.batch_size}, 累积={args.grad_accum}, '
                  f'全局有效Batch={world*args.batch_size*args.grad_accum}, '
                  f'方位分块={args.angle_chunk}, LR={args.lr:g}, 优化器=AdamW', flush=True)
            print('训练/验证：完整TL网格450000点/样本，标准化MSE、真实dB MAE/RMSE；无任何掩码。', flush=True)
            if args.stage == "correct":
                print("警告：本网络额外使用训练集历史TL参考库，属于reference-assisted对比，非完全同信息输入。")
        if args.stage == "correct" and args.resume is None:
            source = args.pretrained or (args.output_dir / 'pretrain_best.pt')
            if not source.is_file():
                raise FileNotFoundError(f"Stage I weights not found: {source}. Run --stage pretrain first")
            pre = torch.load(source, map_location="cpu", weights_only=False)
            if pre.get('loss_policy') != LOSS_POLICY:
                raise ValueError('Stage-I检查点不是全网格无掩码目标；需要重新预训练')
            if pre.get("stage") != "pretrain" or pre.get("stats") != stats:
                raise ValueError("Pretrained codec must be Stage I and have identical training stats")
            model.codec.load_state_dict(pre["model"], strict=True)
            if rank == 0:
                print("Loaded Stage-I TL autoencoder:", source)
        opt_args = {"weight_decay": args.weight_decay}
        if device.type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters:
            opt_args['fused'] = True
        optimizer = torch.optim.AdamW(groups, lr=args.lr, **opt_args)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))
        start_epoch, best, bad = 0, float("inf"), 0
        if args.resume is not None:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            if state.get('loss_policy') != LOSS_POLICY:
                raise ValueError('不能从旧版或不同损失策略的检查点续训')
            if state.get("stage") != args.stage or state["stats"] != stats:
                raise ValueError("Resume stage/stats mismatch")
            if state.get("parameter_count") != param_count:
                raise ValueError("Checkpoint param count differs")
            if args.stage == 'correct' and state.get('reference_sha256') != reference_hash:
                raise ValueError("Reference bank differs from checkpoint; refusing resume")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state.get('scaler', {}))
            start_epoch = state['epoch'] + 1
            best, bad = state['best_val_rmse_db'], state['bad_epochs']
            if rank == 0:
                print(f'从epoch={start_epoch+1}继续{args.stage}阶段训练',flush=True)
        if world > 1:
            # Buffer sync disabled: reference bank is immutable and identical on all ranks.
            model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                        broadcast_buffers=False)
        unwrapped = model.module if isinstance(model, DDP) else model
        # Stage II produces the final prediction checkpoint.
        prefix = 'pretrain_' if args.stage == 'pretrain' else ''
        last_path = args.output_dir / f'{prefix}last.pt'
        best_path = args.output_dir / f'{prefix}best.pt'
        history_path = args.output_dir / f'{prefix}history.csv'
        if rank == 0:
            print(f'AdamW融合实现={bool(optimizer.param_groups[0].get("fused",False))}', flush=True)
        if rank == 0 and start_epoch == 0 and history_path.exists():
            raise FileExistsError(f"{history_path} exists; use --resume or fresh output dir")
        for epoch in range(start_epoch, args.epochs):
            tick = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)
            lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs)))
            if args.stage == "correct":
                optimizer.param_groups[0]['lr'] = 0.0 if epoch < args.decoder_freeze_epochs else lr
                optimizer.param_groups[1]['lr'] = lr
                if epoch == args.decoder_freeze_epochs:
                    bad = 0  # don't let the frozen-decoder plateau cause premature stopping
                    for p in decoder_params(unwrapped.codec):
                        optimizer.state.pop(p, None)  # true fresh Adam state after freeze
            else:
                optimizer.param_groups[0]['lr'] = lr
            result_train = train_epoch(model, train_loader, optimizer, scaler, stats, args, device, world)
            result_val = validate(unwrapped, val_loader, stats, args, device)
            val_rmse = result_val['overall']['rmse_db']
            if val_rmse is None:
                raise ValueError("Validation set has zero finite TL labels")
            improved = val_rmse < best - 1e-8
            best = min(best, val_rmse)
            bad = 0 if improved else bad+1
            if rank == 0:
                row = {'epoch': epoch+1, 'lr': lr,
                       'train_mse_normalized': result_train['loss_mse_normalized'],
                       "train_mae_db":result_train['mae_db'],
                       "train_rmse_db":result_train['rmse_db'],
                       "val_mae_db":result_val['overall']['mae_db'],
                       "val_rmse_db":val_rmse, "seconds":round(time.time()-tick,2)}
                existed = history_path.exists()
                with history_path.open('a',newline='',encoding='utf-8') as f:
                    w = csv.DictWriter(f,fieldnames=list(row));
                    if not existed: w.writeheader()
                    w.writerow(row)
                save_checkpoint(last_path, model, optimizer, stats, epoch, best, bad,
                                args, param_count, result_val, scaler, reference_hash)
                if improved:
                    clone_best(last_path, best_path)
                print(f'Epoch {epoch+1:03d}/{args.epochs} | '
                      f'train RMSE {result_train["rmse_db"]:.4f} dB | '
                      f'val MAE {result_val["overall"]["mae_db"]:.4f} dB | '
                      f'val RMSE {val_rmse:.4f} dB | '
                      f'best {best:.4f} dB | {row["seconds"]}s', flush=True)
                (args.output_dir / f'{prefix}last_val_metrics.json').write_text(
                    json.dumps(result_val, ensure_ascii=False, indent=2), encoding='utf-8')
            barrier()
            if bad >= args.patience and (args.stage != "correct" or epoch >= args.decoder_freeze_epochs):
                if rank == 0:
                    print(f"Early stop after {bad} validation epochs without improvement")
                break
        if rank == 0:
            print("Finished; best checkpoint:", best_path)
            if args.stage == 'pretrain':
                print('下一步：运行 --stage correct，加载 pretrain_best.pt')
            else:
                print('下一步：python evaluate.py --split test --full-split')
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
