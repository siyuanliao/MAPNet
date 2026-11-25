#!/usr/bin/env python3
"""DD-PUNet training.

Training and validation use all TL cells, including those below the seabed.
Final test evaluation uses the common water-column mask in evaluate.py.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
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

from dd_punet.data import (TLDataset, DistributedEvalSampler,
                                  check_files, compute_train_stats, save_stats)
from dd_punet.losses import fullgrid_error_sums
from dd_punet.metrics import FullGridMetrics
from dd_punet.model import (MODEL_ARCHITECTURE, CHECKPOINT_MODEL_NAMES, REFERENCE_PARAMETERS,
                                    PARAMETER_TOLERANCE, build_model, count_parameters)
from dd_punet.paths import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


def parse_args():
    p = argparse.ArgumentParser(description='DD-PUNet full-grid TL training (NO masking)')
    p.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=2, help='每张GPU的微批样本数')
    p.add_argument('--angle-chunk', type=int, default=1, help='分段预测方位数量，1最节省显存')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--amp', choices=('bf16','fp16','off'), default='bf16')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--range-max-m', type=float, default=50000.)
    p.add_argument('--depth-max-m', type=float, default=4000.)
    p.add_argument('--tl-invalid-raw', type=int, default=None,
                   help='仅当TL整数使用特殊哨兵编码时填写；必须配合固定填充值')
    p.add_argument('--invalid-target-fill-db', type=float, default=None,
                   help='将原始TL哨兵替换为此有限dB值，仍然参与全网格MSE')
    p.add_argument('--stats-chunk-size', type=int, default=16)
    p.add_argument('--stats-from', type=Path, default=None,
                   help='仅允许复用DD-PUNet的全网格统计，不允许复用MAP-Net水体统计')
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--deterministic', action='store_true')
    return p.parse_args()


def setup_distributed():
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if world > 1:
        kwargs = {'backend': 'nccl' if device.type == 'cuda' else 'gloo'}
        if device.type == 'cuda' and 'device_id' in inspect.signature(dist.init_process_group).parameters:
            kwargs['device_id'] = device
        dist.init_process_group(**kwargs)
    return rank, world, local_rank, device


def seed_everything(seed, rank, deterministic=False):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision('high')


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
    kw = dict(dataset=dataset, batch_size=args.batch_size,
              shuffle=training and sampler is None, sampler=sampler,
              num_workers=args.workers, pin_memory=device.type=='cuda',
              persistent_workers=args.workers>0, drop_last=False)
    if args.workers:
        kw['prefetch_factor'] = 2
    return DataLoader(**kw), sampler


def amp_context(device, name):
    if device.type != 'cuda' or name == 'off':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.bfloat16 if name == 'bf16' else torch.float16)


def angle_chunks(n):
    for start in range(0, 9, n):
        yield list(range(start, min(9, start+n)))


def train_one_epoch(model, loader, optimizer, scaler, stats, args, device, world):
    """Full-grid normalized MSE with exact count weighting across chunks and DDP."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(3, dtype=torch.float64, device=device)  # SSE, SAE, cells
    group_local_count = torch.zeros((), dtype=torch.float64, device=device)
    sum_scale = 1_000_000.0  # avoid big unnormalized gradients until optimizer update
    angle_groups = list(angle_chunks(args.angle_chunk))

    for step, (ssp, bathy, target_db, _ids) in enumerate(loader):
        ssp = ssp.to(device, non_blocking=True)
        bathy = bathy.to(device, non_blocking=True)
        target_db = target_db.to(device, non_blocking=True)
        if not bool(torch.isfinite(target_db).all()):
            raise ValueError('Full-grid loss requires finite target TL at every grid cell')
        target_norm = (target_db.float() - stats['tl_mean']) / stats['tl_std']
        update = (step+1) % args.grad_accum == 0 or step+1 == len(loader)

        for j, ai in enumerate(angle_groups):
            sync = update and j == len(angle_groups)-1
            no_sync = model.no_sync() if world > 1 and not sync else contextlib.nullcontext()
            with no_sync:
                with amp_context(device, args.amp):
                    pred_norm = model(ssp, bathy, angle_indices=ai)
                sq, ab, count = fullgrid_error_sums(pred_norm, target_norm[:, ai])
                loss = sq / sum_scale
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            with torch.no_grad():
                sums[0] += sq.detach().double()
                sums[1] += ab.double()
                sums[2] += count
                group_local_count += count

        if update:
            global_count = allreduce_(group_local_count.clone())
            if global_count.item() <= 0:
                raise ValueError('Empty optimizer update group')
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            factor = world * sum_scale / global_count
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if gradients:
                torch._foreach_mul_(gradients, factor)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float('inf'),
                error_if_nonfinite=not scaler.is_enabled())
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_local_count.zero_()

    allreduce_(sums)
    n = float(sums[2].item())
    if n == 0:
        raise ValueError('No training TL cells')
    return {'loss_fullgrid_mse_normalized': float(sums[0].item()/n),
            'train_mae_db': float(sums[1].item()/n * stats['tl_std']),
            'train_rmse_db': float(math.sqrt(sums[0].item()/n) * stats['tl_std']),
            'grid_points': int(n)}


@torch.no_grad()
def validate(model_unwrapped, loader, stats, args, device):
    model_unwrapped.eval()
    metrics = FullGridMetrics(device)
    for ssp, bathy, target_db, _ids in loader:
        ssp = ssp.to(device, non_blocking=True)
        bathy = bathy.to(device, non_blocking=True)
        target_db = target_db.to(device, non_blocking=True)
        predicted_db = torch.empty_like(target_db)
        for ai in angle_chunks(args.angle_chunk):
            with amp_context(device, args.amp):
                out = model_unwrapped.predict_db(ssp, bathy, angle_indices=ai)
            predicted_db[:,ai] = out.float()
        metrics.add(predicted_db, target_db)
    metrics.sync()
    return metrics.result()


def append_history(path, row):
    path = Path(path)
    exists = path.is_file()
    if exists:
        with path.open('r', newline='', encoding='utf8') as f:
            header = next(csv.reader(f), None)
        if header != list(row):
            raise ValueError('Existing history.csv has an incompatible column schema; use a new output directory')
    with path.open('a', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, *, epoch, model, optimizer, scheduler, scaler, stats,
                    best_rmse, bad_epochs, args, val_metrics):
    unwrapped = model.module if isinstance(model, DDP) else model
    state = {'epoch': epoch, 'model': unwrapped.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'scaler': scaler.state_dict(), 'stats': stats,
             'best_val_rmse_db': best_rmse, 'bad_epochs': bad_epochs,
             'args': vars(args), 'validation_metrics_db': val_metrics,
             'model_name': MODEL_ARCHITECTURE,
             'model_parameters': count_parameters(unwrapped),
             'year_split': 'years 1-11 / 12 / 13',
             'loss_definition': 'unmasked full-grid normalized TL MSE; ALL [9,200,250] cells'}
    tmp = path.with_name(path.name+'.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def update_best_checkpoint(last_file, best_file):
    tmp = best_file.with_name(best_file.name+'.tmp')
    tmp.unlink(missing_ok=True)
    try:
        os.link(last_file, tmp)
    except OSError:
        shutil.copyfile(last_file, tmp)
    os.replace(tmp, best_file)


def validate_stats(stats, args):
    if stats.get('tl_scope') != 'full_grid':
        raise ValueError('Stats must use DD-PUNet full-grid TL normalization; MAP-Net water-only stats are NOT compatible')
    if stats.get('train_samples') != 15972:
        raise ValueError('Stats must come from first 11 training years')
    for key in ('ssp_mean','ssp_std','bathy_mean','bathy_std','tl_mean','tl_std'):
        if key not in stats or not math.isfinite(float(stats[key])):
            raise ValueError(f'Invalid normalization stats: {key}')
    for key in ('ssp_std','bathy_std','tl_std'):
        if stats[key] <= 0:
            raise ValueError(f'Normalization std must be positive: {key}')
    if stats.get('tl_invalid_raw') != args.tl_invalid_raw or stats.get('invalid_target_fill_db') != args.invalid_target_fill_db:
        raise ValueError('TL sentinel/fill value differs from stats.json; use a new output directory')


def main():
    args = parse_args()
    if (args.batch_size<1 or args.grad_accum<1 or args.workers<0 or args.epochs<1 or
            not 1<=args.angle_chunk<=9 or args.stats_chunk_size<1 or args.patience<1):
        raise ValueError('Invalid batch-size/grad-accum/workers/epochs/angle-chunk/stats-chunk-size/patience')
    if (args.tl_invalid_raw is None) != (args.invalid_target_fill_db is None):
        raise ValueError('--tl-invalid-raw and --invalid-target-fill-db must be set together')
    if args.invalid_target_fill_db is not None and not math.isfinite(args.invalid_target_fill_db):
        raise ValueError('invalid-target-fill-db must be finite')
    rank, world, local_rank, device = setup_distributed()
    try:
        seed_everything(args.seed, rank, args.deterministic)
        if device.type=='cuda' and args.amp=='bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('当前GPU不支持BF16，请设--amp fp16或off')
        if rank==0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print(f'设备={device}, GPU进程数={world}, 共享数据目录={args.data_dir}, 输出目录={args.output_dir}', flush=True)
            check_files(args.data_dir)
            stats_path = args.output_dir/'stats.json'
            if stats_path.exists():
                stats = json.loads(stats_path.read_text(encoding='utf8'))
                validate_stats(stats, args)
                print('使用已有训练集统计量:', stats_path, flush=True)
            elif args.stats_from is not None:
                stats = json.loads(args.stats_from.read_text(encoding='utf8'))
                validate_stats(stats, args)
                save_stats(stats, stats_path)
                print('复制既有全网格统计量:', args.stats_from, flush=True)
            else:
                print('开始扫描前11年的全部TL网格，计算训练集标准化统计量（无掩码）...', flush=True)
                stats = compute_train_stats(args.data_dir,
                    tl_invalid_raw=args.tl_invalid_raw,
                    invalid_target_fill_db=args.invalid_target_fill_db,
                    chunk_size=args.stats_chunk_size)
                validate_stats(stats, args)
                save_stats(stats, stats_path)
            print('标准化统计量:', json.dumps(stats, ensure_ascii=False), flush=True)
        barrier()
        stats = json.loads((args.output_dir/'stats.json').read_text(encoding='utf8'))
        validate_stats(stats,args)
        train_set = TLDataset(args.data_dir, 'train', tl_invalid_raw=args.tl_invalid_raw,
                              invalid_target_fill_db=args.invalid_target_fill_db, check=False)
        val_set = TLDataset(args.data_dir, 'val', tl_invalid_raw=args.tl_invalid_raw,
                            invalid_target_fill_db=args.invalid_target_fill_db, check=False)
        train_loader, sampler = loader_for(train_set, training=True, args=args,
                                           rank=rank, world=world, device=device)
        val_loader, _ = loader_for(val_set, training=False, args=args,
                                   rank=rank, world=world, device=device)
        model = build_model(args.data_dir, stats, width=args.width,
                            range_max_m=args.range_max_m, depth_max_m=args.depth_max_m).to(device)
        count = count_parameters(model)
        if abs(count-REFERENCE_PARAMETERS)/REFERENCE_PARAMETERS > PARAMETER_TOLERANCE:
            raise ValueError(f'DD-PUNet has {count:,} parameters; outside ±10% budget')
        if rank==0:
            print(f'{MODEL_ARCHITECTURE} 可训练参数量: {count:,}', flush=True)
            print(f'参考MAP-Net参数量={REFERENCE_PARAMETERS:,}, 差异={(count/REFERENCE_PARAMETERS-1)*100:+.2f}%', flush=True)
            print(f'方位分块={args.angle_chunk}, AMP={args.amp}, 每GPU微批={args.batch_size}, 累积={args.grad_accum}', flush=True)
        kw = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if device.type=='cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters:
            kw['fused'] = True
        optimizer = torch.optim.AdamW(model.parameters(), **kw)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda' and args.amp=='fp16')
        first_epoch, best_rmse, bad_epochs = 0, math.inf, 0
        if args.resume is not None:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state.get('model_name') not in CHECKPOINT_MODEL_NAMES or state.get('stats') != stats:
                raise ValueError('Resume checkpoint architecture or normalization mismatch')
            if state.get('loss_definition') != 'unmasked full-grid normalized TL MSE; ALL [9,200,250] cells':
                raise ValueError('Resume checkpoint was trained with a different objective')
            if state.get('model_parameters') != count:
                raise ValueError('Resume checkpoint model parameter count mismatch')
            previous = state.get('args',{})
            for k in ('width','range_max_m','depth_max_m','tl_invalid_raw','invalid_target_fill_db',
                      'grad_accum','batch_size','angle_chunk'):
                if previous.get(k) != getattr(args,k):
                    raise ValueError(f'Resume setting mismatch for {k}; keep original settings')
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state.get('scaler', {}))
            first_epoch=int(state['epoch'])+1
            best_rmse=float(state['best_val_rmse_db'])
            bad_epochs=int(state['bad_epochs'])
            if rank==0:
                print(f'从epoch={first_epoch+1}恢复训练', flush=True)
        if rank==0:
            print(f'AdamW融合实现={bool(optimizer.param_groups[0].get("fused",False))}',flush=True)
        if world>1:
            model = DDP(model, device_ids=[local_rank] if device.type=='cuda' else None,
                        broadcast_buffers=False)
        unwrapped = model.module if isinstance(model, DDP) else model
        for epoch in range(first_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            started=time.time()
            train_result=train_one_epoch(model,train_loader,optimizer,scaler,stats,args,device,world)
            val_result=validate(unwrapped,val_loader,stats,args,device)
            val_rmse=val_result['overall']['rmse_db']
            if val_rmse is None:
                raise ValueError('Validation split has zero TL grid cells')
            improved=val_rmse < best_rmse-1e-8
            best_rmse=min(best_rmse,val_rmse)
            bad_epochs=0 if improved else bad_epochs+1
            lr_current=optimizer.param_groups[0]['lr']
            scheduler.step()
            if rank==0:
                row = {'epoch': epoch+1,'lr':lr_current,
                       'train_mse_normalized':train_result['loss_fullgrid_mse_normalized'],
                       'train_mae_db':train_result['train_mae_db'],
                       'train_rmse_db':train_result['train_rmse_db'],
                       'val_mae_db':val_result['overall']['mae_db'],
                       'val_rmse_db':val_rmse,'seconds':round(time.time()-started,2)}
                append_history(args.output_dir/'history.csv', row)
                save_checkpoint(args.output_dir/'last.pt',epoch=epoch,model=model,
                     optimizer=optimizer,scheduler=scheduler,scaler=scaler,stats=stats,
                     best_rmse=best_rmse,bad_epochs=bad_epochs,args=args,val_metrics=val_result)
                if improved:
                    update_best_checkpoint(args.output_dir/'last.pt',args.output_dir/'best.pt')
                (args.output_dir/'last_val_metrics.json').write_text(
                    json.dumps(val_result,indent=2,ensure_ascii=False),encoding='utf8')
                print(f'Epoch {epoch+1:03d}/{args.epochs} | '
                      f'train RMSE {train_result["train_rmse_db"]:.4f} dB | '
                      f'val MAE {val_result["overall"]["mae_db"]:.4f} dB | '
                      f'val RMSE {val_rmse:.4f} dB | '
                      f'best {best_rmse:.4f} dB | {row["seconds"]}s',flush=True)
            barrier()
            if bad_epochs>=args.patience:
                if rank==0:print(f'验证集RMSE连续{args.patience}轮未改善，Early Stopping。',flush=True)
                break
        if rank==0:
            print(f'训练结束。最佳验证集全网格RMSE={best_rmse:.4f} dB，模型位于 {args.output_dir/"best.pt"}')
            print('测试集不参与训练：运行 evaluate.py --split test --full-split 进行独立评估。')
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__=='__main__':
    main()
