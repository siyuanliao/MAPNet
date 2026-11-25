#!/usr/bin/env python3
"""DD-PUNet: evaluate RMSE/MAE only within valid water, consistent with MAP-Net.

Training uses full-grid MSE; final evaluation uses the water-column mask.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import math
import torch.distributed as dist
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dd_punet.data import TLDataset, SPLIT_SLICES, check_files
from dd_punet.model import MODEL_ARCHITECTURE, CHECKPOINT_MODEL_NAMES, build_model, count_parameters
from dd_punet.paths import DEFAULT_DATA_DIR, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR

# ============== 可直接在 Spyder 中编辑的默认配置 ==============
ROOT = Path(__file__).resolve().parent
DATA_DIR = DEFAULT_DATA_DIR
CHECKPOINT = DEFAULT_CHECKPOINT
OUTPUT_DIR = DEFAULT_OUTPUT_DIR/'evaluation'
# Set EVALUATE_FULL_SPLIT=False to evaluate SAMPLE_IDS only.
SPLIT = 'test'
SAMPLE_IDS = [18875]
EVALUATE_FULL_SPLIT = True
BATCH_SIZE = 1
NUM_WORKERS = 0
ANGLE_CHUNK = 1
AMP = 'auto'


class CrossPlatformUnpickler(pickle.Unpickler):
    """Allow checkpoints saved on a different operating system."""
    def find_class(self, module, name):
        if module in ('pathlib', 'pathlib._local'):
            if name == 'PosixPath':
                return PurePosixPath
            if name == 'WindowsPath':
                return PureWindowsPath
        return super().find_class(module, name)


class CrossPlatformPickle:
    Unpickler = CrossPlatformUnpickler


def _angle_chunks(chunk_size):
    for start in range(0, 9, chunk_size):
        yield list(range(start, min(9, start+chunk_size)))




# Water-column evaluation
# Water-column metrics are implemented locally for standalone evaluation.
def water_valid_mask(target, sample_ids, bottom_idx):
    """按原始样本编号生成掩码，并排除真实TL的非有限值。"""
    if target.ndim != 4 or target.shape[1:] != (9, 200, 250):
        raise ValueError(f"TL shape must be [B,9,200,250], got {tuple(target.shape)}")
    # 注意：sample_ids为18876条数据的原始样本编号，而不是当前batch内的索引。
    src = sample_ids.to(device=target.device, dtype=torch.long) % 121
    h = bottom_idx[src].to(torch.long)
    z = torch.arange(200, device=target.device).view(1, 1, 200, 1)


    return (z <= h.unsqueeze(2)) & torch.isfinite(target)

class MaskedMetrics:
    """按所有有效水体网格点聚合误差，避免对每个Batch的RMSE求平均。"""

    def __init__(self, device):
        self.device = torch.device(device)
        self.sse = torch.zeros((), device=device, dtype=torch.float64)
        self.sae = torch.zeros((), device=device, dtype=torch.float64)
        self.n = torch.zeros((), device=device, dtype=torch.float64)
        self.angle_sse = torch.zeros(9, device=device, dtype=torch.float64)
        self.angle_sae = torch.zeros(9, device=device, dtype=torch.float64)
        self.angle_n = torch.zeros(9, device=device, dtype=torch.float64)
        self.range_sse = torch.zeros(5, device=device, dtype=torch.float64)
        self.range_sae = torch.zeros(5, device=device, dtype=torch.float64)
        self.range_n = torch.zeros(5, device=device, dtype=torch.float64)
        self.depth_sse = torch.zeros(4, device=device, dtype=torch.float64)
        self.depth_sae = torch.zeros(4, device=device, dtype=torch.float64)
        self.depth_n = torch.zeros(4, device=device, dtype=torch.float64)
        self.nonfinite = torch.zeros((), device=device, dtype=torch.int64)

    @torch.no_grad()
    def add(self, pred_db, target_db, valid):
        # 水体外及真实TL非有限点不纳入误差，分组边界沿用原始物理网格。
        self.nonfinite += (valid & ~torch.isfinite(pred_db)).sum()
        err = torch.where(valid, pred_db.float(), 0.0) - torch.where(valid, target_db.float(), 0.0)
        abs_err = err.abs().double()
        sq_err = err.square().double()
        self.sse += sq_err.sum()
        self.sae += abs_err.sum()
        self.n += valid.sum().double()
        self.angle_sse += sq_err.sum(dim=(0, 2, 3))
        self.angle_sae += abs_err.sum(dim=(0, 2, 3))
        self.angle_n += valid.sum(dim=(0, 2, 3)).double()

        self.range_sse += sq_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_sae += abs_err.reshape(-1, 5, 50).sum(dim=(0, 2))
        self.range_n += valid.reshape(-1, 5, 50).sum(dim=(0, 2)).double()
        self.depth_sse += sq_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_sae += abs_err.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3))
        self.depth_n += valid.reshape(-1, 4, 50, 250).sum(dim=(0, 2, 3)).double()

    def sync(self):
        if dist.is_available() and dist.is_initialized():
            # 统一规约，减少分布式验证时的通信次数。
            names = (
                "sse", "sae", "n", "angle_sse", "angle_sae", "angle_n",
                "range_sse", "range_sae", "range_n", "depth_sse", "depth_sae", "depth_n",
            )
            shape = [getattr(self, name).shape for name in names]
            packed = torch.cat([getattr(self, name).reshape(-1) for name in names])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            idx = 0
            for name, size in zip(names, shape):
                length = math.prod(size)
                getattr(self, name).copy_(packed[idx:idx + length].reshape(size))
                idx += length
            dist.all_reduce(self.nonfinite, op=dist.ReduceOp.SUM)

    def result(self):
        if int(self.nonfinite.item()) > 0:
            raise FloatingPointError("有效水体区域的预测TL含NaN/Inf，无法计算评价指标")

        def one(sse, sae, n):
            n = float(n.item())
            if n == 0:
                return {"mae_db": None, "rmse_db": None, "valid_points": 0}
            return {
                "mae_db": float(sae.item() / n),
                "rmse_db": math.sqrt(max(0.0, float(sse.item() / n))),
                "valid_points": int(n),
            }

        return {
            "overall": one(self.sse, self.sae, self.n),
            "per_angle": {f"{k*40}deg": one(self.angle_sse[k], self.angle_sae[k], self.angle_n[k]) for k in range(9)},
            "per_range_km": {f"{10*k}-{10*(k+1)}": one(self.range_sse[k], self.range_sae[k], self.range_n[k]) for k in range(5)},
            "per_depth_m": {f"{1000*k}-{1000*(k+1)}": one(self.depth_sse[k], self.depth_sae[k], self.depth_n[k]) for k in range(4)},
        }


def _load_bottom_index(data_dir, device):
    """读取同 MAP-Net 的海底深度索引；不存在或形状错误时立即终止。"""
    bottom_path = Path(data_dir) / "bottom_depth_index_r2.npy"
    if not bottom_path.is_file():
        raise FileNotFoundError(
            f"缺少用于有效水体评价的文件：{bottom_path}。"
            " 请使用与 MAP-Net 评价相同的 bottom_depth_index_r2.npy。"
        )
    bottom = np.load(bottom_path, allow_pickle=False)
    if bottom.shape != (121, 9, 250):
        raise ValueError(f"海底深度索引必须是 (121,9,250)，实际为 {bottom.shape}")
    # 与 MAP-Net 保持一致：不改变原文件类型，water_valid_mask 内统一转换为 long。
    return torch.from_numpy(np.array(bottom, copy=True)).to(device)


def _sample_rmse_mae_db(pred_db, true_db, valid_mask):
    """逐样本有效水体 RMSE 和 MAE（dB）；必须显式传入 water_valid_mask。"""
    if pred_db.shape != true_db.shape or valid_mask.shape != true_db.shape:
        raise ValueError("pred_db、true_db、valid_mask 形状必须完全一致。")
    valid = valid_mask.to(torch.bool)
    counts = valid.flatten(start_dim=1).sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("存在无有效水体评价网格的样本。")
    if bool(((~torch.isfinite(pred_db) | ~torch.isfinite(true_db)) & valid).any()):
        raise FloatingPointError("有效水体区域出现 NaN/Inf，不能计算误差。")
    # 将无效点先替换为零，再作差，避免 NaN 在无效点传播到误差统计。
    p = torch.where(valid, pred_db.detach().to(torch.float64), 0.0)
    t = torch.where(valid, true_db.detach().to(torch.float64), 0.0)
    residual = (p - t).flatten(start_dim=1)
    n = counts.to(torch.float64)
    rmse = (residual.square().sum(dim=1) / n).sqrt()
    mae = residual.abs().sum(dim=1) / n
    return rmse.cpu().numpy(), mae.cpu().numpy()


def _save_sample_errors(output_dir, filename, model_name, split, lo, hi,
                        evaluate_full_split, collected_ids, collected_rmse,
                        collected_mae, metric_scope):
    """按原始样本编号升序保存 RMSE、MAE，并检查三者严格一一对应。"""
    sample_ids = np.asarray(collected_ids, dtype=np.int64)
    rmse_db = np.asarray(collected_rmse, dtype=np.float64)
    mae_db = np.asarray(collected_mae, dtype=np.float64)
    if not (len(sample_ids) == len(rmse_db) == len(mae_db)) or len(sample_ids) == 0:
        raise RuntimeError("逐样本编号、RMSE 和 MAE 数量不一致或为空。")

    order = np.argsort(sample_ids)
    sample_ids = sample_ids[order]
    rmse_db = rmse_db[order]
    mae_db = mae_db[order]
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise RuntimeError("测试数据中出现重复的原始样本编号，已停止导出。")
    if evaluate_full_split and not np.array_equal(sample_ids, np.arange(lo, hi, dtype=np.int64)):
        raise RuntimeError(
            f"完整划分样本编号不连续或不完整：预期 [{lo}, {hi-1}] 共 {hi-lo} 条，"
            f"实际 {len(sample_ids)} 条。"
        )
    if not np.isfinite(rmse_db).all() or not np.isfinite(mae_db).all():
        raise RuntimeError("逐样本 RMSE/MAE 中出现 NaN/Inf，已停止导出。")

    path = output_dir / filename
    np.savez_compressed(
        path,
        sample_ids=sample_ids,   # [N] 0-based 原始全局编号
        rmse_db=rmse_db,         # [N] RMSE，单位 dB
        mae_db=mae_db,           # [N] MAE，单位 dB
        model_name=np.asarray(model_name),
        split=np.asarray(split),
        metric_scope=np.asarray(metric_scope),
    )
    print(f"已保存逐样本 RMSE 和 MAE：{path}（{len(sample_ids)} 条）", flush=True)
    return path


@torch.inference_mode()
def run_evaluation(split=SPLIT, sample_ids=None,
                   evaluate_full_split=EVALUATE_FULL_SPLIT,
                   data_dir=None, checkpoint=None, output_dir=None,
                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                   angle_chunk=ANGLE_CHUNK, amp=AMP):
    """Evaluate chosen sample IDs or whole year split, optionally export NPZ."""
    if split not in SPLIT_SLICES:
        raise ValueError('split must be train, val or test')
    if batch_size < 1 or num_workers < 0 or not 1 <= angle_chunk <= 9:
        raise ValueError('batch-size>=1, workers>=0 and angle-chunk in [1,9] required')
    if amp not in ('auto','bf16','fp16','off'):
        raise ValueError('amp must be auto, bf16, fp16, or off')
    data_dir = Path(DATA_DIR if data_dir is None else data_dir)
    checkpoint = Path(CHECKPOINT if checkpoint is None else checkpoint)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    lo, hi = SPLIT_SLICES[split]
    ids = sorted(set(map(int,
        ([] if evaluate_full_split and sample_ids is None else
         SAMPLE_IDS if sample_ids is None else sample_ids))))
    if any(not lo <= i < hi for i in ids):
        raise ValueError(f'sample IDs must be within {split} split [{lo},{hi-1}]')
    if not evaluate_full_split and not ids:
        raise ValueError('Subset evaluation requires at least one sample ID')
    check_files(data_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if amp == 'auto':
        amp = 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp16'
    if amp == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support BF16; use fp16 or off')

    # Load only checkpoints you trust. Their args can contain pickled pathlib objects.
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False,
                      pickle_module=CrossPlatformPickle)
    if ckpt.get('model_name') not in CHECKPOINT_MODEL_NAMES:
        raise ValueError('Checkpoint belongs to a different network architecture')
    stats = ckpt['stats']
    if stats.get('tl_scope') != 'full_grid':
        raise ValueError('Checkpoint was not trained with the required full-grid TL statistics')
    saved = ckpt.get('args', {})
    model = build_model(data_dir, stats, width=int(saved.get('width',32)),
                        range_max_m=float(saved.get('range_max_m',50000.)),
                        depth_max_m=float(saved.get('depth_max_m',4000.))).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    if ckpt.get('model_parameters',count_parameters(model)) != count_parameters(model):
        raise ValueError('Checkpoint parameter count mismatch')
    model.eval()
    ds = TLDataset(data_dir,split,
                   tl_invalid_raw=stats.get('tl_invalid_raw'),
                   invalid_target_fill_db=stats.get('invalid_target_fill_db'),check=False)
    if not evaluate_full_split:
        ds = Subset(ds, [i-lo for i in ids])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type=='cuda')
    bottom = _load_bottom_index(data_dir, device)
    metrics = MaskedMetrics(device)
    output_dir.mkdir(parents=True,exist_ok=True)
    exported = set(ids)
    # 逐样本 RMSE/MAE 仅保存三个长度为 N 的向量，不缓存整个 TL 预测张量。
    collected_sample_ids = []
    collected_sample_rmse = []
    collected_sample_mae = []
    for step,(ssp,bathy,truth_db,raw_ids) in enumerate(loader,1):
        ssp = ssp.to(device,non_blocking=True)
        bathy = bathy.to(device,non_blocking=True)
        truth_db = truth_db.to(device,non_blocking=True)
        pred_db = torch.empty_like(truth_db)
        for ai in _angle_chunks(angle_chunk):
            ctx = (torch.autocast('cuda', dtype=torch.bfloat16 if amp=='bf16' else torch.float16)
                   if device.type=='cuda' and amp!='off' else nullcontext())
            with ctx:
                pred_db[:,ai] = model.predict_db(ssp,bathy,angle_indices=ai).float()
        mask = water_valid_mask(truth_db, raw_ids, bottom)
        metrics.add(pred_db, truth_db, mask)
        batch_rmse, batch_mae = _sample_rmse_mae_db(pred_db, truth_db, mask)
        collected_sample_ids.extend(raw_ids.detach().cpu().tolist())
        collected_sample_rmse.extend(batch_rmse.tolist())
        collected_sample_mae.extend(batch_mae.tolist())

        for j,sid in enumerate(raw_ids.tolist()):
            if sid not in exported:
                continue
            out_path=output_dir/f'{split}_sample_{sid}.npz'
            m = mask[j].cpu().numpy()
            np.savez_compressed(out_path,
                truth_db=np.where(m, truth_db[j].cpu().numpy(), np.nan),
                pred_db=np.where(m, pred_db[j].cpu().numpy(), np.nan),
                valid_mask=m,
                sample_id=sid, source_idx=sid%121,
                angles_deg=np.arange(9)*40,
                depth_m=np.linspace(0,float(saved.get('depth_max_m',4000.)),200),
                range_m=np.linspace(0,float(saved.get('range_max_m',50000.)),250),
                grid_scope='water_mask',
                checkpoint_epoch=int(ckpt.get('epoch',-1))+1)
            print('已导出：',out_path,flush=True)
        if evaluate_full_split and (step%100==0 or step==len(loader)):
            print(f'评估进度：{min(step*batch_size,len(ds))}/{len(ds)}',flush=True)
    # 整个 test 集按全局样本编号排序，一一对应保存 RMSE/MAE(dB)。
    _save_sample_errors(
        output_dir, "DD_PUNet_sample_errors.npz", "DD-PUNet",
        split, lo, hi, evaluate_full_split,
        collected_sample_ids, collected_sample_rmse, collected_sample_mae,
        metric_scope="water_mask",
    )

    report_metrics=metrics.result()
    report={'split':split,'checkpoint':str(checkpoint),
            'checkpoint_epoch':int(ckpt.get('epoch',-1))+1,
            'model_name':MODEL_ARCHITECTURE,
            'model_parameters':count_parameters(model),
            'evaluated_samples':len(ds),
            'is_exported_subset':not evaluate_full_split,
            'sample_ids_exported':ids,
            'units':'dB',
            'evaluation_scope':'water_mask',
            **report_metrics}
    name=f'{split}_{"metrics_db" if evaluate_full_split else "exported_subset_metrics_db"}.json'
    (output_dir/name).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    print('有效水体dB误差：',report_metrics['overall'],flush=True)
    if not evaluate_full_split:
        print('注意：以上仅为指定样本误差，不代表整个测试集。',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description='Evaluate DD-PUNet on valid water cells (same mask as MAP-Net)')
    p.add_argument('--split',choices=list(SPLIT_SLICES),default=SPLIT)
    p.add_argument('--full-split',action='store_true', default=EVALUATE_FULL_SPLIT)
    p.add_argument("--subset", dest="full_split", action="store_false",
                          help="仅评价 --export-ids 指定样本（不评价整个划分）")
    p.add_argument('--export-ids','--sample-ids',dest='export_ids',default=None,
                   help='comma-separated original IDs; required for selected subset')
    p.add_argument('--batch-size',type=int,default=BATCH_SIZE)
    p.add_argument('--workers',type=int,default=NUM_WORKERS)
    p.add_argument('--angle-chunk',type=int,default=ANGLE_CHUNK)
    p.add_argument('--amp',choices=('auto','bf16','fp16','off'),default=AMP)
    p.add_argument('--data-dir',type=Path,default=DATA_DIR)
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    args=p.parse_args()
    ids=([] if args.full_split and args.export_ids is None else
         SAMPLE_IDS if args.export_ids is None else
         [int(x.strip()) for x in args.export_ids.split(',') if x.strip()])
    run_evaluation(split=args.split, sample_ids=ids,
                   evaluate_full_split=args.full_split,
                   data_dir=args.data_dir,checkpoint=args.checkpoint,
                   output_dir=args.output_dir,batch_size=args.batch_size,
                   num_workers=args.workers,angle_chunk=args.angle_chunk,amp=args.amp)


if __name__=='__main__':
    main()
