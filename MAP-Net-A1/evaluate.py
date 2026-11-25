
"""三维传播损失预测与水体掩码评价。"""
import argparse
from contextlib import nullcontext
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickle
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from map_net_a1.data import TLDataset, SPLIT_SLICES, check_files
from map_net_a1.metrics import MaskedMetrics, water_valid_mask
from map_net_a1.model import build_model, MODEL_ARCHITECTURE, checkpoint_matches

# User configuration (including Spyder execution)
from map_net_a1.paths import PROJECT_DIR, DEFAULT_DATA_DIR, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT_DIR

ROOT = PROJECT_DIR
DATA_DIR = DEFAULT_DATA_DIR
CHECKPOINT = DEFAULT_CHECKPOINT
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "evaluation"

# Full-split evaluation exports sample-level RMSE and MAE.
MODEL_OUTPUT_NAME = "MAP_Net_A1"
MODEL_LABEL = "MAP-Net-A1"
METRIC_SCOPE = "water_mask"
SPLIT = "test"
SAMPLE_IDS = [17424, 17600]
EVALUATE_FULL_SPLIT = True

BATCH_SIZE = 4
NUM_WORKERS = 0
AMP = "auto"


class CrossPlatformUnpickler(pickle.Unpickler):
    """兼容 Linux/Windows 保存的路径对象，不改变模型权重。"""

    def find_class(self, module, name):
        if module in ("pathlib", "pathlib._local"):
            if name == "PosixPath":
                return PurePosixPath
            if name == "WindowsPath":
                return PureWindowsPath
        return super().find_class(module, name)


class CrossPlatformPickle:
    Unpickler = CrossPlatformUnpickler



def _sample_rmse_mae_db(pred_db, true_db, valid_mask=None):
    """逐样本计算 RMSE 和 MAE (dB)，各返回形状为 (B,) 的 float64 数组。

    与原脚本一致：valid_mask=None 时统计全网格；提供 mask 时只统计
    有效网格。无效点不会参与 MAE 或 RMSE，累计过程使用 float64。
    """
    if pred_db.shape != true_db.shape:
        raise ValueError(f"预测/真实 TL 的形状不一致：{pred_db.shape} / {true_db.shape}")

    pred = pred_db.detach().to(torch.float64)
    true = true_db.detach().to(torch.float64)
    if valid_mask is None:
        # Full-grid metrics require finite targets.
        if not torch.isfinite(pred).all() or not torch.isfinite(true).all():
            raise ValueError("完整网格中出现 NaN/Inf，无法计算 full-grid RMSE/MAE，请检查数据填充。")
        residual = pred - true
        flat = residual.flatten(start_dim=1)
        rmse = flat.square().mean(dim=1).sqrt()
        mae = flat.abs().mean(dim=1)
    else:
        valid = valid_mask.bool()
        if valid.shape != true.shape:
            raise ValueError(f"Mask/TL 的形状不一致：{valid.shape} / {true.shape}")
        counts = valid.flatten(start_dim=1).sum(dim=1)
        if (counts == 0).any():
            raise ValueError("有样本没有任何有效评价网格点，无法计算该样本的 RMSE/MAE。")
        if not torch.isfinite(true[valid]).all() or not torch.isfinite(pred[valid]).all():
            raise ValueError("有效评价区域中出现 NaN/Inf，无法计算该样本的 RMSE/MAE。")
        # 先屏蔽后求绝对值/平方，避免 NaN * 0 传播。
        residual = torch.where(valid, pred - true, torch.zeros_like(pred))
        flat = residual.flatten(start_dim=1)
        rmse = (flat.square().sum(dim=1) / counts).sqrt()
        mae = flat.abs().sum(dim=1) / counts
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
                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, amp=AMP):
    """预测指定样本或整个划分；所有误差均在真实dB尺度计算。"""

    data_dir = Path(DATA_DIR if data_dir is None else data_dir)
    checkpoint = Path(CHECKPOINT if checkpoint is None else checkpoint)
    output_dir = Path(OUTPUT_DIR if output_dir is None else output_dir)
    # 全集评价默认不另存 SAMPLE_IDS 所列的单个三维声场。需要时通过
    # --export-ids 显式指定；--subset 会按 SAMPLE_IDS 评价少量样本。
    ids = sorted(set(map(int,
        ([] if evaluate_full_split and sample_ids is None else
         SAMPLE_IDS if sample_ids is None else sample_ids))))

    if split not in SPLIT_SLICES:
        raise ValueError("SPLIT 必须是 train、val 或 test")

    lo, hi = SPLIT_SLICES[split]
    if any(not lo <= i < hi for i in ids) or (not ids and not evaluate_full_split):
        raise ValueError(f"样本编号应位于 [{lo}, {hi - 1}]，且指定样本模式不能为空")

    if amp not in ("auto", "bf16", "fp16", "off"):
        raise ValueError("AMP 必须是 auto、bf16、fp16 或 off")

    check_files(data_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if amp == "auto":
        amp = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp16"

    if device.type == "cuda" and amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持BF16，请使用AMP='auto'或'fp16'")

    use_amp = device.type == "cuda" and amp != "off"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    # 兼容Linux超算保存的best.pt，仅加载可信任的自训练模型

    ckpt = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        pickle_module=CrossPlatformPickle
    )

    if not checkpoint_matches(ckpt):
        raise ValueError(
            f"checkpoint不是A1模型：记录的是 {ckpt.get('model_name')!r}；"
            f"需要 {MODEL_ARCHITECTURE!r}。"
        )
    stats = ckpt["stats"]

    model = build_model(data_dir, stats).to(device)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "无法加载权重：checkpoint必须来自当前MAP-Net-A1模型。"
        ) from exc
    model.eval()


    ds = TLDataset(
        data_dir, split,
        tl_invalid_raw=stats.get("tl_invalid_raw"),
        check=False
    )

    if not evaluate_full_split:
        ds = Subset(ds, [i - lo for i in ids])

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda"
    )

    bottom = torch.from_numpy(
        np.array(
            np.load(data_dir / "bottom_depth_index_r2.npy", allow_pickle=False),
            copy=True
        )
    ).to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, export_ids = MaskedMetrics(device), set(ids)

    # 仅累计每个样本的编号、RMSE、MAE，不保存全测试集的预测 TL 张量。
    collected_sample_ids = []
    collected_sample_rmse = []
    collected_sample_mae = []

    for step, (ssp, bathy, true_db, sample_ids_t) in enumerate(loader, 1):

        ssp, bathy, true_db, sample_ids_t = (
            x.to(device, non_blocking=True)
            for x in (ssp, bathy, true_db, sample_ids_t)
        )

        ctx = torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext()

        with ctx:

            pred_db = model.predict_db(ssp, bathy)

        # 动态生成水体掩码，同时排除真实TL中的NaN/Inf
        mask = water_valid_mask(true_db, sample_ids_t, bottom)

        metrics.add(pred_db, true_db, mask)

        batch_rmse, batch_mae = _sample_rmse_mae_db(pred_db, true_db, mask)
        collected_sample_ids.extend(sample_ids_t.detach().cpu().tolist())
        collected_sample_rmse.extend(batch_rmse.tolist())
        collected_sample_mae.extend(batch_mae.tolist())


        for j, sid in enumerate(sample_ids_t.tolist()):
            if sid not in export_ids:
                continue

            m = mask[j].cpu().numpy()
            true = np.where(m, true_db[j].cpu().numpy(), np.nan)
            pred = np.where(m, pred_db[j].float().cpu().numpy(), np.nan)

            path = output_dir / f"{split}_sample_{sid}.npz"

            np.savez_compressed(
                path,
                truth_db=true,
                pred_db=pred,
                valid_mask=m,
                sample_id=sid,
                source_idx=sid % 121,
                angles_deg=np.arange(9) * 40,
                depth_m=np.linspace(0, 4000, 200),
                range_m=np.linspace(0, 50000, 250)
            )
            print(f"已导出：{path}")

        if evaluate_full_split and (step % 100 == 0 or step == len(loader)):
            print(f"评估进度：{min(step * batch_size, len(ds))}/{len(ds)}")


    # 每个原始样本 ID 恰好一条 RMSE 和 MAE；全划分检查无重复/缺失。
    sample_errors_file = f"{MODEL_OUTPUT_NAME}_sample_errors.npz"
    _save_sample_errors(
        output_dir, sample_errors_file, MODEL_LABEL,
        split, lo, hi, evaluate_full_split,
        collected_sample_ids, collected_sample_rmse, collected_sample_mae,
        metric_scope=METRIC_SCOPE,
    )

    result = metrics.result()

    report = {
        "split": split,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) + 1,
        "model_name": MODEL_ARCHITECTURE,
        "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "evaluated_samples": len(ds),
        "is_exported_subset": not evaluate_full_split,
        "sample_ids_exported": ids,
        "units": "dB",
        "sample_errors_file": sample_errors_file,
        **result
    }

    name = f"{split}_{'metrics_db' if evaluate_full_split else 'exported_subset_metrics_db'}.json"

    (output_dir / name).write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("真实dB误差：", result["overall"])

    if not evaluate_full_split:
        print("注意：以上仅为选定样本的误差，不代表整个测试集。")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAP-Net-A1 in dB using water masks")
    parser.add_argument("--split", choices=list(SPLIT_SLICES), default=SPLIT)
    parser.add_argument("--full-split", action="store_true", default=EVALUATE_FULL_SPLIT,
                        help="evaluate all 1452 examples; do not export all TL tensors")
    parser.add_argument("--subset", dest="full_split", action="store_false",
                        help="仅评价 --export-ids 指定样本；如未提供则使用顶部 SAMPLE_IDS")
    parser.add_argument("--export-ids", "--sample-ids", dest="export_ids", default=None,
                        help="comma-separated original sample IDs for export/subset evaluation")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default=AMP)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be >=1 and --workers must be >=0")
    ids = (None if args.export_ids is None else
           [int(x.strip()) for x in args.export_ids.split(",") if x.strip()])
    run_evaluation(
        split=args.split, sample_ids=ids, evaluate_full_split=args.full_split,
        data_dir=args.data_dir, checkpoint=args.checkpoint, output_dir=args.output_dir,
        batch_size=args.batch_size, num_workers=args.workers, amp=args.amp,
    )


if __name__ == "__main__":
    main()
