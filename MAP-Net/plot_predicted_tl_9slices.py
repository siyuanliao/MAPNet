#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 MAP-Net 对指定样本预测得到的 9 个方位角 2D TL 切片图（3×3布局，共享一个颜色棒）。

使用说明
--------
1. 将本脚本放在 MAP-Net 项目根目录（与 train.py 和 map_net/ 同级）附近，
   或保证 Python 能够 import map_net.* 。
2. 修改【Configuration】中的 SAMPLE_ID、DATA_DIR、CHECKPOINT 等参数。
3. 在 Spyder 中按 F5 运行，或命令行执行：

   python plot_predicted_tl_9slices.py --sample-id 17644

输出
----
- 1 张 3×3 子图组成的 JPG（可选PDF），内容为 9 个方位角的预测 TL 2D 切片。
- 9 个子图共用 1 根颜色棒，放在整张图最右侧。

说明
----
- 每个子图对应一个方位角（0°, 40°, ..., 320°）。
- 子图上方仅标注角度，不再标注(a)(b)等编号。
- 默认使用 jet_r 色图。
- 默认仅显示有效水体区域；海底以下和无效点可显示为浅灰色背景。
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import pickle
from pathlib import Path, PurePosixPath, PureWindowsPath

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import torch

from map_net.data import N_SAMPLES, SPLIT_SLICES, TLDataset, check_files
from map_net.metrics import water_valid_mask
from map_net.model import build_model
from map_net.paths import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


# Configuration
SAMPLE_ID = 17644
DATA_DIR = DEFAULT_DATA_DIR
CHECKPOINT = DEFAULT_CHECKPOINT
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / 'single_sample_tl_slices'

DEVICE = 'auto'            # 'auto' / 'cuda' / 'cpu'
AMP = 'auto'               # 'auto' / 'bf16' / 'fp16' / 'off'
SHOW_FIGURES = False
SAVE_PDF = False
DPI = 300

# 图片整体宽:高 = 1.8
FIGURE_WIDTH = 12.6
FIGURE_HEIGHT = 7.0

# 颜色设置
TL_CMAP = 'jet_r'
TL_VMIN_DB = 60          # 例如 60.0；None 表示自动
TL_VMAX_DB = 95          # 例如 95.0；None 表示自动
INVALID_COLOR = '#d9d9d9'  # 海底以下/无效区域颜色

# 坐标范围
DEPTH_MAX_M = 4000.0       # 200层对应0~4000 m
RANGE_MAX_KM = 50.0        # 250点对应0~50 km

# 子图和颜色棒布局
WSPACE = 0.10
HSPACE = 0.18
COLORBAR_PAD = 0.02        # 与绘图区间距（图像坐标系）
COLORBAR_WIDTH = 0.018
COLORBAR_LABEL = 'TL (dB)'

# 是否叠加海底线（基于 local bathymetry 沿对应方位抽样）
SHOW_SEABED_LINE = True
SEABED_LINE_COLOR = 'k'
SEABED_LINE_WIDTH = 1.0
FILL_BELOW_SEABED = True
BELOW_SEABED_COLOR = '#d9d9d9'

# 字体
TITLE_SIZE = 16            # 整体标题
SUBTITLE_SIZE = 12         # 每个子图角度标题
LABEL_SIZE = 11
TICK_SIZE = 9
# ====================================================================

ANGLES_DEG = np.arange(9, dtype=np.int64) * 40
RECEIVER_DEPTH_M = np.linspace(0.0, DEPTH_MAX_M, 200)
RANGE_M = np.linspace(0.0, RANGE_MAX_KM * 1000.0, 250)


class CrossPlatformUnpickler(pickle.Unpickler):
    """兼容跨操作系统保存的可信 checkpoint 中的 Path 对象。"""
    def find_class(self, module, name):
        if module in ('pathlib', 'pathlib._local'):
            if name == 'PosixPath':
                return PurePosixPath
            if name == 'WindowsPath':
                return PureWindowsPath
        return super().find_class(module, name)


class CrossPlatformPickle:
    Unpickler = CrossPlatformUnpickler


def sample_split(sample_id: int):
    if not (0 <= sample_id < N_SAMPLES):
        raise ValueError(f'样本编号必须在 0~{N_SAMPLES-1}，实际: {sample_id}')
    for name, (start, stop) in SPLIT_SLICES.items():
        if start <= sample_id < stop:
            return name, sample_id - start
    raise RuntimeError('SPLIT_SLICES 配置错误')


def device_and_amp(device_arg: str, amp_arg: str):
    if device_arg not in ('auto', 'cuda', 'cpu'):
        raise ValueError(f'非法 DEVICE: {device_arg}')
    if amp_arg not in ('auto', 'bf16', 'fp16', 'off'):
        raise ValueError(f'非法 AMP: {amp_arg}')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定 CUDA，但当前环境没有可用 GPU')
    cuda = torch.cuda.is_available() if device_arg == 'auto' else (device_arg == 'cuda')
    device = torch.device('cuda' if cuda else 'cpu')
    amp = amp_arg
    if amp == 'auto':
        amp = 'bf16' if cuda and torch.cuda.is_bf16_supported() else ('fp16' if cuda else 'off')
    if cuda and amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU不支持 BF16，请改为 fp16 或 off')
    return device, amp


def _resolve_color_limits(array, user_vmin, user_vmax, *, label, default_percentiles=(1, 99)):
    arr = np.asarray(array, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f'{label}没有有效数值，无法设置色标')
    vmin = float(np.percentile(arr, default_percentiles[0])) if user_vmin is None else float(user_vmin)
    vmax = float(np.percentile(arr, default_percentiles[1])) if user_vmax is None else float(user_vmax)
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError(f'{label} 色标范围无效: vmin={vmin}, vmax={vmax}')
    return vmin, vmax


def sample_bathymetry_along_rays(bathymetry_m: np.ndarray,
                                 range_km: np.ndarray,
                                 angles_deg: np.ndarray = ANGLES_DEG) -> np.ndarray:
    """61×61地形网格按真实地理方向双线性插值，返回 [9, 250] 海底水深（m）。"""
    bathy = np.asarray(bathymetry_m, dtype=np.float64)
    if bathy.shape != (61, 61):
        raise ValueError(f'地形维度应为(61,61)，实际 {bathy.shape}')
    if not np.isfinite(bathy).all():
        raise ValueError('局部海底地形含有非有限数值')

    theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))[:, None]
    r = np.asarray(range_km, dtype=np.float64)[None, :]
    col_f = (r * np.sin(theta) + 60.0) / 2.0
    row_f = (r * np.cos(theta) + 60.0) / 2.0
    col_f = np.clip(col_f, 0, 60)
    row_f = np.clip(row_f, 0, 60)

    c0 = np.floor(col_f).astype(int)
    r0 = np.floor(row_f).astype(int)
    c1 = np.minimum(c0 + 1, 60)
    r1 = np.minimum(r0 + 1, 60)
    tc = col_f - c0
    tr = row_f - r0

    depth = ((1 - tr) * ((1 - tc) * bathy[r0, c0] + tc * bathy[r0, c1]) +
             tr * ((1 - tc) * bathy[r1, c0] + tc * bathy[r1, c1]))
    return np.maximum(depth, 0.0).astype(np.float32)


def evaluate_sample(sample_id: int, data_dir: Path, checkpoint: Path,
                    device_arg: str = DEVICE, amp_arg: str = AMP) -> dict:
    split, local_id = sample_split(sample_id)
    check_files(data_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'没有找到权重文件: {checkpoint}')
    device, amp = device_and_amp(device_arg, amp_arg)

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False,
                      pickle_module=CrossPlatformPickle)
    if 'model' not in ckpt or 'stats' not in ckpt:
        raise KeyError("best.pt 必须包含 'model' 和 'stats'")

    model = build_model(data_dir, ckpt['stats']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    dataset = TLDataset(data_dir, split,
                        tl_invalid_raw=ckpt['stats'].get('tl_invalid_raw'),
                        check=False)
    ssp_t, bathy_t, truth_t, index_t = dataset[local_id]
    if index_t.item() != sample_id:
        raise RuntimeError('数据与全局样本编号不一致')

    with torch.inference_mode():
        autocast_ctx = (torch.autocast('cuda', dtype=torch.bfloat16 if amp == 'bf16' else torch.float16)
                        if device.type == 'cuda' and amp != 'off' else nullcontext())
        with autocast_ctx:
            pred_t = model.predict_db(
                ssp_t.unsqueeze(0).to(device),
                bathy_t.unsqueeze(0).to(device)
            ).float().cpu()

    if pred_t.shape != (1, 9, 200, 250):
        raise ValueError(f'模型预测形状有误: {tuple(pred_t.shape)}')

    bottom_index_all = np.load(data_dir / 'bottom_depth_index_r2.npy', allow_pickle=False)
    bottom_index_t = torch.from_numpy(np.array(bottom_index_all, copy=True))
    truth_b = truth_t.unsqueeze(0)
    mask_b = water_valid_mask(truth_b, index_t.reshape(1), bottom_index_t)
    if not bool(mask_b.any()):
        raise ValueError(f'样本 {sample_id} 没有有效水体TL网格点')

    valid = mask_b[0].numpy()
    pred_db = np.where(valid, pred_t[0].numpy(), np.nan).astype(np.float32)

    return dict(
        sample_id=sample_id,
        split=split,
        pred_db=pred_db,
        valid_mask=valid,
        bathymetry_m=bathy_t.numpy(),
        checkpoint=str(checkpoint),
    )


def plot_predicted_tl_slices(data: dict, filepath: Path, dpi: int, show: bool, save_pdf: bool):
    pred = np.asarray(data['pred_db'], dtype=np.float32)
    if pred.shape != (9, 200, 250):
        raise ValueError(f'预测数组形状错误: {pred.shape}')

    tl_vmin, tl_vmax = _resolve_color_limits(pred, TL_VMIN_DB, TL_VMAX_DB,
                                             label='Predicted TL', default_percentiles=(1, 99))
    cmap = plt.get_cmap(TL_CMAP).copy()
    cmap.set_bad(INVALID_COLOR)
    norm = Normalize(vmin=tl_vmin, vmax=tl_vmax, clip=True)

    range_km = RANGE_M / 1000.0
    depth_km = RECEIVER_DEPTH_M / 1000.0
    bathy_rays_km = sample_bathymetry_along_rays(data['bathymetry_m'], range_km) / 1000.0

    fig, axes = plt.subplots(3, 3, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), sharex=True, sharey=True)
    axes = np.asarray(axes)

    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.10, top=0.95,
                        wspace=WSPACE, hspace=HSPACE)

    im = None
    extent = [range_km[0], range_km[-1], depth_km[-1], depth_km[0]]
    for i, ax in enumerate(axes.ravel()):
        arr = np.ma.masked_invalid(pred[i])
        im = ax.imshow(arr, origin='upper', extent=extent,
                       cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')

        if SHOW_SEABED_LINE:
            seabed = np.clip(bathy_rays_km[i], 0.0, depth_km[-1])
            if FILL_BELOW_SEABED:
                ax.fill_between(range_km, seabed, depth_km[-1], color=BELOW_SEABED_COLOR, zorder=2)
            ax.plot(range_km, seabed, color=SEABED_LINE_COLOR, lw=SEABED_LINE_WIDTH, zorder=3)

        ax.set_title(f'{int(ANGLES_DEG[i])}°', fontsize=SUBTITLE_SIZE, pad=6)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(False)

    # 统一坐标标签：只保留最左列和最下排，可读性更好
    for r in range(3):
        axes[r, 0].set_ylabel('Depth (km)', fontsize=LABEL_SIZE)
    for c in range(3):
        axes[2, c].set_xlabel('Range (km)', fontsize=LABEL_SIZE)

    # 共享颜色棒（右侧）
    cax = fig.add_axes([0.90 + COLORBAR_PAD, 0.15, COLORBAR_WIDTH, 0.70])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(COLORBAR_LABEL, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, facecolor='white', bbox_inches='tight')
    if save_pdf:
        fig.savefig(filepath.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    print(f'绘图完成: {filepath}')
    if show:
        plt.show()
    plt.close(fig)


def run(sample_id: int = SAMPLE_ID,
        data_dir=DATA_DIR,
        checkpoint=CHECKPOINT,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        amp=AMP,
        show=SHOW_FIGURES,
        dpi=DPI,
        save_pdf=SAVE_PDF):
    sample_id = int(sample_id)
    data_dir = Path(data_dir)
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)

    data = evaluate_sample(sample_id, data_dir, checkpoint, device, amp)
    result_dir = output_dir / f'sample_{sample_id:05d}'
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_predicted_tl_slices(data, result_dir / 'predicted_tl_9_slices.jpg', dpi, show, save_pdf)
    print(f'结果已保存至: {result_dir.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='绘制指定样本的MAP-Net预测3D TL九方位2D切片图')
    parser.add_argument('--sample-id', type=int, default=SAMPLE_ID, help='全局0-based样本编号 0~18875')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default=DEVICE)
    parser.add_argument('--amp', choices=['auto', 'bf16', 'fp16', 'off'], default=AMP)
    parser.add_argument('--show', action='store_true', help='显示图片')
    parser.add_argument('--pdf', action='store_true', help='同时保存PDF')
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()

    run(sample_id=args.sample_id,
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        amp=args.amp,
        show=args.show or SHOW_FIGURES,
        dpi=args.dpi,
        save_pdf=args.pdf or SAVE_PDF)


if __name__ == '__main__':
    main()
