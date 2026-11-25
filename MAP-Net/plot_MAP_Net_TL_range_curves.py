# -*- coding: utf-8 -*-
"""MAP-Net 单样本 / 单方位角 / 多接收深度的 TL 距离曲线对比。

每个指定深度生成一张图：
  上图：真实 TL（蓝色）与 MAP-Net 预测 TL（红色）
  下图：预测 TL - 真实 TL（绿色）及其与零线之间的填充区域

使用方法：把本文件放在 MAP-Net 工程根目录（与 map_net/ 同级），
在配置区设置 SAMPLE_ID、AZIMUTH_DEG、DEPTHS_M，
在 Spyder 中按 F5，或命令行执行：
    python plot_MAP_Net_TL_range_curves.py

注意：
  - 只使用已计算的9个离散方位角：0、40、...、320 度。
  - 模型推理仅执行1次，多个深度共用这次推理的结果。
  - 仅选择距离指定深度最近的原始接收层（不进行插值）。
  - 水体掩码与已有 evaluate.py 一致，海底以下/无效区域不显示。
"""

from __future__ import annotations

# Configuration

# 样本编号（0-based）：0~18875，测试集为17424~18875
SAMPLE_ID = 17928

# 方位角：只能填写 0, 40, 80, 120, 160, 200, 240, 280, 320 之一
AZIMUTH_DEG = 0

# 一个或多个接收深度（m），每个深度输出一张上下两层的图
DEPTHS_M = [1000, 2000, 3000]

# 深度提取：固定为最近的原始接收深度网格层，不进行任何插值。

# 数据、checkpoint 与结果保存目录
# 默认目录由 map_net.paths 定义。
# None 表示使用默认路径。
DATA_DIR = None
CHECKPOINT = None
OUTPUT_DIR = None

DEVICE = 'auto'              # 'auto' / 'cuda' / 'cpu'
AMP = 'auto'                 # 'auto' / 'bf16' / 'fp16' / 'off'

# 画布/输出
FIGSIZE = (9.5, 4.6)        # 单张图宽、高（英寸），压缩后缩小总画布高度
DPI = 300
FILE_FORMAT = 'png'         # 'png'、'jpg'、'pdf' 等 Matplotlib 支持的格式
SHOW_FIGURES = False        # True：绘制后显示；False：只保存
SAVE_CSV = False            # True：每个深度另存距离/真实值/预测值/误差CSV

# 主图显示及配色
PRED_COLOR = 'red'
TRUE_COLOR = 'blue'
ERROR_COLOR = 'forestgreen'
ERROR_FILL_COLOR = '#81c995'
ERROR_FILL_ALPHA = 0.32
PRED_LINEWIDTH = 1.8
TRUE_LINEWIDTH = 1.5
ERROR_LINEWIDTH = 1.35
ZERO_LINE_COLOR = '0.3'

# 轴和版式
RANGE_XLIM_KM = (0.0, 50.0)  # 可改为 None 自动
TL_YLIM_DB = None           # 例如 (50, 110)，None 自动
ERROR_YLIM_DB = None        # 例如 (-10, 10)，None 自动
# 精确控制两幅子图的实际高度/距离（相对上一版）：
# 上层高度缩小30%，下层高度缩小20%，图间距离缩小50%。
# 旧版 FIGSIZE=(9.5,6.2)、height_ratios=(2,1)、hspace=.48、top=.96、bottom=.12
# 对应的物理尺寸分别是 2.800 / 1.400 / 1.008 英寸。
ORIGINAL_TOP_HEIGHT_IN = 2.800
ORIGINAL_BOTTOM_HEIGHT_IN = 1.400
ORIGINAL_GAP_IN = 1.008
TOP_HEIGHT_SCALE = 0.6
BOTTOM_HEIGHT_SCALE = 0.80
INTERPANEL_GAP_SCALE = 0.38  # 缩小间距，令下层图相对上移约0.12英寸
BOTTOM_MARGIN_IN = 0.77    # 将下层图上移约0.12英寸，保留底部Range (km)标签
TOP_MARGIN_MIN_IN = 0.22  # 上方最小留白（英寸）
PLOT_GRID = True
GRID_ALPHA = 0.30
LEGEND_LOC = 'lower right'  # 上图右下角；不再标注方位角和深度
FONT_SIZE = 11
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10

# 物理坐标：与当前 MAP-Net/evaluate.py 保持一致
# 如原始数值仿真使用不同坐标，须先核对模型本身的坐标定义。
RECEIVER_DEPTH_M = (0.0, 4000.0, 200)  # (起点m, 终点m, 层数)
RANGE_COORD_M = (0.0, 50000.0, 250)    # (起点m, 终点m, 网格点数)

# ========================= End configuration =========================

import csv
import pickle
from contextlib import nullcontext
from pathlib import Path, PurePosixPath, PureWindowsPath

import matplotlib.pyplot as plt
import numpy as np
import torch

from map_net.data import N_SAMPLES, SPLIT_SLICES, TLDataset, check_files
from map_net.metrics import water_valid_mask
from map_net.model import build_model
from map_net.paths import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR

ANGLES_DEG = np.arange(9) * 40
DEPTH_GRID_M = np.linspace(*RECEIVER_DEPTH_M, dtype=np.float64)
RANGE_KM = np.linspace(*RANGE_COORD_M, dtype=np.float64) / 1000.0

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'axes.spines.top': True,
    'axes.spines.right': True,
})


class CrossPlatformUnpickler(pickle.Unpickler):
    """兼容Linux/Windows路径对象，仅用于加载自己训练的可信checkpoint。"""
    def find_class(self, module, name):
        if module in ('pathlib', 'pathlib._local'):
            if name == 'PosixPath':
                return PurePosixPath
            if name == 'WindowsPath':
                return PureWindowsPath
        return super().find_class(module, name)


class CrossPlatformPickle:
    Unpickler = CrossPlatformUnpickler


def get_split_and_index(sample_id):
    sample_id = int(sample_id)
    if not (0 <= sample_id < N_SAMPLES):
        raise ValueError(f'SAMPLE_ID 只能是 0~{N_SAMPLES-1}，实际 {sample_id}')
    for split, (lo, hi) in SPLIT_SLICES.items():
        if lo <= sample_id < hi:
            return split, sample_id - lo
    raise RuntimeError('无法确定样本所在的数据集')


def resolve_device_and_amp():
    if DEVICE not in ('auto', 'cuda', 'cpu'):
        raise ValueError('DEVICE 必须是 auto / cuda / cpu')
    if AMP not in ('auto', 'bf16', 'fp16', 'off'):
        raise ValueError('AMP 必须是 auto / bf16 / fp16 / off')
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('DEVICE=cuda，但没有可用CUDA设备')
    use_cuda = torch.cuda.is_available() if DEVICE == 'auto' else (DEVICE == 'cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')
    amp = AMP
    if amp == 'auto':
        amp = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if use_cuda else 'off'
    if use_cuda and amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU不支持BF16，请改用 AMP=auto / fp16 / off')
    return device, amp


def load_and_predict(sample_id, data_dir, checkpoint):
    """仅做一次推理，返回完整[9,200,250]真实/预测TL（dB）及有效水体掩码。"""
    split, local_id = get_split_and_index(sample_id)
    data_dir, checkpoint = Path(data_dir), Path(checkpoint)
    check_files(data_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'未找到训练权重: {checkpoint}')

    device, amp = resolve_device_and_amp()
    # 仅加载自己训练的、可信任的 checkpoint 文件。
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False,
                      pickle_module=CrossPlatformPickle)
    if 'model' not in ckpt or 'stats' not in ckpt:
        raise KeyError("checkpoint必须包含 'model' 和 'stats'")

    model = build_model(data_dir, ckpt['stats']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    ds = TLDataset(data_dir, split, tl_invalid_raw=ckpt['stats'].get('tl_invalid_raw'),
                   check=False)
    ssp_t, bathy_t, true_t, index_t = ds[local_id]
    if int(index_t.item()) != int(sample_id):
        raise RuntimeError('数据索引与指定样本编号不一致')

    with torch.inference_mode():
        autocast_ctx = (torch.autocast('cuda', dtype=(torch.bfloat16 if amp == 'bf16'
                                            else torch.float16))
                        if device.type == 'cuda' and amp != 'off' else nullcontext())
        with autocast_ctx:
            pred_t = model.predict_db(ssp_t.unsqueeze(0).to(device),
                                      bathy_t.unsqueeze(0).to(device)).float().cpu()
    if pred_t.shape != (1, 9, 200, 250):
        raise RuntimeError(f'MAP-Net预测输出形状错误：{tuple(pred_t.shape)}')

    bottom = np.load(data_dir / 'bottom_depth_index_r2.npy', allow_pickle=False)
    bottom_t = torch.from_numpy(np.array(bottom, copy=True))
    mask_t = water_valid_mask(true_t.unsqueeze(0), index_t.view(1), bottom_t)[0]
    if not bool(mask_t.any()):
        raise RuntimeError(f'样本{sample_id}没有有效水体网格点')
    if not bool(torch.isfinite(pred_t[0][mask_t]).all()):
        raise FloatingPointError('预测TL在有效水体内包含NaN/Inf')

    valid = mask_t.numpy()
    return {
        'sample_id': int(sample_id),
        'split': split,
        'true': np.where(valid, true_t.numpy(), np.nan).astype(np.float32),
        'pred': np.where(valid, pred_t[0].numpy(), np.nan).astype(np.float32),
        'mask': valid,
    }


def extract_depth_curve(data, angle_index, requested_depth_m):
    """按深度提取距离曲线，不跨越海底/无效值区域连线。"""
    depth_m = float(requested_depth_m)
    if not np.isfinite(depth_m) or not (DEPTH_GRID_M[0] <= depth_m <= DEPTH_GRID_M[-1]):
        raise ValueError(f'深度必须在 {DEPTH_GRID_M[0]:g}–{DEPTH_GRID_M[-1]:g} m：{depth_m}')
    truth_section = data['true'][angle_index]
    pred_section = data['pred'][angle_index]
    water_section = data['mask'][angle_index]

    # 直接选用最靠近指定深度的实际200层网格点（不进行插值）。
    index = int(np.argmin(np.abs(DEPTH_GRID_M - depth_m)))
    actual_depth_m = float(DEPTH_GRID_M[index])
    truth = np.asarray(truth_section[index], dtype=np.float64).copy()
    pred = np.asarray(pred_section[index], dtype=np.float64).copy()
    valid = np.asarray(water_section[index], dtype=bool).copy()

    valid &= np.isfinite(truth) & np.isfinite(pred)
    truth[~valid] = np.nan
    pred[~valid] = np.nan
    error = pred - truth
    if valid.any():
        errors = error[valid]
        rmse = float(np.sqrt(np.mean(np.square(errors, dtype=np.float64))))
        mae = float(np.mean(np.abs(errors)))
    else:
        rmse = mae = None

    return {
        'requested_depth_m': depth_m,
        'actual_depth_m': actual_depth_m,
        'range_km': RANGE_KM.copy(),
        'true_db': truth,
        'pred_db': pred,
        'error_db': error,
        'valid': valid,
        'rmse_db': rmse,
        'mae_db': mae,
    }


def save_curve_figure(curve, filepath):
    filepath = Path(filepath)
    # 手动指定坐标轴在画布中的实际高度和图间距，确保压缩比例精确。
    # 与仅调整 GridSpec 的 hspace 不同，这里的缩放不会被 FIGSIZE 隐式改变。
    fig_height_in = float(FIGSIZE[1])
    if len(FIGSIZE) != 2 or not all(np.isfinite(FIGSIZE)) or any(v <= 0 for v in FIGSIZE):
        raise ValueError('FIGSIZE 必须为 (正宽度, 正高度)')
    top_height_in = ORIGINAL_TOP_HEIGHT_IN * TOP_HEIGHT_SCALE
    bottom_height_in = ORIGINAL_BOTTOM_HEIGHT_IN * BOTTOM_HEIGHT_SCALE
    gap_in = ORIGINAL_GAP_IN * INTERPANEL_GAP_SCALE
    if min(top_height_in, bottom_height_in, gap_in) <= 0:
        raise ValueError('上图高度、下图高度和图间距离必须为正数')
    remaining_top_in = (fig_height_in - BOTTOM_MARGIN_IN - bottom_height_in
                        - gap_in - top_height_in)
    if remaining_top_in < TOP_MARGIN_MIN_IN:
        min_height = (BOTTOM_MARGIN_IN + bottom_height_in + gap_in
                      + top_height_in + TOP_MARGIN_MIN_IN)
        raise ValueError(f'FIGSIZE 的高度不足，至少需要 {min_height:.2f} 英寸')

    fig = plt.figure(figsize=FIGSIZE)
    left, width = 0.115, 0.860
    ax_bottom = fig.add_axes([
        left, BOTTOM_MARGIN_IN / fig_height_in,
        width, bottom_height_in / fig_height_in,
    ])
    ax_top = fig.add_axes([
        left, (BOTTOM_MARGIN_IN + bottom_height_in + gap_in) / fig_height_in,
        width, top_height_in / fig_height_in,
    ], sharex=ax_bottom)

    x = curve['range_km']
    yt = curve['true_db']
    yp = curve['pred_db']
    e = curve['error_db']
    valid = curve['valid']

    # 上图：先红色预测，再蓝色真实，统一配色。
    ax_top.plot(x, yp, lw=PRED_LINEWIDTH, color=PRED_COLOR, label='Predicted value')
    ax_top.plot(x, yt, lw=TRUE_LINEWIDTH, color=TRUE_COLOR, label='Real value')
    ax_top.set_ylabel('TL (dB)')
    ax_top.legend(loc=LEGEND_LOC, frameon=True)
    if TL_YLIM_DB is not None:
        ax_top.set_ylim(*TL_YLIM_DB)

    # 下图：残差 = 预测 - 真实，以零线为基准填充。
    ax_bottom.axhline(0.0, color=ZERO_LINE_COLOR, lw=0.9, zorder=1)
    ax_bottom.fill_between(x, 0.0, e, where=valid, color=ERROR_FILL_COLOR,
                           alpha=ERROR_FILL_ALPHA, interpolate=False, zorder=2)
    ax_bottom.plot(x, e, color=ERROR_COLOR, lw=ERROR_LINEWIDTH, zorder=3)
    ax_bottom.set_ylabel('Error (dB)')
    if ERROR_YLIM_DB is not None:
        ax_bottom.set_ylim(*ERROR_YLIM_DB)

    for ax in (ax_top, ax_bottom):
        ax.set_xlim(*(RANGE_XLIM_KM if RANGE_XLIM_KM is not None else
                      (float(x.min()), float(x.max()))))
        ax.grid(PLOT_GRID, color='0.75', linestyle='--', alpha=GRID_ALPHA)
        ax.set_axisbelow(True)

    # 只在下层图保留Range (km)，上图保留x刻度但不显示横坐标标题。
    ax_bottom.set_xlabel('Range (km)')
    ax_top.set_xlabel('')
    ax_top.tick_params(axis='x', labelbottom=True)

    if not valid.any():
        ax_top.text(0.5, 0.5, 'No valid water-column points at this depth',
                    ha='center', va='center', transform=ax_top.transAxes,
                    fontsize=11, color='0.4')

    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=DPI, facecolor='white', bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def save_curve_csv(curve, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['range_km', 'true_tl_db', 'predicted_tl_db',
                         'pred_minus_true_db', 'valid_water'])
        for i in range(curve['range_km'].size):
            writer.writerow([
                float(curve['range_km'][i]),
                float(curve['true_db'][i]),
                float(curve['pred_db'][i]),
                float(curve['error_db'][i]),
                int(curve['valid'][i]),
            ])


def main():
    sample_id = int(SAMPLE_ID)
    azimuth_deg = int(AZIMUTH_DEG)
    if azimuth_deg != AZIMUTH_DEG or azimuth_deg not in ANGLES_DEG.tolist():
        raise ValueError(f'AZIMUTH_DEG 只能从 {ANGLES_DEG.tolist()} 中选择')
    if not DEPTHS_M:
        raise ValueError('DEPTHS_M 不能为空，请提供至少一个深度，例如 [100, 500]')
    if len(set(map(float, DEPTHS_M))) != len(DEPTHS_M):
        raise ValueError('DEPTHS_M 存在重复深度，请去重后再运行')
    if FILE_FORMAT.lower() not in ('png', 'jpg', 'jpeg', 'pdf', 'svg', 'tif', 'tiff'):
        raise ValueError('FILE_FORMAT 只能是 png / jpg / jpeg / pdf / svg / tif / tiff')
    if len(RECEIVER_DEPTH_M) != 3 or len(RANGE_COORD_M) != 3:
        raise ValueError('RECEIVER_DEPTH_M / RANGE_COORD_M 必须是(起点,终点,点数)')
    if len(DEPTH_GRID_M) != 200 or len(RANGE_KM) != 250:
        raise ValueError('物理网格必须是200接收深度层和250传播距离点')

    data_dir = Path(DATA_DIR) if DATA_DIR is not None else DEFAULT_DATA_DIR
    checkpoint = Path(CHECKPOINT) if CHECKPOINT is not None else DEFAULT_CHECKPOINT
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR is not None else DEFAULT_OUTPUT_DIR / 'single_sample_tl_range_curves'

    print(f'加载 MAP-Net：sample={sample_id}, azimuth={azimuth_deg}°, depths={DEPTHS_M}')
    data = load_and_predict(sample_id, data_dir, checkpoint)
    angle_index = azimuth_deg // 40
    result_dir = output_dir / f'sample_{sample_id:05d}' / f'azimuth_{azimuth_deg:03d}'

    for depth in DEPTHS_M:
        curve = extract_depth_curve(data, angle_index, float(depth))
        depth_str = f'{float(depth):g}'.replace('.', 'p')
        prefix = f'sample_{sample_id:05d}_azimuth_{azimuth_deg:03d}_depth_{depth_str}m'
        figure_path = result_dir / f'{prefix}.{FILE_FORMAT.lower()}'
        save_curve_figure(curve, figure_path)
        if SAVE_CSV:
            save_curve_csv(curve, result_dir / f'{prefix}.csv')
        rmse = f"{curve['rmse_db']:.4f}" if curve['rmse_db'] is not None else 'N/A'
        mae = f"{curve['mae_db']:.4f}" if curve['mae_db'] is not None else 'N/A'
        print(f'  Requested depth: {depth:g} m | plotted depth: '
              f'{curve["actual_depth_m"]:.2f} m | valid points: '
              f'{curve["valid"].sum()} | RMSE: {rmse} dB | MAE: {mae} dB')
        print(f'    Saved: {figure_path}')

    print(f'绘图完成。输出目录：{result_dir.resolve()}')


if __name__ == '__main__':
    main()
