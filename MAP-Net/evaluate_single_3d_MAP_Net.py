# -*- coding: utf-8 -*-
"""MAP-Net 单样本预测评价 + 五张论文风格图片（九方位三维径向剖面）。

放在 MAP-Net 项目根目录（与 train.py 和 map_net/ 同级），修改【Configuration】后在 Spyder 按 F5。
也可：python evaluate_single_3d_MAP_Net.py --sample-id 17424

五张图：
  01_SSP.jpg                   声源声速剖面
  02_Bathymetry.jpg            局部海底地形三维曲面（自定义色标/cmap）
  03_True_3D_TL.jpg            九方位真实TL径向垂直剖面与海底轮廓
  04_Predicted_3D_TL.jpg       九方位MAP-Net预测TL（相同色标）
  05_Absolute_Error_3D_TL.jpg 九方位绝对误差（非负、不同色标）
另存 metrics.json 和 single_sample_data.npz。

误差口径完全沿用 map_net.metrics：bottom_depth_index_r2.npy + TL有效性掩码。
单样本RMSE/MAE取9*200*250张量中全部有效水体点的总体误差，不是9个方位误差的算术均值。

九方位图严格基于原始9张距离-深度二维TL剖面，而非凭空插值为全三维体场。
采用自定义正交投影（径向水平坐标+垂直深度）绘制，视觉风格匹配参考图：
九个彩色竖直'幕布'、各方向真实地形灰色填充、方位标签、中心红色声源、左侧深度尺、右侧色标。
图中径向/垂直缩放是可视化比例，不代表水平与垂直的等比例物理距离。

注意：与 MAP-Net 模型自带的固定坐标保持一致：接收深度 0~4000 m (200)，距离 0~50000 m (250)。
如果你实际TL仿真使用了不同的接收坐标，必须先核对模型本身及评价坐标，不能只换轴标签。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import pickle
from pathlib import Path, PurePosixPath, PureWindowsPath

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; 注册3D投影（兼容老版Matplotlib）
import numpy as np
import torch

from map_net.data import N_SAMPLES, SPLIT_SLICES, TLDataset, check_files
from map_net.metrics import MaskedMetrics, water_valid_mask
from map_net.model import MODEL_ARCHITECTURE, MODEL_FULL_NAME, build_model
from map_net.paths import DEFAULT_CHECKPOINT, DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR


# Configuration
SAMPLE_ID = 18754                                 # 0-based，允许 0~18875
DATA_DIR = DEFAULT_DATA_DIR                       # 6个_r2.npy文件所在目录
CHECKPOINT = DEFAULT_CHECKPOINT                   # 训练得到的best.pt
OUTPUT_DIR = DEFAULT_OUTPUT_DIR / 'single_sample_3d_evaluation'

DEVICE = 'auto'                                   # 'auto'/'cuda'/'cpu'
AMP = 'auto'                                      # 'auto'/'bf16'/'fp16'/'off' (GPU)
SHOW_FIGURES = False                             # True: Spyder绘图窗口显示
DPI = 260                                        # 图片分辨率
SAVE_PDF = False                                # True: 同时保存可缩放PDF

# =================== ① 五张图片的尺寸（单位：英寸） ===================
# (宽, 高)，可以分别修改；DPI控制输出清晰度而非尺寸
FIGSIZE_SSP = (5/1.2, 7.6/1.2)                        # 图1：SSP
FIGSIZE_BATHY = (10.5, 8.0)                     # 图2：3D海底地形
FIGSIZE_TRUE_TL = (14.5, 10.5)                  # 图3：真实九方位TL
FIGSIZE_PRED_TL = (14.5, 10.5)                  # 图4：预测九方位TL
FIGSIZE_ERROR_TL = (14.5, 10.5)                 # 图5：九方位绝对误差

# Plotting grid
ANGLES_DEG = np.arange(9, dtype=np.int64) * 40
RECEIVER_DEPTH_M = np.linspace(0.0, 4000.0, 200)
RANGE_M = np.linspace(0.0, 50000.0, 250)

# =================== ② 色条（colorbar）的颜色范围与cmap ===================
# None = 自动估计；填入数字 = 固定色标范围，超出范围的数值只做颜色截断，不修改数据
# 图3真实TL、图4预测TL必须共用同一色标，便于公平比较
TL_VMIN_DB = 60                               # 例如 40.0
TL_VMAX_DB = 95                               # 例如 140.0
TL_CMAP = 'jet_r'                                 # 'jet' / 'turbo' / 'viridis' 等

# 图5绝对误差（单位dB）；原始误差数据不因色标截断而改变
ERROR_VMIN_DB = 0.0                             # 例如 0.0
ERROR_VMAX_DB = 6                           # 例如 20.0；None取99%分位
ERROR_CMAP = 'viridis'                            # 如 'turbo' / 'hot' / 'inferno'

# 图2：三维海底地形（深度向下为正，单位m），可独立选择色标与cmap
BATHY_VMIN_M = None                            # 例如 0.0 或 1000.0
BATHY_VMAX_M = None                            # 例如 5000.0
BATHY_CMAP = 'terrain'                          # 如 'terrain_r' / 'viridis' / 'turbo'
BATHY_VIEW_ELEV = 28                           # 三维视角：仰角（°）
BATHY_VIEW_AZIM = -60                          # 三维视角：方位（°）
BATHY_Z_ASPECT = 0.48                          # 深度轴在图中的相对显示比例

# 九方位竖直剖面的正交投影缩放（只影响绘图形状，不更改原始物理数据）
PROJECT_X_PER_KM = 7.8 * 1.2                     # 东西方向每千米的图上宽度（放大20%）
PROJECT_Y_PER_KM = 6.2 * 1.2                     # 南北方向每千米的图上高度（放大20%）
PROJECT_Z_PER_KM = 78.0                          # 深度方向每千米的图上高度
PROJECT_DEPTH_X_SHEAR = 5.0                      # 避免正北方向幕布退化为零宽度

# 九方位TL图的【显示旋转角度】（单位：度，顺时针为正）
# 只旋转绘图上的径向幕布位置，不修改原始TL[9,200,250]方位顺序、地形采样和RMSE/MAE。
# 设置80°时：真实0°绘制在原80°的位置；真实80°绘制在原160°的位置。
RADIAL_DISPLAY_ROTATION_DEG = 0

# 九方位图的自动布局参数：投影比例改变后，标注、深度尺和绘图范围同步调整。
# 下面这些只影响绘图排版，不改变TL结果、海底水深或误差统计。
RADIAL_AX_RECT = (0.10, 0.09, 0.76, 0.80)       # 九方位主图位置 (left,bottom,width,height)
RADIAL_COLORBAR_RECT = (0.88, 0.24, 0.033, 0.56)
RADIAL_LABEL_EXTRA_KM = 4.0                       # 常规方位标签比幕布外缘多延伸的距离
RADIAL_DEPTH_AXIS_GAP_RATIO = 0.125               # 深度标尺到幕布最左边的相对间隔
RADIAL_X_MARGIN_RATIO = 0.042                     # 自适应绘图范围的水平留白
RADIAL_Y_MARGIN_RATIO = 0.032                     # 自适应绘图范围的垂直留白
RADIAL_LABEL_FONTSIZE = 17
# ================================================================================


class CrossPlatformUnpickler(pickle.Unpickler):
    """兼容跨操作系统保存的可信checkpoint中的Path对象。"""
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
        raise ValueError(f'样本编号必须在0~{N_SAMPLES-1}，实际: {sample_id}')
    for name, (start, stop) in SPLIT_SLICES.items():
        if start <= sample_id < stop:
            return name, sample_id - start
    raise RuntimeError('SPLIT_SLICES有误')


def device_and_amp(device_arg: str, amp_arg: str):
    if device_arg not in ('auto', 'cuda', 'cpu'):
        raise ValueError(f'非法DEVICE: {device_arg}')
    if amp_arg not in ('auto', 'bf16', 'fp16', 'off'):
        raise ValueError(f'非法AMP: {amp_arg}')
    if device_arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定CUDA，但当前环境没有可用GPU')
    cuda = torch.cuda.is_available() if device_arg == 'auto' else device_arg == 'cuda'
    device = torch.device('cuda' if cuda else 'cpu')
    amp = amp_arg
    if amp == 'auto':
        amp = 'bf16' if cuda and torch.cuda.is_bf16_supported() else ('fp16' if cuda else 'off')
    if cuda and amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU不支持BF16：改为AMP=auto、fp16或off')
    return device, amp


def evaluate_sample(sample_id: int, data_dir: Path, checkpoint: Path,
                    device_arg: str = DEVICE, amp_arg: str = AMP) -> dict:
    """从原始数据读取单样本，加载checkpoint并推理；严格沿用原评估水体掩码。"""
    split, local_id = sample_split(sample_id)
    check_files(data_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'没有找到MAP-Net权重: {checkpoint}')
    device, amp = device_and_amp(device_arg, amp_arg)

    # weights_only=False仅用于本人训练且可信的best.pt，不要加载未知网络权重。
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False,
                      pickle_module=CrossPlatformPickle)
    if 'model' not in ckpt or 'stats' not in ckpt:
        raise KeyError("best.pt 必须包含 'model' 和 'stats'")
    model = build_model(data_dir, ckpt['stats']).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    dataset = TLDataset(data_dir, split,
                        tl_invalid_raw=ckpt['stats'].get('tl_invalid_raw'), check=False)
    ssp_t, bathy_t, truth_t, index_t = dataset[local_id]
    if index_t.item() != sample_id:
        raise RuntimeError('数据与全局样本编号不一致')

    with torch.inference_mode():
        autocast_ctx = (torch.autocast('cuda', dtype=torch.bfloat16 if amp == 'bf16'
                                      else torch.float16)
                        if device.type == 'cuda' and amp != 'off' else nullcontext())
        with autocast_ctx:
            # predict_db 已按checkpoint的tl_mean/tl_std反标准化，输出dB
            pred_t = model.predict_db(ssp_t.unsqueeze(0).to(device),
                                      bathy_t.unsqueeze(0).to(device)).float().cpu()

    if pred_t.shape != (1, 9, 200, 250):
        raise ValueError(f'模型预测形状有误: {tuple(pred_t.shape)}')

    bottom_index_all = np.load(data_dir / 'bottom_depth_index_r2.npy', allow_pickle=False)
    bottom_index_t = torch.from_numpy(np.array(bottom_index_all, copy=True))
    truth_b = truth_t.unsqueeze(0)
    mask_b = water_valid_mask(truth_b, index_t.reshape(1), bottom_index_t)
    if not bool(mask_b.any()):
        raise ValueError(f'样本{sample_id}没有有效水体TL网格点')
    if not bool(torch.isfinite(pred_t[mask_b]).all()):
        raise FloatingPointError('模型在有效水体区域的预测含NaN/Inf')

    # 复用项目MaskedMetrics；真实dB尺度计算，按有效点数汇总
    metric_accumulator = MaskedMetrics(torch.device('cpu'))
    metric_accumulator.add(pred_t, truth_b, mask_b)
    metrics = metric_accumulator.result()

    valid = mask_b[0].numpy()
    true_db = np.where(valid, truth_t.numpy(), np.nan).astype(np.float32)
    pred_db = np.where(valid, pred_t[0].numpy(), np.nan).astype(np.float32)
    ssp_depth = np.load(data_dir / 'ssp_depth_m_r2.npy', allow_pickle=False)
    valid_count = int(np.load(data_dir / 'ssp_valid_count_r2.npy', mmap_mode='r',
                                  allow_pickle=False)[sample_id])
    return dict(
        sample_id=sample_id, split=split, truth_db=true_db, pred_db=pred_db,
        valid_mask=valid, ssp_mps=ssp_t.numpy(), ssp_depth_m=ssp_depth,
        ssp_valid_count=valid_count, bathymetry_m=bathy_t.numpy(),
        bottom_index=bottom_index_all[sample_id % 121].copy(),
        metrics=metrics, checkpoint=str(checkpoint),
        checkpoint_epoch=int(ckpt.get('epoch', -1)) + 1,
        device=str(device), amp=amp,
    )


def _resolve_color_limits(array, user_vmin, user_vmax, *, label, default_percentiles=(0, 100)):
    """只设置颜色映射范围，不修改数组值。自动范围可由分位数估算。"""
    finite = np.asarray(array, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f'{label}没有有效数值，无法设置colorbar')
    vmin = (float(np.percentile(finite, default_percentiles[0]))
            if user_vmin is None else float(user_vmin))
    vmax = (float(np.percentile(finite, default_percentiles[1]))
            if user_vmax is None else float(user_vmax))
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError(f'{label}色标上下限必须是有限数值')
    # 全常数场 / 完美预测造成的全零误差，在自动色标下也应能绘图。
    if vmin == vmax and user_vmin is None and user_vmax is None:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    elif vmin == vmax and user_vmax is None:
        vmax = vmin + max(0.01, abs(vmin) * 0.01)
    if vmin >= vmax:
        raise ValueError(f'{label}色标范围无效: vmin={vmin} >= vmax={vmax}')
    return vmin, vmax


def choose_color_limits(data: dict) -> dict:
    """分别确定TL、绝对误差、三维地形的色标范围。"""
    valid = data['valid_mask']
    tru, pred = data['truth_db'], data['pred_db']
    both = np.concatenate((tru[valid], pred[valid]))
    tl_vmin, tl_vmax = _resolve_color_limits(
        both, TL_VMIN_DB, TL_VMAX_DB, label='TL', default_percentiles=(1, 99))
    err = np.abs(pred[valid].astype(np.float64) - tru[valid].astype(np.float64))
    err_vmin, err_vmax = _resolve_color_limits(
        err, ERROR_VMIN_DB, ERROR_VMAX_DB, label='绝对误差', default_percentiles=(0, 99))
    bathy_vmin, bathy_vmax = _resolve_color_limits(
        data['bathymetry_m'], BATHY_VMIN_M, BATHY_VMAX_M, label='Bathymetry')
    return {
        'tl': (tl_vmin, tl_vmax),
        'error': (err_vmin, err_vmax),
        'bathymetry': (bathy_vmin, bathy_vmax),
    }


def _validate_figure_size(figsize, label):
    if (len(figsize) != 2 or not np.isfinite(figsize).all()
            or any(float(size) <= 0 for size in figsize)):
        raise ValueError(f'{label} figsize必须是两个正的有限数字，例如(10, 8)')
    return tuple(float(size) for size in figsize)


def save_figure(fig, filename: Path, show: bool, dpi: int, save_pdf: bool):
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=dpi, facecolor='white', bbox_inches='tight')
    if save_pdf:
        fig.savefig(filename.with_suffix('.pdf'), facecolor='white', bbox_inches='tight')
    print(f'  绘图完成: {filename.name}', flush=True)
    if show:
        plt.show()
    plt.close(fig)


def plot_ssp(data: dict, filepath: Path, dpi: int, show: bool, pdf: bool,
             figsize=FIGSIZE_SSP):
    ssp, dep = data['ssp_mps'], data['ssp_depth_m']
    cnt = data['ssp_valid_count']
    valid = (np.arange(len(ssp)) < cnt) & np.isfinite(ssp) & np.isfinite(dep)
    fig, ax = plt.subplots(figsize=_validate_figure_size(figsize, 'SSP'),
                           layout='constrained')
    ax.plot(ssp[valid], dep[valid] / 1000., color='#17669c', lw=2.2)
    ax.scatter(ssp[valid], dep[valid] / 1000., s=14, color='#17669c', zorder=3)
    ax.set(xlabel='Sound speed (m/s)', ylabel='Depth (km)',
           title=f'Source SSP | Sample {data["sample_id"]}\nValid levels: {valid.sum()}/50')
    ax.invert_yaxis()
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
    save_figure(fig, filepath, show, dpi, pdf)


def plot_bathymetry(data: dict, filepath: Path, dpi: int, show: bool, pdf: bool,
                    *, figsize=FIGSIZE_BATHY, cmap_name=BATHY_CMAP,
                    vmin=None, vmax=None):
    """图2：将61×61真实海底水深绘制成三维曲面，颜色表示水深（m）。

    x从西到东，y从南到北；z轴数值向下为正，在显示上反转z轴，深海在下。
    声源只标记水平位置，红星落在(0,0)对应的海底表面，不表示真实声源深度。
    """
    bathy = np.asarray(data['bathymetry_m'], dtype=np.float64)
    if bathy.shape != (61, 61) or not np.isfinite(bathy).all():
        raise ValueError('地形图需要无NaN/Inf的(61,61)海底水深数据')
    fig = plt.figure(figsize=_validate_figure_size(figsize, 'Bathymetry'),
                     layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False  # 手动将声源标记放到地形表面上层
    xy_km = np.linspace(-60., 60., 61)
    xx, yy = np.meshgrid(xy_km, xy_km, indexing='xy')
    zz_km = bathy / 1000.

    if vmin is None or vmax is None:
        vmin, vmax = _resolve_color_limits(
            bathy, BATHY_VMIN_M if vmin is None else vmin,
            BATHY_VMAX_M if vmax is None else vmax,
            label='Bathymetry')
    norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    cmap = plt.get_cmap(cmap_name)
    ax.plot_surface(xx, yy, zz_km, facecolors=cmap(norm(bathy)),
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=False, shade=False, zorder=1)

    # 红色五角星仅代表声源的水平位置，不暗示声源位于海底。
    ax.scatter([0.], [0.], [zz_km[30, 30]], c='red', marker='*',
               s=180, depthshade=False, edgecolors='black', linewidths=.7,
               zorder=10, label='Source horizontal position')
    ax.set(xlabel='East / West distance (km)',
           ylabel='South / North distance (km)',
           zlabel='Seafloor depth (km)',
           title=f'3D Local Bathymetry | Sample {data["sample_id"]}\n'
                 '61×61 grid, 2 km spacing')
    ax.set_xlim(-60., 60.)
    ax.set_ylim(-60., 60.)
    zlo, zhi = float(np.nanmin(zz_km)), float(np.nanmax(zz_km))
    if np.isclose(zlo, zhi):
        zlo -= .25
        zhi += .25
    else:
        margin = (zhi-zlo) * .06
        zlo -= margin
        zhi += margin
    ax.set_zlim(zlo, zhi)
    ax.invert_zaxis()  # 海面较浅处在上，深海在下
    ax.set_box_aspect((1, 1, BATHY_Z_ASPECT))
    ax.view_init(elev=BATHY_VIEW_ELEV, azim=BATHY_VIEW_AZIM)
    ax.legend(loc='upper left', fontsize=9)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=.66, pad=.095, aspect=22)
    cbar.set_label('Seafloor depth (m)')
    save_figure(fig, filepath, show, dpi, pdf)


def sample_bathymetry_along_rays(bathymetry_m: np.ndarray,
                                  range_km: np.ndarray,
                                  angles_deg: np.ndarray = ANGLES_DEG) -> np.ndarray:
    """61×61经纬规则网格按真实地理方向双线性插值，返回[9,250]真实海底水深(m)。

    row: 南→北，col: 西→东，声源中心为(row=30, col=30)，网格间距2km。
    x[km] = r*sin(theta), y[km] = r*cos(theta), 0°正北顺时针增加。
    """
    bathy = np.asarray(bathymetry_m, dtype=np.float64)
    if bathy.shape != (61, 61):
        raise ValueError(f'地形维度应为(61,61)，实际{bathy.shape}')
    if not np.isfinite(bathy).all():
        raise ValueError('局部海底地形含有非有限数值')
    theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))[:, None]
    r = np.asarray(range_km, dtype=np.float64)[None, :]
    col_f = (r * np.sin(theta) + 60.) / 2.
    row_f = (r * np.cos(theta) + 60.) / 2.
    if np.any(col_f < -1.e-8) or np.any(col_f > 60. + 1.e-8) or np.any(row_f < -1.e-8) or np.any(row_f > 60. + 1.e-8):
        raise ValueError('传播路径超出了地形的±60km范围')
    col_f = np.clip(col_f, 0, 60)
    row_f = np.clip(row_f, 0, 60)
    c0 = np.floor(col_f).astype(int)
    r0 = np.floor(row_f).astype(int)
    c1, r1 = np.minimum(c0 + 1, 60), np.minimum(r0 + 1, 60)
    tc, tr = col_f - c0, row_f - r0
    depth = ((1-tr)*((1-tc)*bathy[r0,c0] + tc*bathy[r0,c1])
             + tr*((1-tc)*bathy[r1,c0] + tc*bathy[r1,c1]))
    # 海面以上的地形在水声TL图中按0m深度处理
    return np.maximum(depth, 0.0).astype(np.float32)


def radial_display_angle(physical_angle_deg):
    """物理方位角 -> 图上的方向（北0°，顺时针为正）。

    此函数只用于九方位图的几何位置。图上文字仍标注原始物理角度；
    TL数组索引、传播路径对应的海底深度始终不变。
    """
    return (float(physical_angle_deg) + float(RADIAL_DISPLAY_ROTATION_DEG)) % 360.0


def project_coordinates(r_km, depth_km, theta_deg):
    """将3D径向垂直剖面的空间位置投影到示意图的二维绘图平面。"""
    angle = np.deg2rad(float(theta_deg))
    x = (PROJECT_X_PER_KM * np.sin(angle) * np.asarray(r_km)
         + PROJECT_DEPTH_X_SHEAR * np.asarray(depth_km))
    y = (PROJECT_Y_PER_KM * np.cos(angle) * np.asarray(r_km)
         - PROJECT_Z_PER_KM * np.asarray(depth_km))
    return x, y


def ray_affine(theta_deg):
    """仿射变换：模型原坐标(range km, depth km)→图上正交投影(x, y)。"""
    angle = np.deg2rad(float(theta_deg))
    # x=a*r+c*depth ; y=b*r+d*depth
    return Affine2D.from_values(PROJECT_X_PER_KM * np.sin(angle),
                                PROJECT_Y_PER_KM * np.cos(angle),
                                PROJECT_DEPTH_X_SHEAR,
                                -PROJECT_Z_PER_KM,
                                0.0, 0.0)


def _radial_annotation_layout(r_max_km: float, z_max_km: float):
    """根据九方位幕布的真实投影范围自动布置角度标签、深度标尺和视图边界。

    旧代码把深度尺X=-505、Y=240、xlim=(-600,580)、ylim=(-685,390)
    设为固定值；在水平投影放大时会裁掉图像。本函数避免这些硬编码。
    全部坐标只依赖于投影、距离/深度网格，不依赖于TL数值：真值、预测、
    误差三幅图使用完全相同的布局。
    """
    corners = []
    for physical_angle in ANGLES_DEG:
        visual_angle = radial_display_angle(physical_angle)
        for range_km in (0.0, float(r_max_km)):
            for depth_km in (0.0, float(z_max_km)):
                x, y = project_coordinates(range_km, depth_km, visual_angle)
                corners.append((float(x), float(y)))
    corners = np.array(corners, dtype=float)
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    span_x, span_y = x_max - x_min, y_max - y_min
    if span_x <= 0 or span_y <= 0:
        raise ValueError('投影比例不合理：九方位图无法形成有效的绘图范围')

    label_xy = {}
    for angle in ANGLES_DEG:
        physical_angle = int(angle)
        visual_angle = radial_display_angle(physical_angle)
        # 标签位置按【旋转后的视觉方向】判断，但标签文字仍为物理方位角。
        # 这样在80°旋转后，原本160°/200°位置的特殊标注规则也随之旋转。
        if np.isclose(visual_angle, 160) or np.isclose(visual_angle, 200):
            # 将朝南的两条标签放在幕布底部之外，随投影宽高同步偏移。
            lx, ly = project_coordinates(r_max_km + 2.0, z_max_km + 0.35, visual_angle)
            lx += (+1 if np.isclose(visual_angle, 160) else -1) * 0.023 * span_x
            ly -= 0.013 * span_y
        elif np.isclose(visual_angle, 120) or np.isclose(visual_angle, 240):
            # 视觉120°/240°标签放在幕布侧下方，避免压在彩色图面内部。
            lx, ly = project_coordinates(r_max_km + RADIAL_LABEL_EXTRA_KM, 0.0, visual_angle)
            lx += (+1 if np.isclose(visual_angle, 120) else -1) * 0.038 * span_x
            ly -= 0.95 * PROJECT_Z_PER_KM * z_max_km
        else:
            lx, ly = project_coordinates(r_max_km + RADIAL_LABEL_EXTRA_KM, 0.0, visual_angle)
        label_xy[physical_angle] = (float(lx), float(ly))

    # 深度标尺始终布置在所有幕布的左侧，图形横向拉伸后也保持间距。
    depth_axis_x = x_min - RADIAL_DEPTH_AXIS_GAP_RATIO * span_x
    depth_axis_top = 0.65 * y_max  # 深度尺顶部随北向投影范围动态抬升
    # 深度标尺与幕布的数值深度同缩放，避免视觉上深度刻度失真。
    depth_ticks_km = np.arange(0., z_max_km + 1.e-9, 1.)
    if len(depth_ticks_km) == 0 or abs(depth_ticks_km[-1] - z_max_km) > 1.e-7:
        depth_ticks_km = np.r_[depth_ticks_km, z_max_km]
    depth_y = depth_axis_top - PROJECT_Z_PER_KM * depth_ticks_km
    tick_length = 0.010 * span_x
    tick_text_x = depth_axis_x - 0.017 * span_x
    depth_caption_x = depth_axis_x - 0.054 * span_x

    all_x = np.r_[corners[:, 0], [p[0] for p in label_xy.values()],
                  [depth_axis_x, tick_text_x, depth_caption_x]]
    all_y = np.r_[corners[:, 1], [p[1] for p in label_xy.values()], depth_y]
    # 留出文字半宽、刻度文字和标记空间；内容只要缩放仍位于画布内部。
    x_pad = max(RADIAL_X_MARGIN_RATIO * span_x, 20.0)
    y_pad = max(RADIAL_Y_MARGIN_RATIO * span_y, 20.0)
    xlim = (float(all_x.min() - x_pad), float(all_x.max() + x_pad))
    ylim = (float(all_y.min() - y_pad), float(all_y.max() + y_pad))
    return {
        'angle_labels': label_xy,
        'depth_axis_x': depth_axis_x,
        'depth_y': depth_y,
        'depth_ticks_km': depth_ticks_km,
        'tick_length': tick_length,
        'tick_text_x': tick_text_x,
        'depth_caption_x': depth_caption_x,
        'xlim': xlim,
        'ylim': ylim,
    }


def plot_radial_curtains(field: np.ndarray, bathy_rays_m: np.ndarray, *,
                         sample_id: int, figure_title: str, cbar_label: str,
                         cmap_name: str, vmin: float, vmax: float,
                         filepath: Path, dpi: int, show: bool, pdf: bool,
                         figsize=FIGSIZE_TRUE_TL):
    """绘制9方向彩色垂直幕布 + 灰色海底。

    为减少3D工具默认透视遮挡带来的问题，使用精确的3D→2D正交坐标变换。
    每幅幕布仍对应原始真实[200,250]值，无角度插值和数值平滑。
    """
    if field.shape != (9, 200, 250) or bathy_rays_m.shape != (9, 250):
        raise ValueError('期望field[9,200,250]、bathy_rays[9,250]')
    zmax_km = RECEIVER_DEPTH_M[-1] / 1000.
    r_km = RANGE_M / 1000.
    cm = plt.get_cmap(cmap_name).copy()
    cm.set_bad((1,1,1,0))                         # 掩码为透明，仅灰色地形代表海底
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    fig = plt.figure(figsize=_validate_figure_size(figsize, 'Radial TL'),
                     facecolor='white')
    ax = fig.add_axes(RADIAL_AX_RECT)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()

    # 先按旋转后的【视觉方向】计算遮挡顺序；每个物理方位始终对应原来的TL/海底数组。
    order = sorted(range(9), key=lambda idx: (
        -np.cos(np.deg2rad(radial_display_angle(ANGLES_DEG[idx]))), idx))
    for draw_rank, k in enumerate(order):
        ang = int(ANGLES_DEG[k])  # 真实物理方位：决定使用field[k]和bathy_rays_m[k]
        visual_angle = radial_display_angle(ang)
        trans = ray_affine(visual_angle) + ax.transData
        # imshow()输入的纵坐标使用以km为单位的向下正深度；extent匹配行0=0km。
        arr = np.ma.masked_invalid(field[k])
        ax.imshow(arr, origin='upper',
                  extent=(r_km[0], r_km[-1], zmax_km, 0.),
                  cmap=cm, norm=norm, interpolation='nearest',
                  transform=trans, aspect='auto', zorder=10 + 2 * draw_rank)

        # 真实61×61地形沿该方向双线性采样；深于绘图底界时裁剪于底界。
        seabed_km = np.clip(bathy_rays_m[k] / 1000., 0., zmax_km)
        # 在(range, depth)局部坐标建立海底以下封闭多边形，再投影到图上。
        curtain_polygon = np.concatenate((
            np.column_stack((r_km, seabed_km)),
            np.array([[r_km[-1], zmax_km], [r_km[0], zmax_km]]),
        ))
        ax.add_patch(Polygon(curtain_polygon, closed=True, facecolor='#d5d5d5',
                             edgecolor='black', linewidth=.85, transform=trans,
                             zorder=11 + 2 * draw_rank))
        # 清晰描出海底起伏轮廓，作为两种TL场共同的真实地形叠加。
        xx, yy = project_coordinates(r_km, seabed_km, visual_angle)
        seabed_visible = bathy_rays_m[k] < RECEIVER_DEPTH_M[-1]
        ax.plot(xx, np.where(seabed_visible, yy, np.nan), color='black', lw=.85,
                zorder=12 + 2 * draw_rank, solid_joinstyle='round')
        # 每个方向的剖面水平海面外缘（较细，与幕布边缘一致）
        top_x, top_y = project_coordinates(r_km, np.zeros_like(r_km), visual_angle)
        ax.plot(top_x, top_y, color='black', lw=.65, alpha=.9,
                zorder=12 + 2 * draw_rank)

    # 所有装饰元素使用同一套动态布局，随PROJECT_X/Y_PER_KM自动调整。
    layout = _radial_annotation_layout(float(r_km[-1]), float(zmax_km))
    ax.plot(0., 0., color='red', marker='*', ms=29, mec='#d62828', mew=.5,
            linestyle='none', zorder=100)

    for ang in ANGLES_DEG:
        lx, ly = layout['angle_labels'][int(ang)]
        ax.text(lx, ly, f'{int(ang)}°', fontsize=RADIAL_LABEL_FONTSIZE,
                fontweight='bold', ha='center', va='center', zorder=110)

    depth_axis_x = layout['depth_axis_x']
    depth_y = layout['depth_y']
    ax.plot([depth_axis_x, depth_axis_x], [depth_y[-1], depth_y[0]],
            color='black', lw=1.7, zorder=100)
    for depth_km, yy in zip(layout['depth_ticks_km'], depth_y):
        ax.plot([depth_axis_x - layout['tick_length'], depth_axis_x], [yy, yy],
                color='black', lw=1.5, zorder=100)
        ax.text(layout['tick_text_x'], yy, f'{depth_km:g}',
                ha='right', va='center', fontsize=14)
    ax.text(layout['depth_caption_x'], float(depth_y.mean()), 'Depth (km)',
            ha='center', va='center', rotation=90, fontsize=19,
            fontweight='bold')

    # 根据图形元素计算坐标轴范围。
    # 同样的投影与网格，三张TL图一定得到同样的坐标范围。
    ax.set_xlim(*layout['xlim'])
    ax.set_ylim(*layout['ylim'])
    fig.text(.49, .945, figure_title, ha='center', va='top', fontsize=20, fontweight='bold')

    # 注意：图3、4固定共用同一Normalize范围。图5误差色条从0开始。
    color_ax = fig.add_axes(RADIAL_COLORBAR_RECT)
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=color_ax)
    cbar.ax.tick_params(labelsize=12)
    color_ax.set_title(cbar_label, fontsize=15, fontweight='bold', pad=15)
    save_figure(fig, filepath, show, dpi, pdf)


def export_results(data: dict, result_dir: Path, color_limits):
    result_dir.mkdir(parents=True, exist_ok=True)
    sample_id = data['sample_id']
    err = (data['pred_db'] - data['truth_db']).astype(np.float32)
    np.savez_compressed(result_dir / 'single_sample_data.npz',
                        sample_id=np.int64(sample_id),
                        angles_deg=ANGLES_DEG, visual_angles_deg=np.asarray([radial_display_angle(a) for a in ANGLES_DEG]),
                        depth_m=RECEIVER_DEPTH_M, range_m=RANGE_M,
                        ssp_mps=data['ssp_mps'], ssp_depth_m=data['ssp_depth_m'],
                        bathymetry_m=data['bathymetry_m'],
                        truth_db=data['truth_db'], pred_db=data['pred_db'],
                        signed_error_db=err, absolute_error_db=np.abs(err),
                        valid_mask=data['valid_mask'],
                        bottom_depth_index=data['bottom_index'])
    report = {
        'model': MODEL_ARCHITECTURE, 'model_full_name': MODEL_FULL_NAME,
        'sample_id': int(sample_id), 'split': data['split'],
        'year_index_0based': int(sample_id // 1452),
        'month_index_0based': int((sample_id // 121) % 12),
        'latitude_index_0based': int((sample_id // 11) % 11),
        'longitude_index_0based': int(sample_id % 11),
        'source_idx': int(sample_id % 121),
        'ssp_valid_count': data['ssp_valid_count'],
        'checkpoint': data['checkpoint'], 'checkpoint_epoch': data['checkpoint_epoch'],
        'device': data['device'], 'amp': data['amp'],
        'units': 'dB', 'evaluation_scope': 'valid_water_only',
        'error_plot_type': 'absolute_prediction_minus_truth',
        'tl_display_clim_db': list(color_limits['tl']),
        'absolute_error_display_clim_db': list(color_limits['error']),
        'bathymetry_display_clim_m': list(color_limits['bathymetry']),
        'colormaps': {'tl': TL_CMAP, 'error': ERROR_CMAP, 'bathymetry': BATHY_CMAP},
        'figure_size_inch': {
            'ssp': list(FIGSIZE_SSP), 'bathymetry': list(FIGSIZE_BATHY),
            'true_tl': list(FIGSIZE_TRUE_TL), 'pred_tl': list(FIGSIZE_PRED_TL),
            'absolute_error': list(FIGSIZE_ERROR_TL),
        },
        'radial_display_rotation_clockwise_deg': float(RADIAL_DISPLAY_ROTATION_DEG),
        'grid_assumptions': {'depth_m': [0, 4000, 200], 'range_m': [0, 50000, 250]},
        **data['metrics'],
    }
    (result_dir / 'metrics.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n' + '=' * 60)
    print(f'MAP-Net 单样本预测评价 | 样本 {sample_id} | {data["split"]}')
    print(f'RMSE : {report["overall"]["rmse_db"]:.6f} dB')
    print(f'MAE  : {report["overall"]["mae_db"]:.6f} dB')
    print(f'有效水体网格点: {report["overall"]["valid_points"]:,}')
    print('各方位 (RMSE / MAE, dB):')
    for a in ANGLES_DEG:
        item = report['per_angle'][f'{a}deg']
        if item['valid_points']:
            print(f'  {a:3d}° : {item["rmse_db"]:.4f} / {item["mae_db"]:.4f}')
        else:
            print(f'  {a:3d}° : no valid water cells')
    print('=' * 60)
    return report


def run(sample_id: int = SAMPLE_ID, data_dir=DATA_DIR, checkpoint=CHECKPOINT,
        output_dir=OUTPUT_DIR, device=DEVICE, amp=AMP,
        show=SHOW_FIGURES, dpi=DPI, save_pdf=SAVE_PDF):
    sample_id = int(sample_id)
    data_dir, checkpoint, output_dir = Path(data_dir), Path(checkpoint), Path(output_dir)
    data = evaluate_sample(sample_id, data_dir, checkpoint, device, amp)
    limits = choose_color_limits(data)
    result_dir = output_dir / f'sample_{sample_id:05d}'
    report = export_results(data, result_dir, limits)
    r_km = RANGE_M / 1000.
    ray_depths = sample_bathymetry_along_rays(data['bathymetry_m'], r_km)

    plot_ssp(data, result_dir / '01_SSP.jpg', dpi, show, save_pdf,
             figsize=FIGSIZE_SSP)
    plot_bathymetry(data, result_dir / '02_Bathymetry.jpg', dpi, show, save_pdf,
                    figsize=FIGSIZE_BATHY, cmap_name=BATHY_CMAP,
                    vmin=limits['bathymetry'][0], vmax=limits['bathymetry'][1])
    plot_radial_curtains(data['truth_db'], ray_depths,
                         sample_id=sample_id,
                         figure_title=f'True Transmission Loss | Sample {sample_id}',
                         cbar_label='TL (dB)', cmap_name=TL_CMAP,
                         vmin=limits['tl'][0], vmax=limits['tl'][1],
                         filepath=result_dir / '03_True_3D_TL.jpg',
                         dpi=dpi, show=show, pdf=save_pdf,
                         figsize=FIGSIZE_TRUE_TL)
    plot_radial_curtains(data['pred_db'], ray_depths,
                         sample_id=sample_id,
                         figure_title=f'MAP-Net Predicted Transmission Loss | Sample {sample_id}',
                         cbar_label='TL (dB)', cmap_name=TL_CMAP,
                         vmin=limits['tl'][0], vmax=limits['tl'][1],
                         filepath=result_dir / '04_Predicted_3D_TL.jpg',
                         dpi=dpi, show=show, pdf=save_pdf,
                         figsize=FIGSIZE_PRED_TL)
    absolute_error = np.abs(data['pred_db'] - data['truth_db'])
    plot_radial_curtains(absolute_error, ray_depths,
                         sample_id=sample_id,
                         figure_title=f'Absolute TL Prediction Error | Sample {sample_id}',
                         cbar_label='|Error| (dB)', cmap_name=ERROR_CMAP,
                         vmin=limits['error'][0], vmax=limits['error'][1],
                         filepath=result_dir / '05_Absolute_Error_3D_TL.jpg',
                         dpi=dpi, show=show, pdf=save_pdf,
                         figsize=FIGSIZE_ERROR_TL)
    print(f'\n5张图、评估报告和数值已保存至: {result_dir.resolve()}')
    return report


def main():
    parser = argparse.ArgumentParser(description='MAP-Net单样本RMSE/MAE与九方位三维径向TL绘图')
    parser.add_argument('--sample-id', type=int, default=SAMPLE_ID,
                        help='全局0-based样本编号 0~18875')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default=DEVICE)
    parser.add_argument('--amp', choices=['auto','bf16','fp16','off'], default=AMP)
    parser.add_argument('--show', action='store_true', help='显示5张图片')
    parser.add_argument('--pdf', action='store_true', help='同时保存PDF')
    parser.add_argument('--dpi', type=int, default=DPI)
    args = parser.parse_args()
    run(sample_id=args.sample_id, data_dir=args.data_dir, checkpoint=args.checkpoint,
        output_dir=args.output_dir, device=args.device, amp=args.amp,
        show=args.show or SHOW_FIGURES, dpi=args.dpi, save_pdf=args.pdf or SAVE_PDF)


if __name__ == '__main__':
    main()
