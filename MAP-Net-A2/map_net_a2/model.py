"""MAP-Net-A2 (without DPF).

DPF is replaced by a minimal global-context and coordinate adapter. Center-ray bathymetry sampling supplies the decoder geometry; no learned directional terrain sampling or FiLM is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MODEL_CONFIG = {
    "ssp_channels": (48, 96, 192), "ssp_blocks": (2, 2, 3),
    "bathy_channels": (48, 96, 192), "bathy_blocks": (3, 3, 4),
    "path_channels": 128, "context_channels": 256, "skip_channels": 24,
    "base_channels": 256, "base_blocks": 9,
    "up_channels": (192, 128, 48), "up_blocks": (3, 2, 0),
}
MODEL_ARCHITECTURE = "MAP-Net-A2"
LEGACY_MODEL_ARCHITECTURE = "DBF-TLNet-MLA-A2-noDPF"


def checkpoint_matches(state):
    """Check the variant identity before loading state-dictionary weights."""
    name_matches = state.get("model_name") in (MODEL_ARCHITECTURE, LEGACY_MODEL_ARCHITECTURE)
    return name_matches

# A2功能边界：M1、M3、M4均使用完整网络原有子层；仅M2换成最小接口适配。
# path_adapter 中的FC不是原M2的地形采样编码器，不能访问多尺度地形图。
PAPER_MODULES = {
    "M1_EFE": ("ssp", "bathy"),
    "A2_M2_INTERFACE_ONLY": ("global_context", "path_adapter"),
    "M3_CAI": ("angular", "angular_mla"),
    "M4_MTR": ("context_proj", "path_skip", "start", "base", "up1", "up2", "up3", "tail"),
}


# ============================================================================


# ============================================================================
class ConvGNAct(nn.Sequential):
    def __init__(self, n_in, n_out, k=3, stride=1, ndim=2):
        layer = nn.Conv1d if ndim == 1 else nn.Conv2d
        super().__init__(
            layer(n_in, n_out, k, stride=stride, padding=k // 2, bias=False),
            nn.GroupNorm(8, n_out),
            nn.SiLU(inplace=True),
        )


class Residual1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = ConvGNAct(channels, channels, ndim=1)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.gn2(self.c2(self.c1(x))))


class Residual2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = ConvGNAct(channels, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.gn2(self.c2(self.c1(x))))


def residual_stack(residual_cls, channels, count):
    if count < 0:
        raise ValueError("residual block count cannot be negative")
    return nn.Sequential(*[residual_cls(channels) for _ in range(count)])


# ============================================================================
# 模块1 / EFE：双分支环境特征编码
# 边界：从原始SSP与61×61局部地形，编码为SSP全局向量、三尺度地形特征
#       和地形全局向量；此处不采样传播方位、不生成TL。
# 调用位置：MAPNetA2._module1_encode_environment()。
# ============================================================================
class SSPEncoder(nn.Module):
    """M1/SSP支路：50层SSP -> [B,192]；缺测信息由valid通道与池化掩码保留。"""
    def __init__(self, depth_m, channels, blocks):
        super().__init__()
        depth = torch.as_tensor(depth_m, dtype=torch.float32).reshape(-1)
        if depth.numel() != 50 or not torch.isfinite(depth).all():
            raise ValueError("ssp_depth_m must contain 50 finite depths")
        self.register_buffer("depth_norm", depth / depth.max().clamp_min(1.0))
        c0, c1, c2 = channels
        n0, n1, n2 = blocks
        self.stem = ConvGNAct(3, c0, k=5, ndim=1)
        self.layers = nn.Sequential(
            *[Residual1D(c0) for _ in range(n0)],
            ConvGNAct(c0, c1, ndim=1),
            *[Residual1D(c1) for _ in range(n1)],
            ConvGNAct(c1, c2, ndim=1),
            *[Residual1D(c2) for _ in range(n2)],
        )
        self.attention = nn.Conv1d(c2, 1, kernel_size=1)

    def forward(self, ssp, mean, std):
        valid = torch.isfinite(ssp)
        speed = torch.where(valid, (ssp - mean) / std, 0.0)
        z = self.depth_norm.to(ssp.dtype).view(1, 50).expand_as(ssp)
        x = torch.stack((speed, z, valid.to(ssp.dtype)), dim=1)
        x = self.layers(self.stem(x))
        weights = self.attention(x).masked_fill(~valid[:, None, :], -1e4)
        weights = torch.softmax(weights, dim=-1)
        return (x * weights).sum(dim=-1)


class BathyEncoder(nn.Module):
    """M1/地形支路：61×61水深 -> 多尺度特征(f0,f1,f2)和[B,192]全局向量。"""
    def __init__(self, channels, blocks):
        super().__init__()
        c0, c1, c2 = channels
        self.stem = ConvGNAct(1, c0)
        self.b0 = residual_stack(Residual2D, c0, blocks[0])
        self.down1 = ConvGNAct(c0, c1, stride=2)
        self.b1 = residual_stack(Residual2D, c1, blocks[1])
        self.down2 = ConvGNAct(c1, c2, stride=2)
        self.b2 = residual_stack(Residual2D, c2, blocks[2])

    def forward(self, x):
        f0 = self.b0(self.stem(x[:, None]))
        f1 = self.b1(self.down1(f0))
        f2 = self.b2(self.down2(f1))
        return (f0, f1, f2), F.adaptive_avg_pool2d(f2, 1).flatten(1)


# ============================================================================
# A2 / M2消融：最小的接口适配，不保留DPF的可学习方向路径编码和FiLM
# 模块边界：
# - 全局特征的384 -> 256线性投影由MAPNetA2.global_context承担；
# - 本类唯一可训练层 Linear(256+3,128)，给M3生成形状相同的path。
# - 3个额外输入只是固定的 sin(angle)、cos(angle)、归一化距离；
# - 原始海底水深经无参数中心路径grid_sample提供给M4的coords。
# 这里没有多尺度地形采样、三条平行路径编码、1D残差压缩或FiLM。
# ============================================================================
class MinimalM2Interface(nn.Module):
    """A2替代接口：仅用全局环境与固定位置编码生成path，采样中心水深供M4。"""

    def __init__(self, context_channels=256, path_channels=128):
        super().__init__()
        # A2新增的唯一线性层：只把输入整理成M3需要的128通道路径特征。
        self.to_path = nn.Linear(context_channels + 3, path_channels)

        ang = torch.arange(9, dtype=torch.float32) * (torch.pi * 40.0 / 180.0)
        ranges = torch.linspace(0.0, 50000.0, 250)
        radius = ranges / 50000.0
        sin_a = ang.sin()[:, None].expand(9, 250)
        cos_a = ang.cos()[:, None].expand(9, 250)
        norm_r = (2 * radius - 1)[None, :].expand(9, 250)
        fixed_position = torch.stack((sin_a, cos_a, norm_r), dim=-1)
        self.register_buffer("position", fixed_position[None], persistent=False)

        # 只取中心路径（偏移量0）；严格沿用原M2的方位、0-50km、±60km归一化。
        x = ranges[None, :] * sin_a
        y = ranges[None, :] * cos_a
        center_grid = torch.stack((x / 60000.0, y / 60000.0), dim=-1)
        self.register_buffer("center_grid", center_grid[None], persistent=False)

    def forward(self, context, bathy_raw):
        b = context.shape[0]
        if bathy_raw.shape != (b, 61, 61):
            raise ValueError("A2 adapter expects bathy [B,61,61]")
        # 无参数全局特征广播 + 固定角度/距离，不沿路径读取地形CNN特征。
        global_feature = context[:, None, None, :].expand(b, 9, 250, -1)
        positions = self.position.to(dtype=context.dtype).expand(b, -1, -1, -1)
        inp = torch.cat((global_feature, positions), dim=-1)
        path = self.to_path(inp).permute(0, 1, 3, 2).contiguous()  # [B,9,128,250]

        # 仅满足M4坐标通道中的bottom_ray需求，完全没有可训练地形提取算子。
        bottom_ray = F.grid_sample(
            bathy_raw[:, None], self.center_grid.expand(b, -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(1)  # [B,9,250]，原始物理水深(m)
        return path, bottom_ray


# ============================================================================
# 模块3 / CAI：跨方位特征交互
# 输入/输出均为[B,9,128,250]，不更改距离分辨率，也不负责TL解码。
# 顺序：周期性局部方向卷积 -> MLA在每个距离位置融合全部9个方位。
# ============================================================================
class AngularInteraction(nn.Module):
    """M3/局部交互：9方位环形相接，联合方位邻域和距离邻域做卷积。"""
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, (3, 5), groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, path):
        x = path.permute(0, 2, 1, 3)
        padded = F.pad(x, (2, 2, 0, 0), mode="replicate")
        padded = torch.cat((padded[:, :, -1:, :], padded, padded[:, :, :1, :]), dim=2)
        residual = F.silu(self.norm(self.pointwise(self.depthwise(padded))))
        return (x + self.scale * residual).permute(0, 2, 1, 3)


class AngularLatentAttention(nn.Module):
    """M3/全方位交互：借鉴MLA的低秩Q/KV和物理角度周期性RoPE。"""
    def __init__(self, channels=128, num_heads=4, q_rank=64, kv_rank=64,
                 content_dim=32, rotary_dim=16, value_dim=32):
        super().__init__()
        if channels <= 0 or num_heads <= 0 or rotary_dim % 2:
            raise ValueError("MLA channels/heads must be positive and rotary_dim must be even")
        self.num_heads = num_heads
        self.content_dim = content_dim
        self.rotary_dim = rotary_dim
        self.value_dim = value_dim
        self.pre_norm = nn.LayerNorm(channels)
        self.q_down = nn.Linear(channels, q_rank, bias=False)
        self.q_norm = nn.LayerNorm(q_rank)
        self.q_content = nn.Linear(q_rank, num_heads * content_dim, bias=False)
        self.q_rotary = nn.Linear(q_rank, num_heads * rotary_dim, bias=False)
        self.kv_down = nn.Linear(channels, kv_rank, bias=False)
        self.kv_norm = nn.LayerNorm(kv_rank)
        self.k_content = nn.Linear(kv_rank, num_heads * content_dim, bias=False)
        self.v_up = nn.Linear(kv_rank, num_heads * value_dim, bias=False)

        self.k_rotary = nn.Linear(channels, rotary_dim, bias=False)
        self.out = nn.Linear(num_heads * value_dim, channels, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        ang = torch.arange(9, dtype=torch.float32) * (2 * torch.pi / 9)
        harmonics = torch.arange(1, rotary_dim // 2 + 1, dtype=torch.float32)
        self.register_buffer("rope_cos", torch.cos(ang[:, None] * harmonics[None, :]))
        self.register_buffer("rope_sin", torch.sin(ang[:, None] * harmonics[None, :]))

    def _apply_rotary(self, x):

        pairs = x.reshape(*x.shape[:-1], self.rotary_dim // 2, 2)
        c = self.rope_cos.to(dtype=x.dtype)[None, None]
        s = self.rope_sin.to(dtype=x.dtype)[None, None]
        a, b = pairs.unbind(dim=-1)
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def forward(self, path):


        if path.ndim != 4 or path.shape[1] != 9:
            raise ValueError("angular MLA expects [B,9,C,R]")
        b, angles, channels, ranges = path.shape
        seq = path.permute(0, 3, 1, 2).reshape(b * ranges, angles, channels)
        x = self.pre_norm(seq)
        q_latent = self.q_norm(self.q_down(x))
        kv_latent = self.kv_norm(self.kv_down(x))
        h = self.num_heads
        qc = self.q_content(q_latent).view(-1, angles, h, self.content_dim).transpose(1, 2)
        qr = self.q_rotary(q_latent).view(-1, angles, h, self.rotary_dim).transpose(1, 2)
        kc = self.k_content(kv_latent).view(-1, angles, h, self.content_dim).transpose(1, 2)

        kr = self.k_rotary(x)[:, None, :, :].expand(-1, h, -1, -1)
        v = self.v_up(kv_latent).view(-1, angles, h, self.value_dim).transpose(1, 2)
        q = torch.cat((qc, self._apply_rotary(qr)), dim=-1)
        k = torch.cat((kc, self._apply_rotary(kr)), dim=-1)

        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).contiguous().reshape(b * ranges, angles, h * self.value_dim)
        delta = self.out(attended).reshape(b, ranges, angles, channels).permute(0, 2, 3, 1)
        return path + self.residual_scale * delta


# ============================================================================
# 模块4 / MTR：多尺度传播损失场重建
# 将M3的[B,9,128,250]路径特征结合M2的context与bottom_ray，逐尺度
# 重建二维深度-距离场；9个方位共享相同的解码器参数。
# ============================================================================
class DecoderUp(nn.Module):
    """M4的逐级上采样单元：融合路径skip和6个物理/几何坐标通道。"""
    def __init__(self, n_in, n_out, n_residual, skip_channels=16):
        super().__init__()
        self.up = ConvGNAct(n_in, n_out)
        self.fuse = ConvGNAct(n_out + skip_channels + 6, n_out)
        self.res = residual_stack(Residual2D, n_out, n_residual)
        self.skip_channels = skip_channels

    def forward(self, x, size, skip, coords):
        x = self.up(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        n, _, z, r = x.shape
        p = F.interpolate(skip, size=r, mode="linear", align_corners=True)
        p = p[:, :, None, :].expand(n, self.skip_channels, z, r)
        return self.res(self.fuse(torch.cat([x, p, coords], dim=1)))


class MAPNetA2(nn.Module):
    """A2消融网络：完整M1 -> M2最小适配 -> 完整M3 -> 完整M4。

    不再定义self.path或self.film。M4完全保持原有可学习层和计算；
    其128个通道使用固定MODEL_CONFIG而非从已删除的self.path读取。
    """
    def __init__(self, ssp_depth_m, *, ssp_mean, ssp_std,
                 bathy_mean, bathy_std, tl_mean, tl_std,
                 gradient_checkpointing=False):
        super().__init__()
        cfg = MODEL_CONFIG
        self.gradient_checkpointing = bool(gradient_checkpointing)
        for label, val in [("ssp_std", ssp_std), ("bathy_std", bathy_std), ("tl_std", tl_std)]:
            if val <= 0:
                raise ValueError(f"{label} must be positive")
        for label, val in [
            ("ssp_mean", ssp_mean), ("ssp_std", ssp_std),
            ("bathy_mean", bathy_mean), ("bathy_std", bathy_std),
            ("tl_mean", tl_mean), ("tl_std", tl_std),
        ]:
            self.register_buffer(label, torch.tensor(float(val)))
        # M1/EFE（未修改）: 两个残差CNN环境编码器，参数量4,365,505。
        self.ssp = SSPEncoder(ssp_depth_m, cfg["ssp_channels"], cfg["ssp_blocks"])
        self.bathy = BathyEncoder(cfg["bathy_channels"], cfg["bathy_blocks"])
        path_width = cfg["path_channels"]
        context_width = cfg["context_channels"]
        global_in = cfg["ssp_channels"][-1] + cfg["bathy_channels"][-1]

        # A2/M2接口（替代原DPF）: 仅2个FC，一个将全局特征映射为context，
        # 另一个将context与固定角度/距离映射为path；并进行无参数中心水深采样。

        self.global_context = nn.Sequential(nn.Linear(global_in, context_width), nn.SiLU())
        self.path_adapter = MinimalM2Interface(context_width, path_width)

        # M3/CAI（未修改）：原周期性局部卷积和角度低秩MLA保持一致。
        self.angular = AngularInteraction(path_width)
        self.angular_mla = AngularLatentAttention(path_width)
        # M4/MTR：保留context和路径细节旁路，向二维TL解码器注入环境信息。
        self.context_proj = nn.Linear(context_width, path_width)
        skip_width = cfg["skip_channels"]
        self.path_skip = nn.Conv1d(path_width, skip_width, 1)
        bottleneck = cfg["base_channels"]
        self.start = ConvGNAct(2 * path_width + 6, bottleneck)
        self.base = residual_stack(Residual2D, bottleneck, cfg["base_blocks"])
        u0, u1, u2 = cfg["up_channels"]
        r0, r1, r2 = cfg["up_blocks"]
        self.up1 = DecoderUp(bottleneck, u0, r0, skip_channels=skip_width)
        self.up2 = DecoderUp(u0, u1, r1, skip_channels=skip_width)
        self.up3 = DecoderUp(u1, u2, r2, skip_channels=skip_width)

        self.tail = nn.Sequential(
            nn.Conv2d(u2, u2, 3, padding=1, groups=u2, bias=False),
            nn.Conv2d(u2, u2, 1, bias=False),
            nn.GroupNorm(8, u2), nn.SiLU(inplace=True),
            nn.Conv2d(u2, 1, 1),
        )
        # M4/MTR使用的固定坐标；不属于单独的可学习网络模块。
        self.register_buffer("angles", torch.arange(9, dtype=torch.float32)
                             * torch.pi * 40.0 / 180.0)
        # 固定物理坐标只计算一次，不参与训练，也不写入权重文件。
        self.register_buffer("angle_sin", self.angles.sin(), persistent=False)
        self.register_buffer("angle_cos", self.angles.cos(), persistent=False)
        for z_count, r_count in ((25, 32), (50, 63), (100, 125), (200, 250)):
            self.register_buffer(
                f"z_grid_{z_count}_{r_count}",
                torch.linspace(0, 1, z_count).view(1, 1, z_count, 1),
                persistent=False,
            )
            self.register_buffer(
                f"r_grid_{z_count}_{r_count}",
                torch.linspace(0, 1, r_count).view(1, 1, 1, r_count),
                persistent=False,
            )


        for module in self.modules():
            if isinstance(module, (Residual1D, Residual2D)):
                nn.init.zeros_(module.gn2.weight)

    def _coords(self, bottom_ray, z_count, r_count):
        """M4内部工具：构造深度、距离、方位、路径底深等6个坐标通道。

        bottom_ray来自M2的地形中心路径采样（单位m），不同于训练用的
        bottom_depth_index水体掩码；后者不参加模型的forward。
        """
        b = bottom_ray.shape[0]
        n = b * 9
        dtype = bottom_ray.dtype
        z = getattr(self, f"z_grid_{z_count}_{r_count}").to(dtype=dtype)
        r = getattr(self, f"r_grid_{z_count}_{r_count}").to(dtype=dtype)
        z = z.expand(n, 1, z_count, r_count)
        r = r.expand(n, 1, z_count, r_count)
        si = self.angle_sin.to(dtype=dtype).view(1, 9, 1, 1, 1)
        co = self.angle_cos.to(dtype=dtype).view(1, 9, 1, 1, 1)
        si = si.expand(b, 9, 1, z_count, r_count).reshape(n, 1, z_count, r_count)
        co = co.expand(b, 9, 1, z_count, r_count).reshape(n, 1, z_count, r_count)
        bottom = bottom_ray.reshape(n, 1, 250)
        if r_count != 250:
            bottom = F.interpolate(bottom, size=r_count, mode="linear", align_corners=True)
        bottom = (bottom / 4000.0).unsqueeze(2).expand(n, 1, z_count, r_count)
        return torch.cat((z, r, si, co, bottom, bottom - z), dim=1)

    def _apply_block(self, block, *tensors):
        """M4内部工具：在不改变模块计算的前提下选择是否梯度检查点重算。"""
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, *tensors, use_reentrant=False)
        return block(*tensors)

    # ------------------------------------------------------------------------
    # M1 / EFE : 双分支环境特征编码（原始输入的唯一入口）
    # 输入：ssp[B,50]、bathy[B,61,61]，均保留原始物理单位。
    # 输出：s[B,192]、features=(f0,f1,f2)、terrain_global[B,192]。
    # 下游：三种环境表示由M2接收，M1不参与方向注意力或TL解码。
    # ------------------------------------------------------------------------
    def _module1_encode_environment(self, ssp, bathy):
        """M1/EFE：分别编码垂向SSP和二维海底地形。"""
        s = self.ssp(ssp, self.ssp_mean, self.ssp_std)
        features, terrain_global = self.bathy((bathy - self.bathy_mean) / self.bathy_std)
        return s, features, terrain_global

    # ------------------------------------------------------------------------
    # A2 / M2消融的接口桥（不是原有DPF）
    # 输入：M1 s[B,192]、terrain_global[B,192]和原始bathy[B,61,61]。
    # M1仍计算f0/f1/f2，但删去M2后，这些空间特征不再作为定向采样输入；
    # 其深层全局输出g仍由M1的完整地形编码器产生、支持梯度传播。
    # 输出：path[B,9,128,250]给完整M3；context[B,256]和bottom_ray[B,9,250]
    # 直接旁路到完整M4。bottom_ray仅为M4物理坐标输入。
    # 删除的M2关键机制：三路径可学习地形提取、方向特征压缩、FiLM条件调制。
    # ------------------------------------------------------------------------
    def _module2_interface_without_dpf(self, s, terrain_global, bathy):
        """A2必要的张量接口适配（仅2个FC + 无参数中心路径采样）。"""
        context = self.global_context(torch.cat([s, terrain_global], dim=1))
        path, bottom_ray = self.path_adapter(context, bathy)
        return path, bottom_ray, context

    # ------------------------------------------------------------------------
    # M3 / CAI : 跨方位交互（输入/输出shape严格相同）
    # 输入：M2的path[B,9,128,250]。
    # 输出：交互增强的path[B,9,128,250]。
    # 局部周期性卷积先交互相邻方位，再用MLA在每个距离点融合9个方位；
    # 本模块不消费bottom_ray/context，也不改变深度或距离网格。
    # ------------------------------------------------------------------------
    def _module3_interact_azimuths(self, path):
        """M3/CAI：周期性局部角度卷积 + 低秩双向角度注意力。"""
        return self.angular_mla(self.angular(path))

    # ------------------------------------------------------------------------
    # M4 / MTR : 多尺度TL场重建
    # 输入：M3增强path + M2旁路context、bottom_ray。
    # 输出：归一化TL[B,9,200,250]；转换回dB的predict_db不在四模块内。
    # 路径的9个方位暂时折叠进batch维，使用共享的二维解码器。
    # ------------------------------------------------------------------------
    def _module4_reconstruct_tl(self, path, bottom_ray, context):
        """M4/MTR：方向径向特征 -> 多尺度深度-距离传播损失矩阵。"""
        b = path.shape[0]
        # A2已删除self.path；M4通道数沿用原完整网络的固定配置128。
        channels = MODEL_CONFIG["path_channels"]
        n = b * 9
        p = path.reshape(n, channels, 250)
        skip = self.path_skip(p)
        p_low = F.adaptive_avg_pool1d(p, 32)[:, :, None, :].expand(n, channels, 25, 32)
        context_low = self.context_proj(context)[:, None, :]
        context_low = context_low.expand(b, 9, channels).reshape(n, channels, 1, 1)
        context_low = context_low.expand(n, channels, 25, 32)
        coords = self._coords(bottom_ray, 25, 32)
        x = self._apply_block(self.base, self.start(torch.cat([p_low, context_low, coords], dim=1)))
        x = self._apply_block(self.up1, x, (50, 63), skip, self._coords(bottom_ray, 50, 63))
        x = self._apply_block(self.up2, x, (100, 125), skip, self._coords(bottom_ray, 100, 125))
        x = self._apply_block(self.up3, x, (200, 250), skip, self._coords(bottom_ray, 200, 250))
        return self.tail(x).reshape(b, 9, 200, 250)

    def forward(self, ssp, bathy):
        """A2路径：完整M1 -> 极简M2接口 -> 完整M3 -> 完整M4。"""
        if ssp.ndim != 2 or ssp.shape[1] != 50:
            raise ValueError("ssp must be [B,50]")
        if bathy.ndim != 3 or bathy.shape[1:] != (61, 61):
            raise ValueError("bathy must be [B,61,61]")
        if ssp.shape[0] != bathy.shape[0]:
            raise ValueError("batch size mismatch")

        # M1不变。f0/f1/f2仍由完整地形CNN产生，但A2不再采样它们。
        s, _features, terrain_global = self._module1_encode_environment(ssp, bathy)
        # 删除DPF，仅保留必要形状接口：全局FC上下文+固定位置/中心水深。
        path, bottom_ray, context = self._module2_interface_without_dpf(
            s, terrain_global, bathy
        )
        # M3 -> M4：结合邻近及全局方位依赖的增强路径特征。
        path = self._module3_interact_azimuths(path)
        # M4输出归一化TL；实际dB由predict_db转换。
        return self._module4_reconstruct_tl(path, bottom_ray, context)

    def predict_db(self, ssp, bathy):
        return self.forward(ssp, bathy).float() * self.tl_std + self.tl_mean


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(data_dir, stats, *, gradient_checkpointing=False):
    """根据训练集统计量和实际SSP深度构建固定网络。"""
    import numpy as np
    from pathlib import Path
    depth = np.load(Path(data_dir) / "ssp_depth_m_r2.npy", allow_pickle=False)
    return MAPNetA2(
        depth,
        ssp_mean=stats["ssp_mean"], ssp_std=stats["ssp_std"],
        bathy_mean=stats["bathy_mean"], bathy_std=stats["bathy_std"],
        tl_mean=stats["tl_mean"], tl_std=stats["tl_std"],
        gradient_checkpointing=gradient_checkpointing,
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    model = MAPNetA2(
        torch.linspace(0, 5000, 50), ssp_mean=1500, ssp_std=30,
        bathy_mean=2500, bathy_std=1000, tl_mean=75, tl_std=20,
    )
    print(f"{MODEL_ARCHITECTURE}: {count_parameters(model):,} trainable parameters")
    model.eval()
    with torch.no_grad():
        ssp = torch.full((1, 50), 1500.0)
        ssp[:, 40:] = float("nan")
        bathy = torch.full((1, 61, 61), 2500.0)
        pred = model.predict_db(ssp, bathy)
    print(f"shape={tuple(pred.shape)}, finite={bool(torch.isfinite(pred).all())}")
