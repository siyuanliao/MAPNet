"""MAP-Net：按论文功能划分的四阶段三维传播损失预测网络。

这里的“四模块”是方法层面的功能边界，不是四套独立训练的子网络：
  M1/EFE 环境特征编码（SSPEncoder + BathyEncoder）
      SSP[B,50]、Bathy[B,61,61] -> s[B,192]、多尺度地形特征、g_bathy[B,192]
  M2/DPF 方向感知特征构建与融合（DirectionalPathEncoder + GlobalContext + FiLM）
      M1特征 + 原始地形 -> path[B,9,128,250]、bottom_ray[B,9,250]、context[B,256]
  M3/CAI 跨方位特征交互（AngularInteraction + AngularLatentAttention）
      path[B,9,128,250] -> enriched_path[B,9,128,250]
  M4/MTR 多尺度传播损失重建（坐标嵌入 + 残差解码器 + Tail）
      enriched_path、context、bottom_ray -> 归一化TL[B,9,200,250]

训练用的water mask不是第五个网络模块；它在metrics/train中排除海底下的TL。
为兼容既有checkpoint，保留全部子层属性名、参数和buffer及其注册次序。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MODEL_CONFIG = {
    "ssp_channels": (48, 96, 192), "ssp_blocks": (2, 2, 3),
    "bathy_channels": (48, 96, 192), "bathy_blocks": (3, 3, 4),
    "path_projection": (24, 40, 56), "path_channels": 128,
    "path_blocks": 3, "context_channels": 256, "skip_channels": 24,
    "base_channels": 256, "base_blocks": 9,
    "up_channels": (192, 128, 48), "up_blocks": (3, 2, 0),
}
MODEL_ARCHITECTURE = "MAP-Net"
MODEL_FULL_NAME = "Multi-Azimuth Path-Aware Network"

# 论文功能模块清单：使用子层属性名，不额外封装nn.Module，避免改变权重键名。
# M1产生环境特征；M2构造逐方位路径并融合环境；M3只做跨方位交互；
# M4把增强后的方位-距离特征还原为方位-深度-距离TL场。
PAPER_MODULES = {
    "M1_EFE": ("ssp", "bathy"),
    "M2_DPF": ("path", "global_context", "film"),
    "M3_CAI": ("angular", "angular_mla"),
    "M4_MTR": ("context_proj", "path_skip", "start", "base", "up1", "up2", "up3", "tail"),
}


# ============================================================================
# 共享基础层（基础构件，不单独算作论文第五个模块）
# ConvGNAct / Residual1D / Residual2D 被上述模块内部重复使用。
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
# 调用位置：MAPNet._module1_encode_environment()。
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
# 模块2 / DPF：方向感知的路径特征构建与环境融合
# 本类只负责“路径采样和压缩”；SSP/地形全局上下文与FiLM在MAPNet中执行。
# 边界：输入M1的三尺度地形特征和原始地形；输出9方位路径特征与
#       每个方位中心路径的海底水深。FiLM后才是完整的M2输出。
# ============================================================================
class DirectionalPathEncoder(nn.Module):
    """M2的几何采样子层：9个方位×3条偏移路径×250距离点。"""
    def __init__(self, feature_channels, projection_channels, path_channels, path_blocks):
        super().__init__()

        ang = torch.arange(9, dtype=torch.float32) * (torch.pi * 40.0 / 180.0)
        radius = torch.linspace(0.0, 50000.0, 250)
        offset = torch.tensor([-3000.0, 0.0, 3000.0])
        xx = (radius[None, None, :] * ang.sin()[:, None, None]
              + offset[None, :, None] * ang.cos()[:, None, None])
        yy = (radius[None, None, :] * ang.cos()[:, None, None]
              - offset[None, :, None] * ang.sin()[:, None, None])
        grid = torch.stack((xx / 60000.0, yy / 60000.0), dim=-1)
        if not bool((grid.abs() <= 1).all()):
            raise ValueError("bathy sampling outside +-60 km")
        self.register_buffer("grid", grid.reshape(1, 27, 250, 2))
        p0, p1, p2 = projection_channels
        c0, c1, c2 = feature_channels
        self.proj0 = nn.Conv2d(c0, p0, 1)
        self.proj1 = nn.Conv2d(c1, p1, 1)
        self.proj2 = nn.Conv2d(c2, p2, 1)
        self.compress = nn.Sequential(
            ConvGNAct(3 * (p0 + p1 + p2) + 3, path_channels, k=1, ndim=1),
            *[Residual1D(path_channels) for _ in range(path_blocks)],
        )
        self.path_channels = path_channels

    def _sample(self, feature):
        b, c = feature.shape[:2]
        sampled = F.grid_sample(
            feature, self.grid.expand(b, -1, -1, -1),
            mode="bilinear", padding_mode="border", align_corners=True,
        )
        return (sampled.reshape(b, c, 9, 3, 250)
                .permute(0, 2, 1, 3, 4).reshape(b, 9, 3 * c, 250))

    def forward(self, features, bathy_raw):
        f0, f1, f2 = features
        raw = self._sample(bathy_raw[:, None])
        pieces = [
            self._sample(self.proj0(f0)),
            self._sample(self.proj1(f1)),
            self._sample(self.proj2(f2)),
            raw / 4000.0,
        ]
        x = torch.cat(pieces, dim=2)
        b = bathy_raw.shape[0]
        path = self.compress(x.reshape(b * 9, x.shape[2], 250))
        path = path.reshape(b, 9, self.path_channels, 250)
        center_depth = raw[:, :, 1, :]
        return path, center_depth


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


class MAPNet(nn.Module):
    """四功能模块串联的完整模型，forward返回归一化TL[B,9,200,250]。

    模块交接详见 _module1_encode_environment() 到 _module4_reconstruct_tl()。
    这里只改变代码分段与注释；原有所有nn.Module仍注册在self原属性下，
    不引入额外的父级封装，确保历史MLA checkpoint仍可strict=True加载。
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
        # M1/EFE：原始环境输入 -> SSP全局特征 + 地形多尺度/全局特征。
        self.ssp = SSPEncoder(ssp_depth_m, cfg["ssp_channels"], cfg["ssp_blocks"])
        self.bathy = BathyEncoder(cfg["bathy_channels"], cfg["bathy_blocks"])
        path_width = cfg["path_channels"]
        # M2/DPF（部分）：地形多尺度特征 -> 沿方位的路径采样与压缩。
        self.path = DirectionalPathEncoder(
            cfg["bathy_channels"], cfg["path_projection"],
            path_width, cfg["path_blocks"],
        )
        # M3/CAI：方位特征之间的局部周期性交互与全方位MLA交互。
        # Preserve module registration order for checkpoint compatibility.
        self.angular = AngularInteraction(path_width)
        self.angular_mla = AngularLatentAttention(path_width)
        # M2/DPF（续）：SSP和地形全局特征形成context，再通过FiLM调制路径。
        # 这些层虽在M3之后定义，但在forward中属于先于M3执行的模块2。
        context_width = cfg["context_channels"]
        global_in = cfg["ssp_channels"][-1] + cfg["bathy_channels"][-1]
        self.global_context = nn.Sequential(nn.Linear(global_in, context_width), nn.SiLU())
        self.film = nn.Linear(context_width, 2 * path_width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
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
    # M2 / DPF : 传播方向路径构建及SSP/地形条件融合
    # 输入：M1的s/features/terrain_global，以及原始bathy（路径真实水深）。
    # 输出：FiLM调制后的path[B,9,128,250]、bottom_ray[B,9,250]、
    #       context[B,256]。path交给M3；bottom_ray/context旁路交给M4。
    # 注意：FiLM属于M2，不属于后面的角度交互模块。
    # ------------------------------------------------------------------------
    def _module2_fuse_directional_paths(self, s, features, terrain_global, bathy):
        """M2/DPF：物理方位采样、全局环境融合与FiLM条件调制。"""
        path, bottom_ray = self.path(features, bathy)
        context = self.global_context(torch.cat([s, terrain_global], dim=1))
        gamma, beta = self.film(context).chunk(2, dim=1)
        path = path * (1 + gamma[:, None, :, None]) + beta[:, None, :, None]
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
        channels = self.path.path_channels
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
        """四模块唯一主路径：M1 -> M2 -> M3 -> M4（没有新增层或参数）。"""
        if ssp.ndim != 2 or ssp.shape[1] != 50:
            raise ValueError("ssp must be [B,50]")
        if bathy.ndim != 3 or bathy.shape[1:] != (61, 61):
            raise ValueError("bathy must be [B,61,61]")
        if ssp.shape[0] != bathy.shape[0]:
            raise ValueError("batch size mismatch")

        # M1 -> M2：SSP全局特征与地形多尺度/全局特征。
        s, features, terrain_global = self._module1_encode_environment(ssp, bathy)
        # M2 -> M3：FiLM调制后的逐方位路径特征。
        # M2 -> M4（旁路）：全局环境context与路径海底水深bottom_ray。
        path, bottom_ray, context = self._module2_fuse_directional_paths(
            s, features, terrain_global, bathy
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
    return MAPNet(
        depth,
        ssp_mean=stats["ssp_mean"], ssp_std=stats["ssp_std"],
        bathy_mean=stats["bathy_mean"], bathy_std=stats["bathy_std"],
        tl_mean=stats["tl_mean"], tl_std=stats["tl_std"],
        gradient_checkpointing=gradient_checkpointing,
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    model = MAPNet(
        torch.linspace(0, 5000, 50), ssp_mean=1500, ssp_std=30,
        bathy_mean=2500, bathy_std=1000, tl_mean=75, tl_std=20,
    )
    print(f"{MODEL_ARCHITECTURE} ({MODEL_FULL_NAME}): {count_parameters(model):,} trainable parameters")
    model.eval()
    with torch.no_grad():
        ssp = torch.full((1, 50), 1500.0)
        ssp[:, 40:] = float("nan")
        bathy = torch.full((1, 61, 61), 2500.0)
        pred = model.predict_db(ssp, bathy)
    print(f"shape={tuple(pred.shape)}, finite={bool(torch.isfinite(pred).all())}")
