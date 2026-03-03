"""Full PyTorch port of wan_vae: CausalConv3d, RMS_norm, ResidualBlock, Resample, Encoder3d, Decoder3d, WanVAE_."""

import math

import torch
import torch.nn as nn
from einops import rearrange

CACHE_T = 2

VAE_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
VAE_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]
VAE_SCALE = [torch.tensor(VAE_MEAN, dtype=torch.float32), torch.tensor([1.0 / s for s in VAE_STD], dtype=torch.float32)]


class CausalConv3d(nn.Module):
    """3D conv with causal padding on time (axis 1): pad left only."""

    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0, 0), stride=(1, 1, 1)):
        super().__init__()
        self._pad = padding
        if isinstance(padding, int):
            pad = (padding, padding, padding)
        else:
            pad = tuple(padding)
        # (time_left, time_right), (h_left, h_right), (w_left, w_right)
        self._padding = (
            (0, 0),
            (2 * pad[0], 0) if pad[0] > 0 else (0, 0),
            (pad[1], pad[1]),
            (pad[2], pad[2]),
            (0, 0),
        )
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0,
        )

    def forward(self, x, cache_x=None, time_padding=None):
        # x: (N, C, T, H, W)
        padding = list(self._padding)
        if cache_x is not None and self._padding[1][0] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[1] = (padding[1][0] - cache_x.shape[2], padding[1][1])
        if time_padding is not None:
            padding[1] = time_padding
        # Pad (T, H, W) for tensor (N, C, T, H, W) -> (W_l, W_r, H_l, H_r, T_l, T_r)
        pl, pr = padding[1][0], padding[1][1]
        hl, hr = padding[2][0], padding[2][1]
        wl, wr = padding[3][0], padding[3][1]
        x = torch.nn.functional.pad(x, (wl, wr, hl, hr, pl, pr))
        return self.conv(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, images=True, bias=False):
        super().__init__()
        self.scale = dim ** 0.5
        self.images = images
        shape = (dim, 1, 1, 1) if images else (dim, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        # x: (N,...,C) or (N,T,H,W,C) for 5d
        norm = x.float().norm(dim=-1, keepdim=True).clamp(min=1e-12)
        out = (x.float() / norm * self.scale * self.gamma.view(-1))
        if self.bias is not None:
            out = out + self.bias.view(-1)
        return out.to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.ModuleList([
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, (3, 3, 3), padding=(1, 1, 1)),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, (3, 3, 3), padding=(1, 1, 1)),
        ])
        self.shortcut = CausalConv3d(in_dim, out_dim, (1, 1, 1), padding=(0, 0, 0)) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, new_cache=None, feat_idx=None):
        h = self.shortcut(x) if not isinstance(self.shortcut, nn.Identity) else x
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    last_frame = feat_cache[idx][:, :, -1:, :, :]
                    cache_x = torch.cat([last_frame, cache_x], dim=2)
                x = layer(x, cache_x)
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif isinstance(layer, RMS_norm):
                x = x.permute(0, 2, 3, 4, 1)
                x = layer(x)
                x = x.permute(0, 4, 1, 2, 3)
            else:
                x = layer(x)
        return x + h


class Upsample(nn.Module):
    def __init__(self, scale_factor=(2.0, 2.0)):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (N, H, W, C) -> need (N, C, H, W) for F.interpolate
        n, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)
        new_h, new_w = int(h * self.scale_factor[0]), int(w * self.scale_factor[1])
        x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="nearest")
        return x.permute(0, 2, 3, 1)


class Resample(nn.Module):
    def __init__(self, dim, mode, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.upsample = Upsample(scale_factor=(2.0, 2.0))
            self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)
        elif mode == "upsample3d":
            self.upsample = Upsample(scale_factor=(2.0, 2.0))
            self.conv = nn.Conv2d(dim, dim // 2, 3, padding=1)
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=0)
        elif mode == "downsample3d":
            self.conv = nn.Conv2d(dim, dim, 3, stride=2, padding=0)
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.conv = None

    def forward(self, x, feat_cache=None, new_cache=None, feat_idx=None):
        # x: (B, C, T, H, W) from encoder/decoder
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> b t h w c")
        if self.mode == "upsample3d" and hasattr(self, "time_conv"):
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    new_cache[idx] = torch.zeros(b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype)
                    feat_idx[0] += 1
                else:
                    x_ct = rearrange(x, "b t h w c -> b c t h w")
                    cache_x = x_ct[:, :, -CACHE_T:, :, :]
                    cache_x = torch.cat([feat_cache[idx], cache_x], dim=2)[:, :, -2:, :, :]
                    x = self.time_conv(x_ct, cache_x)
                    new_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = rearrange(x, "b (r c) t h w -> b (t r) h w c", r=2)
            t = x.shape[1]
        x_flat = rearrange(x, "b t h w c -> (b t) h w c")
        if self.mode in ["upsample2d", "upsample3d"]:
            x_flat = self.upsample(x_flat)
            x_flat = self.conv(x_flat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif self.mode in ["downsample2d", "downsample3d"]:
            x_flat = torch.nn.functional.pad(x_flat.permute(0, 3, 1, 2), (0, 1, 0, 1)).permute(0, 2, 3, 1)
            x_flat = self.conv(x_flat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = rearrange(x_flat, "(b t) h w c -> b t h w c", t=t)
        if self.mode == "downsample3d" and hasattr(self, "time_conv"):
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    new_cache[idx] = rearrange(x, "b t h w c -> b c t h w")
                    feat_idx[0] += 1
                else:
                    x_ct = rearrange(x, "b t h w c -> b c t h w")
                    cache_x = x_ct[:, :, -1:, :, :]
                    last_frame = feat_cache[idx][:, :, -1:, :, :]
                    x_concat = torch.cat([last_frame, x_ct], dim=2)
                    x = self.time_conv(x_concat, feat_cache[idx])
                    new_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = rearrange(x, "b c t h w -> b t h w c")
            else:
                x_ct = rearrange(x, "b t h w c -> b c t h w")
                first_frame = x_ct[:, :, :1, :, :]
                x = self.time_conv(x_ct, time_padding=(2, 0))
                x = rearrange(x, "b c t h w -> b t h w c")
                first_frame = rearrange(first_frame, "b c t h w -> b t h w c")
                x = torch.cat([first_frame, x[:, 1:, :, :, :]], dim=1)
        return rearrange(x, "b t h w c -> b c t h w")


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = rearrange(x, "b c t h w -> b t h w c")
        identity = x
        b, t, h, w, c = x.shape
        x = rearrange(x, "b t h w c -> (b t) h w c")
        x = self.norm(x)
        qkv = self.to_qkv(x.permute(0, 3, 1, 2))
        qkv = rearrange(qkv, "b (c nh) h w -> b (h w) nh c", nh=1)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = 1.0 / math.sqrt(c)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = rearrange(x, "b (h w) nh c -> b h w (c nh)", h=h, w=w)
        x = self.proj(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = rearrange(x, "(b t) h w c -> b t h w c", t=t)
        return rearrange(x + identity, "b t h w c -> b c t h w")


def _count_conv3d(module):
    count = 0
    if isinstance(module, CausalConv3d):
        return 1
    for name, child in module.named_children():
        count += _count_conv3d(child)
    for m in getattr(module, "downsamples", []) or []:
        count += _count_conv3d(m)
    for m in getattr(module, "upsamples", []) or []:
        count += _count_conv3d(m)
    for m in getattr(module, "middle", []) or []:
        count += _count_conv3d(m)
    for m in getattr(module, "head", []) or []:
        count += _count_conv3d(m) if hasattr(m, "__len__") else (1 if isinstance(m, CausalConv3d) else 0)
    return count


def get_cache(module):
    n = _count_conv3d(module)
    return [None] * n


def _pad_cache_time_to(cache_list, T=2):
    out = []
    for a in cache_list:
        if a is None:
            out.append(None)
            continue
        t = a.shape[2]
        if t == T:
            out.append(a)
        else:
            pad = T - t
            zeros = torch.zeros(a.shape[0], a.shape[1], pad, *a.shape[3:], device=a.device, dtype=a.dtype)
            out.append(torch.cat([zeros, a], dim=2))
    return out


class Encoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2, attn_scales=None, temperal_downsample=None, dropout=0.0):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        attn_scales = attn_scales or []
        temperal_downsample = temperal_downsample or [True, True, False]
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(3, dims[0], (3, 3, 3), padding=(1, 1, 1))
        self.downsamples = nn.ModuleList()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.downsamples.append(ResidualBlock(in_d, out_d, dropout))
                if scale in attn_scales:
                    self.downsamples.append(AttentionBlock(out_d))
                in_d = out_d
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.downsamples.append(Resample(out_d, mode, dropout))
                scale /= 2.0
        self.middle = nn.ModuleList([
            ResidualBlock(out_d, out_d, dropout),
            AttentionBlock(out_d),
            ResidualBlock(out_d, out_d, dropout),
        ])
        self.head = nn.ModuleList([
            RMS_norm(out_d, images=False),
            nn.SiLU(),
            CausalConv3d(out_d, z_dim, (3, 3, 3), padding=(1, 1, 1)),
        ])

    def forward(self, x, feat_cache=None):
        # x: (B, T, H, W, C) -> (B, C, T, H, W) for conv
        x = x.permute(0, 4, 1, 2, 3)
        new_cache = [None] * len(feat_cache) if feat_cache else None
        feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                last_frame = feat_cache[idx][:, :, -1:, :, :]
                cache_x = torch.cat([last_frame, cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            new_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    last_frame = feat_cache[idx][:, :, -1:, :, :]
                    cache_x = torch.cat([last_frame, cache_x], dim=2)
                x = layer(x, cache_x)
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif isinstance(layer, RMS_norm):
                x = x.permute(0, 2, 3, 4, 1)
                x = layer(x)
                x = x.permute(0, 4, 1, 2, 3)
            elif callable(layer):
                x = layer(x)
            else:
                x = layer(x)
        return x.permute(0, 2, 3, 4, 1), new_cache  # (B, C, T, H, W) -> (B, T, H, W, C)


class Decoder3d(nn.Module):
    def __init__(self, dim=128, z_dim=4, dim_mult=None, num_res_blocks=2, attn_scales=None, temperal_upsample=None, dropout=0.0):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        attn_scales = attn_scales or []
        temperal_upsample = temperal_upsample or [False, True, True]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / (2 ** (len(dim_mult) - 2))
        self.conv1 = CausalConv3d(z_dim, dims[0], (3, 3, 3), padding=(1, 1, 1))
        self.middle = nn.ModuleList([
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        ])
        self.upsamples = nn.ModuleList()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_d = in_d // 2
            for _ in range(num_res_blocks + 1):
                self.upsamples.append(ResidualBlock(in_d, out_d, dropout))
                if scale in attn_scales:
                    self.upsamples.append(AttentionBlock(out_d))
                in_d = out_d
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                self.upsamples.append(Resample(out_d, mode, dropout))
                scale *= 2.0
        self.head = nn.ModuleList([
            RMS_norm(dims[-1], images=False),
            nn.SiLU(),
            CausalConv3d(dims[-1], 3, (3, 3, 3), padding=(1, 1, 1)),
        ])

    def forward(self, x, feat_cache=None):
        # x: (B, T, H, W, C) -> (B, C, T, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        feat_idx = [0]
        new_cache = [None] * len(feat_cache) if feat_cache else []
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                last_frame = feat_cache[idx][:, :, -1:, :, :]
                cache_x = torch.cat([last_frame, cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            new_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, new_cache, feat_idx)
            else:
                x = layer(x)
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    last_frame = feat_cache[idx][:, :, -1:, :, :]
                    cache_x = torch.cat([last_frame, cache_x], dim=2)
                x = layer(x, cache_x)
                new_cache[idx] = cache_x
                feat_idx[0] += 1
            elif isinstance(layer, RMS_norm):
                x = x.permute(0, 2, 3, 4, 1)
                x = layer(x)
                x = x.permute(0, 4, 1, 2, 3)
            elif callable(layer):
                x = layer(x)
            else:
                x = layer(x)
        return x.permute(0, 2, 3, 4, 1), new_cache  # (B, C, T, H, W) -> (B, T, H, W, C)


class WanVAETorch(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=None, num_res_blocks=2, attn_scales=None, temperal_downsample=None, dropout=0.0):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [False, True, True]
        self.temperal_upsample = list(reversed(temperal_downsample))
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales or [], temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, (1, 1, 1), padding=(0, 0, 0))
        self.conv2 = CausalConv3d(z_dim, z_dim, (1, 1, 1), padding=(0, 0, 0))
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks, attn_scales or [], self.temperal_upsample, dropout)
        self._conv_num = _count_conv3d(self.decoder)
        self._enc_conv_num = _count_conv3d(self.encoder)

    def _apply_scale(self, x, scale):
        if scale is None or len(scale) != 2:
            return x
        s0, s1 = scale[0], scale[1]
        if isinstance(s0, torch.Tensor):
            s0 = s0.to(x.device).to(x.dtype)
            s1 = s1.to(x.device).to(x.dtype)
            while s0.dim() < x.dim():
                s0 = s0.unsqueeze(0)
                s1 = s1.unsqueeze(0)
        return (x - s0) * s1

    def _unapply_scale(self, x, scale):
        if scale is None or len(scale) != 2:
            return x
        s0, s1 = scale[0], scale[1]
        if isinstance(s0, torch.Tensor):
            s0 = s0.to(x.device).to(x.dtype)
            s1 = s1.to(x.device).to(x.dtype)
            while s0.dim() < x.dim():
                s0 = s0.unsqueeze(0)
                s1 = s1.unsqueeze(0)
        return x / s1 + s0

    def cacheless_encode(self, x, scale=None):
        out, _ = self.encoder(x)
        # out: (B, T, H, W, C) -> conv1 (B, C, T, H, W)
        out = out.permute(0, 4, 1, 2, 3)
        conv1_out = self.conv1(out)
        mu = conv1_out[:, : self.z_dim].permute(0, 2, 3, 4, 1)
        return self._apply_scale(mu, scale)

    def encode(self, x, scale=None):
        """x: (N, T, H, W, C). Chunked encode with cache for temporal causality."""
        t = x.shape[1]
        iter_ = 1 + (t - 1) // 4
        cache = get_cache(self.encoder)
        out0, cache = self.encoder(x[:, :1, :, :, :], feat_cache=cache)
        cache = _pad_cache_time_to(cache, T=CACHE_T)
        out = torch.zeros(out0.shape[0], iter_, *out0.shape[2:], device=x.device, dtype=out0.dtype)
        out[:, :1] = out0
        for i in range(1, iter_):
            start = 1 + 4 * (i - 1)
            chunk = x[:, start : start + 4, :, :, :]
            out_i, cache = self.encoder(chunk, feat_cache=cache)
            cache = _pad_cache_time_to(cache, T=CACHE_T)
            out[:, i : i + 1] = out_i
        # out: (B, iter_, H, W, C) -> conv1 expects (B, C, T, H, W)
        conv1_in = out.permute(0, 4, 1, 2, 3)
        conv1_out = self.conv1(conv1_in)
        mu = conv1_out[:, : conv1_out.shape[1] // 2].permute(0, 2, 3, 4, 1)
        return self._apply_scale(mu, scale)

    def decode(self, x, scale=None):
        """x: (N, T, H, W, z_dim). Chunked decode with cache."""
        x = self._unapply_scale(x, scale)
        # conv2 per "frame" in time (1x1x1 so we can do all at once)
        n, t, h, w, c = x.shape
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        iter_ = x.shape[1]
        cache = get_cache(self.decoder)
        out0, cache = self.decoder(x[:, :1, :, :, :], feat_cache=cache)
        cache = _pad_cache_time_to(cache, T=CACHE_T)
        out = torch.zeros(
            out0.shape[0], (iter_ - 1) * 4 + 1, *out0.shape[2:],
            device=x.device, dtype=out0.dtype,
        )
        out[:, :1] = out0
        for i in range(1, iter_):
            chunk = x[:, i : i + 1, :, :, :]
            out_i, cache = self.decoder(chunk, feat_cache=cache)
            cache = _pad_cache_time_to(cache, T=CACHE_T)
            start = 1 + 4 * (i - 1)
            end = start + out_i.shape[1]
            out[:, start:end] = out_i
        return out
