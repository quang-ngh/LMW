"""PyTorch equivalents of transformer building blocks (Flax/NNX naming not used; load via state_dict_utils)."""

import math

import torch
import torch.nn as nn
from einops import rearrange


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, num_features, eps=1e-6, elementwise_affine=True):
        super().__init__(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # dtype = x.dtype
        # if dtype == torch.bfloat16:
        #     x = super().forward(x.float()).to(dtype)
        #     return x
        # return super().forward(x)
        return super().forward(x)


class Conv3d(nn.Module):
    """3D conv matching JAX layout: input (B, C, D, H, W), kernel (D, H, W, in_c, out_c) in JAX -> (out_c, in_c, D, H, W) in PyTorch."""

    def __init__(self, in_channels, out_channels, kernel_size, strides=(1, 1, 1), padding="valid"):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=0 if str(padding).lower() == "valid" else kernel_size[0] // 2,
        )
        # init like JAX
        with torch.no_grad():
            fan_in = in_channels * math.prod(kernel_size)
            self.conv.weight.data.normal_(0, 2.0 / math.sqrt(fan_in))
            self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


def _rope_apply_torch(x, grid_sizes, freqs, start_frame=0):
    """x: (B, S, N, D), freqs complex (seq, D/2). Apply 3D RoPE from grid (F,H,W)."""
    import torch
    f, h, w = grid_sizes
    _, c = x.shape[2], x.shape[3] // 2
    split_boundaries = [c - 2 * (c // 3), c - 2 * (c // 3) + c // 3]
    freqs_list = torch.split(freqs, split_boundaries, dim=1)
    seq_len = f * h * w
    # freqs: (max_seq, total_dim) complex
    sliced_t = freqs_list[0][start_frame : start_frame + f]  # (F, d0)
    # Build 3D freqs (1, S, 1, D) for broadcast
    # repeat to (F, H, W, d0), (F, H, W, d1), (F, H, W, d2) then concat
    d0, d1, d2 = [u.shape[1] for u in freqs_list]
    t_3d = sliced_t.reshape(f, 1, 1, d0).expand(f, h, w, d0).reshape(1, seq_len, 1, d0)
    h_3d = freqs_list[1][:h].reshape(1, h, 1, d1).expand(1, h, w, d1).reshape(1, seq_len, 1, d1)
    w_3d = freqs_list[2][:w].reshape(1, 1, w, d2).expand(1, h, w, d2).reshape(1, seq_len, 1, d2)
    freqs_3d = torch.cat([t_3d, h_3d, w_3d], dim=-1)  # (1, S, 1, D)
    # x: (B, S, N, D) -> view as complex
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.complex(x_float[..., 0], x_float[..., 1])
    freqs_c = torch.complex(freqs_3d.real, freqs_3d.imag) if freqs_3d.is_complex() else freqs_3d
    if not freqs_3d.is_complex():
        freqs_c = torch.view_as_complex(freqs_3d.reshape(*freqs_3d.shape[:-1], -1, 2))
    out = x_complex * freqs_c
    out = torch.stack([out.real, out.imag], dim=-1).flatten(-2)
    return out.to(x.dtype)


def rope_params_torch(max_seq_len, dim, theta=10000.0, device=None):
    """RoPE freqs complex (max_seq_len, dim//2). Matches JAX: arange(0, dim, 2) / dim."""
    assert dim % 2 == 0, "dim must be even"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)  # e^(i*freqs)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs, block_mask=None):
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = _rope_apply_torch(q, grid_sizes, freqs)
        k = _rope_apply_torch(k, grid_sizes, freqs)
        scale = d ** -0.5
        # scaled_dot_product_attention expects (B, S, N, D); uses Flash Attention when available
        attn_mask = None if block_mask is None else ~block_mask  # SDPA: True = mask out
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale
        )
        x = x.reshape(b, s, self.dim)
        return self.o(x)


class WanI2VCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x_bsd, context_btd):
        q = rearrange(self.norm_q(self.q(x_bsd)), "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(self.norm_k(self.k(context_btd)), "b t (n d) -> b t n d", n=self.num_heads)
        v = rearrange(self.v(context_btd), "b t (n d) -> b t n d", n=self.num_heads)
        scale = self.head_dim ** -0.5
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, scale=scale
        )
        x = rearrange(x, "b s n d -> b s (n d)")
        return self.o(x)


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(in_dim, eps=1e-5)
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim, eps=1e-5)

    def forward(self, x):
        x = self.norm0(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.norm1(x)
        return x
