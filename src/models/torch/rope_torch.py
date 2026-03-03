"""PyTorch RoPE utilities for ActionModule and world model."""

import torch


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    if len(x) == dim:
        return tuple(x)
    raise ValueError(f"Expected length {dim} or int, got {len(x)}")


def get_meshgrid_nd(start, *args, dim=2, device=None):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, got {len(args)}")
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, device=device, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    return torch.stack(grid, dim=0)


def reshape_for_broadcast(freqs_cis, x, head_first=False):
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis[0], freqs_cis[1]
        if head_first:
            shape = [1] * x.dim()
            shape[-2] = cos.shape[0]
            shape[-1] = cos.shape[1]
        else:
            shape = [1, cos.shape[0], 1, cos.shape[1]]
        return cos.reshape(*shape), sin.reshape(*shape)
    else:
        shape = [1] * x.dim()
        shape[1] = freqs_cis.shape[0]
        shape[-1] = freqs_cis.shape[1]
        return freqs_cis.reshape(*shape)


def rotate_half(x):
    x = x.float()
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = x_reshaped[..., 0], x_reshaped[..., 1]
    return torch.stack([-x_imag, x_real], dim=-1).reshape(x.shape)


def apply_rotary_emb(xq, xk, freqs_cis, head_first=False, start_offset=0):
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    seq_len = xq.shape[1]
    cos = cos[:, start_offset : start_offset + seq_len, :]
    sin = sin[:, start_offset : start_offset + seq_len, :]
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).to(xq.dtype)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).to(xk.dtype)
    return xq_out, xk_out


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, use_real=False, theta_rescale_factor=1.0, interpolation_factor=1.0, device=None):
    if isinstance(pos, int):
        pos = torch.arange(pos, device=device, dtype=torch.float32)
    if theta_rescale_factor != 1.0:
        theta = theta * (theta_rescale_factor ** (dim / (dim - 2)))
    half = dim // 2
    # Match JAX: arange(0, dim, 2)[:dim//2] / dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2 / dim))
    freqs = torch.outer(pos * interpolation_factor, inv_freq)
    if use_real:
        freqs_cos = torch.cos(freqs).repeat(1, 2)
        freqs_sin = torch.sin(freqs).repeat(1, 2)
        return freqs_cos, freqs_sin
    return torch.polar(torch.ones_like(freqs), freqs)


def get_nd_rotary_pos_embed(rope_dim_list, start, *args, theta=10000.0, use_real=False, theta_rescale_factor=1.0, interpolation_factor=1.0, device=None):
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list), device=device)
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i], grid[i].reshape(-1), theta,
            use_real=use_real, theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i], device=device,
        )
        embs.append(emb)
    if use_real:
        cos = torch.cat([e[0] for e in embs], dim=1)
        sin = torch.cat([e[1] for e in embs], dim=1)
        return cos, sin
    return torch.cat(embs, dim=1)
