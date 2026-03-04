"""PyTorch versions of transformer_utils for world model (sinusoidal, RoPE)."""

import torch


def sinusoidal_embedding_1d(dim, position, device=None):
    """Sinusoidal positional embeddings [seq_len, dim]. position: (B*P*F,) or 1d."""
    assert dim % 2 == 0
    half = dim // 2
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32, device=device)
    position = position.float()
    freqs = torch.outer(
        position,
        torch.pow(10000.0, -torch.arange(half, dtype=torch.float32, device=position.device) / half),
    )
    x = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000.0, device=None):
    """RoPE freqs complex [max_seq_len, dim//2]. Matches JAX: arange(0, dim, 2) / dim."""
    assert dim % 2 == 0
    # JAX: jnp.arange(0, dim, 2) / dim -> dim/2 elements; do not use half = dim//2 then arange(0, half, 2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_params_mp(head_dim, max_seq_len=1024, device=None):
    """Concat of 3 RoPE parts for world model. head_dim = dim // num_heads (real).
    JAX: d - 4*(d//6) + 2*(d//6) + 2*(d//6) = d, freqs (max_seq_len, d/2) complex."""
    part0 = head_dim - 4 * (head_dim // 6)
    part1 = 2 * (head_dim // 6)
    part2 = 2 * (head_dim // 6)
    f0 = rope_params(max_seq_len, part0, device=device)
    f1 = rope_params(max_seq_len, part1, device=device)
    f2 = rope_params(max_seq_len, part2, device=device)
    return torch.cat([f0, f1, f2], dim=1)


def _rope_apply_3d(x, grid_sizes, freqs, start_frame=0, device=None):
    """x: (B, S, N, D). Apply 3D RoPE from concatenated freqs (3 parts)."""
    f, h, w = grid_sizes
    _, _, n, d_total = x.shape
    # RoPE is applied in complex form: x has d_total real dims = d_total//2 complex; freqs must match
    c = d_total // 2
    if freqs.shape[1] < c:
        # Caller passed freqs for a smaller head_dim (e.g. wrong config); build correct one from x
        max_seq_len = freqs.shape[0]
        dev = device if device is not None else x.device
        freqs = rope_params_mp(d_total, max_seq_len=max_seq_len, device=dev)
    freqs = freqs[:, :c]
    part0 = c - 2 * (c // 3)
    part1 = c // 3
    part2 = c // 3
    freqs_list = torch.split(freqs, [part0, part1, part2], dim=1)
    seq_len = f * h * w
    sliced_t = freqs_list[0][start_frame : start_frame + f]
    d0, d1, d2 = [u.shape[1] for u in freqs_list]
    t_3d = sliced_t.reshape(f, 1, 1, d0).expand(f, h, w, d0).reshape(1, seq_len, 1, d0)
    # Expand spatial RoPE over frames so (1, h, w) -> (f, h, w) before reshaping to seq_len
    h_3d = freqs_list[1][:h].reshape(1, h, 1, d1).expand(f, h, w, d1).reshape(1, seq_len, 1, d1)
    w_3d = freqs_list[2][:w].reshape(1, 1, w, d2).expand(f, h, w, d2).reshape(1, seq_len, 1, d2)
    freqs_3d = torch.cat([t_3d, h_3d, w_3d], dim=-1)
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.complex(x_float[..., 0], x_float[..., 1])
    if freqs_3d.is_complex():
        freqs_c = freqs_3d
    else:
        freqs_3d = freqs_3d.to(x.device)
        freqs_c = torch.view_as_complex(freqs_3d.reshape(*freqs_3d.shape[:-1], -1, 2))
    out = x_complex * freqs_c
    out = torch.stack([out.real, out.imag], dim=-1).flatten(-2)
    return out.to(x.dtype)


def apply_rope_mp_torch(x, grid_sizes, freqs, f_arg, s_arg, current_start=0):
    """x: (B, F*P*S, N, D). Apply RoPE with multiplayer reshape."""
    from einops import rearrange

    b = x.shape[0]
    out = rearrange(
        x,
        "b (f p s) n d -> (b p) (f s) n d",
        f=f_arg,
        s=s_arg,
    )
    out = _rope_apply_3d(out, grid_sizes, freqs, start_frame=current_start, device=x.device)
    out = rearrange(
        out,
        "(b p) (f s) n d -> b (f p s) n d",
        b=b,
        f=f_arg,
    )
    return out
