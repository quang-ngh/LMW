"""PyTorch port of ActionModule for causal mouse/keyboard conditioning."""

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from .rope_torch import apply_rotary_emb, get_nd_rotary_pos_embed
from .transformer import WanRMSNorm
from .kv_cache_torch import KVCache


class ActionModuleTorch(nn.Module):
    def __init__(
        self,
        mouse_dim_in=2,
        keyboard_dim_in=6,
        hidden_size=128,
        img_hidden_size=1536,
        keyboard_hidden_dim=1024,
        mouse_hidden_dim=1024,
        vae_time_compression_ratio=4,
        windows_size=3,
        heads_num=16,
        patch_size=(1, 2, 2),
        qk_norm=True,
        qkv_bias=False,
        rope_dim_list=(8, 28, 28),
        rope_theta=256,
        mouse_qk_dim_list=(8, 28, 28),
        enable_mouse=True,
        enable_keyboard=True,
        local_attn_size=6,
        left_action_padding=11,
    ):
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard
        self.rope_dim_list = list(rope_dim_list)
        self.rope_theta = rope_theta
        self.mouse_qk_dim_list = list(mouse_qk_dim_list)
        self.heads_num = heads_num
        self.mouse_head_dim = mouse_hidden_dim // heads_num
        self.keyboard_head_dim = keyboard_hidden_dim // heads_num
        self.keyboard_dims = keyboard_dim_in
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.latents_window_size = windows_size
        self.patch_size = list(patch_size) if not isinstance(patch_size, int) else [patch_size] * 3

        if self.enable_keyboard:
            self.keyboard_embed = nn.Sequential(
                nn.Linear(keyboard_dim_in, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

        if self.enable_mouse:
            c = mouse_hidden_dim
            frames_in = mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size
            self.mouse_mlp = nn.Sequential(
                nn.Linear(frames_in, c, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(c, c),
                nn.LayerNorm(c, eps=1e-5, elementwise_affine=True),
            )
            self.t_qkv = nn.Linear(c, c * 3, bias=qkv_bias)
            head_dim = c // heads_num
            self.img_attn_q_norm = WanRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.img_attn_k_norm = WanRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.proj_mouse = nn.Linear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
            self.key_attn_k_norm = WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else nn.Identity()
            self.mouse_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = nn.Linear(
                hidden_size * windows_size * vae_time_compression_ratio,
                keyboard_hidden_dim * 2,
                bias=qkv_bias,
            )
            self.proj_keyboard = nn.Linear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

    def get_rotary_pos_embed(
        self,
        video_length,
        height,
        width,
        head_dim,
        rope_dim_list=None,
        start_offset=0,
        device=None,
    ):
        target_ndim = 3
        latents_size = [video_length + start_offset, height, width]
        ps = self.patch_size
        if isinstance(ps, int):
            rope_sizes = [s // ps for s in latents_size]
        else:
            rope_sizes = [latents_size[idx] // ps[idx] for idx in range(len(latents_size))]
        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
            device=device,
        )
        slice_len = video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0]
        return (
            freqs_cos[-slice_len:],
            freqs_sin[-slice_len:],
        )

    def forward(
        self,
        x,
        tt,
        th,
        tw,
        mouse_condition=None,
        keyboard_condition=None,
        block_mask_mouse=None,
        block_mask_keyboard=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        start_frame=0,
        num_frame_per_block=1,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        new_kv_cache_mouse = None
        new_kv_cache_keyboard = None
        B, N_frames, C = keyboard_condition.shape
        _freqs_cos, _freqs_sin = self.get_rotary_pos_embed(
            7500,
            self.patch_size[1],
            self.patch_size[2],
            64,
            self.mouse_qk_dim_list,
            start_offset=0,
            device=x.device,
        )
        freqs_cis = (_freqs_cos, _freqs_sin)

        assert tt * th * tw == x.shape[1]

        if self.enable_mouse and mouse_condition is not None:
            hidden_states = (
                x.reshape(B, tt, th * tw, x.shape[-1])
                .transpose(0, 2, 1, 3)
                .reshape(B * th * tw, tt, x.shape[-1])
            )
        else:
            hidden_states = x

        frames_window_size = self.vae_time_compression_ratio * self.latents_window_size
        if self.enable_mouse and mouse_condition is not None:
            S = th * tw
            if kv_cache_mouse is not None:
                group_mouse = [
                    mouse_condition[
                        :,
                        i * self.vae_time_compression_ratio : i * self.vae_time_compression_ratio + frames_window_size,
                        :,
                    ]
                    for i in range(num_frame_per_block)
                ]
            else:
                if teacher_forcing:
                    group_mouse = [
                        mouse_condition[
                            :,
                            i * self.vae_time_compression_ratio : i * self.vae_time_compression_ratio + frames_window_size,
                            :,
                        ]
                        for i in range(tt // 2)
                    ]
                    group_mouse = group_mouse + group_mouse
                else:
                    group_mouse = [
                        mouse_condition[
                            :,
                            i * self.vae_time_compression_ratio : i * self.vae_time_compression_ratio + frames_window_size,
                            :,
                        ]
                        for i in range(tt)
                    ]
            group_mouse_btwd = torch.stack(group_mouse, dim=1)
            group_mouse_btd = rearrange(group_mouse_btwd, "B T W D -> B T (W D)")
            group_mouse_btd = group_mouse_btd.repeat_interleave(S, dim=0)

            group_mouse_btd = torch.cat([hidden_states, group_mouse_btd], dim=-1)
            group_mouse_btd = self.mouse_mlp(group_mouse_btd)
            mouse_qkv = self.t_qkv(group_mouse_btd)
            q_blhd, k_blhd, v_blhd = rearrange(
                mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )

            q_blhd = self.img_attn_q_norm(q_blhd).to(v_blhd.dtype)
            k_blhd = self.img_attn_k_norm(k_blhd).to(v_blhd.dtype)

            if teacher_forcing:
                q_blhd = rearrange(q_blhd, "b (r s) n d -> (b r) s n d", r=2)
                k_blhd = rearrange(k_blhd, "b (r s) n d -> (b r) s n d", r=2)
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd, k_blhd, freqs_cis, start_offset=start_frame, head_first=False
                )
                q_blhd = rearrange(q_blhd, "(b r) s n d -> b (r s) n d", r=2)
                k_blhd = rearrange(k_blhd, "(b r) s n d -> b (r s) n d", r=2)
            else:
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd, k_blhd, freqs_cis, start_offset=start_frame, head_first=False
                )

            attn, new_kv_cache_mouse = self._compute_attention_causal(
                q_blhd,
                k_blhd,
                v_blhd,
                repeat_kv=False,
                kv_cache=kv_cache_mouse,
                block_mask=block_mask_mouse,
                teacher_forcing=teacher_forcing,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )
            attn = rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)
            attn = self.proj_mouse(attn)
            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            keyboard_condition = self.keyboard_embed(keyboard_condition)
            if kv_cache_keyboard is not None:
                group_keyboard = [
                    keyboard_condition[
                        :,
                        self.vae_time_compression_ratio * (i - self.latents_window_size) + frames_window_size : i * self.vae_time_compression_ratio + frames_window_size,
                        :,
                    ]
                    for i in range(num_frame_per_block)
                ]
            else:
                if teacher_forcing:
                    group_keyboard = [
                        keyboard_condition[
                            :,
                            self.vae_time_compression_ratio * (i - self.latents_window_size) + frames_window_size : i * self.vae_time_compression_ratio + frames_window_size,
                            :,
                        ]
                        for i in range(tt // 2)
                    ]
                    group_keyboard = group_keyboard + group_keyboard
                else:
                    group_keyboard = [
                        keyboard_condition[
                            :,
                            self.vae_time_compression_ratio * (i - self.latents_window_size) + frames_window_size : i * self.vae_time_compression_ratio + frames_window_size,
                            :,
                        ]
                        for i in range(tt)
                    ]

            group_keyboard = torch.stack(group_keyboard, dim=1)
            group_keyboard = group_keyboard.reshape(group_keyboard.shape[0], group_keyboard.shape[1], -1)

            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)

            B, L, HD = mouse_q.shape
            D = HD // self.heads_num
            q_blhd = mouse_q.reshape(B, L, self.heads_num, D)

            B, L, KHD = keyboard_kv.shape
            keyboard_kv = keyboard_kv.reshape(B, L, 2, self.heads_num, D)
            k_blhd = keyboard_kv[:, :, 0, :, :]
            v_blhd = keyboard_kv[:, :, 1, :, :]
            q_blhd = self.key_attn_q_norm(q_blhd).to(v_blhd.dtype)
            k_blhd = self.key_attn_k_norm(k_blhd).to(v_blhd.dtype)

            S = th * tw
            B, TS, H, D = q_blhd.shape
            T_ = TS // S
            q_blhd = (
                q_blhd.reshape(B, T_, S, H, D)
                .transpose(0, 2, 1, 3, 4)
                .reshape(B * S, T_, H, D)
            )

            if teacher_forcing:
                q_blhd = rearrange(q_blhd, "b (r s) n d -> (b r) s n d", r=2)
                k_blhd = rearrange(k_blhd, "b (r s) n d -> (b r) s n d", r=2)
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd, k_blhd, freqs_cis, start_offset=start_frame, head_first=False
                )
                q_blhd = rearrange(q_blhd, "(b r) s n d -> b (r s) n d", r=2)
                k_blhd = rearrange(k_blhd, "(b r) s n d -> b (r s) n d", r=2)
            else:
                q_blhd, k_blhd = apply_rotary_emb(
                    q_blhd, k_blhd, freqs_cis, start_offset=start_frame, head_first=False
                )

            attn, new_kv_cache_keyboard = self._compute_attention_causal(
                q_blhd,
                k_blhd,
                v_blhd,
                repeat_kv=True,
                kv_cache=kv_cache_keyboard,
                block_mask=block_mask_keyboard,
                teacher_forcing=teacher_forcing,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )
            attn = rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)
            attn = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn

        return hidden_states, new_kv_cache_mouse, new_kv_cache_keyboard

    def _compute_attention_causal(
        self,
        q_blhd,
        k_blhd,
        v_blhd,
        repeat_kv=False,
        kv_cache=None,
        block_mask=None,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        if kv_cache is None:
            if teacher_forcing and no_kv_backprop_teacher_forcing:
                Tq = q_blhd.shape[1]
                past_k = k_blhd[:, : Tq // 2].detach()
                past_v = v_blhd[:, : Tq // 2].detach()
                present_k = k_blhd[:, Tq // 2 :]
                present_v = v_blhd[:, Tq // 2 :]
                k_blhd = torch.cat([past_k, present_k], dim=1)
                v_blhd = torch.cat([past_v, present_v], dim=1)
            if repeat_kv:
                repeat_factor = q_blhd.shape[0] // k_blhd.shape[0]
                k_blhd = k_blhd.repeat(repeat_factor, 1, 1, 1)
                v_blhd = v_blhd.repeat(repeat_factor, 1, 1, 1)
            q = q_blhd.transpose(1, 2)
            k = k_blhd.transpose(1, 2)
            v = v_blhd.transpose(1, 2)
            scale = q.shape[-1] ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if block_mask is not None:
                attn = attn.masked_fill(~block_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)
            x = x.transpose(1, 2).reshape(q_blhd.shape[0], q_blhd.shape[1], -1)
            return x, None
        else:
            new_kv_cache = kv_cache.update(k_blhd, v_blhd)
            k = new_kv_cache.k
            v = new_kv_cache.v
            if repeat_kv:
                repeat_factor = q_blhd.shape[0] // k.shape[0]
                k = k.repeat(repeat_factor, 1, 1, 1)
                v = v.repeat(repeat_factor, 1, 1, 1)
            kv_len = k.shape[1]
            # Valid cache positions: last `length` positions; mask out the rest for SDPA (True = mask out).
            mask_invalid = torch.arange(kv_len, device=k.device) < (
                kv_len - new_kv_cache.length.item()
            )
            mask_bool = mask_invalid.view(1, 1, 1, kv_len).expand(
                q_blhd.shape[0], 1, q_blhd.shape[1], kv_len
            )
            q = q_blhd.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = q.shape[-1] ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(mask_bool, float("-inf"))
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)
            x = x.transpose(1, 2).reshape(q_blhd.shape[0], q_blhd.shape[1], -1)
            return x, new_kv_cache
