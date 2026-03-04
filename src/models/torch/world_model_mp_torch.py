"""Full PyTorch port of Solaris MP world model (MPSelfAttention, blocks, OutputHead)."""

import math

import torch
import torch.nn as nn
from einops import rearrange

from .transformer import (
    Conv3d,
    WanLayerNorm,
    WanRMSNorm,
    WanI2VCrossAttention,
    MLPProj,
)
from .action_module_torch import ActionModuleTorch
from .kv_cache_torch import KVCache, KVCacheDict
from .transformer_utils_torch import (
    sinusoidal_embedding_1d,
    rope_params_mp,
    apply_rope_mp_torch,
)


class MPSelfAttentionTorch(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x_BTD,
        grid_sizes,
        freqs,
        eff_p,
        block_mask=None,
        kv_cache=None,
        current_start=0,
        bidirectional=False,
        teacher_forcing=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        q_BTHD = rearrange(self.norm_q(self.q(x_BTD)), "b s (n d) -> b s n d", n=self.num_heads)
        k_BTHD = rearrange(self.norm_k(self.k(x_BTD)), "b s (n d) -> b s n d", n=self.num_heads)
        v_BTHD = rearrange(self.v(x_BTD), "b s (n d) -> b s n d", n=self.num_heads)
        f0, s0 = grid_sizes[0], grid_sizes[1] * grid_sizes[2]
        new_kv_cache = None

        if bidirectional:
            roped_q = apply_rope_mp_torch(q_BTHD, grid_sizes, freqs, f0, s0)
            roped_k = apply_rope_mp_torch(k_BTHD, grid_sizes, freqs, f0, s0)
            scale = self.head_dim ** -0.5
            q = roped_q.transpose(1, 2)  # (B, N, S, D)
            k = roped_k.transpose(1, 2)
            v = v_BTHD.transpose(1, 2)
            # Use SDPA to avoid materializing full S×S attention matrix (OOM with 100k+ seq len)
            x_BTHD = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=block_mask, scale=scale, dropout_p=0.0
            ).transpose(1, 2)
        elif teacher_forcing:
            grid_sizes_tf = (grid_sizes[0] // 2, grid_sizes[1], grid_sizes[2])
            f_tf, s_tf = grid_sizes_tf[0], grid_sizes_tf[1] * grid_sizes_tf[2]
            q_BTHD = rearrange(q_BTHD, "b (r s) n d -> (b r) s n d", r=2)
            k_BTHD = rearrange(k_BTHD, "b (r s) n d -> (b r) s n d", r=2)
            roped_q = apply_rope_mp_torch(q_BTHD, grid_sizes_tf, freqs, f_tf, s_tf)
            roped_k = apply_rope_mp_torch(k_BTHD, grid_sizes_tf, freqs, f_tf, s_tf)
            roped_q = rearrange(roped_q, "(b r) s n d -> b (r s) n d", r=2)
            roped_k = rearrange(roped_k, "(b r) s n d -> b (r s) n d", r=2)
            if no_kv_backprop_teacher_forcing:
                Tq = roped_q.shape[1]
                past_k, past_v = roped_k[:, : Tq // 2].detach(), v_BTHD[:, : Tq // 2].detach()
                roped_k = torch.cat([past_k, roped_k[:, Tq // 2 :]], dim=1)
                v_BTHD = torch.cat([past_v, v_BTHD[:, Tq // 2 :]], dim=1)
            scale = self.head_dim ** -0.5
            q, k, v = roped_q.transpose(1, 2), roped_k.transpose(1, 2), v_BTHD.transpose(1, 2)
            x_BTHD = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=block_mask, scale=scale, dropout_p=0.0
            ).transpose(1, 2)
        elif kv_cache is None:
            roped_q = apply_rope_mp_torch(q_BTHD, grid_sizes, freqs, f0, s0)
            roped_k = apply_rope_mp_torch(k_BTHD, grid_sizes, freqs, f0, s0)
            scale = self.head_dim ** -0.5
            q, k, v = roped_q.transpose(1, 2), roped_k.transpose(1, 2), v_BTHD.transpose(1, 2)
            x_BTHD = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=block_mask, scale=scale, dropout_p=0.0
            ).transpose(1, 2)
        else:
            roped_q = apply_rope_mp_torch(q_BTHD, grid_sizes, freqs, f0, s0, current_start=current_start)
            roped_k = apply_rope_mp_torch(k_BTHD, grid_sizes, freqs, f0, s0, current_start=current_start)
            new_kv_cache = kv_cache.update(roped_k, v_BTHD)
            k, v = new_kv_cache.k, new_kv_cache.v
            kv_len = k.shape[1]
            # SDPA: True = attend, False = mask out. Valid keys are the last `length` positions.
            valid_len = new_kv_cache.length.item() if isinstance(new_kv_cache.length, torch.Tensor) else new_kv_cache.length
            attn_mask = (torch.arange(kv_len, device=k.device) >= (kv_len - valid_len))[None, None, None, :]
            scale = self.head_dim ** -0.5
            q = roped_q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            x_BTHD = torch.nn.functional.scaled_dot_product_attention(
                q, k_t, v_t, attn_mask=attn_mask, scale=scale, dropout_p=0.0
            ).transpose(1, 2)

        x_BTD = rearrange(x_BTHD, "b s n d -> b s (n d)")
        return self.o(x_BTD), new_kv_cache


class SolarisMPBlockTorch(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        action_config=None,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        action_config = action_config or {}
        self.action_model = ActionModuleTorch(**action_config, local_attn_size=local_attn_size) if action_config else None
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.self_attn = MPSelfAttentionTorch(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        context,
        player_embed_PD,
        block_mask=None,
        block_mask_mouse=None,
        block_mask_keyboard=None,
        num_frame_per_block=1,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        current_start=0,
        use_action_module=True,
        teacher_forcing=False,
        bidirectional=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        b, p = x.shape[0], x.shape[1]
        num_frames = e.shape[2]
        unpack = lambda t: rearrange(t, "b p (f s) c -> b p f s c", f=num_frames)
        pack = lambda t: rearrange(t, "b p f s c -> b p (f s) c", f=num_frames)
        x = unpack(x)
        mod = self.modulation.to(x.dtype)[:, None, None, :, :]
        e = (mod + e).split(1, dim=3)
        e = [e[i].squeeze(3).unsqueeze(3) for i in range(6)]  # (b,p,f,1,c) to broadcast with (b,p,f,s,c)
        #   Casting
        e = [element.to(x.dtype) for element in e]
        player_embed = player_embed_PD[None, :, None, None, :].to(x.dtype)
        x = x + player_embed
        inp = self.norm1(x) * (1 + e[1]) + e[0] 
        y_packed, new_kv_cache = self.self_attn(
            rearrange(inp, "b p f s c -> b (f p s) c"),
            grid_sizes,
            freqs,
            p,
            block_mask=block_mask,
            kv_cache=kv_cache,
            current_start=current_start,
            teacher_forcing=teacher_forcing,
            bidirectional=bidirectional,
            no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
        )
        y = rearrange(y_packed, "b (f p s) c -> b p f s c", f=num_frames, p=p)
        x = x + y * e[2]
        x_norm = self.norm3(x)
        x = x + rearrange(
            self.cross_attn(
                rearrange(x_norm, "b p f s c -> (b p) (f s) c"),
                rearrange(context, "b p f d -> (b p) f d"),
            ),
            "(b p) (f s) c -> b p f s c",
            b=b,
            f=num_frames,
        )
        if self.action_model is not None and use_action_module and (mouse_cond is not None or keyboard_cond is not None):
            x_flat, new_kv_mouse, new_kv_keyboard = self.action_model(
                rearrange(x.to(context.dtype), "b p f s c -> (b p) (f s) c"),
                grid_sizes[0],
                grid_sizes[1],
                grid_sizes[2],
                rearrange(mouse_cond, "b p f d -> (b p) f d") if mouse_cond is not None else None,
                rearrange(keyboard_cond, "b p f d -> (b p) f d") if keyboard_cond is not None else None,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                start_frame=current_start,
                num_frame_per_block=num_frame_per_block,
                teacher_forcing=teacher_forcing,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )
            x = rearrange(x_flat, "(b p) (f s) c -> b p f s c", b=b, f=num_frames)
        else:
            new_kv_mouse = kv_cache_mouse
            new_kv_keyboard = kv_cache_keyboard
        y_ffn = self.norm2(x).float() * (1 + e[4]) + e[3]
        y_ffn = self.ffn(y_ffn.to(x.dtype)) * e[5]
        y_ffn = y_ffn.to(x.dtype)
        x = x + y_ffn
        return pack(x), new_kv_cache, new_kv_mouse, new_kv_keyboard


class OutputHeadTorch(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(self, x, e):
        f = e.shape[2]
        mod = self.modulation.to(x.dtype)[:, None, None, :]
        e = (mod + e).split(1, dim=3)
        e = [e[i].squeeze(3).unsqueeze(3) for i in range(2)]  # (b,p,f,1,dim) to broadcast with (b,p,f,s,c)
        e = [element.to(x.dtype) for element in e]
        norm_x = self.norm(x)
        norm_x_frames = rearrange(norm_x, "b p (f s) c -> b p f s c", f=f)
        modulated = norm_x_frames * (1 + e[1]) + e[0]
        return self.head(modulated)


class SolarisMPModelTorch(nn.Module):
    """Full PyTorch Solaris MP world model."""

    def __init__(
        self,
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=36,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        local_attn_size=6,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        action_config=None,
        inject_sample_info=False,
        eps=1e-6,
        multiplayer_method="multiplayer_attn",
        num_players=2,
    ):
        super().__init__()
        self.model_type = model_type
        self.multiplayer_method = multiplayer_method
        self.num_players = num_players
        self.patch_size = list(patch_size)
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.eps = eps
        self.d = dim // num_heads
        self.expected_p_size = 1 if multiplayer_method == "concat_c" else 2
        c_factor = num_players if multiplayer_method == "concat_c" else 1
        in_dim_eff = in_dim * c_factor
        out_dim_eff = out_dim * c_factor
        self.out_dim = out_dim_eff
        self.num_frame_per_block = 1

        self.patch_embedding = Conv3d(
            in_dim_eff, dim, kernel_size=patch_size, strides=patch_size, padding="valid"
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        action_config_eff = dict(action_config) if action_config else {}
        if action_config and multiplayer_method == "concat_c":
            if "mouse_dim_in" in action_config_eff:
                action_config_eff["mouse_dim_in"] = action_config_eff["mouse_dim_in"] * num_players
            if "keyboard_dim_in" in action_config_eff:
                action_config_eff["keyboard_dim_in"] = action_config_eff["keyboard_dim_in"] * num_players
        self.blocks = nn.ModuleList([
            SolarisMPBlockTorch(
                dim,
                ffn_dim,
                num_heads,
                local_attn_size,
                sink_size,
                qk_norm,
                cross_attn_norm,
                action_config=action_config_eff,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.head = OutputHeadTorch(dim, out_dim_eff, patch_size, eps)
        eff_clip_dim = 1280 * num_players if multiplayer_method == "concat_c" else 1280
        self.img_emb = MLPProj(eff_clip_dim, dim)
        self.player_embed = nn.Embedding(num_players, dim)

    @staticmethod
    def get_causal_attn_mask(num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size=-1):
        i = torch.arange(num_q_blocks, device=None)[:, None]
        j = torch.arange(num_k_blocks, device=None)[None, :]
        if sliding_block_size == -1:
            block_mask = j <= i
        else:
            block_mask = (j >= i - sliding_block_size + 1) & (j <= i)
        block_mask = block_mask.repeat_interleave(q_block_size, dim=0).repeat_interleave(k_block_size, dim=1)
        return block_mask

    @staticmethod
    def get_causal_attn_mask_no_diagonals(num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size=-1):
        i = torch.arange(num_q_blocks, device=None)[:, None]
        j = torch.arange(num_k_blocks, device=None)[None, :]
        if sliding_block_size == -1:
            block_mask = j < i
        else:
            block_mask = (j >= i - sliding_block_size + 1) & (j < i)
        block_mask = block_mask.repeat_interleave(q_block_size, dim=0).repeat_interleave(k_block_size, dim=1)
        return block_mask

    @staticmethod
    def get_block_mask_teacher_forcing(num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size=-1):
        clean_clean = SolarisMPModelTorch.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
        )
        unclean_unclean = SolarisMPModelTorch.get_causal_attn_mask(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, 1
        )
        unclean_clean = SolarisMPModelTorch.get_causal_attn_mask_no_diagonals(
            num_q_blocks, num_k_blocks, q_block_size, k_block_size, sliding_block_size
        )
        clean_unclean = torch.zeros_like(unclean_clean)
        top = torch.cat([clean_clean, clean_unclean], dim=1)
        bottom = torch.cat([unclean_clean, unclean_unclean], dim=1)
        return torch.cat([top, bottom], dim=0)

    def unpatchify(self, x, grid_sizes):
        f, h, w = grid_sizes
        p1, p2, p3 = self.patch_size
        return rearrange(
            x,
            "b p f (h w) (p1 p2 p3 c) -> b p (f p1) (h p2) (w p3) c",
            f=f, h=h, w=w,
            p1=p1, p2=p2, p3=p3,
            c=self.out_dim,
        )

    def forward(
        self,
        frame_BPFHWC,
        t_BPT,
        visual_context_BPFD,
        cond_concat_BPFHWC,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        current_start=0,
        matrix_game_forward=True,
        teacher_forcing=False,
        bidirectional=False,
        no_kv_backprop_teacher_forcing=False,
    ):
        b, p = frame_BPFHWC.shape[0], frame_BPFHWC.shape[1]
        assert p == self.expected_p_size
        x_BPFHWC = torch.cat([frame_BPFHWC, cond_concat_BPFHWC], dim=-1)
        x_BPFHWC = rearrange(
            self.patch_embedding(rearrange(x_BPFHWC, "b p f h w c -> (b p) c f h w")),
            "(b p) c f h w -> b p f h w c",
            b=b,
        )
        grid_sizes = (x_BPFHWC.shape[2], x_BPFHWC.shape[3], x_BPFHWC.shape[4])
        x_BPFC = rearrange(x_BPFHWC, "b p f h w c -> b p (f h w) c")
        t_flat = t_BPT.flatten()
        e_BD = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat, device=t_flat.device).to(x_BPFC.dtype))
        e0 = rearrange(
            self.time_projection(e_BD),
            "(b p f) (r d) -> b p f r d",
            b=t_BPT.shape[0],
            p=p,
            r=6,
        )
        context = rearrange(
            self.img_emb(rearrange(visual_context_BPFD, "b p f d -> (b p) f d")),
            "(b p) f d -> b p f d",
            b=b,
        )
        n_frames, num_patches = grid_sizes[0], grid_sizes[1] * grid_sizes[2]
        player_embed_PD = self.player_embed(torch.arange(p, device=x_BPFC.device)).to(x_BPFC.dtype)
        # Use the attention layer's head_dim so RoPE matches q/k after load_state_dict(strict=False)
        head_dim = self.blocks[0].self_attn.head_dim
        freqs = rope_params_mp(head_dim, device=x_BPFC.device)

        if bidirectional:
            block_mask = block_mask_mouse = block_mask_keyboard = None
        elif kv_cache is None and not teacher_forcing:
            block_mask = self.get_causal_attn_mask(
                n_frames, n_frames, num_patches * p, num_patches * p, self.local_attn_size
            ).to(x_BPFC.device)
            block_mask_mouse = self.get_causal_attn_mask(
                n_frames, n_frames, 1, 1, self.local_attn_size
            ).to(x_BPFC.device)
            block_mask_keyboard = self.get_causal_attn_mask(
                n_frames, n_frames, 1, 1, self.local_attn_size
            ).to(x_BPFC.device)
        elif kv_cache is None and teacher_forcing:
            block_mask = self.get_block_mask_teacher_forcing(
                n_frames // 2, n_frames // 2, num_patches * p, num_patches * p, self.local_attn_size
            ).to(x_BPFC.device)
            block_mask_mouse = self.get_block_mask_teacher_forcing(
                n_frames // 2, n_frames // 2, 1, 1, self.local_attn_size
            ).to(x_BPFC.device)
            block_mask_keyboard = self.get_block_mask_teacher_forcing(
                n_frames // 2, n_frames // 2, 1, 1, self.local_attn_size
            ).to(x_BPFC.device)
        else:
            block_mask = block_mask_mouse = block_mask_keyboard = None

        action_module_zipper = [True] * 15 + [False] * 15 if matrix_game_forward else [True] * 30
        kv_c, kv_m, kv_k = kv_cache, kv_cache_mouse, kv_cache_keyboard
        for i, block in enumerate(self.blocks):
            x_BPFC, kv_c, kv_m, kv_k = block(
                x_BPFC,
                e0,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                player_embed_PD=player_embed_PD,
                block_mask=block_mask,
                block_mask_mouse=block_mask_mouse,
                block_mask_keyboard=block_mask_keyboard,
                num_frame_per_block=self.num_frame_per_block,
                mouse_cond=mouse_cond,
                keyboard_cond=keyboard_cond,
                kv_cache=kv_c,
                kv_cache_mouse=kv_m,
                kv_cache_keyboard=kv_k,
                current_start=current_start,
                use_action_module=action_module_zipper[i] if i < len(action_module_zipper) else True,
                teacher_forcing=teacher_forcing,
                bidirectional=bidirectional,
                no_kv_backprop_teacher_forcing=no_kv_backprop_teacher_forcing,
            )

        e_head = rearrange(
            e_BD,
            "(b p f) (r d) -> b p f r d",
            b=t_BPT.shape[0],
            p=t_BPT.shape[1],
            r=1,
        )

        if isinstance(e_head, list) or isinstance(e_head, tuple):
            e_head = [element.to(x_BPFC.dtype) for element in e_head]
        else:
            e_head = e_head.to(x_BPFC.dtype)

        x_BPFC = self.head(x_BPFC, e_head)
        out = self.unpatchify(x_BPFC, grid_sizes)
        new_kv = (kv_c, kv_m, kv_k) if kv_cache is not None else (None, None, None)
        return out, new_kv[0], new_kv[1], new_kv[2]

    def initialize_kv_cache(self, batch_size, latent_height, latent_width, dtype=torch.bfloat16, num_players=2):
        frames_block_size = self.local_attn_size * latent_height * latent_width * num_players
        head_dim = self.dim // self.num_heads
        action_model = self.blocks[0].action_model
        actions_head_num = action_model.heads_num if action_model else 16
        mouse_head_dim = action_model.mouse_head_dim if action_model else 64
        keyboard_head_dim = action_model.keyboard_head_dim if action_model else 64
        caches = []
        for _ in range(self.num_layers):
            caches.append(KVCacheDict(
                kv_cache=KVCache(
                    torch.zeros(batch_size, frames_block_size, self.num_heads, head_dim, dtype=dtype),
                    torch.zeros(batch_size, frames_block_size, self.num_heads, head_dim, dtype=dtype),
                ),
                kv_cache_mouse=KVCache(
                    torch.zeros(
                        num_players * batch_size * latent_height * latent_width,
                        self.local_attn_size,
                        actions_head_num,
                        mouse_head_dim,
                        dtype=dtype,
                    ),
                    torch.zeros(
                        num_players * batch_size * latent_height * latent_width,
                        self.local_attn_size,
                        actions_head_num,
                        mouse_head_dim,
                        dtype=dtype,
                    ),
                ),
                kv_cache_keyboard=KVCache(
                    torch.zeros(
                        num_players * batch_size,
                        self.local_attn_size,
                        actions_head_num,
                        keyboard_head_dim,
                        dtype=dtype,
                    ),
                    torch.zeros(
                        num_players * batch_size,
                        self.local_attn_size,
                        actions_head_num,
                        keyboard_head_dim,
                        dtype=dtype,
                    ),
                ),
            ))
        return caches
