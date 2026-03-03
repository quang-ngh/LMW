"""PyTorch CLIP vision encoder matching JAX state dict structure for converted checkpoints."""

import math

import torch
import torch.nn as nn
from einops import rearrange


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, causal=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, c = x.shape
        n, d = self.num_heads, self.head_dim
        qkv = self.to_qkv(x).reshape(b, s, 3, n, d)
        q, k, v = qkv.unbind(2)
        scale = d ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.causal:
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = rearrange(x, "b s n d -> b s (n d)")
        return self.proj(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, num_heads, post_norm=False, causal=False, activation="gelu", norm_eps=1e-5):
        super().__init__()
        self.post_norm = post_norm
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        mlp_ratio=4,
        out_dim=512,
        num_heads=12,
        num_layers=12,
        pool_type="token",
        pre_norm=True,
        post_norm=False,
        activation="gelu",
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.pool_type = pool_type
        self.num_patches = (image_size // patch_size) ** 2
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )
        if pool_type in ("token", "token_fc"):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        pos_embed_len = self.num_patches + (1 if pool_type in ("token", "token_fc") else 0)
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, pos_embed_len, dim))
        self.pre_norm = nn.LayerNorm(dim, eps=norm_eps) if pre_norm else nn.Identity()
        self.transformer = nn.ModuleList([
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False, activation, norm_eps)
            for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(dim, eps=norm_eps)
        if pool_type == "token":
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == "token_fc":
            self.head = nn.Linear(dim, out_dim)
        else:
            self.head = nn.Identity()

    def forward(self, x, interpolation=False, use_31_block=False):
        b = x.shape[0]
        x = rearrange(x, "b c h w -> b h w c")
        x = self.patch_embedding(x.permute(0, 3, 1, 2))
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.pool_type in ("token", "token_fc"):
            cls_token = self.cls_embedding.expand(b, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.pre_norm(x)
        blocks = self.transformer[:-1] if use_31_block else self.transformer
        for block in blocks:
            x = block(x)
        x = self.post_norm(x)
        if self.pool_type == "token":
            x = x[:, 0] @ self.head
        elif self.pool_type == "token_fc":
            x = self.head(x[:, 0])
        return x


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def clip_preprocess_videos_torch(videos_bcfhw, image_size=224, device=None):
    """videos_bcfhw: (B, C, F, H, W)."""
    b, c, f, h, w = videos_bcfhw.shape
    x = videos_bcfhw.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
    x = x * 0.5 + 0.5
    x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode="bicubic", align_corners=False)
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.view(b, f, c, image_size, image_size).permute(0, 2, 1, 3, 4)
    return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        vision_pre_norm=True,
        vision_post_norm=False,
        activation="gelu",
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            norm_eps=norm_eps,
        )

    def encode_video(self, videos_bcfhw):
        """videos_bcfhw: (B, C, F, H, W) -> (B, F, D)."""
        videos = clip_preprocess_videos_torch(videos_bcfhw, 224)
        b, c, f, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        out = self.model(videos, use_31_block=True)
        return out.view(b, f, -1)
