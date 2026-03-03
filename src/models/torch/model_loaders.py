"""Torch model constructors for loading converted checkpoints (same config as JAX)."""

from models.torch.clip_torch import CLIPModel as TorchCLIPModel
from models.torch.wan_vae_torch import WanVAETorch
from src.models.torch.world_model_mp_torch import SolarisMPModelTorch


def get_torch_clip_model():
    config = dict(
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
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
        norm_eps=1e-5,
    )
    return TorchCLIPModel(**config)


def get_torch_vae_model():
    cfg = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    return WanVAETorch(**cfg)


def get_torch_world_model(network_config):
    """Build torch Solaris MP model from the same config as JAX."""
    p = network_config.params
    action_config = getattr(p, "action_config", None)
    if action_config is not None and not isinstance(action_config, dict):
        action_config = vars(action_config)
    return SolarisMPModelTorch(
        model_type="i2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=p.in_dim,
        dim=p.dim,
        ffn_dim=p.ffn_dim,
        freq_dim=p.freq_dim,
        text_dim=p.text_dim,
        out_dim=p.out_dim,
        num_heads=p.num_heads,
        num_layers=p.num_layers,
        local_attn_size=p.local_attn_size,
        sink_size=p.sink_size,
        qk_norm=p.qk_norm,
        cross_attn_norm=p.cross_attn_norm,
        action_config=action_config,
        inject_sample_info=p.inject_sample_info,
        eps=p.eps,
        multiplayer_method=getattr(p, "multiplayer_method", "multiplayer_attn"),
        num_players=getattr(p, "num_players", 2),
    )
