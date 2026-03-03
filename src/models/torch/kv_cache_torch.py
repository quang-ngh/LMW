"""PyTorch KV cache for causal rollout."""

from dataclasses import dataclass

import torch


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor
    length: torch.Tensor

    def __init__(self, k: torch.Tensor, v: torch.Tensor, length: int | torch.Tensor = 0):
        self.k = k
        self.v = v
        self.length = torch.tensor(length, dtype=torch.int32, device=k.device) if isinstance(length, int) else length

    def update(self, k_blhd: torch.Tensor, v_blhd: torch.Tensor):
        kv_cache_size = self.k.shape[1]
        new_k = torch.cat([self.k, k_blhd], dim=1)[:, -kv_cache_size:]
        new_v = torch.cat([self.v, v_blhd], dim=1)[:, -kv_cache_size:]
        length_val = self.length.item() if isinstance(self.length, torch.Tensor) else self.length
        new_length = min(length_val + k_blhd.shape[1], kv_cache_size)
        new_length = torch.tensor(new_length, dtype=torch.int32, device=self.k.device)
        return KVCache(new_k, new_v, new_length)


@dataclass
class KVCacheDict:
    kv_cache: KVCache
    kv_cache_mouse: KVCache
    kv_cache_keyboard: KVCache

    def zeros_like(self):
        return KVCacheDict(
            kv_cache=KVCache(torch.zeros_like(self.kv_cache.k), torch.zeros_like(self.kv_cache.v), 0),
            kv_cache_mouse=KVCache(torch.zeros_like(self.kv_cache_mouse.k), torch.zeros_like(self.kv_cache_mouse.v), 0),
            kv_cache_keyboard=KVCache(torch.zeros_like(self.kv_cache_keyboard.k), torch.zeros_like(self.kv_cache_keyboard.v), 0),
        )
