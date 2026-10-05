"""HEVM: hierarchical episodic visual memory modules for pi0.5 PyTorch."""

from __future__ import annotations

import torch
from torch import nn


# Scope indices for KeyframeResampler / gist resampler.
EPISODIC_SCOPE_ANCHOR = 0
EPISODIC_SCOPE_TASK = 1
EPISODIC_SCOPE_GIST = 2


class MemoryCrossAttnFusion(nn.Module):
    """Mem-0 style fusion: current visual tokens cross-attend compact memory tokens."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)

    def forward(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if memory_tokens is None or memory_tokens.shape[1] == 0:
            return query_tokens

        out_dtype = query_tokens.dtype
        module_dtype = self.query_norm.weight.dtype
        query_tokens = query_tokens.to(dtype=module_dtype)
        memory_tokens = memory_tokens.to(dtype=module_dtype)
        key_padding_mask = None
        if memory_mask is not None:
            memory_mask = memory_mask.to(device=memory_tokens.device, dtype=torch.bool)
            if not memory_mask.any():
                return query_tokens.to(dtype=out_dtype)
            key_padding_mask = ~memory_mask

        attn_out, _ = self.attn(
            self.query_norm(query_tokens),
            self.memory_norm(memory_tokens),
            self.memory_norm(memory_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return (query_tokens + attn_out).to(dtype=out_dtype)
