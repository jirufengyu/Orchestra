"""MEM-style space-time separable attention helpers for SigLIP ViT."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_sinusoidal_temporal_pe(max_frames: int, width: int) -> torch.Tensor:
    """Temporal PE where the current frame (last index) gets a zero embedding."""
    pe = torch.zeros(max_frames, width)
    if max_frames <= 1:
        return pe
    half = width // 2
    if half == 0:
        return pe
    fraction = torch.linspace(0.0, 1.0, half)
    period = 1.0 * (10000.0**fraction)
    for idx in range(max_frames):
        age = max_frames - 1 - idx
        if age == 0:
            continue
        angles = age / period * 2.0 * math.pi
        pe[idx, :half] = torch.sin(angles)
        pe[idx, half:] = torch.cos(angles)
    return pe


def _build_temporal_attention_mask(
    num_frames: int,
    frame_mask: torch.Tensor | None,
    batch_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build additive attention mask for causal temporal attention over frames."""
    min_dtype = torch.finfo(dtype).min
    mask = torch.triu(torch.full((num_frames, num_frames), min_dtype, device=device, dtype=dtype), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_tokens, 1, num_frames, num_frames).clone()
    if frame_mask is not None:
        frame_mask = frame_mask.to(device=device, dtype=torch.bool)
        batch_size = frame_mask.shape[0]
        num_patches = batch_tokens // batch_size
        key_valid = frame_mask[:, None, :].expand(batch_size, num_patches, num_frames).reshape(batch_tokens, num_frames)
        key_invalid = (~key_valid)[:, None, None, :]
        mask = mask.masked_fill(key_invalid, min_dtype)
    return mask


def apply_mem_temporal_attention(
    hidden_states: torch.Tensor,
    encoder_layer: nn.Module,
    *,
    num_frames: int,
    batch_size: int,
    temporal_pe: torch.Tensor | None,
    frame_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Add causal temporal attention over frames at the same patch index (MEM-style)."""
    if num_frames <= 1:
        return hidden_states

    _, num_patches, embed_dim = hidden_states.shape
    x = hidden_states.reshape(batch_size, num_frames, num_patches, embed_dim)
    x = x.permute(0, 2, 1, 3).reshape(batch_size * num_patches, num_frames, embed_dim)

    if temporal_pe is not None:
        window_pe = temporal_pe[-num_frames:].to(device=x.device, dtype=x.dtype)
        x = x + window_pe.unsqueeze(0)

    residual = x
    x = encoder_layer.layer_norm1(x)
    attn_mask = _build_temporal_attention_mask(
        num_frames,
        frame_mask,
        batch_size * num_patches,
        x.device,
        x.dtype,
    )
    attn_out, _ = encoder_layer.self_attn(x, attention_mask=attn_mask)
    x = residual + attn_out

    return x.reshape(batch_size, num_patches, num_frames, embed_dim).permute(0, 2, 1, 3).reshape(
        batch_size * num_frames, num_patches, embed_dim
    )


def select_current_frame_tokens(hidden_states: torch.Tensor, *, num_frames: int, batch_size: int) -> torch.Tensor:
    """Keep only the current-frame patch tokens after video encoding."""
    if num_frames <= 1:
        return hidden_states
    _, num_patches, embed_dim = hidden_states.shape
    return (
        hidden_states.reshape(batch_size, num_frames, num_patches, embed_dim)[:, -1, :, :].reshape(
            batch_size, num_patches, embed_dim
        )
    )
