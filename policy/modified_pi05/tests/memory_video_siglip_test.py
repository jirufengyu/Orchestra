import inspect
from unittest import mock

import pytest
import torch
import torch.nn as nn

from openpi.models_pytorch import memory_video_siglip as mem_video
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


def _video_encoder_installed() -> bool:
  try:
      from transformers.models.siglip.modeling_siglip import SiglipEncoder

      params = inspect.signature(SiglipEncoder.forward).parameters
      return "num_frames" in params and "drop_past_after_layer" in params
  except ImportError:
      return False


class _MockEncoderLayer(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, num_heads, batch_first=True)


def test_temporal_pe_last_frame_is_zero():
    pe = mem_video.build_sinusoidal_temporal_pe(max_frames=6, width=64)
    assert pe.shape == (6, 64)
    assert torch.allclose(pe[-1], torch.zeros(64))


def test_temporal_pe_window_uses_current_frame_slot():
    pe = mem_video.build_sinusoidal_temporal_pe(max_frames=6, width=32)
    window = pe[-1:]
    assert torch.allclose(window, torch.zeros(1, 32))


def test_select_current_frame_tokens_k1_is_identity():
    hidden_states = torch.randn(4, 16, 32)
    out = mem_video.select_current_frame_tokens(hidden_states, num_frames=1, batch_size=4)
    assert torch.allclose(hidden_states, out)


def test_apply_mem_temporal_attention_k1_is_identity():
    layer = _MockEncoderLayer(width=32, num_heads=4)
    hidden_states = torch.randn(4, 16, 32)
    pe = mem_video.build_sinusoidal_temporal_pe(6, 32)

    out = mem_video.apply_mem_temporal_attention(
        hidden_states,
        layer,
        num_frames=1,
        batch_size=4,
        temporal_pe=pe,
        frame_mask=torch.ones(4, 1, dtype=torch.bool),
    )
    assert torch.allclose(hidden_states, out)


def test_embed_image_with_memory_k1_delegates_to_embed_image():
    model = mock.Mock(spec=PaliGemmaWithExpertModel)
    model.embed_image = mock.Mock(return_value=torch.ones(2, 256, 1024))
    model.paligemma = mock.Mock()

    wrapper = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    wrapper.paligemma = model.paligemma
    wrapper.embed_image = model.embed_image

    images = torch.randn(2, 1, 3, 224, 224)
    out = PaliGemmaWithExpertModel.embed_image_with_memory(
        wrapper,
        images,
        torch.ones(2, 1, dtype=torch.bool),
        temporal_pe=torch.zeros(6, 1152),
    )

    model.embed_image.assert_called_once()
    called = model.embed_image.call_args.args[0]
    assert called.shape == (2, 3, 224, 224)
    assert torch.allclose(out, torch.ones(2, 256, 1024))


@pytest.mark.skipif(not _video_encoder_installed(), reason="transformers_replace video encoder not installed")
def test_siglip_video_k1_matches_single_frame_forward():
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_channels=3,
        image_size=224,
        patch_size=14,
    )
    model = SiglipVisionTransformer(config)
    model.eval()

    pixel_values = torch.randn(2, 3, 224, 224)
    temporal_pe = mem_video.build_sinusoidal_temporal_pe(max_frames=6, width=config.hidden_size)
    frame_mask = torch.ones(2, 1, dtype=torch.bool)

    with torch.no_grad():
        single = model(pixel_values).last_hidden_state
        video_k1 = model(
            pixel_values,
            num_frames=1,
            frame_mask=frame_mask,
            temporal_pe=temporal_pe,
            temporal_interval=4,
        ).last_hidden_state

    assert single.shape == video_k1.shape
    assert torch.allclose(single, video_k1, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _video_encoder_installed(), reason="transformers_replace video encoder not installed")
def test_siglip_video_k2_changes_output_shape_and_values():
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_channels=3,
        image_size=224,
        patch_size=14,
    )
    model = SiglipVisionTransformer(config)
    model.eval()

    batch_size = 2
    num_frames = 2
    pixel_values = torch.randn(batch_size * num_frames, 3, 224, 224)
    temporal_pe = mem_video.build_sinusoidal_temporal_pe(max_frames=6, width=config.hidden_size)
    frame_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)

    with torch.no_grad():
        single = model(pixel_values[:batch_size]).last_hidden_state
        video = model(
            pixel_values,
            num_frames=num_frames,
            frame_mask=frame_mask,
            temporal_pe=temporal_pe,
            temporal_interval=4,
        ).last_hidden_state

    assert video.shape == single.shape
    assert not torch.allclose(single, video[0:1].expand_as(single), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not _video_encoder_installed(), reason="transformers_replace video encoder not installed")
def test_siglip_video_k2_drop_past_tokens_keeps_current_frame_shape():
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_channels=3,
        image_size=224,
        patch_size=14,
    )
    model = SiglipVisionTransformer(config)
    model.eval()

    batch_size = 2
    num_frames = 2
    pixel_values = torch.randn(batch_size * num_frames, 3, 224, 224)
    temporal_pe = mem_video.build_sinusoidal_temporal_pe(max_frames=6, width=config.hidden_size)
    frame_mask = torch.ones(batch_size, num_frames, dtype=torch.bool)

    with torch.no_grad():
        single = model(pixel_values[:batch_size]).last_hidden_state
        video = model(
            pixel_values,
            num_frames=num_frames,
            frame_mask=frame_mask,
            temporal_pe=temporal_pe,
            temporal_interval=2,
            drop_past_after_layer=2,
        ).last_hidden_state

    assert video.shape == single.shape
