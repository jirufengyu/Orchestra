import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.episodic_memory_pytorch as _episodic_memory
import openpi.models_pytorch.memory_video_siglip as _memory_video_siglip
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
from openpi.models_pytorch.progress_head_pytorch import ProgressEvaluatorHead


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    if not isinstance(device, torch.device):
        device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class KeyframeResampler(nn.Module):
    """Compresses variable-length keyframe features into a fixed number of prefix tokens."""

    def __init__(self, width: int, num_tokens: int, num_heads: int, num_scopes: int = 2):
        super().__init__()
        self.num_tokens = num_tokens
        self.queries = nn.Parameter(torch.randn(num_scopes, num_tokens, width) * 0.02)
        self.scope_embeddings = nn.Parameter(torch.randn(num_scopes, width) * 0.02)
        self.query_norm = nn.LayerNorm(width)
        self.frame_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, frame_features: torch.Tensor, frame_mask: torch.Tensor, *, scope_index: int):
        bsize = frame_features.shape[0]
        dtype = self.queries.dtype
        frame_features = frame_features.to(dtype=dtype)
        queries = self.queries[scope_index].to(dtype=dtype)[None, :, :].expand(bsize, -1, -1)
        frame_features = frame_features + self.scope_embeddings[scope_index].to(dtype=dtype)

        frame_mask = frame_mask.to(device=frame_features.device, dtype=torch.bool)
        has_keyframe = frame_mask.any(dim=1)
        key_padding_mask = ~frame_mask
        # MultiheadAttention returns NaNs when every key is masked. For rows without
        # keyframes, unmask the zero features and mask the produced prefix tokens later.
        key_padding_mask = torch.where(has_keyframe[:, None], key_padding_mask, torch.zeros_like(key_padding_mask))

        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.frame_norm(frame_features),
            self.frame_norm(frame_features),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = queries + attn_out
        out = out + self.mlp(self.out_norm(out))
        token_mask = has_keyframe[:, None].expand(bsize, self.num_tokens)
        out = out * token_mask[:, :, None].to(dtype=out.dtype)
        return out, token_mask


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.use_keyframe_conditioning = getattr(config, "use_keyframe_conditioning", False)
        self.use_short_horizon_memory = getattr(config, "use_short_horizon_memory", False)
        self.use_anchor_temporal_memory = getattr(config, "use_anchor_temporal_memory", False)
        self.use_head_only_vision = getattr(config, "use_head_only_vision", False)
        self.anchor_temporal_num_frames = int(getattr(config, "anchor_temporal_num_frames", 2))
        self.use_episodic_memory = getattr(config, "use_episodic_memory", False)
        self.episodic_memory_cross_attn = getattr(config, "episodic_memory_cross_attn", False)
        self.anchor_memory_cross_attn = getattr(config, "anchor_memory_cross_attn", False)
        self.base_patch_dropout_prob = getattr(config, "base_patch_dropout_prob", 0.0)
        self.base_patch_dropout_ratio = getattr(config, "base_patch_dropout_ratio", 0.0)
        self.wrist_patch_dropout_prob = getattr(config, "wrist_patch_dropout_prob", 0.0)
        self.wrist_patch_dropout_ratio = getattr(config, "wrist_patch_dropout_ratio", 0.0)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_expert_depth = action_expert_config.depth
        self.use_internal_guidance = bool(getattr(config, "use_internal_guidance", False))
        self.ig_layer_idx = getattr(config, "ig_layer_idx", None)
        self.ig_loss_weight = float(getattr(config, "ig_loss_weight", 0.5))
        self.ig_scale = float(getattr(config, "ig_scale", 1.2))
        self.ig_min_t = float(getattr(config, "ig_min_t", 0.3))
        self.ig_max_t = float(getattr(config, "ig_max_t", 1.0))
        if self.use_internal_guidance:
            if self.ig_layer_idx is None:
                raise ValueError("ig_layer_idx must be set when use_internal_guidance is True.")
            if not 0 <= self.ig_layer_idx < self.action_expert_depth:
                raise ValueError(
                    f"ig_layer_idx={self.ig_layer_idx} is out of range for "
                    f"{config.action_expert_variant} depth={self.action_expert_depth}."
                )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        if self.use_internal_guidance:
            self.ig_action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        if self.use_keyframe_conditioning:
            if getattr(config, "keyframe_encoder_type", "paligemma_pool") != "paligemma_pool":
                raise ValueError("Only keyframe_encoder_type='paligemma_pool' is currently implemented")
            num_scopes = 3 if self.use_episodic_memory else 2
            self.keyframe_resampler = KeyframeResampler(
                width=paligemma_config.width,
                num_tokens=getattr(config, "keyframe_token_count", 4),
                num_heads=paligemma_config.num_heads,
                num_scopes=num_scopes,
            )
            if self.use_episodic_memory:
                self.gist_resampler = KeyframeResampler(
                    width=paligemma_config.width,
                    num_tokens=getattr(config, "gist_token_count", 8),
                    num_heads=paligemma_config.num_heads,
                    num_scopes=1,
                )

        if (self.use_episodic_memory and self.episodic_memory_cross_attn) or self.anchor_memory_cross_attn:
            self.memory_cross_attn = _episodic_memory.MemoryCrossAttnFusion(
                paligemma_config.width,
                paligemma_config.num_heads,
            )

        if self.use_short_horizon_memory or self.use_anchor_temporal_memory:
            vision_width = int(self.paligemma_with_expert.paligemma.config.vision_config.hidden_size)
            if self.use_short_horizon_memory:
                self.register_buffer(
                    "memory_temporal_pe",
                    _memory_video_siglip.build_sinusoidal_temporal_pe(
                        getattr(config, "memory_num_frames", 6),
                        vision_width,
                    ),
                    persistent=False,
                )
                self.memory_temporal_interval = int(getattr(config, "memory_temporal_interval", 4))
                self.memory_drop_past_after_layer = getattr(config, "memory_drop_past_after_layer", None)
            if self.use_anchor_temporal_memory:
                self.register_buffer(
                    "anchor_temporal_pe",
                    _memory_video_siglip.build_sinusoidal_temporal_pe(
                        self.anchor_temporal_num_frames,
                        vision_width,
                    ),
                    persistent=False,
                )
                self.anchor_temporal_interval = int(getattr(config, "memory_temporal_interval", 4))
                self.anchor_drop_past_after_layer = getattr(config, "memory_drop_past_after_layer", None)

        self.use_progress_head = getattr(config, "use_progress_head", False)
        self.num_progress_bins = int(getattr(config, "num_progress_bins", 101))
        self.progress_loss_weight = float(getattr(config, "progress_loss_weight", 1.0))
        self.subtask_done_loss_weight = float(getattr(config, "subtask_done_loss_weight", 1.0))
        self.done_focal_gamma = float(getattr(config, "done_focal_gamma", 0.0))
        done_focal_alpha = getattr(config, "done_focal_alpha", None)
        self.done_focal_alpha = float(done_focal_alpha) if done_focal_alpha is not None else None
        if self.use_progress_head:
            self.progress_query = nn.Parameter(torch.randn(1, 1, action_expert_config.width) * 0.02)
            self.progress_evaluator = ProgressEvaluatorHead(
                action_expert_config.width,
                num_progress_bins=self.num_progress_bins,
                dropout=float(getattr(config, "progress_head_dropout", 0.1)),
            )

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _apply_patch_token_dropout(
        self,
        token_mask: torch.Tensor,
        *,
        dropout_prob: float,
        dropout_ratio: float,
    ) -> torch.Tensor:
        """Randomly masks image patch tokens during training only."""
        if not self.training or dropout_prob <= 0 or dropout_ratio <= 0:
            return token_mask

        bsize, num_tokens = token_mask.shape
        device = token_mask.device
        apply_dropout = torch.rand(bsize, device=device) < dropout_prob
        if not apply_dropout.any():
            return token_mask

        keep_tokens = torch.rand(bsize, num_tokens, device=device) >= dropout_ratio
        keep_tokens = torch.where(apply_dropout[:, None], keep_tokens, torch.ones_like(keep_tokens, dtype=torch.bool))
        out_mask = token_mask & keep_tokens

        # Keep at least one token for valid images to avoid fully masked attention rows.
        needs_token = apply_dropout & token_mask.any(dim=1) & ~out_mask.any(dim=1)
        if needs_token.any():
            row_idx = torch.where(needs_token)[0]
            token_idx = torch.randint(num_tokens, (row_idx.shape[0],), device=device)
            out_mask = out_mask.clone()
            out_mask[row_idx, token_idx] = True

        return out_mask

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        image_keys = ("base_0_rgb",) if self.use_head_only_vision else IMAGE_KEYS
        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train, image_keys=image_keys
        )
        keyframes = None
        if self.use_keyframe_conditioning or self.use_anchor_temporal_memory:
            keyframes = {
                "episode": (observation.episode_keyframes, observation.episode_keyframe_mask),
                "task": (observation.task_keyframes, observation.task_keyframe_mask),
            }
            if self.use_episodic_memory:
                keyframes["gist"] = (
                    getattr(observation, "gist_keyframes", None),
                    getattr(observation, "gist_keyframe_mask", None),
                )
        memory = None
        if self.use_short_horizon_memory:
            memory = (getattr(observation, "memory_frames", None), observation.memory_frame_mask)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            keyframes,
            memory,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_keyframes(self, keyframes, keyframe_mask, *, scope_index: int):
        if not self.use_keyframe_conditioning or keyframes is None:
            return None, None
        if keyframes.ndim != 5:
            raise ValueError(f"keyframes must have shape [B, K, C, H, W] or [B, K, H, W, C], got {keyframes.shape}")

        bsize, num_keyframes = keyframes.shape[:2]
        flat = keyframes.reshape(bsize * num_keyframes, *keyframes.shape[2:])
        if flat.shape[1] != 3 and flat.shape[-1] == 3:
            flat = flat.permute(0, 3, 1, 2)

        def keyframe_embed_func(flat_keyframes):
            return self.paligemma_with_expert.embed_image(flat_keyframes)

        patch_tokens = self._apply_checkpoint(keyframe_embed_func, flat)
        frame_features = patch_tokens.mean(dim=1).reshape(bsize, num_keyframes, -1)
        if keyframe_mask is None:
            keyframe_mask = torch.ones((bsize, num_keyframes), dtype=torch.bool, device=keyframes.device)
        keyframe_emb, keyframe_token_mask = self.keyframe_resampler(frame_features, keyframe_mask, scope_index=scope_index)
        return keyframe_emb.to(dtype=patch_tokens.dtype), keyframe_token_mask

    def embed_gist_keyframes(self, gist_keyframes, gist_mask):
        if not self.use_episodic_memory or gist_keyframes is None:
            return None, None
        if gist_keyframes.ndim != 5:
            raise ValueError(f"gist_keyframes must have shape [B, K, C, H, W] or [B, K, H, W, C], got {gist_keyframes.shape}")

        bsize, num_keyframes = gist_keyframes.shape[:2]
        flat = gist_keyframes.reshape(bsize * num_keyframes, *gist_keyframes.shape[2:])
        if flat.shape[1] != 3 and flat.shape[-1] == 3:
            flat = flat.permute(0, 3, 1, 2)

        def gist_embed_func(flat_keyframes):
            return self.paligemma_with_expert.embed_image(flat_keyframes)

        patch_tokens = self._apply_checkpoint(gist_embed_func, flat)
        frame_features = patch_tokens.mean(dim=1).reshape(bsize, num_keyframes, -1)
        if gist_mask is None:
            gist_mask = torch.ones((bsize, num_keyframes), dtype=torch.bool, device=gist_keyframes.device)
        gist_emb, gist_token_mask = self.gist_resampler(frame_features, gist_mask, scope_index=0)
        return gist_emb.to(dtype=patch_tokens.dtype), gist_token_mask

    def _fuse_image_tokens_with_memory(
        self,
        img_emb: torch.Tensor,
        anchor_tokens: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
        gist_tokens: torch.Tensor | None,
        gist_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not hasattr(self, "memory_cross_attn"):
            return img_emb
        if anchor_tokens is not None:
            img_emb = self.memory_cross_attn(img_emb, anchor_tokens, anchor_mask)
        if self.use_episodic_memory and gist_tokens is not None:
            img_emb = self.memory_cross_attn(img_emb, gist_tokens, gist_mask)
        return img_emb

    def embed_memory_frames(self, memory_frames, memory_frame_mask):
        if not self.use_short_horizon_memory or memory_frames is None:
            return None

        def memory_embed_func(frames, mask):
            return self.paligemma_with_expert.embed_image_with_memory(
                frames,
                mask,
                temporal_pe=self.memory_temporal_pe,
                temporal_interval=self.memory_temporal_interval,
                drop_past_after_layer=self.memory_drop_past_after_layer,
            )

        return self._apply_checkpoint(memory_embed_func, memory_frames, memory_frame_mask)

    def embed_anchor_temporal_frames(self, episode_keyframes, episode_keyframe_mask, current_image):
        """Fuse episode-start anchor + current frame via SigLIP temporal attention."""
        if not self.use_anchor_temporal_memory or episode_keyframes is None:
            return None
        if episode_keyframes.ndim != 5:
            raise ValueError(
                f"episode_keyframes must have shape [B, K, C, H, W] or [B, K, H, W, C], got {episode_keyframes.shape}"
            )

        bsize = episode_keyframes.shape[0]
        anchor = episode_keyframes[:, :1, ...]
        if current_image.ndim == 4:
            current = current_image[:, None, ...]
        else:
            current = current_image

        if anchor.shape[1] != 3 and anchor.shape[-1] == 3:
            anchor = anchor.permute(0, 1, 4, 2, 3)
        if current.shape[1] != 3 and current.shape[-1] == 3:
            current = current.permute(0, 1, 4, 2, 3)

        frames = torch.cat([anchor, current], dim=1)
        if episode_keyframe_mask is None:
            frame_mask = torch.ones((bsize, frames.shape[1]), dtype=torch.bool, device=frames.device)
        else:
            anchor_valid = episode_keyframe_mask[:, :1].to(device=frames.device, dtype=torch.bool)
            frame_mask = torch.cat(
                [anchor_valid, torch.ones((bsize, 1), dtype=torch.bool, device=frames.device)],
                dim=1,
            )

        def anchor_embed_func(frames_tensor, mask):
            return self.paligemma_with_expert.embed_image_with_memory(
                frames_tensor,
                mask,
                temporal_pe=self.anchor_temporal_pe,
                temporal_interval=self.anchor_temporal_interval,
                drop_past_after_layer=self.anchor_drop_past_after_layer,
            )

        return self._apply_checkpoint(anchor_embed_func, frames, frame_mask)

    def _append_image_tokens(
        self,
        embs: list,
        pad_masks: list,
        att_masks: list,
        img_emb: torch.Tensor,
        img_mask: torch.Tensor,
        *,
        image_index: int,
    ) -> None:
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        img_token_mask = img_mask[:, None].expand(bsize, num_img_embs)
        if image_index == 0:
            img_token_mask = self._apply_patch_token_dropout(
                img_token_mask,
                dropout_prob=self.base_patch_dropout_prob,
                dropout_ratio=self.base_patch_dropout_ratio,
            )
        elif image_index in (1, 2):
            img_token_mask = self._apply_patch_token_dropout(
                img_token_mask,
                dropout_prob=self.wrist_patch_dropout_prob,
                dropout_ratio=self.wrist_patch_dropout_ratio,
            )
        pad_masks.append(img_token_mask)
        att_masks += [0] * num_img_embs

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, keyframes=None, memory=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        memory_frames_by_key, memory_frame_mask = (None, None)
        if memory is not None:
            memory_frames_by_key, memory_frame_mask = memory

        anchor_prefix_tokens = None
        anchor_prefix_mask = None
        gist_prefix_tokens = None
        gist_prefix_mask = None
        if keyframes is not None and (self.use_episodic_memory or self.anchor_memory_cross_attn):
            episode_kf, episode_mask = keyframes.get("episode", (None, None))
            anchor_scope = (
                _episodic_memory.EPISODIC_SCOPE_ANCHOR
                if self.use_episodic_memory
                else 0
            )
            anchor_prefix_tokens, anchor_prefix_mask = self.embed_keyframes(
                episode_kf,
                episode_mask,
                scope_index=anchor_scope,
            )
            if self.use_episodic_memory:
                gist_kf, gist_mask = keyframes.get("gist", (None, None))
                gist_prefix_tokens, gist_prefix_mask = self.embed_gist_keyframes(gist_kf, gist_mask)

        episode_kf, episode_mask = (None, None)
        if keyframes is not None:
            episode_kf, episode_mask = keyframes.get("episode", (None, None))

        active_indices = (0,) if self.use_head_only_vision else tuple(range(len(images)))
        for image_index in active_indices:
            img = images[image_index]
            img_mask = img_masks[image_index]
            cam_key = IMAGE_KEYS[image_index]

            def image_embed_func(img_tensor):
                return self.paligemma_with_expert.embed_image(img_tensor)

            cam_memory = None
            if isinstance(memory_frames_by_key, dict):
                cam_memory = memory_frames_by_key.get(cam_key)
            elif memory_frames_by_key is not None and image_index == 0:
                cam_memory = memory_frames_by_key

            if image_index == 0 and self.use_anchor_temporal_memory and self.use_short_horizon_memory:
                short_emb = self.embed_memory_frames(cam_memory, memory_frame_mask)
                if short_emb is None:
                    short_emb = self._apply_checkpoint(image_embed_func, img)
                anchor_emb = self.embed_anchor_temporal_frames(episode_kf, episode_mask, img)
                if anchor_emb is None:
                    anchor_emb = self._apply_checkpoint(image_embed_func, img)
                self._append_image_tokens(embs, pad_masks, att_masks, short_emb, img_mask, image_index=0)
                self._append_image_tokens(embs, pad_masks, att_masks, anchor_emb, img_mask, image_index=0)
                continue

            if self.use_short_horizon_memory and cam_memory is not None:
                img_emb = self.embed_memory_frames(cam_memory, memory_frame_mask)
                if img_emb is None:
                    img_emb = self._apply_checkpoint(image_embed_func, img)
            elif image_index == 0 and self.use_anchor_temporal_memory:
                img_emb = self.embed_anchor_temporal_frames(episode_kf, episode_mask, img)
                if img_emb is None:
                    img_emb = self._apply_checkpoint(image_embed_func, img)
            else:
                img_emb = self._apply_checkpoint(image_embed_func, img)

            img_emb = self._fuse_image_tokens_with_memory(
                img_emb,
                anchor_prefix_tokens,
                anchor_prefix_mask,
                gist_prefix_tokens,
                gist_prefix_mask,
            )
            self._append_image_tokens(embs, pad_masks, att_masks, img_emb, img_mask, image_index=image_index)

        if self.use_episodic_memory:
            if anchor_prefix_tokens is not None:
                embs.append(anchor_prefix_tokens)
                pad_masks.append(anchor_prefix_mask)
                att_masks += [0] * anchor_prefix_tokens.shape[1]
            if gist_prefix_tokens is not None:
                embs.append(gist_prefix_tokens)
                pad_masks.append(gist_prefix_mask)
                att_masks += [0] * gist_prefix_tokens.shape[1]
        elif self.use_keyframe_conditioning and keyframes is not None and not self.anchor_memory_cross_attn:
            for scope_index, scope in enumerate(("episode", "task")):
                scope_keyframes, scope_mask = keyframes[scope]
                keyframe_emb, keyframe_token_mask = self.embed_keyframes(
                    scope_keyframes,
                    scope_mask,
                    scope_index=scope_index,
                )
                if keyframe_emb is None:
                    continue
                embs.append(keyframe_emb)
                pad_masks.append(keyframe_token_mask)
                att_masks += [0] * keyframe_emb.shape[1]

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, return_extra=False) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, keyframes, memory = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, keyframes, memory
        )
        bsize = prefix_embs.shape[0]
        device = actions.device
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            if self.use_internal_guidance:
                (_, suffix_out), _, inter_suffix_out = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                    return_intermediate_layer_idx=self.ig_layer_idx,
                )
                return suffix_out, inter_suffix_out
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out, None

        suffix_out, inter_suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        if inter_suffix_out is not None:
            inter_suffix_out = inter_suffix_out[:, -self.config.action_horizon :]
            inter_suffix_out = inter_suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        mse = F.mse_loss(u_t, v_t, reduction="none")
        extra_losses = None
        if self.use_internal_guidance:
            if inter_suffix_out is None:
                raise RuntimeError("Internal guidance was enabled but no intermediate suffix output was returned.")

            def ig_action_out_proj_func(inter_suffix_out):
                return self.ig_action_out_proj(inter_suffix_out)

            v_inter = self._apply_checkpoint(ig_action_out_proj_func, inter_suffix_out)
            ig_mse = F.mse_loss(u_t, v_inter, reduction="none")
            mse = mse + self.ig_loss_weight * ig_mse
            extra_losses = {
                "ig_aux_loss": ig_mse.mean(),
            }

        progress_aux_loss = None
        if self.use_progress_head and hasattr(observation, "progress_bin") and observation.progress_bin is not None:
            progress_bin = observation.progress_bin
            subtask_done = observation.subtask_done
            progress_mask = observation.progress_mask
            if progress_bin is not None and subtask_done is not None and progress_mask is not None:
                prog_hidden = self._progress_step_from_prefix_embs(
                    prefix_embs, prefix_pad_masks, prefix_att_masks, bsize, device
                )
                prog_losses = self.progress_evaluator.loss(
                    prog_hidden,
                    progress_bin,
                    subtask_done,
                    progress_mask,
                    done_focal_gamma=self.done_focal_gamma,
                    done_focal_alpha=self.done_focal_alpha,
                )
                progress_aux_loss = (
                    self.progress_loss_weight * prog_losses["progress_loss"]
                    + self.subtask_done_loss_weight * prog_losses["done_loss"]
                )
                mse = mse + progress_aux_loss / mse.shape[-1]
                extra_losses = {
                    **(extra_losses or {}),
                    **prog_losses,
                    "progress_aux_loss": progress_aux_loss,
                }

        if return_extra:
            return mse, v_t, u_t, extra_losses
        return mse

    def _progress_step_from_prefix_embs(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        bsize: int,
        device: torch.device,
    ) -> Tensor:
        query = self.progress_query.expand(bsize, -1, -1)
        if prefix_embs.dtype == torch.bfloat16:
            query = query.to(dtype=torch.bfloat16)
        suffix_embs = query
        suffix_pad_masks = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        suffix_att_masks = torch.tensor([1], dtype=torch.bool, device=device).expand(bsize, 1)

        suffix_len = 1
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        progress_time = torch.zeros(bsize, device=device, dtype=torch.float32)
        progress_adarms_cond = create_sinusoidal_pos_embedding(
            progress_time,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        progress_adarms_cond = progress_adarms_cond.to(dtype=torch.float32, device=device)
        progress_adarms_cond = self.time_mlp_in(progress_adarms_cond)
        progress_adarms_cond = F.silu(progress_adarms_cond)
        progress_adarms_cond = self.time_mlp_out(progress_adarms_cond)
        progress_adarms_cond = F.silu(progress_adarms_cond)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, progress_adarms_cond],
        )
        return outputs_embeds[1][:, -1, :].to(dtype=torch.float32)

    @torch.no_grad()
    def predict_progress(self, device, observation) -> dict[str, Tensor]:
        """Inference: progress value and subtask-done probability."""
        images, img_masks, lang_tokens, lang_masks, state, keyframes, memory = self._preprocess_observation(
            observation, train=False
        )
        bsize = lang_tokens.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, keyframes, memory
        )
        hidden = self._progress_step_from_prefix_embs(
            prefix_embs, prefix_pad_masks, prefix_att_masks, bsize, device
        )
        progress_logits, done_logit = self.progress_evaluator(hidden)
        progress_value = self.progress_evaluator.logits_to_progress(progress_logits, self.num_progress_bins)
        return {
            "progress_logits": progress_logits,
            "done_logit": done_logit,
            "progress_value": progress_value,
            "done_prob": torch.sigmoid(done_logit),
        }

    @torch.no_grad()
    def sample_actions(
        self, device, observation, noise=None, num_steps=10, num_samples=1, temperature=1.0
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_samples x num_steps x num_motors)"""
        bsize = observation.state.shape[0]

        images, img_masks, lang_tokens, lang_masks, state, keyframes, memory = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, keyframes, memory
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Transformers 4.53+ Gemma expects Cache-like objects (with get_seq_length).
        # PaliGemma may return legacy tuple cache; convert if needed.
        if isinstance(past_key_values, tuple):
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Expand for multiple samples if needed
        if num_samples > 1:
            # Expand cache along batch dimension
            if hasattr(past_key_values, "batch_repeat_interleave"):
                past_key_values.batch_repeat_interleave(num_samples)
            else:
                # Legacy tuple cache fallback
                past_key_values = tuple(
                    tuple(t.repeat_interleave(num_samples, dim=0) for t in layer_kv) for layer_kv in past_key_values
                )
            # Expand masks and state
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(num_samples, dim=0)
            state = state.repeat_interleave(num_samples, dim=0)

        effective_bsize = bsize * num_samples

        if noise is None:
            actions_shape = (effective_bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device) * temperature
        else:
            if noise.shape[0] != effective_bsize:
                raise ValueError(
                    f"Provided noise shape {noise.shape} does not match effective batch size {effective_bsize}"
                )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(effective_bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

        x_t = x_t.view(bsize, num_samples, self.config.action_horizon, self.config.action_dim)
        if num_samples == 1:
            x_t = x_t.squeeze(1)

        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # Ensure cache is in the new Cache format expected by transformers Gemma.
        if isinstance(past_key_values, tuple):
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        if self.use_internal_guidance:
            outputs_embeds, _, inter_suffix_out = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                return_intermediate_layer_idx=self.ig_layer_idx,
            )
        else:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            inter_suffix_out = None

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_final = self.action_out_proj(suffix_out)
        if not self.use_internal_guidance:
            return v_final

        if inter_suffix_out is None:
            raise RuntimeError("Internal guidance was enabled but no intermediate suffix output was returned.")
        inter_suffix_out = inter_suffix_out[:, -self.config.action_horizon :]
        inter_suffix_out = inter_suffix_out.to(dtype=torch.float32)
        v_inter = self.ig_action_out_proj(inter_suffix_out)
        v_guided = v_inter + self.ig_scale * (v_final - v_inter)

        apply_ig = (timestep >= self.ig_min_t) & (timestep <= self.ig_max_t)
        apply_ig = apply_ig[:, None, None].to(dtype=torch.bool)
        return torch.where(apply_ig, v_guided, v_final)
