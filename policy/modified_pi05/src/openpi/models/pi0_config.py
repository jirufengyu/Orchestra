import dataclasses
from typing import Any, TYPE_CHECKING, cast

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    vf_use_action_expert: bool = True
    num_value_heads: int = 1
    value_head_dropout_p: float = 0.1
    value_head_dropout_in_eval: bool = True

    # ── MoE (Mixture-of-Experts) action expert configuration ──
    # If True, replace the single action expert with a task-conditioned MoE.
    use_moe: bool = False
    # Task names for the MoE experts. Each task gets its own Gemma expert.
    moe_task_names: tuple[str, ...] | None = None
    # Whether to maintain a shared expert alongside task-specific experts.
    # The shared expert serves as default/fallback and as init source for new tasks.
    moe_use_shared_expert: bool = True
    # When training incrementally, only train this task's expert (freeze all others).
    # Set to None to train all experts jointly.
    moe_train_task: str | None = None

    # ── Keyframe conditioning ──
    # If True, the PyTorch pi0 implementation can consume episode/task keyframes as
    # compact visual prefix tokens in addition to the current observation images.
    use_keyframe_conditioning: bool = False
    # Number of compact prefix tokens produced per keyframe scope (episode/task).
    keyframe_token_count: int = 4
    # Maximum number of raw keyframes expected in fake specs and padding-friendly batches.
    max_episode_keyframes: int = 8
    max_task_keyframes: int = 8
    # Current implementation reuses the PaliGemma/SigLIP vision tower and pools each
    # frame before a learned query resampler, avoiding full patch-token expansion.
    keyframe_encoder_type: str = "paligemma_pool"
    # If True, model transforms prepend `keyframe_caption` / `task_plan_text` to prompt
    # before tokenization when those fields are present in samples.
    use_keyframe_caption: bool = False

    # ── HEVM: unified episodic memory (anchor + recent + gist) ──
    use_episodic_memory: bool = False
    gist_token_count: int = 8
    max_gist_keyframes: int = 16
    gist_frame_stride: int = 10
    episodic_memory_cross_attn: bool = True
    # Mem-0 style: fuse episode anchor into image patch tokens via cross-attention only
    # (no extra VLM prefix tokens). Requires use_keyframe_conditioning.
    anchor_memory_cross_attn: bool = False

  # ── Short-horizon video memory (MEM-style) ──
    # If True, fuse a short history of base-camera frames into the current image patch
    # tokens via causal temporal attention before passing them to the VLM backbone.
    use_short_horizon_memory: bool = False
    # Number of frames in the short-horizon window, including the current frame.
    memory_num_frames: int = 6
    # Stride between consecutive history frames in dataset frame indices.
    memory_frame_stride: int = 1
    # Apply causal temporal attention every N ViT layers inside SigLIP (MEM default: 4).
    memory_temporal_interval: int = 4
    # Drop past-frame patch tokens after this many SigLIP ViT layers. None keeps them until the end.
    memory_drop_past_after_layer: int | None = None

    # Fuse episode-start anchor + current frame via SigLIP temporal attention (Mem-0 style),
    # producing a second head-camera patch grid alongside short-horizon memory.
    use_anchor_temporal_memory: bool = False
    # Number of frames in the anchor temporal sequence (anchor, current).
    anchor_temporal_num_frames: int = 2
    # If True, only embed base_0_rgb in the VLM prefix (skip wrist cameras entirely).
    use_head_only_vision: bool = False

    # Training-only visual token dropout in the PyTorch model.
    # This masks image patch tokens after SigLIP embedding without changing image pixels.
    base_patch_dropout_prob: float = 0.0
    base_patch_dropout_ratio: float = 0.0
    wrist_patch_dropout_prob: float = 0.0
    wrist_patch_dropout_ratio: float = 0.0

    # ── Subtask progress head (joint training with action flow-matching) ──
    use_progress_head: bool = False
    num_progress_bins: int = 101
    progress_loss_weight: float = 1.0
    subtask_done_loss_weight: float = 1.0
    progress_head_dropout: float = 0.1
    progress_head_use_mlp: bool = True
    progress_boundary_margin: int = 8
    done_focal_gamma: float = 0.0
    done_focal_alpha: float | None = None

    # ── Internal Guidance (optional auxiliary action-flow supervision) ──
    use_internal_guidance: bool = False
    ig_layer_idx: int | None = None
    ig_loss_weight: float = 0.5
    ig_scale: float = 1.2
    ig_min_t: float = 0.3
    ig_max_t: float = 1.0

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if not 0.0 <= self.base_patch_dropout_prob <= 1.0:
            raise ValueError("base_patch_dropout_prob must be in [0, 1].")
        if not 0.0 <= self.base_patch_dropout_ratio <= 1.0:
            raise ValueError("base_patch_dropout_ratio must be in [0, 1].")
        if not 0.0 <= self.wrist_patch_dropout_prob <= 1.0:
            raise ValueError("wrist_patch_dropout_prob must be in [0, 1].")
        if not 0.0 <= self.wrist_patch_dropout_ratio <= 1.0:
            raise ValueError("wrist_patch_dropout_ratio must be in [0, 1].")
        if self.done_focal_gamma < 0.0:
            raise ValueError("done_focal_gamma must be non-negative.")
        if self.done_focal_alpha is not None and not 0.0 <= self.done_focal_alpha <= 1.0:
            raise ValueError("done_focal_alpha must be in [0, 1].")
        if self.use_internal_guidance and self.ig_layer_idx is None:
            raise ValueError("ig_layer_idx must be set when use_internal_guidance is True.")
        if self.ig_layer_idx is not None and self.ig_layer_idx < 0:
            raise ValueError("ig_layer_idx must be non-negative.")
        if self.ig_loss_weight < 0.0:
            raise ValueError("ig_loss_weight must be non-negative.")
        if not 0.0 <= self.ig_min_t <= self.ig_max_t <= 1.0:
            raise ValueError("Expected 0 <= ig_min_t <= ig_max_t <= 1.")
        if self.memory_num_frames < 1:
            raise ValueError("memory_num_frames must be >= 1.")
        if self.memory_frame_stride < 1:
            raise ValueError("memory_frame_stride must be >= 1.")
        if self.memory_temporal_interval < 1:
            raise ValueError("memory_temporal_interval must be >= 1.")
        if self.memory_drop_past_after_layer is not None and self.memory_drop_past_after_layer < 1:
            raise ValueError("memory_drop_past_after_layer must be >= 1 when set.")
        if self.gist_token_count < 1:
            raise ValueError("gist_token_count must be >= 1.")
        if self.max_gist_keyframes < 1:
            raise ValueError("max_gist_keyframes must be >= 1.")
        if self.gist_frame_stride < 1:
            raise ValueError("gist_frame_stride must be >= 1.")
        if self.anchor_memory_cross_attn and not self.use_keyframe_conditioning:
            raise ValueError("anchor_memory_cross_attn=True requires use_keyframe_conditioning=True.")
        if self.anchor_temporal_num_frames < 2:
            raise ValueError("anchor_temporal_num_frames must be >= 2 (anchor + current).")
        if self.use_anchor_temporal_memory and self.anchor_memory_cross_attn:
            raise ValueError("use_anchor_temporal_memory is incompatible with anchor_memory_cross_attn.")
        if self.use_anchor_temporal_memory and self.use_keyframe_conditioning:
            raise ValueError(
                "use_anchor_temporal_memory replaces use_keyframe_conditioning; disable keyframe conditioning."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        observation_cls = cast(Any, _model.Observation)

        with at.disable_typechecking():
            memory_image_keys = (
                ("base_0_rgb",) if self.use_head_only_vision else _model.IMAGE_KEYS
            )
            memory_fields: dict[str, Any] = {}
            if self.use_short_horizon_memory:
                memory_fields = {
                    "memory_frames": {
                        key: jax.ShapeDtypeStruct(
                            [batch_size, self.memory_num_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                        )
                        for key in memory_image_keys
                    },
                    "memory_frame_mask": jax.ShapeDtypeStruct([batch_size, self.memory_num_frames], jnp.bool_),
                }
            anchor_fields: dict[str, Any] = {}
            if self.use_anchor_temporal_memory:
                anchor_fields = {
                    "episode_keyframes": jax.ShapeDtypeStruct(
                        [batch_size, 1, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                    ),
                    "episode_keyframe_mask": jax.ShapeDtypeStruct([batch_size, 1], jnp.bool_),
                }
            gist_fields: dict[str, Any] = {}
            if self.use_episodic_memory:
                gist_fields = {
                    "gist_keyframes": jax.ShapeDtypeStruct(
                        [batch_size, self.max_gist_keyframes, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                    ),
                    "gist_keyframe_mask": jax.ShapeDtypeStruct([batch_size, self.max_gist_keyframes], jnp.bool_),
                }
            if self.use_keyframe_conditioning:
                observation_spec = observation_cls(
                    images={
                        "base_0_rgb": image_spec,
                        "left_wrist_0_rgb": image_spec,
                        "right_wrist_0_rgb": image_spec,
                    },
                    image_masks={
                        "base_0_rgb": image_mask_spec,
                        "left_wrist_0_rgb": image_mask_spec,
                        "right_wrist_0_rgb": image_mask_spec,
                    },
                    state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                    episode_keyframes=jax.ShapeDtypeStruct(
                        [batch_size, self.max_episode_keyframes, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                    ),
                    episode_keyframe_mask=jax.ShapeDtypeStruct([batch_size, self.max_episode_keyframes], jnp.bool_),
                    task_keyframes=jax.ShapeDtypeStruct(
                        [batch_size, self.max_task_keyframes, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                    ),
                    task_keyframe_mask=jax.ShapeDtypeStruct([batch_size, self.max_task_keyframes], jnp.bool_),
                    tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                    tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                    **memory_fields,
                    **anchor_fields,
                    **gist_fields,
                )
            else:
                image_keys_spec = (
                    {"base_0_rgb": image_spec}
                    if self.use_head_only_vision
                    else {
                        "base_0_rgb": image_spec,
                        "left_wrist_0_rgb": image_spec,
                        "right_wrist_0_rgb": image_spec,
                    }
                )
                image_masks_spec = (
                    {"base_0_rgb": image_mask_spec}
                    if self.use_head_only_vision
                    else {
                        "base_0_rgb": image_mask_spec,
                        "left_wrist_0_rgb": image_mask_spec,
                        "right_wrist_0_rgb": image_mask_spec,
                    }
                )
                observation_spec = observation_cls(
                    images=image_keys_spec,
                    image_masks=image_masks_spec,
                    state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                    tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                    tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                    **memory_fields,
                    **anchor_fields,
                    **gist_fields,
                )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
