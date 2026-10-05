import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.shared.normalize as _normalize
import openpi.transforms as transforms
from openpi.policies import rmbench_progress_transforms
from openpi.policies import seg_transforms as _seg_transforms


def _enable_seg_infer_transforms(
    transform_list: list[transforms.DataTransformFn],
    *,
    infer_representation: str | tuple[str, ...] = "mask",
) -> list[transforms.DataTransformFn]:
    """Use deterministic seg conditioning at inference (no dropout / fixed type(s))."""
    infer_reprs = _seg_transforms.parse_seg_representations(infer_representation)
    updated: list[transforms.DataTransformFn] = []
    for transform in transform_list:
        if isinstance(transform, _seg_transforms.SegConditionTransform):
            updated.append(
                dataclasses.replace(
                    transform,
                    infer_mode=True,
                    infer_representations=infer_reprs,
                )
            )
        elif isinstance(transform, _seg_transforms.LoadActorsMeta):
            updated.append(dataclasses.replace(transform, infer_mode=True))
        else:
            updated.append(transform)
    return updated


def create_progress_evaluator_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    default_prompt: str | None = None,
    pytorch_device: str | None = None,
    use_ema: bool = False,
    seg_infer_representation: str = "mask",
):
    """Load a standalone progress evaluator with its own input pipeline."""
    import safetensors.torch
    import torch

    from openpi.models_pytorch.task_progress_evaluator_pytorch import TaskProgressEvaluator
    from openpi.policies.progress_evaluator_policy import ProgressEvaluatorPolicy

    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    weight_name = "ema_model.safetensors" if use_ema else "model.safetensors"
    weight_path = checkpoint_dir / weight_name
    if not weight_path.exists():
        raise FileNotFoundError(f"Progress evaluator weights not found at {weight_path}")

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TaskProgressEvaluator(train_config.model).to(pytorch_device)
    safetensors.torch.load_model(model, weight_path, device=pytorch_device)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    input_transforms = list(data_config.data_transforms.inputs)
    input_transforms = _enable_seg_infer_transforms(
        input_transforms,
        infer_representation=seg_infer_representation,
    )

    metadata = dict(train_config.policy_metadata or {})
    metadata.update(
        {
            "progress_head": True,
            "standalone_progress_evaluator": True,
            "seg_conditioning": any(
                isinstance(transform, _seg_transforms.SegConditionTransform)
                for transform in input_transforms
            ),
        }
    )
    memory_image_keys = getattr(train_config.data, "memory_image_keys", ())
    if memory_image_keys:
        metadata["memory_camera_keys"] = rmbench_progress_transforms.memory_cam_keys_from_lerobot_keys(
            tuple(memory_image_keys)
        )

    return ProgressEvaluatorPolicy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *input_transforms,
        ],
        # State/action normalization is intentionally omitted: this evaluator only consumes
        # images, language tokens, seg conditioning, and short-horizon memory.
        model_input_transforms=list(data_config.model_transforms.inputs),
        metadata=metadata,
        pytorch_device=pytorch_device,
    )


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    use_ema: bool = False,
    q_function_path: str | None = None,
    q_function_config: str | None = None,
    q_norm_stats_path: str | None = None,
    q_infer_mode: bool = True,
    seg_infer_mode: bool = True,
    seg_infer_representation: str = "mask",
    num_candidates: int = 1,
    csv_log_dir: str | None = None,
    risk_sensitive_lambda: float = 0.0,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        use_ema: Whether to load the EMA model instead of the regular model. If True, will look for
                "ema_model.safetensors" instead of "model.safetensors".
        q_function_path: Path to the Q-function checkpoint for re-ranking.
        q_function_config: Config name for the Q-function. If None, uses train_config.
        q_norm_stats_path: Path to the Q-function normalization stats directory.
        q_infer_mode: If True, will set inference-mode flags on data transforms that support it
            (e.g. Aloha VF transform) to avoid requiring dataset-only keys at eval time.
        seg_infer_mode: If True, SegConditionTransform uses fixed representation without dropout.
        seg_infer_representation: Seg mode(s) at inference: mask, bbox, point, all,
            or comma-separated (e.g. mask,point). Overridable per request via obs seg_representation.
        num_candidates: Number of candidates to sample for re-ranking.
        csv_log_dir: If set, save per-inference logs into a timestamped CSV under this directory.
        risk_sensitive_lambda: Risk-sensitive coefficient for uncertainty-aware action selection.
                                  When > 0, uses score = q_mean - lambda * q_std to penalize high variance.
                                  Set to 0.0 to disable (uses standard argmax). Default: 0.0.
    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safetensors" or "ema_model.safetensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors or ema_model.safetensors
    if use_ema:
        weight_path = os.path.join(checkpoint_dir, "ema_model.safetensors")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"EMA model not found at {weight_path}. Make sure the checkpoint was trained with EMA enabled.")
        is_pytorch = True
        logging.info("Loading EMA model...")
    else:
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        is_pytorch = os.path.exists(weight_path)
        logging.info("Loading model...")

    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    env_seg_repr = os.environ.get("OPENPI_SEG_INFER_REPRESENTATION")
    if env_seg_repr:
        seg_infer_representation = env_seg_repr
    elif hasattr(train_config.data, "eval_seg_representation"):
        seg_infer_representation = train_config.data.eval_seg_representation

    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # Load Q-function model if path is provided
    q_function_model = None
    q_input_transform = None
    
    if q_function_path:
        import torch
        import safetensors.torch
        from openpi.models_pytorch.q_function_pytorch import Pi0QFunction
        import openpi.models.pi0_config

        logging.info(f"Loading Q-function from {q_function_path}...")
        
        # Determine Q-function config
        if q_function_config:
            q_train_config = _config.get_config(q_function_config)
            q_model_config = q_train_config.model
        else:
            q_train_config = train_config
            q_model_config = train_config.model
            
        # Ensure we have a valid Pi0Config for Q-function
        if not isinstance(q_model_config, openpi.models.pi0_config.Pi0Config):
             # Fallback or conversion if needed
             q_model_config = openpi.models.pi0_config.Pi0Config(
                dtype=train_config.pytorch_training_precision if hasattr(train_config, 'pytorch_training_precision') else "bfloat16",
                paligemma_variant=getattr(q_model_config, "paligemma_variant", "gemma_300m"),
                action_expert_variant=getattr(q_model_config, "action_expert_variant", "gemma_300m"),
                pi05=getattr(q_model_config, "pi05", True),
            )

        # Initialize Q-function
        q_function_model = Pi0QFunction(q_model_config, num_bins=201)
        
        # Load weights
        q_ckpt_path = pathlib.Path(q_function_path)
        if q_ckpt_path.is_dir():
             if (q_ckpt_path / "ema_model.safetensors").exists():
                 q_weight_path = q_ckpt_path / "ema_model.safetensors"
             elif (q_ckpt_path / "model.safetensors").exists():
                 q_weight_path = q_ckpt_path / "model.safetensors"
             else:
                 raise FileNotFoundError(f"No model weights found in {q_ckpt_path}")
             
             # Also load Q-function norm stats from specific path if provided, or checkpoint assets
             try:
                 q_data_config = q_train_config.data.create(q_train_config.assets_dirs, q_train_config.model)
                 if q_data_config.asset_id:
                     q_norm_stats = None
                     if q_norm_stats_path:
                         q_norm_stats_path_obj = pathlib.Path(q_norm_stats_path)
                         # Check if the user provided the exact directory containing norm_stats.json
                         if (q_norm_stats_path_obj / "norm_stats.json").exists():
                             logging.info(f"Loading Q-function norm stats directly from {q_norm_stats_path}...")
                             q_norm_stats = _normalize.load(q_norm_stats_path_obj)
                         else:
                             # Fallback: assume user provided the parent 'assets' directory
                             logging.info(f"Loading Q-function norm stats from {q_norm_stats_path} for asset {q_data_config.asset_id}...")
                             q_norm_stats = _checkpoints.load_norm_stats(q_norm_stats_path_obj, q_data_config.asset_id)
                     elif (q_ckpt_path / "assets").exists():
                         logging.info(f"Loading Q-function norm stats from checkpoint assets for asset {q_data_config.asset_id}...")
                         q_norm_stats = _checkpoints.load_norm_stats(q_ckpt_path / "assets", q_data_config.asset_id)
                     
                     if q_norm_stats:
                         q_data_inputs = list(q_data_config.data_transforms.inputs)
                         if q_infer_mode:
                             # Flip inference flags for transforms that require dataset-only keys in training.
                             q_data_inputs = [
                                 dataclasses.replace(t, q_infer_mode=True)
                                 if hasattr(t, "q_infer_mode")
                                 else (dataclasses.replace(t, train_mode=False) if hasattr(t, "train_mode") else t)
                                 for t in q_data_inputs
                             ]

                         # Create Q-function input transforms (similar to policy inputs)
                         # We reuse the repack transforms as they usually map hardware names to standard names
                         q_input_transform = transforms.compose([
                            *repack_transforms.inputs,
                            transforms.InjectDefaultPrompt(default_prompt),
                            *q_data_inputs,
                            transforms.Normalize(q_norm_stats, use_quantiles=q_data_config.use_quantile_norm),
                            *q_data_config.model_transforms.inputs,
                         ])
             except Exception as e:
                 logging.warning(f"Could not load Q-function norm stats or create transforms: {e}. Using Policy transforms.")

        else:
             q_weight_path = q_ckpt_path
        # q_function_device = "cuda:1"
        q_function_device = pytorch_device
        safetensors.torch.load_model(q_function_model, q_weight_path)
        q_function_model.to(q_function_device)
        q_function_model.eval()
        logging.info("Q-function loaded successfully.")


    policy_input_transforms = list(data_config.data_transforms.inputs)
    has_seg_transform = any(isinstance(t, _seg_transforms.SegConditionTransform) for t in policy_input_transforms)
    if seg_infer_mode and has_seg_transform:
        policy_input_transforms = _enable_seg_infer_transforms(
            policy_input_transforms,
            infer_representation=seg_infer_representation,
        )

    policy_metadata = dict(train_config.policy_metadata or {})
    if has_seg_transform:
        policy_metadata["seg_conditioning"] = True
        policy_metadata["seg_infer_representation"] = seg_infer_representation
        policy_metadata["seg_requires_client_actors_meta"] = True
        if getattr(train_config.data, "use_episode_union_seg", False):
            policy_metadata["use_episode_union_seg"] = True
        if getattr(train_config.data, "use_subtask_seg", False):
            policy_metadata["use_subtask_seg"] = True
    if is_pytorch and getattr(model, "use_progress_head", False):
        policy_metadata["progress_head"] = True
    if getattr(train_config.model, "use_episodic_memory", False):
        policy_metadata["use_episodic_memory"] = True
    elif getattr(train_config.data, "use_anchor_memory", False):
        policy_metadata["rollout_anchor_memory"] = True
    if getattr(train_config.model, "use_anchor_temporal_memory", False):
        policy_metadata["use_anchor_temporal_memory"] = True
    if (
        getattr(train_config.model, "use_short_horizon_memory", False)
        or getattr(train_config.data, "use_short_horizon_memory", False)
        or getattr(train_config.model, "use_episodic_memory", False)
        or getattr(train_config.data, "use_episodic_memory", False)
    ):
        policy_metadata["use_short_horizon_memory"] = True
        memory_image_keys = getattr(train_config.data, "memory_image_keys", ())
        if memory_image_keys:
            policy_metadata["memory_camera_keys"] = rmbench_progress_transforms.memory_cam_keys_from_lerobot_keys(
                tuple(memory_image_keys)
            )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *policy_input_transforms,
        ],
        model_input_transforms=[
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        q_function_model=q_function_model,
        q_input_transform=q_input_transform,
        num_candidates=num_candidates,
        csv_log_dir=csv_log_dir,
        risk_sensitive_lambda=risk_sensitive_lambda,
    )
