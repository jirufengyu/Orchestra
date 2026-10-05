from collections.abc import Callable, Sequence
import csv
import json
import logging
import pathlib
import time
import datetime
import threading
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
import torch.nn.functional as F
from typing_extensions import override

from matplotlib import image as mpl_image

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import rmbench_progress_transforms
from openpi.serving import obs_tap as _obs_tap
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        model_input_transforms: Sequence[_transforms.DataTransformFn] | None = None,
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda:0",
        is_pytorch: bool = False,
        q_function_model: torch.nn.Module | None = None,
        q_input_transform: _transforms.DataTransformFn | None = None,
        num_candidates: int = 1,
        csv_log_dir: str | None = None,
        risk_sensitive_lambda: float = 0.0,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transforms applied before model preprocessing.
            model_input_transforms: Optional second-stage transforms (normalize/tokenize).
                When omitted, ``transforms`` is applied as a single composed pipeline.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            q_function_model: Optional Q-function model for re-ranking actions.
            q_input_transform: Optional input transforms for Q-function (including normalization).
            num_candidates: Number of action candidates to sample for re-ranking.
            risk_sensitive_lambda: Risk-sensitive coefficient for uncertainty-aware action selection.
                                  When > 0, uses score = q_mean - lambda * q_std to penalize high variance.
                                  Set to 0.0 to disable (uses standard argmax). Default: 0.0.
        """
        self._model = model
        if model_input_transforms is None:
            self._pre_model_transform = _transforms.compose(transforms)
            self._model_input_transform = lambda data: data
        else:
            self._pre_model_transform = _transforms.compose(transforms)
            self._model_input_transform = _transforms.compose(model_input_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._obs_tap_fn: Callable[[dict[str, Any]], None] | None = None
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._memory_camera_keys: tuple[str, ...] | None = None
        memory_camera_keys = self._metadata.get("memory_camera_keys")
        if memory_camera_keys:
            self._memory_camera_keys = tuple(memory_camera_keys)
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._q_function_model = q_function_model
        self._q_input_transform = q_input_transform
        self._num_candidates = num_candidates
        self._risk_sensitive_lambda = risk_sensitive_lambda
        self._q_logged = False
        self._csv_log_dir = csv_log_dir
        self._csv_path: pathlib.Path | None = None
        self._csv_fp = None
        self._csv_writer: csv.DictWriter | None = None
        self._csv_lock = threading.Lock()
        
        # Time statistics CSV logging
        self._inference_time_csv_path = pathlib.Path("temp/inference_time.csv")
        self._inference_time_csv_lock = threading.Lock()
        self._inference_time_csv_header_written = False

        if self._q_function_model:
            self._q_function_model.eval()
            self._q_function_model.to(self._pytorch_device)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            model_config = getattr(model, "config", None)
            self._use_anchor_temporal_memory = bool(getattr(model_config, "use_anchor_temporal_memory", False))
            self._anchor_image: np.ndarray | None = None
            if getattr(model_config, "use_short_horizon_memory", False) or getattr(
                model_config, "use_episodic_memory", False
            ):
                memory_camera_keys = self._memory_camera_keys
                self._memory_buffer = rmbench_progress_transforms.ShortHorizonMemoryBuffer(
                    num_frames=int(getattr(model_config, "memory_num_frames", 6)),
                    frame_stride=int(getattr(model_config, "memory_frame_stride", 1)),
                    camera_keys=memory_camera_keys,
                )
            else:
                self._memory_buffer = None
        else:
            self._memory_buffer = None
            self._use_anchor_temporal_memory = False
            self._anchor_image = None
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def set_obs_tap_fn(self, tap_fn: Callable[[dict[str, Any]], None] | None) -> None:
        """Register a callback invoked after data transforms with a watch-friendly payload."""
        self._obs_tap_fn = tap_fn

    def reset_memory(self) -> None:
        if self._memory_buffer is not None:
            self._memory_buffer.reset()
        self._anchor_image = None

    def _inject_anchor_temporal_memory(self, obs: dict) -> dict:
        if not getattr(self, "_use_anchor_temporal_memory", False):
            return obs

        camera_keys = self._memory_camera_keys or ("cam_high",)
        camera_images = self._extract_camera_images_for_memory(obs, camera_keys=camera_keys)
        if camera_images is None:
            return obs

        base_image = camera_images.get("cam_high")
        if base_image is None:
            return obs

        base_image = np.asarray(base_image)
        if base_image.ndim == 4:
            base_image = base_image[0]
        # Rollout clients (Aloha/RMBench) send CHW; episode_keyframes must be HWC for ResizeImages.
        if base_image.ndim == 3 and base_image.shape[0] == 3 and base_image.shape[-1] != 3:
            base_image = np.transpose(base_image, (1, 2, 0))
        if self._anchor_image is None:
            self._anchor_image = base_image.astype(np.uint8, copy=True)

        anchor = np.asarray(self._anchor_image)
        if anchor.ndim == 4:
            anchor = anchor[0]
        return {
            **obs,
            "episode_keyframes": anchor[None, ...],
            "episode_keyframe_mask": np.asarray([True], dtype=np.bool_),
        }

    @staticmethod
    def _extract_camera_images_for_memory(
        obs: dict,
        *,
        camera_keys: tuple[str, ...] | None = None,
    ) -> dict[str, np.ndarray] | None:
        images = obs.get("images") or obs.get("image")
        if not isinstance(images, dict):
            return None

        candidate_keys = camera_keys or (
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        )
        camera_images = {key: images[key] for key in candidate_keys if key in images}
        if "cam_high" not in camera_images and "base_0_rgb" in camera_images:
            camera_images["cam_high"] = camera_images["base_0_rgb"]
        if "cam_left_wrist" not in camera_images and "left_wrist_0_rgb" in camera_images:
            camera_images["cam_left_wrist"] = camera_images["left_wrist_0_rgb"]
        if "cam_right_wrist" not in camera_images and "right_wrist_0_rgb" in camera_images:
            camera_images["cam_right_wrist"] = camera_images["right_wrist_0_rgb"]
        if camera_keys is not None:
            camera_images = {key: camera_images[key] for key in camera_keys if key in camera_images}
        if not camera_images:
            return None
        return camera_images

    def append_memory(self, obs: dict) -> None:
        """Append camera frames to short-horizon memory without running inference."""
        if self._memory_buffer is None:
            return
        camera_images = self._extract_camera_images_for_memory(
            obs, camera_keys=self._memory_camera_keys
        )
        if camera_images is not None:
            self._memory_buffer.append(camera_images)

    def _inject_short_horizon_memory(self, obs: dict) -> dict:
        if self._memory_buffer is None:
            return obs

        skip_append = bool(obs.pop("skip_memory_append", False))
        camera_images = self._extract_camera_images_for_memory(
            obs, camera_keys=self._memory_camera_keys
        )
        if camera_images is None:
            return obs

        if not skip_append:
            self._memory_buffer.append(camera_images)
        memory_frames, memory_frame_mask = self._memory_buffer.build()
        return {
            **obs,
            "memory_frames": memory_frames,
            "memory_frame_mask": memory_frame_mask,
        }

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Capture raw prompt (before transforms may tokenize/pop it).
        input_prompt = None
        # print(obs)
        # # 可视化三路相机图像到 temp/，失败则打印 shape
        # _temp_dir = pathlib.Path("temp")
        # _temp_dir.mkdir(exist_ok=True)
        # for _cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        #     try:
        #         if "images" not in obs or _cam not in obs["images"]:
        #             print(f"[obs images] {_cam} not in obs['images']")
        #             continue
        #         _arr = np.asarray(obs["images"][_cam])
        #         print(f"[obs images] {_cam} shape={_arr.shape}")
        #         if _arr.ndim == 4:
        #             _arr = _arr[0]
        #         if _arr.ndim == 2:
        #             _arr = np.stack([_arr] * 3, axis=-1)
        #         elif _arr.ndim == 3 and _arr.shape[0] in (1, 3, 4):
        #             _arr = np.transpose(_arr, (1, 2, 0))
        #         if _arr.ndim == 3 and _arr.shape[-1] == 1:
        #             _arr = np.repeat(_arr, 3, axis=-1)
        #         if np.issubdtype(_arr.dtype, np.floating):
        #             _arr = np.clip(_arr, 0, 1).astype(np.float32)
        #         else:
        #             _arr = (_arr.astype(np.float32) / 255.0).clip(0, 1)
        #         mpl_image.imsave(str(_temp_dir / f"{_cam}_lerobot.png"), _arr)
        #         print(f"[obs images] saved -> temp/{_cam}_lerobot.png")
        #     except Exception as _e:
        #         _arr = obs.get("images", {}).get(_cam) if "images" in obs else None
        #         _shape = getattr(_arr, "shape", None) if _arr is not None else "N/A"
        #         print(f"[obs images] {_cam} visualize failed, shape={_shape}, error={_e}")
        if "prompt" in obs:
            try:
                p = obs["prompt"]
                # Normalize numpy scalars to python str
                if isinstance(p, np.ndarray) and p.shape == ():
                    p = p.item()
                input_prompt = str(p)
            except Exception:
                input_prompt = None
        # Make a copy since transformations may modify the inputs in place.
        tap_meta = _obs_tap.extract_episode_step_ids(obs)
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._inject_anchor_temporal_memory(inputs)
        inputs = self._inject_short_horizon_memory(inputs)
        inputs = self._pre_model_transform(inputs)
        if self._obs_tap_fn is not None:
            try:
                self._obs_tap_fn(
                    _obs_tap.build_obs_tap_payload(
                        inputs,
                        raw_obs=obs,
                        input_prompt=input_prompt,
                        tap_meta=tap_meta,
                    )
                )
            except Exception:
                logging.warning("Failed to mirror post-transform observation", exc_info=True)
        inputs = self._model_input_transform(inputs)

        # Determine number of candidates
        num_candidates = self._num_candidates if self._q_function_model is not None else 1

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[np.newaxis, ...], num_candidates, axis=0), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            # Note: For PyTorch we don't repeat inputs here, we use num_samples in sample_actions
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            
            # For PyTorch, candidate count is controlled by sample_actions(num_samples=...).
            # If user provides noise, they must provide per-candidate noise with batch == num_candidates.
            if self._is_pytorch_model and num_candidates > 1 and noise.shape[0] not in (1, num_candidates):
                raise ValueError(
                    f"Provided noise batch {noise.shape[0]} does not match num_candidates={num_candidates}. "
                    "For Q rerank with PyTorch, pass noise shaped [num_candidates, action_horizon, action_dim] "
                    "or omit noise."
                )
            if (not self._is_pytorch_model) and num_candidates > 1 and noise.shape[0] == 1:
                noise = jnp.repeat(noise, num_candidates, axis=0)
            
            sample_kwargs["noise"] = noise

        # For PyTorch, use optimized num_samples argument
        if self._is_pytorch_model and num_candidates > 1:
            sample_kwargs["num_samples"] = num_candidates

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        
        # Sample actions
        sample_action_start = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        sample_action_time = time.monotonic() - sample_action_start

        # Normalize actions shape for PyTorch if num_samples was used
        if self._is_pytorch_model and num_candidates > 1:
            # actions is [1, N, T, D] -> [N, T, D]
            actions = actions.view(-1, *actions.shape[2:])

        # Candidate action stats (mean/var across candidates). Always compute in model output space.
        cand_mean_np = None
        cand_var_np = None
        cand_mean_scalar = None
        cand_var_scalar = None
        if self._is_pytorch_model:
            if num_candidates > 1:
                cand = actions.to(torch.float32)  # [N, T, D]
                cand_mean = cand.mean(dim=0)  # [T, D]
                cand_var = cand.var(dim=0, unbiased=False)  # [T, D]
                cand_mean_np = cand_mean.detach().cpu().numpy()
                cand_var_np = cand_var.detach().cpu().numpy()
                cand_mean_scalar = float(cand_mean.mean().item())
                cand_var_scalar = float(cand_var.mean().item())
            else:
                cand = actions.to(torch.float32)  # [T, D]
                cand_mean_np = cand.detach().cpu().numpy()
                cand_var_np = torch.zeros_like(cand).detach().cpu().numpy()
                cand_mean_scalar = float(cand.mean().item())
                cand_var_scalar = 0.0

        best_idx = 0
        q_values = None
        q_mean = None
        q_var = None
        q_std = None

        if self._q_function_model is not None:
            q_function_time = None  # Will be set during q_function processing
            if not self._is_pytorch_model:
                raise ValueError("Q-function re-ranking is only supported for PyTorch policies.")

            # Ensure actions are [N, T, D]
            if actions.ndim == 2:
                actions_candidates = actions[None, ...]
            else:
                actions_candidates = actions

            # Convert policy-normalized state (batched) to unbatched numpy for output_transform
            state_norm_np = np.asarray(inputs["state"][0].detach().cpu())

            q_scores: list[float] = []
            q_distribution_stds: list[float] = []
            using_q_norm = self._q_input_transform is not None
            
            # Try batch processing first for better performance
            use_batch_processing = True
            num_cands = actions_candidates.shape[0]
            
            if use_batch_processing and num_cands > 1:
                try:
                    actions_cands_np = actions_candidates.detach().cpu().numpy()  # [N, T, D]
                    
                    # Try batch unnormalize first
                    try:
                        # Repeat state for all candidates
                        state_batch = np.repeat(state_norm_np[None, ...], num_cands, axis=0)  # [N, state_dims...]
                        # print(f"state_batch.shape: {state_batch.shape}")
                        # print(f"actions_cands_np.shape before output_transform: {actions_cands_np.shape}")
                        raw_outs = self._output_transform({"state": state_batch, "actions": actions_cands_np})
                        raw_actions_batch = raw_outs["actions"]  # [N, T, D]
                        # print(f"raw_actions_batch.shape after output_transform (batch mode): {raw_actions_batch.shape}")
                    except Exception as e:
                        # _output_transform doesn't support batch processing, do it one by one
                        # print(f"Batch output_transform failed: {e}, trying one by one")
                        raw_actions_list = []
                        for i in range(num_cands):
                            raw_out = self._output_transform({
                                "state": state_norm_np, 
                                "actions": actions_cands_np[i]
                            })
                            raw_actions_list.append(raw_out["actions"])
                        raw_actions_batch = np.stack(raw_actions_list, axis=0)  # [N, T, D]
                        # print(f"raw_actions_batch.shape after output_transform (loop mode): {raw_actions_batch.shape}")
                    
                    if self._q_input_transform is not None:
                        # Q-function input preprocessing - must process one by one as it doesn't support batching
                        # print(f"Processing {num_cands} candidates through q_input_transform (one by one)...")
                        
                        q_proc_list = []
                        q_actions_list = []
                        
                        for i in range(num_cands):
                            # Prepare single candidate data
                            q_data_single = dict(obs)
                            q_data_single["actions"] = raw_actions_batch[i]  # usually [T, D]

                            # Apply Q-function input transform.
                            # Some Q checkpoints compute norm stats over the LAST dim assuming it is horizon (T),
                            # i.e., actions are shaped [D, T] during Q training. In that case, passing [T, D]
                            # will fail normalization broadcasting with shapes (T, D) vs (T,).
                            try:
                                q_proc_single = self._q_input_transform(q_data_single)
                            except Exception as e:
                                act = q_data_single.get("actions", None)
                                if isinstance(act, np.ndarray) and act.ndim == 2:
                                    q_data_single_t = dict(obs)
                                    q_data_single_t["actions"] = act.T  # [D, T]
                                    q_proc_single = self._q_input_transform(q_data_single_t)
                                else:
                                    raise e

                            q_actions_single = np.asarray(q_proc_single.pop("actions"))
                            
                            q_proc_list.append(q_proc_single)
                            q_actions_list.append(q_actions_single)
                        
                        # Stack results to create batch
                        q_actions_batch_np = np.stack(q_actions_list, axis=0)  # [N, T, D]
                        # print(f"q_actions_batch_np.shape after q_input_transform: {q_actions_batch_np.shape}")
                        
                        # Stack observation data
                        # Take first processed obs as template and stack arrays/scalars into batch.
                        def _stack_values(values: list):
                            v0 = values[0]
                            # Numpy arrays -> stack
                            if isinstance(v0, np.ndarray):
                                return np.stack(values, axis=0)
                            # Numpy scalars / python scalars (e.g., bool mask) -> asarray to create batch dim
                            if isinstance(v0, (np.generic, bool, int, float)):
                                return np.asarray(values)
                            # Torch tensors (should be rare here) -> convert to numpy then stack
                            if torch.is_tensor(v0):
                                return torch.stack(values, dim=0).detach().cpu().numpy()
                            # Fallback: keep as-is (e.g., strings/prompts)
                            return v0

                        q_proc_batch = {}
                        for key in q_proc_list[0].keys():
                            if isinstance(q_proc_list[0][key], dict):
                                q_proc_batch[key] = {}
                                for sub_key in q_proc_list[0][key].keys():
                                    values = [q_proc_list[i][key][sub_key] for i in range(num_cands)]
                                    q_proc_batch[key][sub_key] = _stack_values(values)
                            else:
                                values = [q_proc_list[i][key] for i in range(num_cands)]
                                q_proc_batch[key] = _stack_values(values)
                        
                        # Convert to torch tensors with batch dimension
                        def to_torch_multi_batched(x):
                            return torch.from_numpy(np.array(x)).to(self._pytorch_device)
                        
                        q_obs_inputs_batch = jax.tree.map(to_torch_multi_batched, q_proc_batch)
                        q_obs_batch = _model.Observation.from_dict(q_obs_inputs_batch)
                        q_actions_batch = torch.from_numpy(np.array(q_actions_batch_np)).to(self._pytorch_device).to(
                            torch.float32
                        )  # [N, T, D]
                    else:
                        # Fallback: no Q normalization, use policy preprocessing
                        # Repeat observation for all candidates
                        q_obs_inputs_batch = jax.tree.map(
                            lambda x: x.repeat(num_cands, *([1] * (x.ndim - 1))), 
                            inputs
                        )
                        q_obs_batch = _model.Observation.from_dict(q_obs_inputs_batch)
                        q_actions_batch = actions_candidates.to(torch.float32)  # [N, T, D]
                    
                    # Batch Q-function evaluation
                    with torch.no_grad():
                        q_function_start = time.monotonic()
                        logits_batch = self._q_function_model(q_obs_batch, q_actions_batch)  # [N, num_bins]
                        q_function_time = time.monotonic() - q_function_start
                        probs_batch = F.softmax(logits_batch, dim=-1)  # [N, num_bins]
                        bins = torch.linspace(-1.0, 0.0, logits_batch.shape[-1], 
                                            device=logits_batch.device, dtype=probs_batch.dtype)  # [num_bins]
                        
                        # Compute expected value (mean) for all candidates
                        expected_batch = (probs_batch * bins[None, :]).sum(dim=-1)  # [N]
                        q_scores = expected_batch.detach().cpu().numpy().tolist()
                        
                        # Compute standard deviation for all candidates
                        expected_square_batch = (probs_batch * bins[None, :].pow(2)).sum(dim=-1)  # [N]
                        variance_batch = expected_square_batch - expected_batch.pow(2)
                        variance_batch = torch.clamp(variance_batch, min=0.0)
                        dist_std_batch = variance_batch.sqrt()  # [N]
                        q_distribution_stds = dist_std_batch.detach().cpu().numpy().tolist()
                    
                    # Successfully used batch processing
                    use_batch_processing = True
                    # Log time statistics after successful batch processing
                    self._log_inference_time(sample_action_time, q_function_time)
                    
                except Exception as e:
                    # Batch processing failed, fall back to loop-based processing
                    logging.warning(
                        f"Batch Q-function processing failed (candidates={num_cands}), "
                        f"falling back to loop-based processing. Error: {e}"
                    )
                    use_batch_processing = False
                    q_scores = []
                    q_distribution_stds = []
            else:
                use_batch_processing = False
            
            # LOOP-BASED PROCESSING PATH (original implementation or fallback)
            if not use_batch_processing or num_cands == 1:
                q_function_time_total = 0.0
                for i in range(actions_candidates.shape[0]):
                    # Candidate action in policy-normalized space (unbatched numpy)
                    act_norm_np = np.asarray(actions_candidates[i].detach().cpu())

                    # Unnormalize to raw action (match policy output space)
                    raw_out = self._output_transform({"state": state_norm_np, "actions": act_norm_np})
                    raw_action = raw_out["actions"]

                    if self._q_input_transform is not None:
                        # Apply Q-function input preprocessing (should match Q training): repack -> normalize(Q stats) -> tokenize/pad, etc.
                        q_data = dict(obs)
                        q_data["actions"] = raw_action
                        try:
                            q_proc = self._q_input_transform(q_data)
                        except Exception as e:
                            act = q_data.get("actions", None)
                            if isinstance(act, np.ndarray) and act.ndim == 2:
                                q_data_t = dict(obs)
                                q_data_t["actions"] = act.T  # [D, T]
                                q_proc = self._q_input_transform(q_data_t)
                            else:
                                raise e

                        # Extract normalized action for Q-function and build Observation
                        q_actions_np = np.asarray(q_proc.pop("actions"))

                        def to_torch_batched(x):
                            return torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...]

                        q_obs_inputs = jax.tree.map(to_torch_batched, q_proc)
                        q_obs = _model.Observation.from_dict(q_obs_inputs)
                        q_actions = torch.from_numpy(np.array(q_actions_np)).to(self._pytorch_device)[None, ...].to(
                            torch.float32
                        )
                    else:
                        # Fallback: assume Q-function was trained with the same preprocessing as policy.
                        q_obs = observation
                        q_actions = actions_candidates[i : i + 1].to(torch.float32)

                    with torch.no_grad():
                        q_function_start = time.monotonic()
                        logits = self._q_function_model(q_obs, q_actions)  # [1, num_bins]
                        q_function_time_total += time.monotonic() - q_function_start
                        probs = F.softmax(logits, dim=-1)
                        bins = torch.linspace(-1.0, 0.0, logits.shape[-1], device=logits.device, dtype=probs.dtype)
                        
                        # Compute expected value (mean)
                        expected = (probs * bins).sum(dim=-1)  # [1]
                        q_scores.append(float(expected.item()))
                        
                        # Compute standard deviation of the distribution (uncertainty)
                        # Var[X] = E[X^2] - (E[X])^2
                        expected_square = (probs * bins.pow(2)).sum(dim=-1)
                        variance = expected_square - expected.pow(2)
                        # Clamp variance to be non-negative (numerical stability)
                        variance = torch.clamp(variance, min=0.0)
                        dist_std = variance.sqrt()
                        q_distribution_stds.append(float(dist_std.item()))
                
                # Log time statistics after loop processing
                self._log_inference_time(sample_action_time, q_function_time_total)

            q_values = np.asarray(q_scores, dtype=np.float32)
            q_dist_stds = np.asarray(q_distribution_stds, dtype=np.float32)

            # Stats across candidates (for logging)
            q_mean = float(np.mean(q_values))
            q_var = float(np.var(q_values))
            q_std = float(np.std(q_values))
            
            # Risk-sensitive action selection: score = q_mean - lambda * q_distribution_std
            # When lambda > 0, this penalizes actions with high variance (uncertainty)
            if self._risk_sensitive_lambda > 0.0:
                risk_sensitive_scores = q_values - self._risk_sensitive_lambda * q_dist_stds
                best_idx = int(np.argmax(risk_sensitive_scores))
            else:
                # Standard selection: use argmax of Q values
                best_idx = int(np.argmax(q_values))

            if not self._q_logged:
                self._q_logged = True
                if self._risk_sensitive_lambda > 0.0:
                    logging.info(
                        "Q-rerank enabled (risk-sensitive, lambda=%.2f): candidates=%d using_q_norm=%s q_min=%.4f q_max=%.4f q_mean=%.4f q_std=%.4f q_best=%.4f best_idx=%d",
                        self._risk_sensitive_lambda,
                        q_values.shape[0],
                        using_q_norm,
                        float(q_values.min()),
                        float(q_values.max()),
                        q_mean,
                        q_std,
                        float(q_values[best_idx]),
                        best_idx,
                    )
                else:
                    logging.info(
                        "Q-rerank enabled: candidates=%d using_q_norm=%s q_min=%.4f q_max=%.4f q_best=%.4f best_idx=%d",
                        q_values.shape[0],
                        using_q_norm,
                        float(q_values.min()),
                        float(q_values.max()),
                        float(q_values[best_idx]),
                        best_idx,
                    )
                if cand_mean_scalar is not None and cand_var_scalar is not None:
                    logging.info(
                        "Action-candidates stats: mean(avg)=%.6f var(avg)=%.6f | Q stats: mean=%.6f std=%.6f var=%.6f",
                        cand_mean_scalar,
                        cand_var_scalar,
                        q_mean if q_mean is not None else float("nan"),
                        q_std if q_std is not None else float("nan"),
                        q_var if q_var is not None else float("nan"),
                    )
            else:
                logging.debug(
                    "Q-rerank: candidates=%d using_q_norm=%s q_values=%s best_idx=%d",
                    q_values.shape[0],
                    using_q_norm,
                    q_values,
                    best_idx,
                )
        else:
            # No q_function_model, log time with q_function_time=0
            self._log_inference_time(sample_action_time, 0.0)

        # Select the best candidate
        if self._is_pytorch_model:
            # Keep dimensions consistent (batch size 1)
            # inputs is [1, ...], actions is [N, T, D]
            # We want output state [1, ...] and actions [1, T, D]
            outputs = {
                "state": inputs["state"], # inputs is already size 1
                "actions": actions[best_idx:best_idx+1],
            }
        else:
            outputs = {
                "state": jax.tree.map(lambda x: x[best_idx:best_idx+1], inputs["state"]),
                "actions": jax.tree.map(lambda x: x[best_idx:best_idx+1], actions),
            }

        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        outputs["input_prompt"] = input_prompt
        
        if cand_mean_np is not None and cand_var_np is not None:
            outputs["action_candidates_mean"] = cand_mean_np
            outputs["action_candidates_var"] = cand_var_np

        if q_values is not None:
            outputs["q_values"] = q_values
            if q_dist_stds is not None:
                outputs["q_dist_stds"] = q_dist_stds
            if q_mean is not None:
                outputs["q_values_mean"] = q_mean
            if q_var is not None:
                outputs["q_values_var"] = q_var
            if q_std is not None:
                outputs["q_values_std"] = q_std
            outputs["selected_idx"] = best_idx

        # Persist per-inference logs to CSV if enabled.
        if self._csv_log_dir is not None:
            try:
                self._append_csv_log(outputs)
            except Exception:
                logging.exception("Failed to append inference log CSV (ignored).")
            
        return outputs

    def infer_progress(self, obs: dict) -> dict[str, Any]:
        """Run only the joint RMBench progress head for client-server rollout switching."""
        if not self._is_pytorch_model or not getattr(self._model, "use_progress_head", False):
            return {"progress_available": False}

        inputs = dict(obs)
        inputs.pop("return_progress", None)
        inputs = jax.tree.map(lambda x: x, inputs)
        inputs = self._pre_model_transform(inputs)
        inputs = self._model_input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)
        progress = self._model.predict_progress(self._pytorch_device, observation)

        def _scalar(value: Any) -> float:
            if torch.is_tensor(value):
                return float(value.detach().cpu().reshape(-1)[0].item())
            return float(np.asarray(value).reshape(-1)[0])

        return {
            "progress_available": True,
            "done_prob": _scalar(progress["done_prob"]),
            "progress_value": _scalar(progress["progress_value"]),
            "done_logit": _scalar(progress["done_logit"]),
        }

    def _ensure_csv_logger(self) -> None:
        if self._csv_writer is not None:
            return
        log_dir = pathlib.Path(self._csv_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = log_dir / f"policy_infer_{ts}.csv"
        self._csv_fp = self._csv_path.open("w", newline="")
        fieldnames = [
            "ts_unix",
            "infer_ms",
            "input_prompt",
            "selected_idx",
            "q_values",
            "q_dist_stds",
            "q_best",
            "q_values_mean",
            "q_values_std",
            "q_values_var",
            "action",
            "action_candidates_mean",
            "action_candidates_var",
        ]
        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._csv_fp.flush()
        logging.info("Logging per-inference CSV to %s", str(self._csv_path))

    def _append_csv_log(self, outputs: dict) -> None:
        # Thread-safe; websocket server can handle multiple clients.
        with self._csv_lock:
            self._ensure_csv_logger()
            assert self._csv_writer is not None
            assert self._csv_fp is not None

            action = outputs.get("actions", None)
            q_values = outputs.get("q_values", None)
            q_dist_stds = outputs.get("q_dist_stds", None)
            selected_idx = outputs.get("selected_idx", None)

            q_best = None
            if q_values is not None and selected_idx is not None:
                try:
                    q_best = float(np.asarray(q_values)[int(selected_idx)])
                except Exception:
                    q_best = None

            row = {
                "ts_unix": time.time(),
                "infer_ms": float(outputs.get("policy_timing", {}).get("infer_ms", float("nan"))),
                "input_prompt": outputs.get("input_prompt", None),
                "selected_idx": int(selected_idx) if selected_idx is not None else None,
                "q_values": json.dumps(np.asarray(q_values).tolist()) if q_values is not None else None,
                "q_dist_stds": json.dumps(np.asarray(q_dist_stds).tolist()) if q_dist_stds is not None else None,
                "q_best": q_best,
                "q_values_mean": outputs.get("q_values_mean", None),
                "q_values_std": outputs.get("q_values_std", None),
                "q_values_var": outputs.get("q_values_var", None),
                "action": json.dumps(np.asarray(action).tolist()) if action is not None else None,
                "action_candidates_mean": json.dumps(np.asarray(outputs.get("action_candidates_mean")).tolist())
                if outputs.get("action_candidates_mean", None) is not None
                else None,
                "action_candidates_var": json.dumps(np.asarray(outputs.get("action_candidates_var")).tolist())
                if outputs.get("action_candidates_var", None) is not None
                else None,
            }
            self._csv_writer.writerow(row)
            self._csv_fp.flush()

    def _log_inference_time(self, sample_action_time: float, q_function_time: float | None = None) -> None:
        """Log inference time statistics to CSV file."""
        if q_function_time is None:
            q_function_time = 0.0
        
        with self._inference_time_csv_lock:
            # Ensure temp directory exists
            self._inference_time_csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file in append mode
            file_exists = self._inference_time_csv_path.exists()
            with self._inference_time_csv_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "sample_action_time_ms", "q_function_time_ms"])
                
                # Write header if file is new or header hasn't been written yet
                if not file_exists or not self._inference_time_csv_header_written:
                    writer.writeheader()
                    self._inference_time_csv_header_written = True
                
                # Write time statistics
                writer.writerow({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "sample_action_time_ms": sample_action_time * 1000,
                    "q_function_time_ms": q_function_time * 1000,
                })
                f.flush()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    def set_obs_tap_fn(self, tap_fn: Callable[[dict[str, Any]], None] | None) -> None:
        if hasattr(self._policy, "set_obs_tap_fn"):
            self._policy.set_obs_tap_fn(tap_fn)

    def append_memory(self, obs: dict) -> None:
        if hasattr(self._policy, "append_memory"):
            self._policy.append_memory(obs)

    def reset_memory(self) -> None:
        if hasattr(self._policy, "reset_memory"):
            self._policy.reset_memory()

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
