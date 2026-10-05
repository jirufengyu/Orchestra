"""Inference adapter for the standalone task progress evaluator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from typing import Any

import jax
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import rmbench_progress_transforms
from openpi.serving import obs_tap as _obs_tap


def _extract_camera_images(
    obs: dict,
    *,
    camera_keys: tuple[str, ...] | None,
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
    aliases = {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    }
    for camera_key, model_key in aliases.items():
        if camera_key not in camera_images and model_key in camera_images:
            camera_images[camera_key] = camera_images[model_key]
    if camera_keys is not None:
        camera_images = {key: camera_images[key] for key in camera_keys if key in camera_images}
    return camera_images or None


class ProgressEvaluatorPolicy(_base_policy.BasePolicy):
    """Expose a standalone progress evaluator through the policy RPC interface."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        transforms: Sequence[_transforms.DataTransformFn],
        model_input_transforms: Sequence[_transforms.DataTransformFn],
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cuda",
    ):
        self._model = model.to(pytorch_device).eval()
        self._device = pytorch_device
        self._pre_model_transform = _transforms.compose(transforms)
        self._model_input_transform = _transforms.compose(model_input_transforms)
        self._metadata = metadata or {}
        self._obs_tap_fn: Callable[[dict[str, Any]], None] | None = None

        model_config = model.config
        if getattr(model_config, "use_short_horizon_memory", False):
            memory_camera_keys = self._metadata.get("memory_camera_keys")
            self._memory_camera_keys = tuple(memory_camera_keys) if memory_camera_keys else None
            self._memory_buffer = rmbench_progress_transforms.ShortHorizonMemoryBuffer(
                num_frames=int(getattr(model_config, "memory_num_frames", 6)),
                frame_stride=int(getattr(model_config, "memory_frame_stride", 1)),
                camera_keys=self._memory_camera_keys,
            )
        else:
            self._memory_camera_keys = None
            self._memory_buffer = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def set_obs_tap_fn(self, tap_fn: Callable[[dict[str, Any]], None] | None) -> None:
        """Register a callback for the evaluator's post-data-transform input."""
        self._obs_tap_fn = tap_fn

    def reset_memory(self) -> None:
        if self._memory_buffer is not None:
            self._memory_buffer.reset()

    def append_memory(self, obs: dict) -> None:
        if self._memory_buffer is None:
            return
        camera_images = _extract_camera_images(
            obs,
            camera_keys=self._memory_camera_keys,
        )
        if camera_images is not None:
            self._memory_buffer.append(camera_images)

    def _inject_memory(self, obs: dict) -> dict:
        if self._memory_buffer is None:
            return obs
        inputs = dict(obs)
        skip_append = bool(inputs.pop("skip_memory_append", False))
        camera_images = _extract_camera_images(
            inputs,
            camera_keys=self._memory_camera_keys,
        )
        if camera_images is None:
            return inputs
        if not skip_append:
            self._memory_buffer.append(camera_images)
        memory_frames, memory_frame_mask = self._memory_buffer.build()
        return {
            **inputs,
            "memory_frames": memory_frames,
            "memory_frame_mask": memory_frame_mask,
        }

    @override
    def infer(self, obs: dict) -> dict[str, Any]:
        return self.infer_progress(obs)

    def infer_progress(self, obs: dict) -> dict[str, Any]:
        input_prompt = obs.get("prompt")
        if input_prompt is not None and not isinstance(input_prompt, str):
            input_prompt = str(input_prompt)
        tap_meta = _obs_tap.extract_episode_step_ids(obs)
        inputs = dict(obs)
        inputs.pop("return_progress", None)
        inputs = self._inject_memory(inputs)
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
                logging.warning("Failed to mirror progress evaluator input", exc_info=True)
        inputs = self._model_input_transform(inputs)
        inputs = jax.tree.map(
            lambda value: torch.from_numpy(np.asarray(value)).to(self._device)[None, ...],
            inputs,
        )
        observation = _model.Observation.from_dict(inputs)
        prediction = self._model.predict(observation)

        def scalar(value: Any) -> float:
            if torch.is_tensor(value):
                return float(value.detach().cpu().reshape(-1)[0].item())
            return float(np.asarray(value).reshape(-1)[0])

        progress_logits = prediction["progress_logits"].detach().float().cpu().reshape(-1).numpy()
        progress_probs = (
            torch.softmax(prediction["progress_logits"], dim=-1).detach().float().cpu().reshape(-1).numpy()
        )
        progress_percent = scalar(prediction["progress_percent"])
        return {
            "progress_available": True,
            "done_prob": scalar(prediction["done_prob"]),
            "done_logit": scalar(prediction["done_logit"]),
            "progress_logits": progress_logits,
            "progress_probs": progress_probs,
            "progress_bin": int(scalar(prediction["progress_bin"])),
            "progress_percent": progress_percent,
            # Preserve the old joint-head convention used by rollout clients.
            "progress_value": progress_percent / 100.0 - 1.0,
        }
