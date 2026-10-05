"""Helpers for mirroring post-transform policy observations to disk."""

from __future__ import annotations

from typing import Any

import numpy as np

_MODEL_IMAGE_TO_CAMERA = {
    "base_0_rgb": "cam_high",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}


def _scalar_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.dtype.kind in {"U", "S", "O"}:
        return _scalar_int(arr.reshape(-1)[0])
    try:
        if arr.ndim == 0:
            return int(arr.item())
        if arr.size == 1:
            return int(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None
    return None


def extract_episode_step_ids(obs: dict[str, Any]) -> dict[str, int | None]:
    """Resolve episode/step ids from common client key aliases."""
    episode_id = _scalar_int(obs.get("episode_id"))
    if episode_id is None:
        episode_id = _scalar_int(obs.get("episode_index", obs.get("episode_idx")))
    step_id = _scalar_int(
        obs.get("step_id", obs.get("frame_index", obs.get("step_index", obs.get("global_step"))))
    )
    return {
        "episode_id": episode_id,
        "step_id": step_id,
        "episode_index": episode_id,
        "frame_index": step_id,
    }


def _images_for_tap(data: dict[str, Any]) -> dict[str, np.ndarray]:
    nested = data.get("images")
    if isinstance(nested, dict):
        return {str(name): np.asarray(image) for name, image in nested.items()}

    model_images = data.get("image")
    if isinstance(model_images, dict):
        out: dict[str, np.ndarray] = {}
        for src, dest in _MODEL_IMAGE_TO_CAMERA.items():
            if src in model_images:
                out[dest] = np.asarray(model_images[src])
        return out
    return {}


def _memory_frames_for_tap(data: dict[str, Any]) -> dict[str, np.ndarray]:
    memory = data.get("memory_frames")
    if not isinstance(memory, dict):
        return {}

    out: dict[str, np.ndarray] = {}
    for src, value in memory.items():
        name = _MODEL_IMAGE_TO_CAMERA.get(str(src), str(src))
        out[name] = np.asarray(value)
    return out


def build_obs_tap_payload(
    transformed: dict[str, Any],
    *,
    raw_obs: dict[str, Any],
    input_prompt: str | None,
    tap_meta: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Build a watch-friendly payload after data transforms (before tokenization)."""
    meta = tap_meta or extract_episode_step_ids(raw_obs)
    prompt = transformed.get("prompt", "")
    if not isinstance(prompt, str):
        prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

    payload: dict[str, Any] = {
        "prompt": str(prompt),
        "input_prompt": input_prompt,
        "episode_id": meta.get("episode_id"),
        "step_id": meta.get("step_id"),
        "episode_index": meta.get("episode_index"),
        "frame_index": meta.get("frame_index"),
        "tap_stage": "post_data_transform",
        "images": _images_for_tap(transformed),
    }

    memory_frames = _memory_frames_for_tap(transformed)
    if memory_frames:
        payload["memory_frames"] = memory_frames
    if "memory_frame_mask" in transformed:
        payload["memory_frame_mask"] = np.asarray(transformed["memory_frame_mask"])
    if "seg_cam_high" in raw_obs:
        payload["seg_cam_high"] = np.asarray(raw_obs["seg_cam_high"])
    elif "observation.segmentation.cam_high" in raw_obs:
        payload["seg_cam_high"] = np.asarray(raw_obs["observation.segmentation.cam_high"])
    if "seg_representation" in raw_obs:
        payload["seg_representation"] = raw_obs["seg_representation"]
    actors_frame_meta = raw_obs.get("actors_frame_meta")
    if actors_frame_meta is not None:
        from openpi.policies.seg_transforms import normalize_actors_frame_meta

        payload["actors_frame_meta"] = normalize_actors_frame_meta(actors_frame_meta)
        payload["actors_frame_meta_source"] = "client"
    if "state" in transformed:
        payload["state"] = np.asarray(transformed["state"])
    return payload
