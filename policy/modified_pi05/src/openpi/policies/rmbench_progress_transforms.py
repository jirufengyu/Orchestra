"""Inject RMBench progress / subtask-done labels into training batches."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from openpi import transforms
from openpi.shared.rmbench_progress import progress_labels_for_frame

logger = logging.getLogger(__name__)

LEROBOT_IMAGE_KEY_TO_CAM_KEY: dict[str, str] = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}


def memory_cam_keys_from_lerobot_keys(
    lerobot_keys: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Map LeRobot parquet image keys to rollout ``images`` dict keys."""
    cam_keys: list[str] = []
    for key in lerobot_keys:
        cam_keys.append(LEROBOT_IMAGE_KEY_TO_CAM_KEY.get(key, key.rsplit(".", 1)[-1]))
    return tuple(cam_keys)


@dataclasses.dataclass(frozen=True)
class InjectRmbenchM1GlobalPrompt(transforms.DataTransformFn):
    """Replace per-frame subtask prompts with the fixed M(1) global instruction."""

    prompt: str

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        return {**data, "prompt": self.prompt}


@dataclasses.dataclass(frozen=True)
class InjectRmbenchM1GlobalPromptPerDataset(transforms.DataTransformFn):
    """Replace per-frame prompts with the fixed M(1) global instruction per dataset."""

    prompts_by_dataset: dict[int, str]

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        ds_raw = data.get("dataset_index")
        # MultiLeRobotDataset supplies dataset_index during training. Rollout
        # requests do not, and already contain the task-specific client prompt;
        # preserve it instead of silently treating every task as dataset 0.
        if ds_raw is None:
            return data
        ds_idx = int(np.asarray(ds_raw).item())
        prompt = self.prompts_by_dataset.get(ds_idx)
        if prompt is None:
            raise ValueError(
                f"No RMBench M1 global prompt for dataset_index={ds_idx}. "
                f"Known indices: {sorted(self.prompts_by_dataset)}"
            )
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class InjectRmbenchEpisodeAnchorKeyframes(transforms.DataTransformFn):
    """Inject the episode-start full image as pi0.5 keyframe memory."""

    anchor_images_by_key: dict[tuple[int, int], np.ndarray]

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if data.get("episode_keyframes") is not None:
            return data

        ds_raw = data.get("dataset_index")
        ds_idx = int(np.asarray(ds_raw).item()) if ds_raw is not None else 0
        ep_raw = data.get("episode_index")
        ep_idx = int(np.asarray(ep_raw).item()) if ep_raw is not None else -1

        anchor = self.anchor_images_by_key.get((ds_idx, ep_idx))
        if anchor is None:
            return data

        data["episode_keyframes"] = anchor[None, ...]
        data["episode_keyframe_mask"] = np.asarray([True], dtype=np.bool_)
        return data


def resolve_lerobot_roots(repo_id: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve LeRobot repo ids to local roots for RMBench metadata sidecars."""
    if repo_id is None:
        return ()
    if isinstance(repo_id, str):
        repo_ids = tuple(r.strip() for r in repo_id.split(",") if r.strip())
    else:
        repo_ids = tuple(str(r).strip() for r in repo_id if str(r).strip())

    cache_root = Path(
        os.environ.get(
            "HF_LEROBOT_HOME",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "lerobot",
        )
    )
    roots: list[str] = []
    for repo in repo_ids:
        repo_path = Path(repo)
        roots.append(str(repo_path if repo_path.exists() else cache_root / repo))
    return tuple(roots)


def load_episode_progress_meta(lerobot_roots: list[str] | tuple[str, ...]) -> dict[tuple[int, int], dict[str, Any]]:
    """Load per-episode metadata keyed by (dataset_index, episode_index)."""
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for ds_idx, root in enumerate(lerobot_roots):
        root_path = Path(root)
        episodes_path = root_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            logger.warning("episodes.jsonl not found at %s", episodes_path)
            continue
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = int(ep["episode_index"])
                actors_rel = ep.get("actors_seg_id_meta")
                num_train = ep.get("length", 0)
                language_segments = ep.get("language_segments", [])
                if actors_rel:
                    actors_path = root_path / "meta" / actors_rel
                    if actors_path.exists():
                        with open(actors_path, "r", encoding="utf-8") as af:
                            actors_meta = json.load(af)
                        num_train = int(actors_meta.get("num_train_frames", num_train))
                        language_segments = actors_meta.get("language_segments", language_segments)
                meta_by_key[(ds_idx, ep_idx)] = {
                    "num_train_frames": num_train,
                    "language_segments": language_segments,
                }
    logger.info("Loaded progress meta for %d (dataset, episode) pairs.", len(meta_by_key))
    return meta_by_key


def load_episode_anchor_images(
    lerobot_roots: list[str] | tuple[str, ...],
    *,
    image_key: str = "observation.images.cam_high",
) -> dict[tuple[int, int], np.ndarray]:
    """Load frame-0 RGB images keyed by (dataset_index, episode_index)."""
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for ds_idx, root in enumerate(lerobot_roots):
        root_path = Path(root)
        episodes_path = root_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            logger.warning("episodes.jsonl not found at %s", episodes_path)
            continue

        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = int(ep["episode_index"])
                try:
                    anchors[(ds_idx, ep_idx)] = _load_episode_frame0_image(root_path, ep_idx, image_key=image_key)
                except FileNotFoundError:
                    logger.warning("Episode parquet not found for %s episode %d", root_path, ep_idx)
                except Exception as exc:
                    logger.warning("Failed to load anchor image for %s episode %d: %s", root_path, ep_idx, exc)

    logger.info("Loaded RMBench anchor images for %d (dataset, episode) pairs.", len(anchors))
    return anchors


def load_episodic_anchor_images_all_cams(
    lerobot_roots: list[str] | tuple[str, ...],
    *,
    image_keys: tuple[str, ...] = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ),
) -> dict[tuple[int, int], np.ndarray]:
    """Load frame-0 RGB for all cameras, stacked as [num_cams, H, W, C]."""
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for ds_idx, root in enumerate(lerobot_roots):
        root_path = Path(root)
        episodes_path = root_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            logger.warning("episodes.jsonl not found at %s", episodes_path)
            continue
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = int(ep["episode_index"])
                try:
                    frames = [
                        _load_episode_frame0_image(root_path, ep_idx, image_key=key) for key in image_keys
                    ]
                    anchors[(ds_idx, ep_idx)] = np.stack(frames, axis=0)
                except Exception as exc:
                    logger.warning(
                        "Failed to load episodic anchor for %s episode %d: %s", root_path, ep_idx, exc
                    )
    logger.info("Loaded episodic multi-camera anchors for %d (dataset, episode) pairs.", len(anchors))
    return anchors


@dataclasses.dataclass(frozen=True)
class InjectEpisodicAnchorKeyframes(transforms.DataTransformFn):
    """Inject episode-start frames from all cameras as anchor keyframes."""

    anchor_images_by_key: dict[tuple[int, int], np.ndarray]

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if data.get("episode_keyframes") is not None:
            return data

        ds_raw = data.get("dataset_index")
        ds_idx = int(np.asarray(ds_raw).item()) if ds_raw is not None else 0
        ep_raw = data.get("episode_index")
        ep_idx = int(np.asarray(ep_raw).item()) if ep_raw is not None else -1

        anchor = self.anchor_images_by_key.get((ds_idx, ep_idx))
        if anchor is None:
            return data

        data["episode_keyframes"] = np.asarray(anchor)
        data["episode_keyframe_mask"] = np.ones((anchor.shape[0],), dtype=np.bool_)
        return data


@dataclasses.dataclass(frozen=True)
class InjectEpisodicGistKeyframes(transforms.DataTransformFn):
    """Inject strided head-camera history as gist keyframes for HEVM."""

    lerobot_roots: tuple[str, ...]
    gist_stride: int = 10
    max_gist_frames: int = 16
    image_key: str = "observation.images.cam_high"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_episode_cache", {})

    def _load_episode_frames(self, ds_idx: int, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (ds_idx, ep_idx)
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            return cached

        import pandas as pd

        root_path = Path(self.lerobot_roots[ds_idx])
        parquet_path = _episode_parquet_path(root_path, ep_idx)
        df = pd.read_parquet(parquet_path, columns=[self.image_key, "frame_index"])
        frame_indices = df["frame_index"].to_numpy()
        images = np.stack([_decode_lerobot_image(v) for v in df[self.image_key]], axis=0)
        self._episode_cache[cache_key] = (frame_indices, images)
        return frame_indices, images

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if data.get("gist_keyframes") is not None:
            return data

        ds_raw = data.get("dataset_index")
        ds_idx = int(np.asarray(ds_raw).item()) if ds_raw is not None else 0
        ep_raw = data.get("episode_index")
        ep_idx = int(np.asarray(ep_raw).item()) if ep_raw is not None else -1
        frame_raw = data.get("frame_index")
        frame_idx = int(np.asarray(frame_raw).item()) if frame_raw is not None else -1
        if ep_idx < 0 or frame_idx < 0 or ds_idx >= len(self.lerobot_roots):
            return data

        frame_indices, images = self._load_episode_frames(ds_idx, ep_idx)
        valid = frame_indices <= frame_idx
        if not valid.any():
            return data

        candidate_indices = frame_indices[valid]
        candidate_images = images[valid]
        selected: list[np.ndarray] = []
        last = -self.gist_stride
        for idx, img in zip(candidate_indices, candidate_images, strict=True):
            idx = int(idx)
            if idx == 0 or idx - last >= self.gist_stride:
                selected.append(img)
                last = idx
        if not selected:
            return data

        selected = selected[-self.max_gist_frames :]
        gist = np.stack(selected, axis=0)
        data["gist_keyframes"] = gist
        data["gist_keyframe_mask"] = np.ones((gist.shape[0],), dtype=np.bool_)
        return data


@dataclasses.dataclass(frozen=True)
class EpisodicMemoryScopeDropout(transforms.DataTransformFn):
    """Training-time dropout over anchor / recent / gist memory scopes."""

    p_drop_anchor: float = 0.2
    p_drop_recent: float = 0.1
    p_drop_gist: float = 0.3

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if np.random.random() < self.p_drop_anchor:
            data.pop("episode_keyframes", None)
            data.pop("episode_keyframe_mask", None)
        if np.random.random() < self.p_drop_recent:
            data.pop("memory_frames", None)
            data.pop("memory_frame_mask", None)
        if np.random.random() < self.p_drop_gist:
            data.pop("gist_keyframes", None)
            data.pop("gist_keyframe_mask", None)
        return data


def _load_episode_frame0_image(root_path: Path, episode_index: int, *, image_key: str) -> np.ndarray:
    import pandas as pd

    parquet_path = _episode_parquet_path(root_path, episode_index)
    df = pd.read_parquet(parquet_path, columns=[image_key, "frame_index"])
    row = df[df["frame_index"] == 0]
    image_value = row[image_key].iloc[0] if not row.empty else df[image_key].iloc[0]
    return _decode_lerobot_image(image_value)


class ShortHorizonMemoryBuffer:
    """Rolling buffer for MEM-style short-horizon memory at inference time."""

    CAMERA_TO_MODEL = {
        "cam_high": "base_0_rgb",
        "cam_left_wrist": "left_wrist_0_rgb",
        "cam_right_wrist": "right_wrist_0_rgb",
    }

    def __init__(
        self,
        *,
        num_frames: int = 6,
        frame_stride: int = 1,
        camera_keys: tuple[str, ...] | None = None,
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.camera_keys = camera_keys or tuple(self.CAMERA_TO_MODEL.keys())
        self._frames_by_key: dict[str, list[np.ndarray]] = {}

    def reset(self) -> None:
        self._frames_by_key = {}

    def append(self, images: dict[str, np.ndarray] | np.ndarray, *, source_key: str | None = None) -> None:
        if isinstance(images, dict):
            for key, image in images.items():
                self._append_one(key, image)
            return
        if source_key is None:
            raise ValueError("source_key is required when appending a single image array.")
        self._append_one(source_key, images)

    def _append_one(self, key: str, image: np.ndarray) -> None:
        image = np.asarray(image)
        if image.ndim == 4:
            image = image[0]
        # Rollout clients (Aloha/RMBench) send CHW; memory_frames must be HWC for ResizeImages.
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        self._frames_by_key.setdefault(key, []).append(image.astype(np.uint8, copy=False))

    def build(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        active_model_keys = [
            self.CAMERA_TO_MODEL[cam_key]
            for cam_key in self.camera_keys
            if cam_key in self.CAMERA_TO_MODEL
        ]
        if not self._frames_by_key:
            empty = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            mask = np.zeros((self.num_frames,), dtype=np.bool_)
            return {model_key: empty for model_key in active_model_keys}, mask

        shared_mask: np.ndarray | None = None
        memory_frames: dict[str, np.ndarray] = {}
        for cam_key in self.camera_keys:
            model_key = self.CAMERA_TO_MODEL.get(cam_key)
            if model_key is None:
                continue
            history = self._frames_by_key.get(cam_key, [])
            if not history:
                continue
            current_idx = len(history) - 1
            offsets = [-(self.num_frames - 1 - i) * self.frame_stride for i in range(self.num_frames)]
            frames: list[np.ndarray] = []
            mask: list[bool] = []
            for offset in offsets:
                target_idx = current_idx + offset
                valid = target_idx >= 0
                frames.append(history[target_idx if valid else 0])
                mask.append(valid)
            memory_frames[model_key] = np.stack(frames, axis=0)
            candidate_mask = np.asarray(mask, dtype=np.bool_)
            shared_mask = candidate_mask if shared_mask is None else (shared_mask & candidate_mask)

        if shared_mask is None:
            empty = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            return {model_key: empty for model_key in active_model_keys}, np.zeros(
                (self.num_frames,), dtype=np.bool_
            )
        return memory_frames, shared_mask


def _episode_parquet_path(root_path: Path, episode_index: int) -> Path:
    rel = Path("data") / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    direct = root_path / rel
    if direct.exists():
        return direct
    matches = sorted((root_path / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"episode_{episode_index:06d}.parquet under {root_path / 'data'}")
    return matches[0]


def _decode_lerobot_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        return image.astype(np.uint8, copy=False)

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)

    if isinstance(value, (bytes, bytearray)):
        with Image.open(io.BytesIO(value)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    raise TypeError(f"Unsupported LeRobot image value type: {type(value)}")


def _normalize_lerobot_image_sequence(image: np.ndarray) -> np.ndarray:
    """Convert LeRobot delta_timestamps images to [K, H, W, C] uint8 for downstream resize."""
    image = np.asarray(image)
    if image.ndim != 4:
        return image
    if image.shape[1] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (0, 2, 3, 1))
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class SplitRecentImageSequences(transforms.DataTransformFn):
    """Split LeRobot delta_timestamps image sequences into current images and memory frames."""

    image_keys: tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    num_frames: int = 6
    frame_stride: int = 1

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        images = data.get("images")
        if not isinstance(images, dict):
            return data

        memory_frames: dict[str, np.ndarray] = {}
        for key in self.image_keys:
            image = images.get(key)
            if image is None:
                continue
            image = np.asarray(image)
            if image.ndim < 4 or image.shape[0] != self.num_frames:
                continue
            memory_frames[key] = _normalize_lerobot_image_sequence(image)
            images[key] = image[-1]

        if not memory_frames:
            return data

        frame_raw = data.get("frame_index")
        if frame_raw is None:
            memory_mask = np.ones((self.num_frames,), dtype=np.bool_)
        else:
            frame_idx = int(np.asarray(frame_raw).item())
            memory_mask = np.asarray(
                [frame_idx - (self.num_frames - 1 - i) * self.frame_stride >= 0 for i in range(self.num_frames)],
                dtype=np.bool_,
            )

        data["images"] = images
        data["memory_frames"] = memory_frames
        data["memory_frame_mask"] = memory_mask
        return data


@dataclasses.dataclass(frozen=True)
class InjectRmbenchProgressLabels(transforms.DataTransformFn):
    """Add progress_bin, subtask_done, progress_mask from episode metadata."""

    episode_meta_by_key: dict[tuple[int, int], dict[str, Any]]
    num_progress_bins: int = 101
    boundary_margin: int = 8

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        ds_raw = data.get("dataset_index")
        ds_idx = int(np.asarray(ds_raw).item()) if ds_raw is not None else 0
        ep_raw = data.get("episode_index")
        ep_idx = int(np.asarray(ep_raw).item()) if ep_raw is not None else -1
        frame_raw = data.get("frame_index")
        frame_idx = int(np.asarray(frame_raw).item()) if frame_raw is not None else -1

        ep_meta = self.episode_meta_by_key.get((ds_idx, ep_idx))
        if ep_meta is None or frame_idx < 0:
            data["progress_bin"] = np.int64(0)
            data["subtask_done"] = np.float32(0.0)
            data["progress_mask"] = np.float32(0.0)
            return data

        num_frames = int(ep_meta.get("num_train_frames", 0))
        language_segments = ep_meta.get("language_segments", [])
        progress_bin, _, subtask_done, has_label = progress_labels_for_frame(
            frame_idx,
            num_frames,
            language_segments,
            num_progress_bins=self.num_progress_bins,
            boundary_margin=self.boundary_margin,
        )
        data["progress_bin"] = np.int64(progress_bin)
        data["subtask_done"] = np.float32(1.0 if subtask_done else 0.0)
        data["progress_mask"] = np.float32(1.0 if has_label else 0.0)
        return data
