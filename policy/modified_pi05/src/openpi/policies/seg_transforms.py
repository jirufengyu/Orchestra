"""Segmentation-conditioned transforms for pi0/pi0.5 training.

Training paradigm:
- Seg mask → overlay on RGB image (as visual condition)
- BBox / Point → embedded as text tokens in prompt
- Random dropout of seg condition to preserve original image+text capability
- Optional image dropout to force reliance on seg tokens
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from openpi import transforms

logger = logging.getLogger(__name__)

_SEG_REPRESENTATION_CHOICES = frozenset({"mask", "bbox", "point"})


def parse_seg_representations(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Parse eval-time seg modes.

    Examples:
        ``mask`` | ``bbox`` | ``point`` — single mode (mask = contour overlay + bbox text, per training)
        ``mask,bbox,point`` | ``all`` — overlay + bbox text + point text
        ``mask,point`` — overlay + bbox text + point text (no extra bbox-only pass)
    """
    if value is None:
        return ("mask",)
    if isinstance(value, (tuple, list)):
        items = [str(x).strip().lower() for x in value]
    else:
        text = str(value).strip().lower()
        if text == "all":
            return ("mask", "bbox", "point")
        items = [part.strip() for part in text.replace("+", ",").split(",") if part.strip()]
    if not items:
        return ("mask",)
    ordered: list[str] = []
    for item in items:
        if item not in _SEG_REPRESENTATION_CHOICES:
            raise ValueError(
                f"Invalid seg representation {item!r}; "
                f"expected one of {sorted(_SEG_REPRESENTATION_CHOICES)} or 'all'"
            )
        if item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def normalize_actors_frame_meta(actors_frame_meta: Any) -> dict[str, list[int]]:
    """Normalize client/dataset actor→seg-id maps to ``dict[str, list[int]]``."""
    if not isinstance(actors_frame_meta, dict):
        raise TypeError(
            f"actors_frame_meta must be a dict, got {type(actors_frame_meta).__name__}"
        )
    if not actors_frame_meta:
        raise ValueError("actors_frame_meta must be a non-empty dict.")

    normalized: dict[str, list[int]] = {}
    for name, ids in actors_frame_meta.items():
        id_list = [int(x) for x in np.asarray(ids).reshape(-1)]
        if id_list:
            normalized[str(name)] = id_list
    if not normalized:
        raise ValueError("actors_frame_meta contains no valid segmentation ids.")
    return normalized


@dataclasses.dataclass(frozen=True)
class LoadActorsMeta:
    """Pre-loads all actors_seg_id metadata for seg-conditioned training.

    This transform enriches each sample with per-frame actor-to-segID mapping
    by looking up (dataset_index, episode_index) and frame_index from the
    pre-loaded metadata.

    For MultiLeRobotDataset, each sub-dataset has its own episode_index space.
    We key metadata by (dataset_index, episode_index) to avoid collisions.

    When ``use_subtask_seg`` is True and per-frame ``active_seg_ids`` are
    available (from ``active_actors_meta``), only those actors are exposed to
    downstream seg conditioning. Empty ``active_seg_ids`` fall back to the
    full per-frame actors map.

    When ``use_episode_union_seg`` is True (M1), use the union of all subtask
    active actors in the episode instead of per-frame or full-scene actors.

    Expects data keys: seg_cam_high (H,W uint8 mask), episode_index, frame_index,
                       dataset_index (optional, from MultiLeRobotDataset).
    Produces: actors_frame_meta (dict[str, list[int]]).
    """

    # Keyed by (dataset_index, episode_index) → episode meta dict
    actors_meta_by_key: dict[tuple[int, int], dict[str, Any]]
    active_actors_by_key: dict[tuple[int, int], dict[str, Any]] | None = None
    episode_union_by_key: dict[tuple[int, int], dict[str, list[int]]] | None = None
    use_subtask_seg: bool = False
    use_episode_union_seg: bool = False
    # Rollout / serve: require client-provided actors_frame_meta; never load dataset meta.
    infer_mode: bool = False

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        if self.infer_mode:
            actors_frame_meta = data.get("actors_frame_meta")
            if actors_frame_meta is None:
                raise ValueError(
                    "Seg-conditioned inference requires client-provided actors_frame_meta. "
                    "Dataset actors_seg_id metadata is not loaded at inference time."
                )
            data["actors_frame_meta"] = normalize_actors_frame_meta(actors_frame_meta)
            actors_list = data.get("actors_list")
            if isinstance(actors_list, (list, tuple)) and actors_list:
                data["actors_list"] = [str(name) for name in actors_list]
            else:
                data["actors_list"] = list(data["actors_frame_meta"])
            return data

        ds_raw = data.get("dataset_index")
        ds_idx = int(np.asarray(ds_raw).item()) if ds_raw is not None else 0
        ep_raw = data.get("episode_index")
        ep_idx = int(np.asarray(ep_raw).item()) if ep_raw is not None else -1
        frame_raw = data.get("frame_index")
        frame_idx = int(np.asarray(frame_raw).item()) if frame_raw is not None else -1

        ep_meta = self.actors_meta_by_key.get((ds_idx, ep_idx))
        if ep_meta is not None and frame_idx < len(ep_meta["frames"]):
            actors_frame_meta = ep_meta["frames"][frame_idx]
            actors_list = ep_meta.get("actors", [])

            if self.use_episode_union_seg and self.episode_union_by_key is not None:
                union = self.episode_union_by_key.get((ds_idx, ep_idx))
                if union:
                    actors_frame_meta = union
                    actors_list = list(union.keys())
            elif self.use_subtask_seg and self.active_actors_by_key is not None:
                active_ep = self.active_actors_by_key.get((ds_idx, ep_idx))
                if active_ep is not None and frame_idx < len(active_ep.get("frames", [])):
                    active_seg_ids = active_ep["frames"][frame_idx].get("active_seg_ids", {})
                    if active_seg_ids:
                        actors_frame_meta = active_seg_ids
                        actors_list = list(active_seg_ids.keys())

            data["actors_frame_meta"] = actors_frame_meta
            data["actors_list"] = actors_list
        return data


def episode_union_active_seg_ids_from_meta(
    active_ep: dict[str, Any],
    actors_ep: dict[str, Any] | None,
) -> dict[str, list[int]]:
    """Resolve episode-level union of subtask active actors for M1 seg conditioning."""
    cached = active_ep.get("episode_union_active_seg_ids")
    if cached:
        return cached

    active_frames = active_ep.get("frames", [])
    actors_frames = actors_ep.get("frames", []) if actors_ep else []
    keys: set[str] = set()
    for fr in active_frames:
        keys.update(fr.get("active_keys") or [])
        keys.update((fr.get("active_seg_ids") or {}).keys())
    if not keys or not actors_frames:
        return {}

    merged: dict[str, set[int]] = {k: set() for k in keys}
    for actor_frame in actors_frames:
        for key in keys:
            if key in actor_frame:
                merged[key].update(actor_frame[key])
    return {k: sorted(v) for k, v in merged.items() if v}


def build_episode_union_active_seg_by_key(
    actors_meta_by_key: dict[tuple[int, int], dict[str, Any]],
    active_actors_by_key: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], dict[str, list[int]]]:
    """Precompute per-episode union active seg ids for M1 training."""
    union_by_key: dict[tuple[int, int], dict[str, list[int]]] = {}
    for key, active_ep in active_actors_by_key.items():
        union = episode_union_active_seg_ids_from_meta(active_ep, actors_meta_by_key.get(key))
        if union:
            union_by_key[key] = union
    logger.info(f"Built episode_union_active_seg for {len(union_by_key)} (dataset, episode) pairs.")
    return union_by_key


def load_all_actors_meta(lerobot_roots: list[str] | tuple[str, ...]) -> dict[tuple[int, int], dict[str, Any]]:
    """Load all episode actors_seg_id metadata from multiple LeRobot dataset roots.

    Returns a dict keyed by (dataset_index, episode_index) to handle
    MultiLeRobotDataset where episode indices overlap across sub-datasets.
    """
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    for ds_idx, root in enumerate(lerobot_roots):
        root_path = Path(root)
        episodes_path = root_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            logger.warning(f"episodes.jsonl not found at {episodes_path}, skipping.")
            continue

        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = ep["episode_index"]
                meta_rel = ep.get("actors_seg_id_meta")
                if meta_rel is None:
                    continue
                meta_path = root_path / "meta" / meta_rel
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        ep_data = json.load(mf)
                    ep_data["actors"] = ep.get("actors", [])
                    meta_by_key[(ds_idx, ep_idx)] = ep_data

    logger.info(f"Loaded actors_meta for {len(meta_by_key)} (dataset, episode) pairs.")
    return meta_by_key


def load_all_active_actors_meta(lerobot_roots: list[str] | tuple[str, ...]) -> dict[tuple[int, int], dict[str, Any]]:
    """Load per-frame active actor/seg metadata for MN subtask-scoped seg conditioning."""
    meta_by_key: dict[tuple[int, int], dict[str, Any]] = {}

    for ds_idx, root in enumerate(lerobot_roots):
        root_path = Path(root)
        episodes_path = root_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            logger.warning(f"episodes.jsonl not found at {episodes_path}, skipping.")
            continue

        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = ep["episode_index"]
                meta_rel = ep.get("active_actors_meta")
                if meta_rel is None:
                    continue
                meta_path = root_path / "meta" / meta_rel
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta_by_key[(ds_idx, ep_idx)] = json.load(mf)

    logger.info(f"Loaded active_actors_meta for {len(meta_by_key)} (dataset, episode) pairs.")
    return meta_by_key


# End-effector actor names to exclude from target selection
_EEF_NAMES = {"eef_l", "eef_r", "eef_left", "eef_right"}


@dataclasses.dataclass(frozen=True)
class SegConditionTransform(transforms.DataTransformFn):
    """Core seg-condition transform.

    For each sample:
    1. With prob `p_drop_seg`, drop seg entirely (preserve original capability).
    2. Otherwise, randomly pick a target object from the seg mask.
    3. Randomly choose representation: mask overlay, bbox text, or point text.
    4. Optionally drop base camera image to force seg reliance.

    Expects keys from RepackTransform:
      - seg_cam_high: (H, W) uint8 segmentation mask
      - actors_frame_meta: dict[str, list[int]] (from LoadActorsMeta)
      - images: dict of camera images (C, H, W) uint8
      - prompt: str

    Produces:
      - Modified prompt (with bbox/point tokens appended)
      - Modified images (with seg overlay if mask mode)
      - _drop_base_image: bool flag for downstream AlohaInputs
    """

    p_drop_seg: float = 0.2
    p_drop_image: float = 0.1
    p_keep_wrist_on_drop: float = 0.7
    type_weights: tuple[float, float, float] = (0.34, 0.33, 0.33)  # mask, bbox, point
    # Overlay style: "contour" draws white boundary, "fill" blends a solid color
    overlay_style: str = "contour"
    overlay_alpha: float = 0.4
    overlay_color: tuple[int, int, int] = (255, 255, 255)
    contour_thickness: int = 2
    coord_bins: int = 256
    # Rollout / serve: no dropout; fixed representation(s). Default mask-only.
    # Use parse_seg_representations: "all", "mask,bbox,point", "mask", etc.
    # Per-request override: pass ``seg_representation`` in the observation dict.
    infer_mode: bool = False
    infer_representations: tuple[str, ...] = ("mask",)

    def __call__(self, data: transforms.DataDict) -> transforms.DataDict:
        seg_mask = data.pop("seg_cam_high", None)
        actors_meta = data.pop("actors_frame_meta", None)
        seg_repr_override = data.pop("seg_representation", None)
        data.pop("actors_list", None)
        data.pop("episode_index", None)
        data.pop("frame_index", None)
        data.pop("dataset_index", None)

        if self.infer_mode and actors_meta is None:
            raise ValueError(
                "Seg-conditioned inference requires client-provided actors_frame_meta."
            )

        if seg_mask is None:
            if self.infer_mode:
                raise ValueError(
                    "Seg-conditioned inference requires client-provided seg_cam_high. "
                    "Dataset segmentation is not loaded at inference time."
                )
            return data

        seg_mask = np.asarray(seg_mask)
        if seg_mask.ndim == 3:
            seg_mask = seg_mask.squeeze(0)

        # --- Decide whether to drop seg entirely ---
        if not self.infer_mode and np.random.random() < self.p_drop_seg:
            return data

        # --- Collect all valid objects ---
        objects = self._get_all_objects(seg_mask, actors_meta)
        if not objects:
            if self.infer_mode:
                raise ValueError(
                    "No actors from client actors_frame_meta matched pixels in seg_cam_high."
                )
            return data

        h, w = seg_mask.shape

        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt.item()) if hasattr(prompt, "item") else str(prompt)

        if self.infer_mode:
            reprs = parse_seg_representations(
                seg_repr_override if seg_repr_override is not None else self.infer_representations
            )
            use_mask_overlay = "mask" in reprs
            # Training ``mask`` mode also appends bbox text; explicit ``bbox`` adds text without overlay.
            use_bbox_text = "bbox" in reprs or "mask" in reprs
            use_point_text = "point" in reprs
        else:
            choice = np.random.choice(["mask", "bbox", "point"], p=self.type_weights)
            use_mask_overlay = choice == "mask"
            use_bbox_text = choice in ("mask", "bbox")
            use_point_text = choice == "point"

        # --- Build seg tokens for ALL objects ---
        seg_parts = []
        combined_mask = np.zeros((h, w), dtype=bool)

        for name, ids in objects.items():
            mask_bool = np.isin(seg_mask, ids)
            if not mask_bool.any():
                continue
            ys, xs = np.where(mask_bool)

            if use_bbox_text:
                seg_parts.append(self._make_bbox_text(name, xs, ys, h, w))
            if use_point_text:
                seg_parts.append(self._make_point_text(name, xs, ys, h, w))
            if use_mask_overlay:
                combined_mask |= mask_bool

        if seg_parts:
            data["prompt"] = prompt + "".join(seg_parts)

        if use_mask_overlay and combined_mask.any():
            self._overlay_mask_on_image(data, combined_mask)

        # --- Optionally drop base camera image ---
        if not self.infer_mode and np.random.random() < self.p_drop_image:
            data["_drop_base_image"] = True
            if np.random.random() > self.p_keep_wrist_on_drop:
                data["_drop_wrist_images"] = True

        return data

    def _get_all_objects(
        self, seg_mask: np.ndarray, actors_meta: dict[str, list[int]] | None
    ) -> dict[str, list[int]]:
        """Return all non-eef objects that have pixels in the current seg mask."""
        if actors_meta is not None:
            return {
                name: ids for name, ids in actors_meta.items()
                if name not in _EEF_NAMES and np.isin(seg_mask, ids).any()
            }
        # Fallback: use raw seg IDs (no actor names)
        unique_ids = np.unique(seg_mask)
        object_ids = unique_ids[unique_ids > 0].tolist()
        return {f"obj_{i}": [i] for i in object_ids}

    def _make_bbox_text(self, name: str, xs: np.ndarray, ys: np.ndarray, h: int, w: int) -> str:
        x1 = int(xs.min() / w * self.coord_bins)
        y1 = int(ys.min() / h * self.coord_bins)
        x2 = int(xs.max() / w * self.coord_bins)
        y2 = int(ys.max() / h * self.coord_bins)
        return f" [target: {name}, bbox: {x1} {y1} {x2} {y2}]"

    def _make_point_text(self, name: str, xs: np.ndarray, ys: np.ndarray, h: int, w: int) -> str:
        idx = np.random.randint(len(ys))
        px = int(xs[idx] / w * self.coord_bins)
        py = int(ys[idx] / h * self.coord_bins)
        return f" [target: {name}, point: {px} {py}]"

    def _overlay_mask_on_image(self, data: transforms.DataDict, mask_bool: np.ndarray) -> None:
        images = data.get("images")
        if images is None or "cam_high" not in images:
            return

        img = np.asarray(images["cam_high"])
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0)).copy()
            was_chw = True
        else:
            img = img.copy()
            was_chw = False

        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask_bool.shape

        if (mask_h, mask_w) != (img_h, img_w):
            from PIL import Image as _PILImage
            mask_resized = np.array(
                _PILImage.fromarray(mask_bool.astype(np.uint8) * 255).resize(
                    (img_w, img_h), _PILImage.NEAREST
                )
            ) > 127
        else:
            mask_resized = mask_bool

        color = np.array(self.overlay_color, dtype=np.uint8)

        if self.overlay_style == "contour":
            contour_mask = self._extract_contour(mask_resized, self.contour_thickness)
            img[contour_mask] = color
        else:
            alpha = self.overlay_alpha
            img[mask_resized] = (
                img[mask_resized].astype(np.float32) * (1 - alpha)
                + color.astype(np.float32) * alpha
            ).astype(np.uint8)

        if was_chw:
            img = np.transpose(img, (2, 0, 1))

        images["cam_high"] = img
        data["images"] = images

    @staticmethod
    def _extract_contour(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
        """Extract boundary pixels from a binary mask via erosion."""
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=thickness)
        return mask & ~eroded


@dataclasses.dataclass(frozen=True)
class AlohaSegInputs(transforms.DataTransformFn):
    """Aloha input transform with seg-conditioned image dropout (PyTorch only).

    Same logic as aloha_policy.AlohaInputs but respects _drop_base_image /
    _drop_wrist_images flags set by SegConditionTransform. This class is used
    exclusively in the PyTorch seg training pipeline so the JAX path is untouched.
    """

    adapt_to_pi: bool = True
    use_wrist_cameras: bool = True

    EXPECTED_CAMERAS: tuple[str, ...] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        from openpi.policies.aloha_policy import _decode_aloha, _encode_actions_inv

        drop_base = data.pop("_drop_base_image", False)
        drop_wrist = data.pop("_drop_wrist_images", False)

        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Apply seg-conditioned image dropout
        if drop_base:
            image_masks["base_0_rgb"] = np.False_
        if not self.use_wrist_cameras:
            image_masks["left_wrist_0_rgb"] = np.False_
            image_masks["right_wrist_0_rgb"] = np.False_
        elif drop_wrist:
            image_masks["left_wrist_0_rgb"] = np.False_
            image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for key in ("progress_bin", "subtask_done", "progress_mask"):
            if key in data:
                inputs[key] = data[key]

        for key in (
            "episode_keyframes",
            "episode_keyframe_mask",
            "task_keyframes",
            "task_keyframe_mask",
            "gist_keyframes",
            "gist_keyframe_mask",
            "memory_frames",
            "memory_frame_mask",
        ):
            if key in data:
                inputs[key] = data[key]

        if "memory_frames" in data and isinstance(data["memory_frames"], dict):
            memory_images = {}
            camera_to_model = {
                "cam_high": "base_0_rgb",
                "cam_left_wrist": "left_wrist_0_rgb",
                "cam_right_wrist": "right_wrist_0_rgb",
            }
            for cam_key, model_key in camera_to_model.items():
                if cam_key in data["memory_frames"]:
                    memory_images[model_key] = data["memory_frames"][cam_key]
            if memory_images:
                inputs["memory_frames"] = memory_images

        return inputs


@dataclasses.dataclass(frozen=True)
class MobileSegInputs(transforms.DataTransformFn):
    """Mobile dual-arm seg input transform with image-dropout support (PyTorch only)."""

    adapt_to_pi: bool = True
    use_wrist_cameras: bool = True

    EXPECTED_CAMERAS: tuple[str, ...] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        from openpi.policies.mobile_policy import _decode_aloha, _encode_actions_inv

        drop_base = data.pop("_drop_base_image", False)
        drop_wrist = data.pop("_drop_wrist_images", False)

        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        if drop_base:
            image_masks["base_0_rgb"] = np.False_
        if not self.use_wrist_cameras:
            image_masks["left_wrist_0_rgb"] = np.False_
            image_masks["right_wrist_0_rgb"] = np.False_
        elif drop_wrist:
            image_masks["left_wrist_0_rgb"] = np.False_
            image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        for key in ("progress_bin", "subtask_done", "progress_mask"):
            if key in data:
                inputs[key] = data[key]

        for key in (
            "episode_keyframes",
            "episode_keyframe_mask",
            "task_keyframes",
            "task_keyframe_mask",
            "gist_keyframes",
            "gist_keyframe_mask",
            "memory_frames",
            "memory_frame_mask",
        ):
            if key in data:
                inputs[key] = data[key]

        if "memory_frames" in data and isinstance(data["memory_frames"], dict):
            memory_images = {}
            camera_to_model = {
                "cam_high": "base_0_rgb",
                "cam_left_wrist": "left_wrist_0_rgb",
                "cam_right_wrist": "right_wrist_0_rgb",
            }
            for cam_key, model_key in camera_to_model.items():
                if cam_key in data["memory_frames"]:
                    memory_images[model_key] = data["memory_frames"][cam_key]
            if memory_images:
                inputs["memory_frames"] = memory_images

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaSegOutputs(transforms.DataTransformFn):
    """Output transform for seg-conditioned Aloha (PyTorch only)."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        from openpi.policies.aloha_policy import _encode_actions
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


@dataclasses.dataclass(frozen=True)
class MobileSegOutputs(transforms.DataTransformFn):
    """Output transform for seg-conditioned Mobile dual-arm policies (PyTorch only)."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        from openpi.policies.mobile_policy import Mobile_ARM_DIM, _encode_actions

        actions = np.asarray(data["actions"][:, :Mobile_ARM_DIM])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
