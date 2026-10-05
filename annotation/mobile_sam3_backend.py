"""SAM3.1 multiplex 推理后端：本地引擎与远程 HTTP client 共用工具。"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import inspect
import io
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image


class SamBackendError(ValueError):
    """SAM 后端输入或推理错误。"""


def _direct_json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """直连本机 HTTP，忽略 http_proxy，避免 127.0.0.1 被转到代理口。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not path.startswith("/"):
        path = "/" + path
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            try:
                parsed_body = json.loads(raw)
                message = parsed_body.get("error", raw)
            except json.JSONDecodeError:
                message = raw or f"HTTP {response.status}"
            raise RuntimeError(str(message))
        return json.loads(raw) if raw else {}
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError(f"无法连接 SAM server ({base_url}): {exc}") from exc
    finally:
        conn.close()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def encode_binary_png(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(
        buffer, format="PNG"
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_binary_png(encoded: str, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("L"))
    except Exception as exc:
        raise SamBackendError(f"候选 mask PNG 无效: {exc}") from exc
    if array.shape != expected_shape:
        raise SamBackendError(f"候选 mask 尺寸错误，应为 {expected_shape[::-1]}")
    return array > 0


def validate_pixel_prompts(
    image_size: tuple[int, int],
    points: list[list[float]],
    labels: list[int],
    box: list[float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    width, height = image_size
    point_coords = None
    point_labels = None
    box_array = None
    if points:
        try:
            point_coords = np.asarray(points, dtype=np.float32)
            raw_labels = np.asarray(labels)
        except (TypeError, ValueError) as exc:
            raise SamBackendError("点坐标或标签必须是数值") from exc
        if point_coords.ndim != 2 or point_coords.shape[1] != 2 or raw_labels.ndim != 1:
            raise SamBackendError("点坐标或标签格式无效")
        if len(points) > 100:
            raise SamBackendError("单次推理最多允许 100 个点")
        if len(point_coords) != len(raw_labels) or not np.isin(raw_labels, [0, 1]).all():
            raise SamBackendError("点坐标数量必须与标签一致，标签只能是 0 或 1")
        if not np.isfinite(point_coords).all():
            raise SamBackendError("点坐标必须有限，标签只能是 0 或 1")
        if (
            np.any(point_coords[:, 0] < 0)
            or np.any(point_coords[:, 0] >= width)
            or np.any(point_coords[:, 1] < 0)
            or np.any(point_coords[:, 1] >= height)
        ):
            raise SamBackendError("点坐标超出图像范围")
        point_labels = raw_labels.astype(np.int32)
    elif labels:
        raise SamBackendError("没有点坐标时 labels 必须为空")

    if box is not None:
        try:
            box_array = np.asarray(box, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise SamBackendError("框坐标必须是数值") from exc
        if box_array.shape != (4,):
            raise SamBackendError("框必须为 xyxy 四个数值")
        if not np.isfinite(box_array).all():
            raise SamBackendError("框坐标必须是有限数值")
        x0, y0, x1, y1 = box_array
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise SamBackendError("框坐标顺序无效或超出图像范围")
    if point_coords is None and box_array is None:
        raise SamBackendError("至少需要一个点或一个框")
    return point_coords, point_labels, box_array


def multiplex_relative_prompts(
    image_size: tuple[int, int],
    point_coords: np.ndarray | None,
    box_xyxy: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    width, height = image_size
    scale = np.asarray([width, height], dtype=np.float32)
    relative_points = point_coords / scale if point_coords is not None else None
    relative_box = None
    if box_xyxy is not None:
        x0, y0, x1, y1 = box_xyxy
        relative_box = np.asarray(
            [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
            dtype=np.float32,
        )
    return relative_points, relative_box


def encode_label_map_png(label_map: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(label_map, dtype=np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_label_map_png(encoded: str, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise SamBackendError(f"label map PNG 无效: {exc}") from exc
    if array.shape != expected_shape:
        raise SamBackendError(f"label map 尺寸错误，应为 {expected_shape[::-1]}")
    return array


def decode_rgb_image(encoded: str) -> Image.Image:
    """Decode a base64 encoded RGB frame used by online sessions."""
    try:
        raw = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise SamBackendError(f"在线帧图像无效: {exc}") from exc


def _normalize_mask_batch(masks: np.ndarray, count: int) -> np.ndarray:
    masks = _to_numpy(masks)
    if masks.ndim == 2 and count == 1:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    if masks.ndim != 3 or masks.shape[0] != count:
        raise RuntimeError(f"SAM3.1 mask batch 形状无效: count={count}, masks={masks.shape}")
    return masks


def merge_label_maps(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """把 overlay 中的实例 id 合并进 base（后者覆盖重叠区域）。"""
    if base.shape != overlay.shape:
        raise RuntimeError("label map 尺寸不一致")
    result = base.astype(np.uint8, copy=True)
    for instance_id in np.unique(overlay):
        value = int(instance_id)
        if value == 0:
            continue
        result[overlay == value] = value
    return result


def outputs_to_label_map(
    outputs: dict[str, Any] | None,
    sam_to_instance: dict[int, int],
    expected_shape: tuple[int, int],
) -> np.ndarray:
    """把 SAM3 单帧多对象输出合并为 instance id label map。"""
    label_map = np.zeros(expected_shape, dtype=np.uint8)
    if not isinstance(outputs, dict):
        return label_map
    if "out_obj_ids" not in outputs or "out_binary_masks" not in outputs:
        return label_map
    obj_ids = _to_numpy(outputs["out_obj_ids"]).reshape(-1)
    if len(obj_ids) == 0:
        return label_map
    masks = _normalize_mask_batch(outputs["out_binary_masks"], len(obj_ids))
    ordered = sorted(
        zip(obj_ids.tolist(), masks),
        key=lambda item: sam_to_instance.get(int(item[0]), 999),
    )
    for raw_id, mask in ordered:
        sam_id = int(raw_id)
        instance_id = sam_to_instance.get(sam_id)
        if instance_id is None:
            continue
        bool_mask = np.asarray(mask, dtype=bool)
        if bool_mask.shape != expected_shape:
            raise RuntimeError(
                f"SAM3.1 mask 尺寸错误，应为 {expected_shape[::-1]}，实际为 {bool_mask.shape[::-1]}"
            )
        label_map[bool_mask] = int(instance_id)
    return label_map


def extract_multiplex_mask(
    outputs: dict[str, Any] | None, obj_id: int, expected_shape: tuple[int, int]
) -> np.ndarray:
    if not isinstance(outputs, dict):
        raise RuntimeError("SAM3.1 推理未返回 outputs")
    if "out_obj_ids" not in outputs or "out_binary_masks" not in outputs:
        raise RuntimeError("SAM3.1 outputs 缺少 out_obj_ids 或 out_binary_masks")
    obj_ids = _to_numpy(outputs["out_obj_ids"]).reshape(-1)
    masks = _to_numpy(outputs["out_binary_masks"])
    if len(obj_ids) == 0:
        raise RuntimeError(
            "SAM3.1 未产出有效 mask（输出为空）。请换一个正点再试，或确认 SAM session 已预填帧缓存。"
        )
    if masks.ndim == 2 and len(obj_ids) == 1:
        masks = masks[None, ...]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0, ...]
    if masks.ndim != 3 or masks.shape[0] != len(obj_ids):
        raise RuntimeError(
            f"SAM3.1 mask 输出形状无效: ids={obj_ids.shape}, masks={masks.shape}"
        )
    matches = np.flatnonzero(obj_ids == obj_id)
    if len(matches) == 0 and len(obj_ids) == 1:
        matches = np.asarray([0], dtype=np.int64)
    if len(matches) == 0:
        raise RuntimeError(
            f"SAM3.1 输出中未找到 obj_id={obj_id}，实际返回 {obj_ids.tolist()}"
        )
    mask = np.asarray(masks[int(matches[0])], dtype=bool)
    if mask.shape != expected_shape:
        raise RuntimeError(
            f"SAM3.1 mask 尺寸错误，应为 {expected_shape[::-1]}，实际为 {mask.shape[::-1]}"
        )
    return mask


SAM_FRAMES_DIRNAME = "sam_frames"
SAM_FRAME_MANIFEST = "manifest.json"


def _read_episode_frames(episode_dir: Path, camera: str) -> tuple[list[dict[str, Any]], list[int], list[Path]]:
    with open(episode_dir / "data.json", "r", encoding="utf-8") as stream:
        data_json = json.load(stream)
    frames = data_json.get("data")
    if not isinstance(frames, list) or not frames:
        raise SamBackendError("data.json 缺少有效 data 列表")
    ordered = sorted(frames, key=lambda item: int(item["idx"]))
    frame_indices: list[int] = []
    sources: list[Path] = []
    for frame in ordered:
        idx = int(frame["idx"])
        colors = frame.get("colors")
        if not isinstance(colors, dict) or camera not in colors:
            raise SamBackendError(f"帧 {idx} 缺少 {camera}")
        rel = colors[camera]
        src = (episode_dir / rel).resolve()
        if not src.is_file():
            raise SamBackendError(f"帧 {idx} 图像不存在: {rel}")
        frame_indices.append(idx)
        sources.append(src)
    return ordered, frame_indices, sources


def task_root_from_episode_dir(episode_dir: Path) -> Path:
    return episode_dir.resolve().parent


def cached_episode_frame_dir(task_root: Path, episode_name: str, camera: str) -> Path:
    return task_root / "annotations" / SAM_FRAMES_DIRNAME / episode_name / camera


def write_episode_frame_manifest(
    frame_dir: Path,
    *,
    episode: str,
    camera: str,
    frame_indices: list[int],
    image_size: tuple[int, int],
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0",
        "episode": episode,
        "camera": camera,
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices),
        "image_size": {"width": image_size[0], "height": image_size[1]},
    }
    with open(frame_dir / SAM_FRAME_MANIFEST, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def prepare_episode_frame_dir(
    episode_dir: Path,
    camera: str,
    output_dir: Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """离线构建 SAM3 连续编号帧目录（symlink + manifest）。"""
    episode_path = episode_dir.expanduser().resolve()
    if not (episode_path / "data.json").is_file():
        raise SamBackendError(f"episode 不存在: {episode_path}")
    task_root = task_root_from_episode_dir(episode_path)
    frame_dir = output_dir or cached_episode_frame_dir(task_root, episode_path.name, camera)
    frame_dir = frame_dir.expanduser().resolve()
    _, frame_indices, sources = _read_episode_frames(episode_path, camera)
    image_size: tuple[int, int] | None = None
    if frame_dir.exists() and not force:
        existing = sorted(frame_dir.glob("*.jpg"))
        if len(existing) == len(frame_indices):
            manifest_path = frame_dir / SAM_FRAME_MANIFEST
            if manifest_path.is_file():
                with open(manifest_path, "r", encoding="utf-8") as stream:
                    manifest = json.load(stream)
                if manifest.get("frame_indices") == frame_indices:
                    size = manifest.get("image_size", {})
                    return {
                        "episode": episode_path.name,
                        "camera": camera,
                        "frame_dir": str(frame_dir),
                        "frame_indices": frame_indices,
                        "frame_count": len(frame_indices),
                        "image_size": {
                            "width": int(size.get("width", 0)),
                            "height": int(size.get("height", 0)),
                        },
                        "skipped": True,
                    }
    if frame_dir.exists() and force:
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for session_index, src in enumerate(sources):
        dst = frame_dir / f"{session_index:05d}.jpg"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        if image_size is None:
            with Image.open(src) as image:
                image_size = image.size
    assert image_size is not None
    write_episode_frame_manifest(
        frame_dir,
        episode=episode_path.name,
        camera=camera,
        frame_indices=frame_indices,
        image_size=image_size,
    )
    return {
        "episode": episode_path.name,
        "camera": camera,
        "frame_dir": str(frame_dir),
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "skipped": False,
    }


def resolve_episode_frame_dir(
    episode_dir: Path,
    camera: str,
    task_root: Path | None = None,
) -> tuple[Path, list[int], tuple[int, int], bool]:
    """返回 SAM3 session 使用的帧目录；优先使用任务级预构建缓存。"""
    episode_path = episode_dir.expanduser().resolve()
    task_root = (task_root or task_root_from_episode_dir(episode_path)).resolve()
    cached = cached_episode_frame_dir(task_root, episode_path.name, camera)
    if cached.is_dir():
        jpgs = sorted(cached.glob("*.jpg"))
        manifest_path = cached / SAM_FRAME_MANIFEST
        if jpgs and manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
            frame_indices = [int(value) for value in manifest.get("frame_indices", [])]
            size = manifest.get("image_size", {})
            width = int(size.get("width", 0))
            height = int(size.get("height", 0))
            if len(jpgs) == len(frame_indices) and width > 0 and height > 0:
                return cached, frame_indices, (width, height), True
    _, frame_indices, sources = _read_episode_frames(episode_path, camera)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"sam3-{episode_path.name}-"))
    image_size: tuple[int, int] | None = None
    for session_index, src in enumerate(sources):
        os.symlink(src, temp_dir / f"{session_index:05d}.jpg")
        if image_size is None:
            with Image.open(src) as image:
                image_size = image.size
    assert image_size is not None
    return temp_dir, frame_indices, image_size, False


def build_episode_frame_dir(episode_dir: Path, camera: str) -> tuple[Path, list[int], tuple[int, int]]:
    """为 SAM3 video session 构建连续编号 JPEG 目录，返回 (frame_dir, frame_indices, image_size)。"""
    frame_dir, frame_indices, image_size, _persistent = resolve_episode_frame_dir(
        episode_dir, camera
    )
    return frame_dir, frame_indices, image_size


class Sam3MultiplexEngine:
    """SAM3.1 multiplex 本地引擎：启动时加载模型，按 episode 维护 video session。"""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self._lock = threading.RLock()
        self._model = None
        self._error: str | None = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._online_sessions: dict[str, dict[str, Any]] = {}
        self._online_lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        identity = {"backend": "multiplex", "model_version": "sam3.1"}
        if self._error:
            return {"available": False, "initialized": False, "error": self._error, **identity}
        if not Path(self.checkpoint).is_file():
            return {
                "available": False,
                "initialized": False,
                "error": f"SAM3 checkpoint 不存在: {self.checkpoint}",
                **identity,
            }
        try:
            sam3_spec = importlib.util.find_spec("sam3")
        except (ImportError, ValueError):
            sam3_spec = None
        if sam3_spec is None:
            return {
                "available": False,
                "initialized": False,
                "error": "未找到 sam3 Python 包",
                **identity,
            }
        return {
            "available": True,
            "initialized": self._model is not None,
            "device": self.device,
            "checkpoint": self.checkpoint,
            "active_sessions": len(self._sessions),
            "active_online_sessions": len(self._online_sessions),
            **identity,
        }

    def start_online_session(
        self,
        *,
        camera: str = "color_0",
        max_frames: int = 10_000,
    ) -> dict[str, Any]:
        """Create a growing episode used for causal, online video tracking.

        The first appended frame starts a long-lived predictor session and adds
        Molmo points once. Later frames are injected into that same
        ``inference_state`` and only the new index is propagated forward.
        Offline annotation still uses ``segment_episode`` / ``start_episode_session``.
        """
        if max_frames <= 0:
            raise SamBackendError("max_frames 必须大于 0")
        online_id = uuid.uuid4().hex
        episode_dir = Path(tempfile.mkdtemp(prefix=f"sam3-online-{online_id[:8]}-"))
        (episode_dir / "colors").mkdir(parents=True)
        (episode_dir / "sam_frames").mkdir(parents=True)
        record = {
            "episode_dir": episode_dir,
            "camera": str(camera),
            "max_frames": int(max_frames),
            "frame_indices": [],
            "prompts": None,
            "image_size": None,
            "predictor_session_id": None,
            "sam_to_instance": {},
        }
        self._write_online_manifest(record)
        with self._online_lock:
            self._online_sessions[online_id] = record
        return {"online_session_id": online_id, "camera": str(camera), "frame_count": 0}

    def append_online_frame(
        self,
        online_session_id: str,
        *,
        frame_idx: int,
        image_base64: str,
        prompts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append one frame and return its stable instance-id label map."""
        with self._online_lock:
            record = self._online_sessions.get(online_session_id)
            if record is None:
                raise SamBackendError(f"未知 online session: {online_session_id}")
            frame_indices = record["frame_indices"]
            if frame_indices and frame_idx <= frame_indices[-1]:
                raise SamBackendError("在线 frame_idx 必须严格递增")
            if len(frame_indices) >= record["max_frames"]:
                raise SamBackendError("在线 session 已达到 max_frames")
            if prompts is not None:
                if not isinstance(prompts, list) or not prompts:
                    raise SamBackendError("prompts 必须是非空列表")
                record["prompts"] = prompts
            if not record["prompts"]:
                raise SamBackendError("首帧必须提供 Molmo/SAM prompts")

            image = decode_rgb_image(image_base64)
            image_size = image.size
            if record["image_size"] is None:
                record["image_size"] = image_size
            elif record["image_size"] != image_size:
                raise SamBackendError(
                    f"在线帧尺寸必须保持一致，期望 {record['image_size']}，实际 {image_size}"
                )
            relative = f"colors/{frame_idx:06d}_{record['camera']}.jpg"
            image.save(record["episode_dir"] / relative, format="JPEG", quality=95)
            frame_indices.append(int(frame_idx))
            self._write_online_manifest(record)
            session_index = len(frame_indices) - 1
            self._link_online_sam_frame(record, session_index, frame_idx)
            is_first = record.get("predictor_session_id") is None
            if is_first:
                label_map, sam_to_instance = self._online_start_and_prompt(record)
            else:
                label_map, sam_to_instance = self._online_track_new_frame(
                    record, frame_idx, image
                )

        return {
            "online_session_id": online_session_id,
            "frame": int(frame_idx),
            "frame_count": len(frame_indices),
            "label_map_png_base64": encode_label_map_png(label_map),
            "sam_obj_mapping": {
                str(sam_id): inst_id for sam_id, inst_id in sam_to_instance.items()
            },
        }

    def close_online_session(self, online_session_id: str) -> None:
        with self._online_lock:
            record = self._online_sessions.pop(online_session_id, None)
        if record is None:
            return
        session_id = record.get("predictor_session_id")
        if session_id:
            self.close_session(str(session_id))
        shutil.rmtree(record["episode_dir"], ignore_errors=True)

    @staticmethod
    def _link_online_sam_frame(record: dict[str, Any], session_index: int, frame_idx: int) -> None:
        src = record["episode_dir"] / f"colors/{frame_idx:06d}_{record['camera']}.jpg"
        dst = record["episode_dir"] / "sam_frames" / f"{session_index:05d}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    def _online_start_and_prompt(
        self, record: dict[str, Any]
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Start one predictor session and add Molmo points on frame 0 only."""
        frame_dir = record["episode_dir"] / "sam_frames"
        image_size = record["image_size"]
        assert image_size is not None
        shape = (image_size[1], image_size[0])
        with self._lock:
            self._initialize()
            assert self._model is not None
            started = self._model.handle_request(
                {"type": "start_session", "resource_path": str(frame_dir)}
            )
            session_id = started["session_id"]
            inference_state = (
                getattr(self._model, "_all_inference_states", {})
                .get(session_id, {})
                .get("state")
            )
            if isinstance(inference_state, dict):
                inference_state["is_image_only"] = False
            self._sessions[session_id] = {
                "episode_dir": str(record["episode_dir"]),
                "camera": record["camera"],
                "frame_dir": frame_dir,
                "persistent": True,
                "frame_indices": record["frame_indices"],
                "image_size": image_size,
                "index_by_frame": {
                    idx: pos for pos, idx in enumerate(record["frame_indices"])
                },
                "online": True,
            }
            record["predictor_session_id"] = session_id
            try:
                self._ensure_propagation_frame_cache(session_id)
                session_record = self._session_record(session_id)
                sam_to_instance: dict[int, int] = {}
                seed_label = np.zeros(shape, dtype=np.uint8)
                next_sam_obj_id = 0
                for prompt in record["prompts"]:
                    instance_id = int(prompt["instance_id"])
                    points = prompt.get("points") or []
                    labels = prompt.get("labels") or []
                    if not points:
                        continue
                    sam_obj_id, outputs = self._add_point_prompt(
                        session_id,
                        session_record,
                        0,
                        points,
                        labels,
                        next_sam_obj_id,
                    )
                    next_sam_obj_id = sam_obj_id + 1
                    session_record["next_sam_obj_id"] = next_sam_obj_id
                    sam_to_instance[sam_obj_id] = instance_id
                    if outputs:
                        partial = outputs_to_label_map(
                            outputs, {sam_obj_id: instance_id}, shape
                        )
                        seed_label = merge_label_maps(seed_label, partial)
                if not sam_to_instance:
                    raise SamBackendError("至少一个实例需要打点")
                record["sam_to_instance"] = sam_to_instance
                return seed_label, sam_to_instance
            except Exception:
                record["predictor_session_id"] = None
                self.close_session(session_id)
                raise

    def _online_track_new_frame(
        self,
        record: dict[str, Any],
        frame_idx: int,
        image: Image.Image,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Inject one new frame into the live session and propagate only that index."""
        session_id = str(record["predictor_session_id"])
        sam_to_instance = dict(record["sam_to_instance"])
        image_size = record["image_size"]
        assert image_size is not None
        shape = (image_size[1], image_size[0])
        with self._lock:
            self._initialize()
            assert self._model is not None
            session_record = self._session_record(session_id)
            session_record["index_by_frame"][int(frame_idx)] = len(record["frame_indices"]) - 1
            self._append_frame_to_inference_state(session_id, image)
            self._ensure_propagation_frame_cache(session_id)
            self._keep_online_partial_tracking(session_id)
            frame_maps = self._propagate_all_frames(
                session_id,
                session_record,
                frame_idx,
                sam_to_instance,
                direction="forward",
                max_frame_num_to_track=0,
            )
            label_map = frame_maps.get(int(frame_idx))
            if label_map is None:
                raise RuntimeError(f"SAM3 未返回在线帧 {frame_idx}")
            return label_map, sam_to_instance

    def _keep_online_partial_tracking(self, session_id: str) -> None:
        """Force the next propagate to run SAM2 tracking instead of fetch.

        SAM3.1 interactivity treats two consecutive ``propagation_partial``
        entries as "forward+backward already done" and switches to
        ``propagation_fetch``. Online tracking only ever goes forward one new
        frame, so a fetch would read an empty cache and return a blank mask.
        Dropping prior propagation actions leaves the original add/refine
        prompts, which selects ``propagation_partial`` again.
        """
        session = getattr(self._model, "_all_inference_states", {}).get(session_id)
        if session is None:
            return
        inference_state = session.get("state")
        if not isinstance(inference_state, dict):
            return
        history = inference_state.get("action_history")
        if not isinstance(history, list):
            return
        inference_state["action_history"] = [
            action
            for action in history
            if isinstance(action, dict) and action.get("type") in {"add", "remove", "refine"}
        ]

    def _append_frame_to_inference_state(self, session_id: str, image: Image.Image) -> int:
        """Grow a live SAM3 inference_state by one RGB frame.

        Public ``handle_request`` has no ``add_frame``. Tracker memory already
        lives in the session; this only extends per-frame buffers so
        ``propagate_in_video(start_frame_index=N, max_frame_num_to_track=0)``
        can run the new index.
        """
        session = getattr(self._model, "_all_inference_states", {}).get(session_id)
        if session is None:
            raise SamBackendError(f"未知 session: {session_id}")
        inference_state = session.get("state")
        if not isinstance(inference_state, dict):
            raise SamBackendError(f"session {session_id} 缺少 inference_state")
        new_idx = int(inference_state.get("num_frames", 0))
        input_batch = inference_state.get("input_batch")
        if input_batch is not None:
            self._append_sam3_input_frame(inference_state, image, new_idx)
        inference_state["num_frames"] = new_idx + 1
        inference_state["is_image_only"] = False
        cache = inference_state.setdefault("cached_frame_outputs", {})
        if isinstance(cache, dict):
            cache.setdefault(new_idx, {})
        for key, fill in (
            ("previous_stages_out", None),
            ("per_frame_raw_point_input", None),
            ("per_frame_raw_box_input", None),
            ("per_frame_visual_prompt", None),
            ("per_frame_geometric_prompt", None),
            ("per_frame_cur_step", 0),
        ):
            slots = inference_state.get(key)
            if isinstance(slots, list):
                while len(slots) <= new_idx:
                    slots.append(fill)
        for sam2_state in inference_state.get("sam2_inference_states") or []:
            if isinstance(sam2_state, dict) and "num_frames" in sam2_state:
                sam2_state["num_frames"] = new_idx + 1
        return new_idx

    def _append_sam3_input_frame(
        self,
        inference_state: dict[str, Any],
        image: Image.Image,
        new_idx: int,
    ) -> None:
        import torch
        from sam3.model.data_misc import BatchedPointer, FindStage, convert_my_tensors
        from sam3.model.io_utils import load_resource_as_video_frames
        from sam3.model.sam3_multiplex_tracking import recursive_to

        input_batch = inference_state["input_batch"]
        tensors = input_batch.img_batch.tensors
        model = self._model.model
        offload_video_to_cpu = not (
            hasattr(tensors, "is_cuda") and bool(tensors.is_cuda)
        )
        new_images, _, _ = load_resource_as_video_frames(
            resource_path=[image],
            image_size=int(model.image_size),
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=tuple(model.image_mean),
            img_std=tuple(model.image_std),
        )
        if hasattr(tensors, "device"):
            new_images = new_images.to(device=tensors.device, dtype=tensors.dtype)
        if hasattr(tensors, "shape") and getattr(tensors, "ndim", 0) == 4:
            input_batch.img_batch.tensors = torch.cat([tensors, new_images], dim=0)
        elif isinstance(tensors, list):
            input_batch.img_batch.tensors = list(tensors) + [new_images[0]]
        else:
            raise RuntimeError(f"无法追加 SAM3 img_batch，类型={type(tensors)}")

        dummy_ptrs = BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        )
        stage = FindStage(
            img_ids=[new_idx],
            img_ids_np=np.array([new_idx]),
            text_ids=[0],
            input_boxes=[torch.zeros(258)],
            input_boxes_before_embed=[torch.empty(0, 4)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, 257)],
            input_points_before_embed=[torch.empty(0, 3)],
            input_points_mask=[torch.empty(0)],
            ptrs=dummy_ptrs,
            ptrs_seg=dummy_ptrs,
            object_ids=[],
        )
        stage = convert_my_tensors(stage)
        device = inference_state.get("device")
        if device is not None:
            stage = recursive_to(stage, device, non_blocking=True)
        input_batch.find_inputs.append(stage)
        if hasattr(stage, "text_ids"):
            stage.text_ids[...] = 0
        if isinstance(input_batch.find_targets, list):
            input_batch.find_targets.append(None)
        if isinstance(input_batch.find_metadatas, list):
            input_batch.find_metadatas.append(None)

    @staticmethod
    def _write_online_manifest(record: dict[str, Any]) -> None:
        frames = [
            {
                "idx": frame_idx,
                "colors": {
                    record["camera"]: f"colors/{frame_idx:06d}_{record['camera']}.jpg"
                },
            }
            for frame_idx in record["frame_indices"]
        ]
        payload = {"text": {"goal": "online tracking"}, "data": frames}
        with open(record["episode_dir"] / "data.json", "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)

    def _initialize(self) -> None:
        if self._model is not None:
            return
        if self._error:
            raise RuntimeError(self._error)
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"SAM3 checkpoint 不存在: {self.checkpoint}")
        try:
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            self._model = build_sam3_multiplex_video_predictor(
                checkpoint_path=self.checkpoint,
                use_fa3=False,
                use_rope_real=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
            init_state = self._model.model.init_state
            if "offload_state_to_cpu" not in inspect.signature(init_state).parameters:

                def compatible_init_state(*args, offload_state_to_cpu=False, **kwargs):
                    del offload_state_to_cpu
                    return init_state(*args, **kwargs)

                self._model.model.init_state = compatible_init_state
        except Exception as exc:
            self._model = None
            self._error = f"SAM3 初始化失败: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._error) from exc

    def warm_up(self) -> None:
        with self._lock:
            self._initialize()

    def start_episode_session(self, episode_dir: str | Path, camera: str) -> dict[str, Any]:
        episode_path = Path(episode_dir).expanduser().resolve()
        if not (episode_path / "data.json").is_file():
            raise SamBackendError(f"episode 不存在: {episode_path}")
        task_root = task_root_from_episode_dir(episode_path)
        frame_dir, frame_indices, image_size, persistent = resolve_episode_frame_dir(
            episode_path, camera, task_root
        )
        with self._lock:
            self._initialize()
            assert self._model is not None
            started = self._model.handle_request(
                {"type": "start_session", "resource_path": str(frame_dir)}
            )
            session_id = started["session_id"]
            self._sessions[session_id] = {
                "episode_dir": str(episode_path),
                "camera": camera,
                "frame_dir": frame_dir,
                "persistent": persistent,
                "frame_indices": frame_indices,
                "image_size": image_size,
                "index_by_frame": {idx: pos for pos, idx in enumerate(frame_indices)},
            }
            # SAM3.1 _build_sam2_output 在帧不在 cached_frame_outputs 时直接返回空 dict，
            # 交互式首帧打点会变成 out_obj_ids=[]。新 session 立刻预填空缓存。
            self._ensure_propagation_frame_cache(session_id)
            return {
                "session_id": session_id,
                "episode": episode_path.name,
                "camera": camera,
                "frame_indices": frame_indices,
                "frame_count": len(frame_indices),
                "image_size": {"width": image_size[0], "height": image_size[1]},
            }

    def close_session(self, session_id: str) -> None:
        with self._lock:
            record = self._sessions.pop(session_id, None)
            if record is None:
                return
            if self._model is not None:
                try:
                    self._model.handle_request(
                        {
                            "type": "close_session",
                            "session_id": session_id,
                            "run_gc_collect": False,
                        }
                    )
                except Exception:
                    pass
            if record.get("persistent"):
                return
            frame_dir = record.get("frame_dir")
            if isinstance(frame_dir, Path) and frame_dir.exists():
                shutil.rmtree(frame_dir, ignore_errors=True)

    def _session_record(self, session_id: str) -> dict[str, Any]:
        record = self._sessions.get(session_id)
        if record is None:
            raise SamBackendError(f"未知 session: {session_id}")
        return record

    def _frame_session_index(self, record: dict[str, Any], frame_idx: int) -> int:
        mapping = record["index_by_frame"]
        if frame_idx not in mapping:
            raise SamBackendError(f"frame {frame_idx} 不在当前 episode session 中")
        return int(mapping[frame_idx])

    def _ensure_propagation_frame_cache(self, session_id: str) -> None:
        """为点提示 / 传播预填空的 cached_frame_outputs。

        SAM3.1 `_build_sam2_output` 在帧不在 cached_frame_outputs 时直接返回空 dict，
        丢掉本次 add_prompt 的 refined mask。新 session 没有任何 cache entry，
        交互式「重打当前帧」会得到 out_obj_ids=[]。传播路径同样需要整段预填，
        否则除 seed 帧外 mask 全丢。
        """
        assert self._model is not None
        session = getattr(self._model, "_all_inference_states", {}).get(session_id)
        if session is None:
            return
        inference_state = session.get("state")
        if not isinstance(inference_state, dict):
            return
        cache = inference_state.get("cached_frame_outputs")
        num_frames = inference_state.get("num_frames")
        if not isinstance(cache, dict) or not isinstance(num_frames, int):
            return
        for frame_idx in range(num_frames):
            cache.setdefault(frame_idx, {})

    def _add_point_prompt(
        self,
        session_id: str,
        record: dict[str, Any],
        session_frame: int,
        points: list[list[float]],
        labels: list[int],
        sam_obj_id: int,
    ) -> tuple[int, dict[str, Any] | None]:
        """用 SAM2 点提示添加/细化单个 track；禁止 bootstrap 框以免 reset_state 清掉其它实例。"""
        assert self._model is not None
        image_size = record["image_size"]
        point_coords, point_labels, box_array = validate_pixel_prompts(
            image_size, points, labels, None
        )
        if box_array is not None:
            raise SamBackendError("整段视频分割仅支持点提示")
        relative_points, _ = multiplex_relative_prompts(image_size, point_coords, None)
        assert relative_points is not None and point_labels is not None
        if np.sum(point_labels == 1) == 0:
            raise SamBackendError("每个实例至少需要一个正点")
        response = self._model.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": session_frame,
                "rel_coordinates": True,
                "obj_id": int(sam_obj_id),
                "points": relative_points.tolist(),
                "point_labels": point_labels.tolist(),
            }
        )
        outputs = response.get("outputs") if isinstance(response, dict) else None
        return int(sam_obj_id), outputs if isinstance(outputs, dict) else None

    def predict(
        self,
        session_id: str,
        frame_idx: int,
        points: list[list[float]],
        labels: list[int],
        box: list[float] | None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._initialize()
            assert self._model is not None
            record = self._session_record(session_id)
            image_size = record["image_size"]
            session_frame = self._frame_session_index(record, frame_idx)
            self._ensure_propagation_frame_cache(session_id)
            point_coords, point_labels, box_array = validate_pixel_prompts(
                image_size, points, labels, box
            )
            if point_coords is not None and box_array is not None:
                raise SamBackendError("点和框不能同时提交，请仅保留一种提示")
            if box_array is not None:
                relative_points, relative_box = multiplex_relative_prompts(
                    image_size, None, box_array
                )
                assert relative_box is not None
                target_obj_id = 1
                response = self._model.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": session_frame,
                        "rel_coordinates": True,
                        "bounding_boxes": [relative_box.tolist()],
                        "bounding_box_labels": [1],
                    }
                )
                outputs = response.get("outputs") if isinstance(response, dict) else None
            else:
                next_obj_id = int(record.get("next_sam_obj_id", 0))
                record["next_sam_obj_id"] = next_obj_id + 1
                target_obj_id, outputs = self._add_point_prompt(
                    session_id, record, session_frame, points, labels, next_obj_id
                )
            mask = extract_multiplex_mask(
                outputs,
                obj_id=target_obj_id,
                expected_shape=(image_size[1], image_size[0]),
            )
            record.setdefault("obj_ids", {})[frame_idx] = target_obj_id
            return [{"score": 1.0, "mask_png_base64": encode_binary_png(mask), "obj_id": target_obj_id}]

    def _propagate_all_frames(
        self,
        session_id: str,
        record: dict[str, Any],
        seed_frame_idx: int,
        sam_to_instance: dict[int, int],
        direction: str = "both",
        max_frame_num_to_track: int | None = None,
    ) -> dict[int, np.ndarray]:
        assert self._model is not None
        image_size = record["image_size"]
        shape = (image_size[1], image_size[0])
        session_frame = self._frame_session_index(record, seed_frame_idx)
        self._ensure_propagation_frame_cache(session_id)
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "start_frame_index": session_frame,
            "propagation_direction": direction,
        }
        if max_frame_num_to_track is not None:
            request["max_frame_num_to_track"] = int(max_frame_num_to_track)
        frame_maps: dict[int, np.ndarray] = {}
        for item in self._model.handle_stream_request(request):
            session_index = int(item.get("frame_index", -1))
            if session_index < 0 or session_index >= len(record["frame_indices"]):
                continue
            frame_idx = int(record["frame_indices"][session_index])
            outputs = item.get("outputs") if isinstance(item, dict) else None
            label_map = outputs_to_label_map(outputs, sam_to_instance, shape)
            if frame_idx in frame_maps:
                frame_maps[frame_idx] = merge_label_maps(frame_maps[frame_idx], label_map)
            else:
                frame_maps[frame_idx] = label_map
        return frame_maps

    def segment_episode(
        self,
        episode_dir: str | Path,
        camera: str,
        seed_frame_idx: int,
        prompts: list[dict[str, Any]],
        direction: str = "forward",
    ) -> dict[str, Any]:
        """把 episode 帧序列作为 SAM3 视频输入，首帧多点提示后传播到整段。"""
        if not prompts:
            raise SamBackendError("至少需要一个实例点提示")
        started = self.start_episode_session(episode_dir, camera)
        session_id = started["session_id"]
        try:
            with self._lock:
                self._initialize()
                assert self._model is not None
                record = self._session_record(session_id)
                session_frame = self._frame_session_index(record, seed_frame_idx)
                image_size = record["image_size"]
                shape = (image_size[1], image_size[0])
                self._ensure_propagation_frame_cache(session_id)
                sam_to_instance: dict[int, int] = {}
                seed_label = np.zeros(shape, dtype=np.uint8)
                next_sam_obj_id = 0
                for prompt in prompts:
                    instance_id = int(prompt["instance_id"])
                    points = prompt.get("points") or []
                    labels = prompt.get("labels") or []
                    if not points:
                        continue
                    sam_obj_id, outputs = self._add_point_prompt(
                        session_id,
                        record,
                        session_frame,
                        points,
                        labels,
                        next_sam_obj_id,
                    )
                    next_sam_obj_id = sam_obj_id + 1
                    record["next_sam_obj_id"] = next_sam_obj_id
                    sam_to_instance[sam_obj_id] = instance_id
                    if outputs:
                        partial = outputs_to_label_map(
                            outputs, {sam_obj_id: instance_id}, shape
                        )
                        seed_label = merge_label_maps(seed_label, partial)
                if not sam_to_instance:
                    raise SamBackendError("至少一个实例需要打点")
                frame_maps = self._propagate_all_frames(
                    session_id, record, seed_frame_idx, sam_to_instance, direction
                )
                for frame_idx in record["frame_indices"]:
                    frame_maps.setdefault(frame_idx, np.zeros(shape, dtype=np.uint8))
                if np.any(seed_label):
                    frame_maps[seed_frame_idx] = merge_label_maps(
                        frame_maps.get(seed_frame_idx, np.zeros(shape, dtype=np.uint8)),
                        seed_label,
                    )
                if not frame_maps:
                    raise RuntimeError("SAM3 视频传播未返回任何帧")
                return {
                    "episode": started["episode"],
                    "camera": camera,
                    "seed_frame": seed_frame_idx,
                    "frame_count": len(frame_maps),
                    "sam_obj_mapping": {
                        str(sam_id): inst_id for sam_id, inst_id in sam_to_instance.items()
                    },
                    "frames": [
                        {
                            "frame": frame_idx,
                            "label_map_png_base64": encode_label_map_png(label_map),
                        }
                        for frame_idx, label_map in sorted(frame_maps.items())
                    ],
                }
        finally:
            self.close_session(session_id)

    def propagate(
        self,
        session_id: str,
        start_frame_idx: int,
        obj_id: int,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._initialize()
            assert self._model is not None
            record = self._session_record(session_id)
            image_size = record["image_size"]
            session_frame = self._frame_session_index(record, start_frame_idx)
            request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "start_frame_index": session_frame,
                "propagation_direction": direction,
                "obj_id": obj_id,
            }
            results: list[dict[str, Any]] = []
            for item in self._model.handle_stream_request(request):
                outputs = item.get("outputs") if isinstance(item, dict) else None
                session_index = int(item.get("frame_index", -1))
                if session_index < 0 or session_index >= len(record["frame_indices"]):
                    continue
                frame_idx = int(record["frame_indices"][session_index])
                mask = extract_multiplex_mask(
                    outputs,
                    obj_id=obj_id,
                    expected_shape=(image_size[1], image_size[0]),
                )
                results.append(
                    {
                        "frame": frame_idx,
                        "score": 1.0,
                        "mask_png_base64": encode_binary_png(mask),
                    }
                )
            return results


class Sam3RemoteClient:
    """标注 Web 使用的 SAM HTTP client。"""

    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id: str | None = None
        self._episode: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = _direct_json_request(
            self.base_url,
            method,
            path,
            payload=payload,
            timeout=timeout or self.timeout,
        )
        if not body.get("ok", True):
            raise RuntimeError(body.get("error", "SAM server 请求失败"))
        return body

    def status(self) -> dict[str, Any]:
        try:
            body = self._request("GET", "/api/status")
            return body.get("sam3", {"available": False, "initialized": False, "error": "无状态"})
        except RuntimeError as exc:
            return {"available": False, "initialized": False, "error": str(exc), "backend": "remote"}

    def start_online_session(
        self,
        *,
        camera: str = "color_0",
        max_frames: int = 10_000,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/online/session/start",
            {"camera": camera, "max_frames": int(max_frames)},
        )

    def append_online_frame(
        self,
        online_session_id: str,
        *,
        frame_idx: int,
        image_base64: str,
        prompts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "online_session_id": online_session_id,
            "frame": int(frame_idx),
            "image_base64": image_base64,
        }
        if prompts is not None:
            payload["prompts"] = prompts
        return self._request(
            "POST",
            "/api/online/frame",
            payload,
            timeout=max(self.timeout, 3600.0),
        )

    def close_online_session(self, online_session_id: str) -> None:
        self._request(
            "POST",
            "/api/online/session/close",
            {"online_session_id": online_session_id},
        )

    def _reset_cached_session(self) -> None:
        self._session_id = None
        self._episode = None

    @staticmethod
    def _is_stale_session_error(exc: BaseException) -> bool:
        return "未知 session" in str(exc)

    def ensure_episode(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        force: bool = False,
    ) -> dict[str, Any]:
        if not force and self._episode == episode_name and self._session_id:
            return {"session_id": self._session_id, "episode": episode_name, "camera": camera}
        if self._session_id:
            try:
                self._request(
                    "POST",
                    "/api/session/close",
                    {"session_id": self._session_id},
                )
            except RuntimeError:
                pass
            self._reset_cached_session()
        body = self._request(
            "POST",
            "/api/session/start",
            {"episode_dir": str(episode_dir), "camera": camera},
        )
        self._session_id = body["session_id"]
        self._episode = episode_name
        return body

    def _with_live_session(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        request_fn,
    ):
        session = self.ensure_episode(episode_dir, episode_name, camera)
        try:
            return request_fn(session)
        except RuntimeError as exc:
            if not self._is_stale_session_error(exc):
                raise
            # SAM 重启后内存 session 全丢，网页仍缓存旧 id。清掉后重建一次。
            self._reset_cached_session()
            session = self.ensure_episode(episode_dir, episode_name, camera, force=True)
            return request_fn(session)

    def segment_episode(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        seed_frame_idx: int,
        prompts: list[dict[str, Any]],
        direction: str = "both",
    ) -> dict[str, Any]:
        del episode_name
        return self._request(
            "POST",
            "/api/segment_episode",
            {
                "episode_dir": str(episode_dir),
                "camera": camera,
                "seed_frame": seed_frame_idx,
                "prompts": prompts,
                "direction": direction,
            },
            timeout=max(self.timeout, 3600.0),
        )

    def predict(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        frame_idx: int,
        points: list[list[float]],
        labels: list[int],
        box: list[float] | None,
    ) -> list[dict[str, Any]]:
        def _call(session: dict[str, Any]) -> list[dict[str, Any]]:
            body = self._request(
                "POST",
                "/api/predict",
                {
                    "session_id": session["session_id"],
                    "frame": frame_idx,
                    "points": points,
                    "labels": labels,
                    "box": box,
                },
            )
            return body.get("candidates", [])

        return self._with_live_session(episode_dir, episode_name, camera, _call)

    def propagate(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        start_frame_idx: int,
        obj_id: int,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        def _call(session: dict[str, Any]) -> list[dict[str, Any]]:
            body = self._request(
                "POST",
                "/api/propagate",
                {
                    "session_id": session["session_id"],
                    "frame": start_frame_idx,
                    "obj_id": obj_id,
                    "direction": direction,
                },
            )
            return body.get("frames", [])

        return self._with_live_session(episode_dir, episode_name, camera, _call)
