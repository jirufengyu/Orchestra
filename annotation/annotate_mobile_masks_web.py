#!/usr/bin/env python3
"""真机 episode 视频 mask 标注 Web。

工作流：
1. 创建任务级物体标签 + 配置占位符候选（label_slots）
2. Qwen 自动为每个 episode 选标签，或手动保存 episode 文本
3. 离线 Molmo+SAM 标注
4. 人工校验 mask

改标签名后可「按模板重建全部 instruction」，不必再跑 Qwen。
个别 episode 可定制 instruction / point_prompt。

启动 Web（校验阶段需要 SAM3 server）::

    python -m annotation.sam3_inference_server \\
      --checkpoint /path/to/sam/sam3.1_multiplex.pt --port 8765 --warm-up

    python -m annotation.annotate_mobile_masks_web \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --sam-server-url http://127.0.0.1:8765 --port 7861
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image

try:
    from .mobile_sam3_backend import (
        Sam3RemoteClient,
        SamBackendError,
        decode_binary_png,
        decode_label_map_png,
    )
    from .pipeline_jobs import QUEUE_FILENAME, read_queue, write_queue
    from .prompt_templates import (
        PromptTemplateError,
        build_episode_prompts,
        build_per_instance_point_prompt_summary,
        normalize_label_slots,
        required_placeholder_count,
        slot_letters,
        split_point_prompt_summary,
    )
    from .episode_auto_label import label_episode_with_qwen, label_episodes_with_qwen
    from .qwen_label_backend import QwenLabelBackendError, QwenVLMClient, resolve_qwen_config
except ImportError:
    from mobile_sam3_backend import (
        Sam3RemoteClient,
        SamBackendError,
        decode_binary_png,
        decode_label_map_png,
    )
    from pipeline_jobs import QUEUE_FILENAME, read_queue, write_queue
    from prompt_templates import (
        PromptTemplateError,
        build_episode_prompts,
        build_per_instance_point_prompt_summary,
        normalize_label_slots,
        required_placeholder_count,
        slot_letters,
        split_point_prompt_summary,
    )
    from episode_auto_label import label_episode_with_qwen, label_episodes_with_qwen
    from qwen_label_backend import QwenLabelBackendError, QwenVLMClient, resolve_qwen_config

DEFAULT_ANNOTATION_CAMERA = "color_0"
TASK_INSTANCES_FILENAME = "instances.json"
EPISODE_LABELS_FILENAME = "episode_labels.json"
EPISODE_RE = re.compile(r"^episode_\d+$")
CAMERA_RE = re.compile(r"^color_\d+$")
FRAME_KEY_RE = re.compile(r"^\d{6}$")
DEFAULT_INSTRUCTION_TEMPLATE = "Place {A} on the {B}"
DEFAULT_COLORS = [
    "#ff4d4f", "#40a9ff", "#73d13d", "#faad14", "#9254de", "#13c2c2",
    "#eb2f96", "#a0d911", "#2f54eb", "#fa8c16",
]


class AnnotationError(ValueError):
    """可安全返回给客户端的输入或数据错误。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def foreground_points(
    points: list[Any], labels: list[Any]
) -> list[list[float]]:
    coords: list[list[float]] = []
    for point, label in zip(points, labels):
        if int(label) != 1:
            continue
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        coords.append([float(point[0]), float(point[1])])
    return coords


def replace_instance_items(
    items: list[dict[str, Any]], instance_id: int, new_item: dict[str, Any]
) -> list[dict[str, Any]]:
    """替换某一 instance 的条目；原有多条会收成一条。"""
    replaced = False
    result: list[dict[str, Any]] = []
    for item in items:
        raw_id = item.get("instance_id")
        if raw_id is None:
            result.append(item)
            continue
        if int(raw_id) != instance_id:
            result.append(item)
            continue
        if not replaced:
            result.append(new_item)
            replaced = True
    if not replaced:
        result.append(new_item)
    return result


def drop_instance_items(
    items: list[dict[str, Any]], instance_id: int
) -> list[dict[str, Any]]:
    """移除某一 instance 的全部条目。"""
    return [
        item
        for item in items
        if item.get("instance_id") is None or int(item["instance_id"]) != int(instance_id)
    ]


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_png(path: Path, array: np.ndarray) -> None:
    if array.dtype != np.uint8 or array.ndim != 2:
        raise AnnotationError("mask 必须是二维 uint8 数组")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Image.fromarray(array, mode="L").save(tmp_name, format="PNG")
        with open(tmp_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def merge_instance_mask(label_map: np.ndarray, candidate: np.ndarray, instance_id: int) -> np.ndarray:
    if label_map.ndim != 2 or candidate.shape != label_map.shape:
        raise AnnotationError("候选 mask 尺寸与图像不一致")
    if not 1 <= int(instance_id) <= 255:
        raise AnnotationError("实例 ID 必须在 1..255")
    result = label_map.astype(np.uint8, copy=True)
    result[result == instance_id] = 0
    result[np.asarray(candidate, dtype=bool)] = instance_id
    return result


def clear_instance_mask(label_map: np.ndarray, instance_id: int) -> np.ndarray:
    result = label_map.astype(np.uint8, copy=True)
    result[result == int(instance_id)] = 0
    return result


class AnnotationStore:
    """数据发现和 sidecar 存储层；默认只标注 head camera。"""

    def __init__(self, data_root: str | Path, annotation_camera: str = DEFAULT_ANNOTATION_CAMERA):
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"数据根目录不存在: {self.root}")
        if not CAMERA_RE.fullmatch(annotation_camera):
            raise ValueError(f"无效 annotation camera: {annotation_camera}")
        self.annotation_camera = annotation_camera
        self._lock = threading.RLock()

    def task_annotations_dir(self) -> Path:
        return self.root / "annotations"

    def task_instances_path(self) -> Path:
        return self.task_annotations_dir() / TASK_INSTANCES_FILENAME

    def episode_labels_path(self) -> Path:
        return self.task_annotations_dir() / EPISODE_LABELS_FILENAME

    @staticmethod
    def normalize_instance_colors(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in sorted(instances, key=lambda value: int(value["id"])):
            entry = dict(item)
            entry["color"] = DEFAULT_COLORS[(int(entry["id"]) - 1) % len(DEFAULT_COLORS)].lower()
            normalized.append(entry)
        return normalized

    def load_task_instances(self) -> list[dict[str, Any]]:
        with self._lock:
            path = self.task_instances_path()
            if path.is_file():
                with open(path, "r", encoding="utf-8") as stream:
                    data = json.load(stream)
                instances = data.get("instances")
                if not isinstance(instances, list):
                    raise AnnotationError(f"{TASK_INSTANCES_FILENAME} 缺少 instances 列表")
                instances = sorted(instances, key=lambda item: int(item["id"]))
                colors = {str(item.get("color", "")).lower() for item in instances}
                if len(colors) < len(instances):
                    instances = self.normalize_instance_colors(instances)
                    self.save_task_instances(instances)
                return instances
            for episode_path in sorted(self.root.iterdir()):
                if not episode_path.is_dir() or not EPISODE_RE.fullmatch(episode_path.name):
                    continue
                meta_path = episode_path / "annotations" / "annotation.json"
                if not meta_path.is_file():
                    continue
                with open(meta_path, "r", encoding="utf-8") as stream:
                    meta = json.load(stream)
                legacy = meta.get("instances")
                if isinstance(legacy, list) and legacy:
                    legacy = self.normalize_instance_colors(legacy)
                    self.save_task_instances(legacy)
                    return legacy
            return []

    def save_task_instances(self, instances: list[dict[str, Any]]) -> None:
        atomic_write_json(
            self.task_instances_path(),
            {"version": "1.0", "instances": sorted(instances, key=lambda item: int(item["id"]))},
        )

    def load_episode_labels_file(self) -> dict[str, Any]:
        with self._lock:
            path = self.episode_labels_path()
            if not path.is_file():
                return {
                    "version": "1.2",
                    "instruction_template": DEFAULT_INSTRUCTION_TEMPLATE,
                    "label_slots": {},
                    "episodes": {},
                }
            with open(path, "r", encoding="utf-8") as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise AnnotationError(f"{EPISODE_LABELS_FILENAME} 格式无效")
            data.setdefault("version", "1.2")
            data.setdefault("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE)
            data.setdefault("label_slots", {})
            data.setdefault("episodes", {})
            if not isinstance(data["episodes"], dict):
                raise AnnotationError(f"{EPISODE_LABELS_FILENAME} 缺少 episodes 字典")
            return data

    def save_episode_labels_file(self, data: dict[str, Any]) -> None:
        payload = {
            "version": data.get("version", "1.2"),
            "instruction_template": data.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE),
            "label_slots": data.get("label_slots", {}),
            "episodes": data.get("episodes", {}),
        }
        atomic_write_json(self.episode_labels_path(), payload)

    def get_label_slots(self) -> dict[str, list[int]]:
        data = self.load_episode_labels_file()
        raw = data.get("label_slots") or {}
        if not isinstance(raw, dict):
            raise AnnotationError("label_slots 必须是字典")
        return {str(key): [int(value) for value in values] for key, values in raw.items()}

    def set_label_slots(self, label_slots: dict[str, Any]) -> dict[str, list[int]]:
        instances = self.load_task_instances()
        template = self.get_instruction_template()
        normalized = normalize_label_slots(label_slots, instruction_template=template, instances=instances)
        with self._lock:
            data = self.load_episode_labels_file()
            data["label_slots"] = normalized
            self.save_episode_labels_file(data)
            return normalized

    def get_instruction_template(self) -> str:
        return str(self.load_episode_labels_file().get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE))

    def set_instruction_template(self, template: str) -> str:
        template = str(template).strip()
        if not template:
            raise AnnotationError("instruction 模板不能为空")
        required_placeholder_count(template)
        with self._lock:
            data = self.load_episode_labels_file()
            data["instruction_template"] = template
            self.save_episode_labels_file(data)
            return template

    def load_episode_labels_map(self) -> dict[str, list[int]]:
        data = self.load_episode_labels_file()
        result: dict[str, list[int]] = {}
        for episode, value in data["episodes"].items():
            if not EPISODE_RE.fullmatch(str(episode)):
                continue
            if isinstance(value, dict):
                ids = value.get("instance_ids") or value.get("labels") or []
            else:
                ids = value
            if not isinstance(ids, list):
                raise AnnotationError(f"{episode} 的标签列表无效")
            result[str(episode)] = [int(item) for item in ids]
        return result

    def get_episode_label_record(self, episode: str) -> dict[str, Any]:
        self.episode_dir(episode)
        data = self.load_episode_labels_file()
        value = data["episodes"].get(episode, {})
        if isinstance(value, dict):
            instance_ids = [int(item) for item in value.get("instance_ids") or []]
            point_prompt = str(value.get("point_prompt", ""))
            custom = bool(value.get("custom"))
            if not point_prompt and instance_ids and not custom:
                lookup = {int(item["id"]): str(item["name"]) for item in self.load_task_instances()}
                labels = [lookup[iid] for iid in instance_ids if iid in lookup]
                if labels:
                    try:
                        point_prompt = build_per_instance_point_prompt_summary(labels)
                    except PromptTemplateError:
                        pass
            return {
                "instance_ids": instance_ids,
                "instruction": value.get("instruction", ""),
                "point_prompt": point_prompt,
                "custom": custom,
            }
        if isinstance(value, list):
            return {
                "instance_ids": [int(item) for item in value],
                "instruction": "",
                "point_prompt": "",
                "custom": False,
            }
        return {"instance_ids": [], "instruction": "", "point_prompt": "", "custom": False}

    def refresh_episode_point_prompts(self, *, save: bool = True) -> int:
        """把 episode_labels.json 里旧的合并 point_prompt 刷新为逐 instance 格式。"""
        lookup = {int(item["id"]): str(item["name"]) for item in self.load_task_instances()}
        with self._lock:
            data = self.load_episode_labels_file()
            episodes = data.get("episodes") or {}
            updated = 0
            for value in episodes.values():
                if not isinstance(value, dict) or value.get("custom"):
                    continue
                instance_ids = [int(item) for item in value.get("instance_ids") or []]
                labels = [lookup[iid] for iid in instance_ids if iid in lookup]
                if not labels:
                    continue
                try:
                    new_prompt = build_per_instance_point_prompt_summary(labels)
                except PromptTemplateError:
                    continue
                if str(value.get("point_prompt") or "") != new_prompt:
                    value["point_prompt"] = new_prompt
                    updated += 1
            if save and updated:
                self.save_episode_labels_file(data)
            return updated

    def get_episode_labels(self, episode: str) -> list[int]:
        return list(self.get_episode_label_record(episode)["instance_ids"])

    def set_episode_labels(self, episode: str, instance_ids: list[int]) -> dict[str, Any]:
        instances = self.load_task_instances()
        valid_ids = {int(item["id"]) for item in instances}
        normalized: list[int] = []
        seen: set[int] = set()
        for value in instance_ids:
            instance_id = int(value)
            if instance_id not in valid_ids:
                raise AnnotationError(f"实例 {instance_id} 不存在")
            if instance_id in seen:
                continue
            seen.add(instance_id)
            normalized.append(instance_id)
        with self._lock:
            data = self.load_episode_labels_file()
            template = str(data.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE))
            episodes = data["episodes"]
            if not normalized:
                episodes.pop(episode, None)
                self.save_episode_labels_file(data)
                return {
                    "instance_ids": [],
                    "instruction": "",
                    "point_prompt": "",
                    "custom": False,
                }
            try:
                prompts = build_episode_prompts(
                    instruction_template=template,
                    instances=instances,
                    instance_ids=normalized,
                )
            except PromptTemplateError as exc:
                raise AnnotationError(str(exc)) from exc
            episodes[episode] = {
                "instance_ids": prompts["instance_ids"],
                "instruction": prompts["instruction"],
                "point_prompt": prompts["point_prompt"],
                "custom": False,
            }
            self.save_episode_labels_file(data)
            return dict(episodes[episode])

    def set_episode_prompts(
        self,
        episode: str,
        *,
        instruction: str | None = None,
        point_prompt: str | None = None,
        custom: bool = True,
    ) -> dict[str, Any]:
        """覆盖单个 episode 的 instruction / point_prompt，不改 instance_ids。"""
        self.episode_dir(episode)
        if instruction is None and point_prompt is None:
            raise AnnotationError("请提供 instruction 或 point_prompt")
        instruction_text = None if instruction is None else str(instruction).strip()
        point_text = None if point_prompt is None else str(point_prompt).strip()
        if instruction is not None and (not instruction_text or len(instruction_text) > 800):
            raise AnnotationError("instruction 长度必须为 1..800")
        if point_prompt is not None and (not point_text or len(point_text) > 1500):
            raise AnnotationError("point_prompt 长度必须为 1..1500")
        with self._lock:
            data = self.load_episode_labels_file()
            value = data["episodes"].get(episode)
            if not isinstance(value, dict) or not value.get("instance_ids"):
                raise AnnotationError(f"{episode} 尚未分配标签，无法定制文本")
            instance_ids = [int(item) for item in value["instance_ids"]]
            if point_text is not None:
                parts = split_point_prompt_summary(point_text)
                if len(instance_ids) > 1 and len(parts) != len(instance_ids):
                    raise AnnotationError(
                        f"多个物体时 point_prompt 需用 | 分成 {len(instance_ids)} 段，当前 {len(parts)} 段"
                    )
            if instruction_text is not None:
                value["instruction"] = instruction_text
            if point_text is not None:
                value["point_prompt"] = point_text
            value["custom"] = bool(custom)
            self.save_episode_labels_file(data)
            return {
                "instance_ids": instance_ids,
                "instruction": str(value.get("instruction") or ""),
                "point_prompt": str(value.get("point_prompt") or ""),
                "custom": bool(value.get("custom")),
            }

    def rebuild_episode_prompts(
        self,
        episodes: list[str] | None = None,
        *,
        skip_custom: bool = True,
    ) -> dict[str, Any]:
        """用当前标签名 + instruction 模板重写文本，不重新选标签。"""
        instances = self.load_task_instances()
        with self._lock:
            data = self.load_episode_labels_file()
            template = str(data.get("instruction_template", DEFAULT_INSTRUCTION_TEMPLATE))
            all_episodes = data.get("episodes") or {}
            names = [str(item) for item in episodes] if episodes else list(all_episodes.keys())
            updated: list[str] = []
            skipped: list[dict[str, str]] = []
            failed: list[dict[str, str]] = []
            for name in names:
                value = all_episodes.get(name)
                if not isinstance(value, dict):
                    skipped.append({"episode": name, "reason": "无记录"})
                    continue
                if skip_custom and value.get("custom"):
                    skipped.append({"episode": name, "reason": "已定制"})
                    continue
                instance_ids = [int(item) for item in value.get("instance_ids") or []]
                if not instance_ids:
                    skipped.append({"episode": name, "reason": "无标签"})
                    continue
                try:
                    prompts = build_episode_prompts(
                        instruction_template=template,
                        instances=instances,
                        instance_ids=instance_ids,
                    )
                except PromptTemplateError as exc:
                    failed.append({"episode": name, "error": str(exc)})
                    continue
                value["instruction"] = prompts["instruction"]
                value["point_prompt"] = prompts["point_prompt"]
                value["custom"] = False
                updated.append(name)
            if updated:
                self.save_episode_labels_file(data)
            return {
                "updated": updated,
                "skipped": skipped,
                "failed": failed,
                "counts": {
                    "updated": len(updated),
                    "skipped": len(skipped),
                    "failed": len(failed),
                },
            }

    def list_episodes(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and EPISODE_RE.fullmatch(path.name) and (path / "data.json").is_file():
                try:
                    info = self.episode_info(path.name)
                    result.append({
                        "name": path.name,
                        "frame_count": info["frame_count"],
                        "confirmed": info["progress"]["confirmed"],
                        "total": info["progress"]["total"],
                    })
                except (OSError, ValueError, KeyError):
                    result.append({"name": path.name, "error": "data.json 无效"})
        return result

    def episode_dir(self, episode: str) -> Path:
        if not isinstance(episode, str) or not EPISODE_RE.fullmatch(episode):
            raise AnnotationError("episode 名称无效")
        path = (self.root / episode).resolve()
        if path.parent != self.root or not (path / "data.json").is_file():
            raise AnnotationError("episode 不存在")
        return path

    def _read_data(self, episode: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
        ep_dir = self.episode_dir(episode)
        with open(ep_dir / "data.json", "r", encoding="utf-8") as stream:
            data_json = json.load(stream)
        frames = data_json.get("data")
        if not isinstance(frames, list):
            raise AnnotationError("data.json 缺少 data 列表")
        seen: set[int] = set()
        for frame in frames:
            idx = frame.get("idx") if isinstance(frame, dict) else None
            if not isinstance(idx, int) or idx < 0 or idx in seen:
                raise AnnotationError("帧 idx 必须是唯一的非负整数")
            seen.add(idx)
            colors = frame.get("colors")
            if not isinstance(colors, dict) or self.annotation_camera not in colors:
                raise AnnotationError(f"帧 {idx} 缺少 {self.annotation_camera}")
            relative = colors[self.annotation_camera]
            if not isinstance(relative, str) or not relative:
                raise AnnotationError(f"帧 {idx} 的 {self.annotation_camera} 图像路径无效")
        return ep_dir, data_json, frames

    @staticmethod
    def _frame(frames: list[dict[str, Any]], frame_idx: int) -> dict[str, Any]:
        if not isinstance(frame_idx, int) or frame_idx < 0:
            raise AnnotationError("frame 必须是非负整数")
        for frame in frames:
            if frame["idx"] == frame_idx:
                return frame
        raise AnnotationError("frame 不存在")

    def image_path(self, episode: str, frame_idx: int, camera: str | None = None) -> Path:
        camera = camera or self.annotation_camera
        if not isinstance(camera, str) or not CAMERA_RE.fullmatch(camera):
            raise AnnotationError("camera 名称无效")
        ep_dir, _, frames = self._read_data(episode)
        frame = self._frame(frames, frame_idx)
        rel = frame["colors"].get(camera)
        if not isinstance(rel, str) or not rel:
            raise AnnotationError("该帧不存在此 camera")
        candidate = (ep_dir / rel).resolve()
        try:
            candidate.relative_to(ep_dir)
        except ValueError as exc:
            raise AnnotationError("图像路径越界") from exc
        if candidate.suffix.lower() not in {".jpg", ".jpeg"} or not candidate.is_file():
            raise AnnotationError("JPEG 图像不存在或格式无效")
        return candidate

    def _schema(self, episode: str) -> dict[str, Any]:
        _, _, frames = self._read_data(episode)
        if not frames:
            raise AnnotationError("episode 没有可用图像")
        sample_path = self.image_path(episode, frames[0]["idx"], self.annotation_camera)
        with Image.open(sample_path) as image:
            width, height = image.size
        return {
            "version": "1.1",
            "image_size": {"width": width, "height": height},
            "cameras": {self.annotation_camera: {"name": self.annotation_camera, "role": "head"}},
            "frames": {},
        }

    def _annotation_path(self, episode: str, *parts: str) -> Path:
        ep_dir = self.episode_dir(episode)
        path = (ep_dir / "annotations" / Path(*parts)).resolve()
        try:
            path.relative_to(ep_dir)
        except ValueError as exc:
            raise AnnotationError("annotations 路径越界") from exc
        return path

    def metadata_path(self, episode: str) -> Path:
        return self._annotation_path(episode, "annotation.json")

    def load_metadata(self, episode: str) -> dict[str, Any]:
        with self._lock:
            path = self.metadata_path(episode)
            if not path.exists():
                meta = self._schema(episode)
            else:
                with open(path, "r", encoding="utf-8") as stream:
                    meta = json.load(stream)
                required = {"version", "image_size", "cameras", "frames"}
                if not required.issubset(meta):
                    raise AnnotationError("annotation.json 缺少必要字段")
                if self.annotation_camera not in meta["cameras"]:
                    meta.setdefault("cameras", {})[self.annotation_camera] = {
                        "name": self.annotation_camera,
                        "role": "head",
                    }
            meta["instances"] = self.load_task_instances()
            return meta

    def save_metadata(self, episode: str, meta: dict[str, Any]) -> None:
        payload = {key: meta[key] for key in ("version", "image_size", "cameras", "frames") if key in meta}
        atomic_write_json(self.metadata_path(episode), payload)

    def episode_info(self, episode: str) -> dict[str, Any]:
        _, data_json, frames = self._read_data(episode)
        meta = self.load_metadata(episode)
        indices = [frame["idx"] for frame in frames]
        camera = self.annotation_camera
        confirmed = sum(
            1
            for idx in indices
            if meta["frames"].get(f"{idx:06d}", {}).get(camera, {}).get("status") == "confirmed"
        )
        record = self.get_episode_label_record(episode)
        return {
            "episode": episode,
            "frame_count": len(frames),
            "frame_indices": indices,
            "camera": camera,
            "cameras": [camera],
            "instances": self.load_task_instances(),
            "frames": meta["frames"],
            "image_size": meta["image_size"],
            "goal": data_json.get("text", {}).get("goal", ""),
            "target_instance_ids": record.get("instance_ids") or [],
            "instruction_template": self.get_instruction_template(),
            "label_slots": self.get_label_slots(),
            "instruction": record.get("instruction", ""),
            "point_prompt": record.get("point_prompt", ""),
            "custom": bool(record.get("custom")),
            "auto_annotation": self.get_auto_job_record(episode),
            "progress": {"confirmed": confirmed, "total": len(indices)},
        }

    def get_auto_job_record(self, episode: str) -> dict[str, Any]:
        path = self.root / "annotations" / QUEUE_FILENAME
        if not path.is_file():
            return {"available": False}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            job = json.loads(line)
            if job.get("episode") != episode:
                continue
            return {
                "available": True,
                "status": job.get("status"),
                "seed_frame": job.get("seed_frame"),
                "camera": job.get("camera") or self.annotation_camera,
                "molmo_points": job.get("molmo_points") or [],
                "sam_prompts": job.get("sam_prompts") or [],
                "warnings": job.get("warnings") or [],
                "point_prompt": job.get("point_prompt", ""),
                "molmo_elapsed_s": job.get("molmo_elapsed_s"),
                "sam_elapsed_s": job.get("sam_elapsed_s"),
                "total_elapsed_s": job.get("total_elapsed_s"),
            }
        return {"available": False}

    def overwrite_instance_seed_points(
        self,
        episode: str,
        instance_id: int,
        points: list[Any],
        labels: list[Any],
    ) -> dict[str, Any]:
        """用手工前景点覆盖该实例的 Molmo / SAM 种子点。"""
        fg_points = foreground_points(points, labels)
        if not fg_points:
            return {"updated": False, "reason": "需要至少一个前景点"}
        path = self.root / "annotations" / QUEUE_FILENAME
        with self._lock:
            if not path.is_file():
                return {"updated": False, "reason": "无自动标注记录"}
            jobs = read_queue(path)
            job = next((item for item in jobs if item.get("episode") == episode), None)
            if job is None:
                return {"updated": False, "reason": "无自动标注记录"}
            seed = fg_points[0]
            molmo_item = {
                "image_num": 0,
                "label": 1,
                "object_id": 1,
                "point": seed,
                "x": seed[0],
                "y": seed[1],
                "instance_id": instance_id,
                "source": "manual",
            }
            sam_item = {
                "instance_id": instance_id,
                "points": fg_points,
                "labels": [1] * len(fg_points),
                "source": "manual",
            }
            molmo_points = replace_instance_items(
                list(job.get("molmo_points") or []), instance_id, molmo_item
            )
            sam_prompts = replace_instance_items(
                list(job.get("sam_prompts") or []), instance_id, sam_item
            )
            job["molmo_points"] = molmo_points
            job["sam_prompts"] = sam_prompts
            job["updated_at"] = utc_now()
            result = job.get("result")
            if isinstance(result, dict):
                result = dict(result)
                result["sam_prompts"] = sam_prompts
                job["result"] = result
            write_queue(path, jobs)
        return {
            "updated": True,
            "molmo_points": molmo_points,
            "sam_prompts": sam_prompts,
        }

    def _validate_target(
        self, episode: str, frame_idx: int, camera: str | None = None
    ) -> tuple[dict[str, Any], tuple[int, int]]:
        camera = camera or self.annotation_camera
        self.image_path(episode, frame_idx, camera)
        meta = self.load_metadata(episode)
        if camera not in meta["cameras"]:
            raise AnnotationError("camera 不在标注元数据中")
        size = meta["image_size"]
        shape = (int(size["height"]), int(size["width"]))
        with Image.open(self.image_path(episode, frame_idx, camera)) as image:
            if image.size != shape[::-1]:
                raise AnnotationError(f"图像尺寸不一致，应为 {shape[::-1]}，实际为 {image.size}")
        return meta, shape

    def mask_path(self, episode: str, frame_idx: int, camera: str | None = None) -> Path:
        camera = camera or self.annotation_camera
        self.image_path(episode, frame_idx, camera)
        return self._annotation_path(episode, "masks", f"{frame_idx:06d}_{camera}.png")

    def load_mask(self, episode: str, frame_idx: int, camera: str | None = None) -> np.ndarray:
        camera = camera or self.annotation_camera
        meta, shape = self._validate_target(episode, frame_idx, camera)
        del meta
        path = self.mask_path(episode, frame_idx, camera)
        if not path.exists():
            return np.zeros(shape, dtype=np.uint8)
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.uint8)
        if array.shape != shape:
            raise AnnotationError(f"已有 mask 尺寸错误: {path}")
        return array

    @staticmethod
    def _instance(instances: list[dict[str, Any]], instance_id: int) -> dict[str, Any]:
        if not isinstance(instance_id, int) or not 1 <= instance_id <= 255:
            raise AnnotationError("实例 ID 无效")
        for item in instances:
            if item["id"] == instance_id:
                return item
        raise AnnotationError("实例不存在")

    def create_instance(self, name: str, color: str | None = None) -> dict[str, Any]:
        name = str(name).strip()
        if not name or len(name) > 80:
            raise AnnotationError("实例名称长度必须为 1..80")
        with self._lock:
            instances = self.load_task_instances()
            if any(item["name"] == name for item in instances):
                raise AnnotationError("实例名称已存在")
            used = {int(item["id"]) for item in instances}
            instance_id = next((value for value in range(1, 256) if value not in used), None)
            if instance_id is None:
                raise AnnotationError("实例数量已达到 255")
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = DEFAULT_COLORS[(instance_id - 1) % len(DEFAULT_COLORS)]
            item = {"id": instance_id, "name": name, "color": color.lower()}
            instances.append(item)
            self.save_task_instances(instances)
            return item

    def rename_instance(self, instance_id: int, name: str) -> dict[str, Any]:
        name = str(name).strip()
        if not name or len(name) > 80:
            raise AnnotationError("实例名称长度必须为 1..80")
        with self._lock:
            instances = self.load_task_instances()
            if any(item["name"] == name and item["id"] != instance_id for item in instances):
                raise AnnotationError("实例名称已存在")
            item = self._instance(instances, instance_id)
            item["name"] = name
            self.save_task_instances(instances)
            return item

    def delete_instance(self, instance_id: int) -> None:
        with self._lock:
            instances = self.load_task_instances()
            self._instance(instances, instance_id)
            for episode in self.list_episodes():
                if "error" in episode:
                    continue
                masks_dir = self.episode_dir(episode["name"]) / "annotations" / "masks"
                if not masks_dir.is_dir():
                    continue
                for path in masks_dir.glob(f"*_{self.annotation_camera}.png"):
                    with Image.open(path) as image:
                        if np.any(np.asarray(image.convert("L")) == instance_id):
                            raise AnnotationError(
                                f"实例 {instance_id} 在 {episode['name']} 仍有 mask 像素，请先清除后再删除"
                            )
            instances = [item for item in instances if item["id"] != instance_id]
            self.save_task_instances(instances)

    @staticmethod
    def _set_status(meta: dict[str, Any], frame_idx: int, camera: str, status: str) -> None:
        key = f"{frame_idx:06d}"
        if not FRAME_KEY_RE.fullmatch(key):
            raise AnnotationError("frame key 无效")
        meta["frames"].setdefault(key, {})[camera] = {
            "status": status,
            "updated_at": utc_now(),
        }

    def accept_candidate(
        self, episode: str, frame_idx: int, instance_id: int, candidate: np.ndarray, camera: str | None = None
    ) -> np.ndarray:
        camera = camera or self.annotation_camera
        with self._lock:
            meta, shape = self._validate_target(episode, frame_idx, camera)
            self._instance(meta["instances"], instance_id)
            current = self.load_mask(episode, frame_idx, camera)
            if candidate.shape != shape:
                raise AnnotationError("候选 mask 尺寸与图像不一致")
            merged = merge_instance_mask(current, candidate, instance_id)
            atomic_write_png(self.mask_path(episode, frame_idx, camera), merged)
            self._set_status(meta, frame_idx, camera, "predicted")
            self.save_metadata(episode, meta)
            return merged

    def clear_instance(
        self, episode: str, frame_idx: int, instance_id: int, camera: str | None = None
    ) -> np.ndarray:
        camera = camera or self.annotation_camera
        with self._lock:
            meta, _ = self._validate_target(episode, frame_idx, camera)
            self._instance(meta["instances"], instance_id)
            result = clear_instance_mask(self.load_mask(episode, frame_idx, camera), instance_id)
            atomic_write_png(self.mask_path(episode, frame_idx, camera), result)
            self._set_status(meta, frame_idx, camera, "manual")
            self.save_metadata(episode, meta)
            return result

    def clear_episode_instance(
        self, episode: str, instance_id: int, camera: str | None = None
    ) -> int:
        """清除该 episode 所有帧上某一实例的 mask 像素。"""
        if not isinstance(instance_id, int) or not 1 <= instance_id <= 255:
            raise AnnotationError("实例 ID 无效")
        camera = camera or self.annotation_camera
        with self._lock:
            meta = self.load_metadata(episode)
            _, _, frames = self._read_data(episode)
            cleared = 0
            for frame in frames:
                frame_idx = int(frame["idx"])
                path = self.mask_path(episode, frame_idx, camera)
                if not path.is_file():
                    continue
                current = self.load_mask(episode, frame_idx, camera)
                if not np.any(current == int(instance_id)):
                    continue
                result = clear_instance_mask(current, instance_id)
                atomic_write_png(path, result)
                self._set_status(meta, frame_idx, camera, "manual")
                cleared += 1
            if cleared:
                self.save_metadata(episode, meta)
            return cleared

    def purge_auto_job_instance(self, episode: str, instance_id: int) -> dict[str, Any]:
        """从 auto_jobs 记录中移除该实例的种子点（不改 episode_labels）。"""
        path = self.root / "annotations" / QUEUE_FILENAME
        with self._lock:
            if not path.is_file():
                return {"updated": False, "reason": "无自动标注记录"}
            jobs = read_queue(path)
            job = next((item for item in jobs if item.get("episode") == episode), None)
            if job is None:
                return {"updated": False, "reason": "无自动标注记录"}
            molmo_points = drop_instance_items(
                list(job.get("molmo_points") or []), instance_id
            )
            sam_prompts = drop_instance_items(
                list(job.get("sam_prompts") or []), instance_id
            )
            instance_ids = [
                int(value)
                for value in (job.get("instance_ids") or [])
                if int(value) != int(instance_id)
            ]
            if (
                molmo_points == job.get("molmo_points")
                and sam_prompts == job.get("sam_prompts")
                and instance_ids == list(job.get("instance_ids") or [])
            ):
                return {"updated": False, "reason": "自动标注记录中无该实例"}
            job["molmo_points"] = molmo_points
            job["sam_prompts"] = sam_prompts
            job["instance_ids"] = instance_ids
            result = job.get("result")
            if isinstance(result, dict):
                result = dict(result)
                result["sam_prompts"] = sam_prompts
                job["result"] = result
            job["updated_at"] = utc_now()
            write_queue(path, jobs)
        return {
            "updated": True,
            "molmo_points": molmo_points,
            "sam_prompts": sam_prompts,
            "instance_ids": instance_ids,
        }

    def confirm(self, episode: str, frame_idx: int, camera: str | None = None) -> None:
        camera = camera or self.annotation_camera
        with self._lock:
            meta, _ = self._validate_target(episode, frame_idx, camera)
            self._set_status(meta, frame_idx, camera, "confirmed")
            self.save_metadata(episode, meta)

    def save_episode_label_maps(
        self,
        episode: str,
        frame_label_maps: dict[int, np.ndarray],
        camera: str | None = None,
        status: str = "predicted",
    ) -> int:
        camera = camera or self.annotation_camera
        with self._lock:
            if not frame_label_maps:
                raise AnnotationError("没有可保存的帧 mask")
            meta = self.load_metadata(episode)
            _, shape = self._validate_target(episode, next(iter(frame_label_maps)), camera)
            saved = 0
            for frame_idx, label_map in sorted(frame_label_maps.items()):
                if label_map.shape != shape:
                    raise AnnotationError(f"帧 {frame_idx} mask 尺寸不一致")
                if label_map.dtype != np.uint8:
                    raise AnnotationError(f"帧 {frame_idx} mask 必须是 uint8")
                atomic_write_png(self.mask_path(episode, frame_idx, camera), label_map)
                self._set_status(meta, frame_idx, camera, status)
                saved += 1
            self.save_metadata(episode, meta)
            return saved

    def confirm_all_frames(self, episode: str, camera: str | None = None) -> int:
        camera = camera or self.annotation_camera
        with self._lock:
            meta = self.load_metadata(episode)
            _, _, frames = self._read_data(episode)
            count = 0
            for frame in frames:
                frame_idx = int(frame["idx"])
                key = f"{frame_idx:06d}"
                record = meta["frames"].get(key, {}).get(camera)
                if record and record.get("status") in {"predicted", "manual"}:
                    self._set_status(meta, frame_idx, camera, "confirmed")
                    count += 1
            self.save_metadata(episode, meta)
            return count


def colorize_mask(mask: np.ndarray, instances: list[dict[str, Any]], binary_id: int | None = None) -> Image.Image:
    if binary_id is not None:
        return Image.fromarray(((mask == binary_id) * 255).astype(np.uint8), mode="L")
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for item in instances:
        color = item["color"].lstrip("#")
        rgb = tuple(int(color[offset : offset + 2], 16) for offset in (0, 2, 4))
        selected = mask == item["id"]
        rgba[selected, :3] = rgb
        rgba[selected, 3] = 135
    return Image.fromarray(rgba, mode="RGBA")


def image_response(image: Image.Image) -> Response:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png", max_age=0)


def create_app(
    data_root: str | Path,
    annotation_camera: str = DEFAULT_ANNOTATION_CAMERA,
    sam_server_url: str | None = None,
    sam_client: Any | None = None,
    qwen_api_url: str | None = None,
    qwen_model: str | None = None,
    qwen_api_key: str | None = None,
    qwen_enable_thinking: bool = False,
) -> Flask:
    app = Flask(__name__)
    store = AnnotationStore(data_root, annotation_camera)
    refreshed = store.refresh_episode_point_prompts()
    if refreshed:
        print(
            f"已刷新 {refreshed} 条 episode 的 point_prompt（逐 instance 格式）",
            flush=True,
        )
    sam = sam_client or Sam3RemoteClient(sam_server_url or "http://127.0.0.1:8765")
    resolved_url, resolved_model, resolved_key = resolve_qwen_config(
        api_url=qwen_api_url,
        model=qwen_model,
        api_key=qwen_api_key,
    )
    qwen = None
    if resolved_url and resolved_key:
        qwen = QwenVLMClient(
            resolved_url,
            resolved_model,
            api_key=resolved_key,
            enable_thinking=qwen_enable_thinking,
        )
    app.config.update(
        STORE=store,
        SAM_CLIENT=sam,
        QWEN_CLIENT=qwen,
        ANNOTATION_CAMERA=annotation_camera,
        SAM_SERVER_URL=sam_server_url,
        QWEN_API_URL=resolved_url,
    )

    @app.errorhandler(AnnotationError)
    @app.errorhandler(SamBackendError)
    @app.errorhandler(QwenLabelBackendError)
    def handle_annotation_error(exc: Exception):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(404)
    def handle_not_found(_exc):
        return jsonify({"ok": False, "error": "接口或资源不存在"}), 404

    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        app.logger.exception("request failed")
        return jsonify({"ok": False, "error": f"服务错误: {type(exc).__name__}: {exc}"}), 500

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/status")
    def api_status():
        return jsonify({
            "ok": True,
            "sam3": sam.status(),
            "qwen": {
                "available": qwen is not None,
                "api_url": resolved_url,
                "model": qwen.model if qwen is not None else resolved_model,
            },
            "data_root": str(store.root),
            "annotation_camera": annotation_camera,
            "instruction_template": store.get_instruction_template(),
            "label_slots": store.get_label_slots(),
            "sam_server_url": getattr(sam, "base_url", None),
        })

    @app.get("/api/episodes")
    def api_episodes():
        return jsonify({"ok": True, "episodes": store.list_episodes()})

    @app.get("/api/episodes/<episode>")
    def api_episode(episode: str):
        return jsonify({"ok": True, **store.episode_info(episode)})

    @app.get("/api/image/<episode>/<int:frame_idx>")
    def api_image(episode: str, frame_idx: int):
        return send_file(
            store.image_path(episode, frame_idx, annotation_camera),
            mimetype="image/jpeg",
            max_age=0,
        )

    @app.get("/api/mask/<episode>/<int:frame_idx>")
    def api_mask(episode: str, frame_idx: int):
        mask = store.load_mask(episode, frame_idx, annotation_camera)
        instance_arg = request.args.get("instance")
        try:
            binary_id = int(instance_arg) if instance_arg is not None else None
        except ValueError as exc:
            raise AnnotationError("instance 查询参数必须是整数") from exc
        meta = store.load_metadata(episode)
        if binary_id is not None:
            store._instance(meta["instances"], binary_id)
        return image_response(colorize_mask(mask, meta["instances"], binary_id))

    @app.get("/api/instances")
    def api_list_instances():
        return jsonify({"ok": True, "instances": store.load_task_instances()})

    @app.post("/api/instances")
    def api_create_instance():
        body = request.get_json(silent=True) or {}
        item = store.create_instance(body.get("name", ""), body.get("color"))
        return jsonify({"ok": True, "instance": item}), 201

    @app.patch("/api/instances/<int:instance_id>")
    def api_rename_instance(instance_id: int):
        body = request.get_json(silent=True) or {}
        item = store.rename_instance(instance_id, body.get("name", ""))
        return jsonify({"ok": True, "instance": item})

    @app.delete("/api/instances/<int:instance_id>")
    def api_delete_instance(instance_id: int):
        store.delete_instance(instance_id)
        return jsonify({"ok": True})

    @app.post("/api/episodes/<episode>/instances")
    def api_create_instance_legacy(episode: str):
        del episode
        body = request.get_json(silent=True) or {}
        item = store.create_instance(body.get("name", ""), body.get("color"))
        return jsonify({"ok": True, "instance": item}), 201

    @app.patch("/api/episodes/<episode>/instances/<int:instance_id>")
    def api_rename_instance_legacy(episode: str, instance_id: int):
        del episode
        body = request.get_json(silent=True) or {}
        item = store.rename_instance(instance_id, body.get("name", ""))
        return jsonify({"ok": True, "instance": item})

    @app.delete("/api/episodes/<episode>/instances/<int:instance_id>")
    def api_delete_instance_legacy(episode: str, instance_id: int):
        del episode
        store.delete_instance(instance_id)
        return jsonify({"ok": True})

    @app.get("/api/task/prompts")
    def api_get_task_prompts():
        return jsonify({
            "ok": True,
            "instruction_template": store.get_instruction_template(),
        })

    @app.put("/api/task/prompts")
    def api_set_task_prompts():
        body = request.get_json(silent=True) or {}
        template = store.set_instruction_template(body.get("instruction_template", ""))
        return jsonify({"ok": True, "instruction_template": template})

    @app.post("/api/preview_episode_prompts")
    def api_preview_episode_prompts():
        body = request.get_json(silent=True) or {}
        instance_ids = body.get("instance_ids") or []
        if not isinstance(instance_ids, list):
            raise AnnotationError("instance_ids 必须是列表")
        template = body.get("instruction_template") or store.get_instruction_template()
        try:
            prompts = build_episode_prompts(
                instruction_template=template,
                instances=store.load_task_instances(),
                instance_ids=[int(value) for value in instance_ids],
            )
        except PromptTemplateError as exc:
            raise AnnotationError(str(exc)) from exc
        return jsonify({"ok": True, **prompts})

    @app.get("/api/task/label_slots")
    def api_get_label_slots():
        return jsonify({"ok": True, "label_slots": store.get_label_slots()})

    @app.put("/api/task/label_slots")
    def api_set_label_slots():
        body = request.get_json(silent=True) or {}
        label_slots = body.get("label_slots") or {}
        if not isinstance(label_slots, dict):
            raise AnnotationError("label_slots 必须是字典")
        saved = store.set_label_slots(label_slots)
        return jsonify({"ok": True, "label_slots": saved})

    @app.post("/api/episodes/auto_label")
    def api_auto_label_episodes():
        if qwen is None:
            raise AnnotationError(
                "未配置 Qwen API。请设置 DASHSCOPE_API_KEY，或用 --qwen-api-url 启动 Web"
            )
        body = request.get_json(silent=True) or {}
        episodes = body.get("episodes")
        overwrite = bool(body.get("overwrite", False))
        if episodes is not None and not isinstance(episodes, list):
            raise AnnotationError("episodes 必须是列表")
        episode_name = body.get("episode")
        if episode_name:
            episodes = [str(episode_name)]
        results = label_episodes_with_qwen(
            store,
            qwen,
            episodes=episodes,
            overwrite=overwrite,
        )
        return jsonify({"ok": True, "results": results})

    @app.get("/api/episodes/<episode>/labels")
    def api_get_episode_labels(episode: str):
        record = store.get_episode_label_record(episode)
        return jsonify({
            "ok": True,
            "episode": episode,
            "instruction_template": store.get_instruction_template(),
            **record,
        })

    @app.put("/api/episodes/<episode>/labels")
    def api_set_episode_labels(episode: str):
        body = request.get_json(silent=True) or {}
        instance_ids = body.get("instance_ids") or []
        if not isinstance(instance_ids, list):
            raise AnnotationError("instance_ids 必须是列表")
        saved = store.set_episode_labels(episode, [int(value) for value in instance_ids])
        return jsonify({"ok": True, "episode": episode, **saved})

    @app.patch("/api/episodes/<episode>/labels")
    def api_patch_episode_labels(episode: str):
        body = request.get_json(silent=True) or {}
        saved = store.set_episode_prompts(
            episode,
            instruction=body.get("instruction"),
            point_prompt=body.get("point_prompt"),
            custom=body.get("custom", True),
        )
        return jsonify({"ok": True, "episode": episode, **saved})

    @app.post("/api/episodes/rebuild_prompts")
    def api_rebuild_episode_prompts():
        body = request.get_json(silent=True) or {}
        episodes = body.get("episodes")
        if episodes is not None and not isinstance(episodes, list):
            raise AnnotationError("episodes 必须是列表")
        result = store.rebuild_episode_prompts(
            [str(item) for item in episodes] if episodes else None,
            skip_custom=bool(body.get("skip_custom", True)),
        )
        return jsonify({"ok": True, **result})

    @app.post("/api/episode/segment")
    def api_episode_segment():
        body = request.get_json(silent=True) or {}
        episode = body.get("episode", "")
        seed_frame = body.get("seed_frame")
        prompts = body.get("prompts") or []
        if not isinstance(prompts, list) or not prompts:
            raise AnnotationError("prompts 必须是非空列表")
        info = store.episode_info(episode)
        if seed_frame is None:
            seed_frame = info["frame_indices"][0]
        if not isinstance(seed_frame, int):
            raise AnnotationError("seed_frame 必须是整数")
        for prompt in prompts:
            instance_id = prompt.get("instance_id")
            points = prompt.get("points") or []
            if not isinstance(instance_id, int) or not points:
                raise AnnotationError("每个 prompt 需要 instance_id 和 points")
            store._instance(store.load_metadata(episode)["instances"], instance_id)
        try:
            result = sam.segment_episode(
                store.episode_dir(episode),
                episode,
                annotation_camera,
                seed_frame,
                prompts,
                body.get("direction", "forward"),
            )
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc), "sam3": sam.status()}), 503
        _, shape = store._validate_target(episode, seed_frame, annotation_camera)
        frame_maps: dict[int, np.ndarray] = {}
        for item in result.get("frames", []):
            frame_idx = int(item["frame"])
            try:
                frame_maps[frame_idx] = decode_label_map_png(
                    item.get("label_map_png_base64", ""), shape
                )
            except SamBackendError as exc:
                raise AnnotationError(str(exc)) from exc
        saved = store.save_episode_label_maps(episode, frame_maps, annotation_camera)
        return jsonify({
            "ok": True,
            "saved_frames": saved,
            "frame_count": result.get("frame_count", saved),
            "sam_obj_mapping": result.get("sam_obj_mapping", {}),
        })

    @app.post("/api/frame/segment")
    def api_frame_segment():
        body = request.get_json(silent=True) or {}
        episode = body.get("episode", "")
        frame_idx = body.get("frame")
        propagate = bool(body.get("propagate", False))
        direction = body.get("direction", "both")
        if not isinstance(frame_idx, int):
            raise AnnotationError("frame 必须是整数")

        raw_prompts = body.get("prompts")
        if isinstance(raw_prompts, list) and raw_prompts:
            batch_prompts: list[dict[str, Any]] = []
            for item in raw_prompts:
                instance_id = item.get("instance_id")
                points = item.get("points") or []
                labels = item.get("labels") or []
                if not isinstance(instance_id, int) or not points:
                    raise AnnotationError("每个 prompt 需要 instance_id 和 points")
                batch_prompts.append(
                    {
                        "instance_id": instance_id,
                        "points": points,
                        "labels": labels or [1] * len(points),
                    }
                )
        else:
            instance_id = body.get("instance_id")
            points = body.get("points") or []
            labels = body.get("labels") or []
            if not isinstance(instance_id, int):
                raise AnnotationError("instance_id 必须是整数")
            if not points:
                raise AnnotationError("请提供至少一个点")
            batch_prompts = [
                {
                    "instance_id": instance_id,
                    "points": points,
                    "labels": labels or [1] * len(points),
                }
            ]

        meta = store.load_metadata(episode)
        for item in batch_prompts:
            store._instance(meta["instances"], item["instance_id"])

        ep_dir = store.episode_dir(episode)
        _, shape = store._validate_target(episode, frame_idx, annotation_camera)
        auto_record = store.get_auto_job_record(episode)
        seed_frame = (
            auto_record.get("seed_frame") if auto_record.get("available") else None
        )
        obj_by_instance: dict[int, int] = {}
        seed_overwrites: list[dict[str, Any]] = []
        foreground_points = 0
        background_points = 0

        for item in batch_prompts:
            instance_id = int(item["instance_id"])
            points = item["points"]
            labels = item["labels"]
            foreground_points += sum(1 for label in labels if int(label) == 1)
            background_points += sum(1 for label in labels if int(label) != 1)
            try:
                candidates = sam.predict(
                    ep_dir,
                    episode,
                    annotation_camera,
                    frame_idx,
                    points,
                    labels,
                    None,
                )
            except RuntimeError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 503
            if not candidates:
                raise AnnotationError(f"实例 {instance_id}：SAM3 未返回候选 mask")
            candidate = decode_binary_png(candidates[0].get("mask_png_base64", ""), shape)
            store.accept_candidate(
                episode, frame_idx, instance_id, candidate, annotation_camera
            )
            obj_id = candidates[0].get("obj_id")
            if obj_id is not None:
                obj_by_instance[instance_id] = int(obj_id)
            if seed_frame is not None and int(seed_frame) == frame_idx:
                seed_overwrites.append(
                    store.overwrite_instance_seed_points(
                        episode, instance_id, points, labels
                    )
                )

        propagated_frames = 0
        if propagate and obj_by_instance:
            frame_maps: dict[int, np.ndarray] = {}
            for instance_id, obj_id in obj_by_instance.items():
                try:
                    frames = sam.propagate(
                        ep_dir,
                        episode,
                        annotation_camera,
                        frame_idx,
                        obj_id,
                        direction,
                    )
                except RuntimeError as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 503
                for item in frames:
                    other_idx = int(item["frame"])
                    if other_idx == frame_idx:
                        continue
                    other_mask = decode_binary_png(item.get("mask_png_base64", ""), shape)
                    if other_idx not in frame_maps:
                        frame_maps[other_idx] = store.load_mask(
                            episode, other_idx, annotation_camera
                        )
                    frame_maps[other_idx] = merge_instance_mask(
                        frame_maps[other_idx], other_mask, instance_id
                    )
            if frame_maps:
                propagated_frames = store.save_episode_label_maps(
                    episode, frame_maps, annotation_camera, status="manual"
                )

        seed_updated = any(item.get("updated") for item in seed_overwrites)
        return jsonify({
            "ok": True,
            "propagated_frames": propagated_frames,
            "segmented_instances": len(batch_prompts),
            "foreground_points": foreground_points,
            "background_points": background_points,
            "obj_ids": obj_by_instance,
            "seed_overwrite": {
                "updated": seed_updated,
                "instances": [
                    item for item in seed_overwrites if item.get("updated")
                ],
            },
        })

    @app.post("/api/mask/clear")
    def api_clear():
        body = request.get_json(silent=True) or {}
        frame_idx, instance_id = body.get("frame"), body.get("instance_id")
        if not isinstance(frame_idx, int) or not isinstance(instance_id, int):
            raise AnnotationError("frame 和 instance_id 必须是整数")
        store.clear_instance(body.get("episode", ""), frame_idx, instance_id, annotation_camera)
        return jsonify({"ok": True})

    @app.post("/api/episode/clear_instance")
    def api_clear_episode_instance():
        body = request.get_json(silent=True) or {}
        episode = body.get("episode", "")
        instance_id = body.get("instance_id")
        purge_auto = bool(body.get("purge_auto_job", True))
        if not isinstance(instance_id, int) or not 1 <= instance_id <= 255:
            raise AnnotationError("instance_id 必须是 1..255 的整数")
        cleared_frames = store.clear_episode_instance(
            episode, instance_id, annotation_camera
        )
        auto_purge = (
            store.purge_auto_job_instance(episode, instance_id)
            if purge_auto
            else {"updated": False, "reason": "未请求清理 auto_jobs"}
        )
        return jsonify({
            "ok": True,
            "cleared_frames": cleared_frames,
            "auto_job_purge": auto_purge,
        })

    @app.post("/api/frame/seed_points")
    def api_overwrite_seed_points():
        body = request.get_json(silent=True) or {}
        episode = body.get("episode", "")
        instance_id = body.get("instance_id")
        points = body.get("points") or []
        labels = body.get("labels") or []
        if not isinstance(instance_id, int):
            raise AnnotationError("instance_id 必须是整数")
        if not points:
            raise AnnotationError("请提供至少一个点")
        store._instance(store.load_metadata(episode)["instances"], instance_id)
        result = store.overwrite_instance_seed_points(
            episode, instance_id, points, labels or [1] * len(points)
        )
        return jsonify({
            "ok": True,
            **result,
            "auto_annotation": store.get_auto_job_record(episode),
        })

    @app.post("/api/frame/confirm")
    def api_confirm():
        body = request.get_json(silent=True) or {}
        frame_idx = body.get("frame")
        if not isinstance(frame_idx, int):
            raise AnnotationError("frame 必须是整数")
        store.confirm(body.get("episode", ""), frame_idx, annotation_camera)
        return jsonify({"ok": True})

    @app.post("/api/episode/confirm_all")
    def api_confirm_all():
        body = request.get_json(silent=True) or {}
        episode = body.get("episode", "")
        count = store.confirm_all_frames(episode, annotation_camera)
        return jsonify({"ok": True, "confirmed_frames": count})

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>视频分割标注</title>
<style>
:root{color-scheme:dark;--bg:#101318;--panel:#1b2028;--line:#343b47}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#e8edf5;font:14px system-ui,sans-serif}
header{padding:12px 18px;background:#171b22;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{display:grid;grid-template-columns:360px 1fr;gap:14px;padding:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
label{display:block;color:#aab4c3;margin:10px 0 4px}select,input,button,textarea{width:100%;padding:8px;border-radius:5px;border:1px solid #465061;background:#252b35;color:#fff}
button{cursor:pointer;margin-top:7px}button.primary{background:#1765c0}button.good{background:#237b4b}button.warn{background:#8a4b19}
.row{display:flex;gap:7px}.row>*{flex:1}.status{padding:7px 10px;border-radius:5px;background:#262c36;font-size:12px}.muted{color:#95a0b0;font-size:12px}
#draw{cursor:crosshair}.badge{display:inline-block;padding:3px 8px;border-radius:10px;background:#424b59}.error{color:#ff7875;white-space:pre-wrap}.ok{color:#73d13d}.pending{color:#faad14}
#segmentStatus.ok{color:#73d13d}#segmentStatus.error{color:#ff7875}#segmentStatus.pending{color:#faad14}
.hint{line-height:1.5;margin-top:9px}.tabs{display:flex;gap:8px;margin-bottom:8px}.tabs button{flex:1;margin-top:0}.tabs button.active{outline:2px solid #69b1ff}
.instance-chip{display:inline-flex;align-items:center;gap:6px;margin:4px 6px 0 0;padding:4px 8px;border-radius:999px;background:#2a3140;font-size:12px;cursor:pointer;border:1px solid transparent}
.instance-chip.active{border-color:#69b1ff;box-shadow:0 0 0 1px #69b1ff inset}
.instance-chip i{display:inline-block;width:10px;height:10px;border-radius:50%}
.instance-chip button{width:auto;margin:0;padding:1px 7px;font-size:11px;border-radius:999px}
.dual-view{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
.view-box .view-title{margin:0 0 8px;color:#aab4c3;font-size:13px}
.frame-nav{margin-bottom:10px}
.layer-toggles{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 10px;font-size:12px;color:#aab4c3}
.layer-toggles label{display:inline-flex;align-items:center;gap:6px;margin:0;cursor:pointer}
.layer-toggles input{width:auto;margin:0}
.seed-preview{margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 10px;font-size:12px;color:#95a0b0}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%}
.stage{position:relative;display:inline-block;max-width:100%;line-height:0}
.stage canvas{position:absolute;left:0;top:0;width:100%;height:100%}
.stage img{display:block;max-width:100%;max-height:calc(100vh - 210px)}
.section-title{margin:14px 0 6px;color:#d7dee8;font-weight:600}
.prompt-box{padding:8px;border:1px solid #343b47;border-radius:6px;background:#151a22;font-size:12px;line-height:1.5;word-break:break-word}
.slot-row{margin-bottom:8px}
@media (max-width:1100px){main{grid-template-columns:1fr}}
</style></head><body>
<header>
<strong>视频分割标注</strong>
<span class="status">① 创建标签</span><span class="status">② Qwen 自动分配</span><span class="status">③ 离线标注</span><span class="status">④ 人工校验</span>
<span id="samStatus" class="status">SAM3 读取中</span>
<span id="qwenStatus" class="status">Qwen 读取中</span>
<span id="progress"></span>
</header>
<main><aside class="panel">
<label>Episode</label><select id="episode"></select>
<p class="muted" id="goal"></p>
<div class="tabs"><button id="tabSetup" class="active">1. 任务配置</button><button id="tabReview">4. 校验标注</button></div>
<div id="setupPanel">
<div class="section-title">任务级物体标签</div>
<div class="row"><input id="newName" placeholder="标签名，如 red apple"><input id="newColor" type="color" value="#ff4d4f" style="max-width:48px"></div>
<button id="create">创建标签</button>
<div id="instanceColors"></div>
<p class="hint muted">先点左侧标签选中物体，再在右侧图上左键打点，即可覆盖该物体的 Molmo 种子点。</p>
<div class="row"><button id="renameSetup">重命名所选标签</button><button id="deleteSetup" class="warn">删除空标签</button></div>
<label>Instruction 模板（任务级）</label>
<input id="instructionTemplate" placeholder="Place {A} on the {B}">
<button id="saveTemplate">保存 instruction 模板</button>
<div class="section-title">占位符候选标签（任务级）</div>
<div id="slotConfig" class="muted">加载中…</div>
<button id="saveSlotConfig" class="primary">保存候选配置</button>
<div class="section-title">本 episode 标签分配</div>
<div id="episodeLabelPickers" class="muted">加载中…</div>
<button id="saveEpisodeLabels" class="primary">保存标签分配</button>
<p class="hint muted">按模板 {A}、{B}… 顺序手动选物体；保存后会重新生成 instruction / Molmo prompt（会取消「定制」标记）。若 Qwen 选错，先改这里再清 mask、重打。</p>
<div class="section-title">当前 episode 文本</div>
<textarea id="episodeInstruction" rows="3" placeholder="instruction，如 Place the red apple on the left bowl"></textarea>
<textarea id="episodePointPrompt" rows="3" placeholder="Point to the red apple. | Point to the left bowl."></textarea>
<p class="hint muted">多个物体的 Molmo prompt 用 | 分隔，顺序与已选标签一致。保存后标记为定制，批量重建时会跳过。</p>
<div class="row"><button id="saveEpisodePrompts" class="primary">保存为定制</button><button id="rebuildCurrent">按模板重建当前</button></div>
<button id="rebuildAllPrompts">按模板重建全部（跳过已定制）</button>
<div class="section-title">Qwen 自动分配 episode 标签</div>
<button id="autoLabelAll" class="good">为全部 episode 自动选标签</button>
<button id="autoLabelCurrent">仅为当前 episode 自动选标签</button>
<div id="episodeLabelTable" class="prompt-box" style="margin-top:8px">尚无 episode 标签</div>
<p class="hint muted">改标签名后点「按模板重建全部」即可刷新 instruction，不必再跑 Qwen。只需改个别句子时用「保存为定制」。</p>
</div>
<div id="reviewPanel" style="display:none">
<label>当前编辑实例</label><select id="instance"></select>
<div class="row"><button id="frameSegment" class="primary">重打当前帧</button><button id="frameSegmentProp" class="warn">重打并传播</button></div>
<p id="segmentStatus" class="hint muted">重打/传播结果会显示在这里</p>
<div class="row"><button id="clearEpisodeInstance" class="warn">清除本 episode 该物体全部 mask</button></div>
<p class="hint muted">换标签后若旧实例（如 6）mask 仍残留，选中该实例点此按钮；会清掉整段视频该 ID 的 mask，并移除 auto_jobs 里对应种子点。</p>
<div class="row"><button id="confirm" class="good">确认当前帧</button><button id="confirmAll" class="good">确认全部帧</button></div>
<p>当前帧状态：<span id="frameStatus" class="badge">未标注</span></p>
<p class="hint muted">左键前景点会保留在画面上直到重打完成。仅在<strong>种子帧</strong>重打时会写入 Molmo 种子点；配置页左键打点会立即覆盖种子。</p>
<div class="row"><button id="rename">重命名标签</button><button id="delete" class="warn">删除空标签</button></div>
</div>
<div id="message"></div>
</aside><section class="panel">
<div class="frame-nav">
<label>浏览帧</label>
<div class="row"><button id="prev" type="button">←</button><select id="frame"></select><button id="next" type="button">→</button></div>
<div class="row"><button id="jumpSeed" type="button">跳到 Molmo 首帧</button><button id="jumpFirst" type="button">首帧</button><button id="jumpLast" type="button">末帧</button></div>
</div>
<div class="layer-toggles">
<label><input type="checkbox" id="showMask" checked> SAM mask</label>
<label><input type="checkbox" id="showMolmo" checked> Molmo 打点</label>
<label><input type="checkbox" id="showSamSeed" checked> SAM 种子点</label>
</div>
<div class="legend">
<span><i style="background:#fff;border:2px solid #69b1ff"></i>Molmo 原始点（蓝圈）</span>
<span><i style="background:#faad14"></i>当前种子点（实例颜色实心，手工覆盖后）</span>
<span><i style="background:#ff4050"></i>未保存的手动点（点击后先出现）</span>
</div>
<p class="muted" id="viewHint">帧 000000 / 共 0 帧</p>
<div class="view-box">
<p class="view-title" id="primaryTitle">当前帧</p>
<div class="stage"><img id="imagePrimary"><canvas id="overlayPrimary"></canvas><canvas id="draw"></canvas></div>
</div>
<div id="seedPreview" class="seed-preview" style="display:none">
<p class="view-title">Molmo 首帧打点对照</p>
<div class="stage"><img id="imageSeed"><canvas id="overlaySeed"></canvas><canvas id="drawSeed"></canvas></div>
</div>
</section></main>
<script>
const $=id=>document.getElementById(id), S={info:null,currentFrame:null,view:'setup',prompts:{},template:'',labelSlots:{},qwenAvailable:false,selectedInstanceId:null};
const LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
async function api(url,opt={}){const r=await fetch(url,opt),j=await r.json();if(!r.ok||j.ok===false)throw Error(j.error||`HTTP ${r.status}`);return j}
function msg(text,kind=false){if(kind==='pending'){$('message').className='pending';$('message').textContent=text;return}$('message').className=kind?'ok':'error';$('message').textContent=text}
function setSegmentStatus(text,kind='muted'){const el=$('segmentStatus');if(!el)return;el.textContent=text||'重打/传播结果会显示在这里';el.className='hint '+(kind==='ok'?'ok':kind==='error'?'error':kind==='pending'?'pending':'muted')}
function episode(){return $('episode').value}
function esc(x){const d=document.createElement('div');d.textContent=x;return d.innerHTML}
function placeholderCount(t){const m=[...String(t).matchAll(/\{([A-Z])\}/g)];if(!m.length)return 0;for(let i=0;i<m.length;i++){if(m[i][1]!==LETTERS[i])throw Error(`占位符须按 {A},{B},... 顺序`)}return m.length}
function activeInstanceId(){return Number(S.selectedInstanceId||$('instance').value)||null}
function ensurePrompt(id){if(!S.prompts[id])S.prompts[id]={points:[],labels:[]};return S.prompts[id]}
function selectInstance(id){S.selectedInstanceId=Number(id)||null;if($('instance')&&S.selectedInstanceId){const opt=[...$('instance').options].find(o=>Number(o.value)===S.selectedInstanceId);if(opt)$('instance').value=String(S.selectedInstanceId)}renderInstanceChips()}
function renderInstanceChips(){if(!S.info)return;$('instanceColors').innerHTML=S.info.instances.map(x=>`<span class="instance-chip ${Number(S.selectedInstanceId)===x.id?'active':''}" data-id="${x.id}"><i style="background:${x.color}"></i><strong>${x.id}</strong>: ${esc(x.name)}<button type="button" class="chip-rename" data-id="${x.id}">改名</button></span>`).join('')}
function renderSlotConfig(){const count=placeholderCount($('instructionTemplate').value||S.template||'');if(!count){$('slotConfig').innerHTML='<span class="muted">请先设置 instruction 模板</span>';return}if(!S.info?.instances?.length){$('slotConfig').innerHTML='<span class="muted">请先创建物体标签</span>';return}const selected=S.labelSlots||{};$('slotConfig').innerHTML=Array.from({length:count},(_,i)=>{const letter=LETTERS[i];const chosen=new Set((selected[letter]||[]).map(Number));const boxes=S.info.instances.map(inst=>`<label style="display:inline-flex;align-items:center;gap:6px;margin:4px 10px 0 0"><input type="checkbox" class="slot-option" data-slot="${letter}" value="${inst.id}" ${chosen.has(inst.id)?'checked':''}><span>${esc(inst.name)}</span></label>`).join('');return `<div class="slot-row"><label>{${letter}} 候选</label><div>${boxes}</div></div>`}).join('')}
function collectSlotConfig(){const result={};for(const el of document.querySelectorAll('.slot-option:checked')){const letter=el.dataset.slot;result[letter]=result[letter]||[];result[letter].push(Number(el.value))}return result}
function renderEpisodeLabelPickers(){const tpl=$('instructionTemplate').value||S.template||'';try{const count=placeholderCount(tpl);if(!count){$('episodeLabelPickers').innerHTML='<span class="muted">请先设置 instruction 模板</span>';return}if(!S.info?.instances?.length){$('episodeLabelPickers').innerHTML='<span class="muted">请先创建物体标签</span>';return}const slots=S.labelSlots||{};const chosen=(S.info.target_instance_ids||[]).map(Number);$('episodeLabelPickers').innerHTML=Array.from({length:count},(_,i)=>{const letter=LETTERS[i];const slotIds=(slots[letter]||[]).map(Number);const candidates=S.info.instances.filter(inst=>!slotIds.length||slotIds.includes(inst.id));const selected=chosen[i];const options=candidates.map(inst=>`<option value="${inst.id}" ${selected===inst.id?'selected':''}>${inst.id}: ${esc(inst.name)}</option>`).join('');return `<label style="margin-top:8px">{${letter}} 物体</label><select class="episode-slot-pick">${options}</select>`}).join('')}catch(e){$('episodeLabelPickers').innerHTML=`<span class="error">${esc(e.message)}</span>`}}
function collectEpisodeLabelIds(){return [...document.querySelectorAll('.episode-slot-pick')].map(el=>Number(el.value))}
function renderEpisodeLabelTable(){if(!S.info)return;const rows=(S.info.all_episodes||[]).map(item=>{const ep=item.name;const ids=(item.instance_ids||[]).join(',');const instruction=item.instruction||'-';const point=item.point_prompt||'-';const custom=item.custom?' <span class="badge">定制</span>':'';return `<div><strong>${esc(ep)}</strong>${custom}<br><span class="muted">标签:</span> ${esc(ids||'-')}<br><span class="muted">instruction:</span> ${esc(instruction)}<br><span class="muted">molmo:</span> ${esc(point)}</div>`}).join('<hr style="border-color:#343b47;margin:8px 0">');$('episodeLabelTable').innerHTML=rows||'尚无 episode 标签'}
async function loadAllEpisodeLabels(){const eps=await api('/api/episodes');const rows=[];for(const item of eps.episodes){if(item.error)continue;const detail=await api('/api/episodes/'+encodeURIComponent(item.name));rows.push({name:item.name,instance_ids:detail.target_instance_ids||[],instruction:detail.instruction,point_prompt:detail.point_prompt,custom:!!detail.custom})}S.info.all_episodes=rows;renderEpisodeLabelTable()}
async function boot(){const [st,eps]=await Promise.all([api('/api/status'),api('/api/episodes')]);S.template=st.instruction_template||'';S.labelSlots=st.label_slots||{};S.qwenAvailable=!!st.qwen?.available;$('instructionTemplate').value=S.template;const s=st.sam3;$('samStatus').textContent=s.initialized?'SAM3 已就绪':(s.available?'SAM3 可连接':'SAM3 不可用');$('qwenStatus').textContent=S.qwenAvailable?`Qwen 已配置 (${st.qwen.model||'model'})`:'Qwen 未配置';$('episode').innerHTML=eps.episodes.map(e=>`<option value="${e.name}">${e.name} (${e.confirmed||0}/${e.total||'?'}帧)</option>`).join('');if(eps.episodes.length)await loadEpisode()}
async function loadEpisode(opts={}){const keepFrame=S.currentFrame,keepView=S.view;S.info=await api('/api/episodes/'+encodeURIComponent(episode()));const idx=S.info.frame_indices||[];S.currentFrame=(keepFrame!=null&&idx.includes(keepFrame))?keepFrame:(idx[0]??0);S.view=keepView||'setup';S.labelSlots=S.info.label_slots||S.labelSlots||{};if(!opts.keepPrompts)S.prompts={};$('goal').textContent=S.info.goal?('任务：'+S.info.goal):'';$('instructionTemplate').value=S.info.instruction_template||S.template;$('episodeInstruction').value=S.info.instruction||'';$('episodePointPrompt').value=S.info.point_prompt||'';$('frame').innerHTML=idx.map(i=>`<option value="${i}">${String(i).padStart(6,'0')}</option>`).join('');if(idx.length)$('frame').value=String(S.currentFrame);$('instance').innerHTML=S.info.instances.map(x=>`<option value="${x.id}">${x.id}: ${esc(x.name)}</option>`).join('');if(S.selectedInstanceId&&S.info.instances.some(x=>x.id===S.selectedInstanceId))$('instance').value=String(S.selectedInstanceId);else if(S.info.instances.length){S.selectedInstanceId=S.info.instances[0].id;$('instance').value=String(S.selectedInstanceId)}renderInstanceChips();renderSlotConfig();renderEpisodeLabelPickers();await loadAllEpisodeLabels();setView(S.view);if(opts.clearMessage)msg('')}
function seedFrame(){const auto=S.info?.auto_annotation;return auto?.available&&auto.seed_frame!=null?Number(auto.seed_frame):S.info?.frame_indices?.[0]??0}
function currentFrame(){return Number($('frame').value??S.currentFrame??0)}
function setFrame(frame){if(!S.info)return;const idx=S.info.frame_indices.indexOf(frame);if(idx<0)return;S.currentFrame=frame;$('frame').selectedIndex=idx;loadViews()}
function instanceById(id){return S.info?.instances?.find(x=>x.id===id)}
function setView(view){S.view=view;$('tabSetup').classList.toggle('active',view==='setup');$('tabReview').classList.toggle('active',view==='review');$('setupPanel').style.display=view==='setup'?'block':'none';$('reviewPanel').style.display=view==='review'?'block':'none';$('draw').style.pointerEvents='auto';if(view==='review'&&$('instance')&&S.selectedInstanceId)$('instance').value=String(S.selectedInstanceId);updateViewHint();loadViews()}
function updateViewHint(){if(!S.info)return;const total=S.info.frame_indices.length,frame=currentFrame(),pos=S.info.frame_indices.indexOf(frame)+1,seed=seedFrame(),auto=S.info.auto_annotation||{};let hint=`帧 ${String(frame).padStart(6,'0')}（${pos}/${total}）`;if(auto.available)hint+=` | 自动标注 ${auto.status||'未知'}`;if(frame===seed)hint+=' | 当前为种子帧';else hint+=` | 种子帧 ${String(seed).padStart(6,'0')}`;$('viewHint').textContent=hint;$('primaryTitle').textContent=`当前帧 ${String(frame).padStart(6,'0')}`}
function setupCanvas(img,overlay,draw){const w=img.naturalWidth,h=img.naturalHeight;if(!w||!h)return;overlay.width=draw.width=w;overlay.height=draw.height=h}
async function loadViews(){updateViewHint();await loadPrimaryView();const seed=seedFrame();const showSeedPanel=currentFrame()!==seed&&($('showMolmo').checked||$('showSamSeed').checked);if(showSeedPanel){$('seedPreview').style.display='block';await loadSeedPreview(seed)}else $('seedPreview').style.display='none';updateState()}
async function loadPrimaryView(){const frame=currentFrame(),img=$('imagePrimary'),overlay=$('overlayPrimary'),draw=$('draw');img.onload=()=>{setupCanvas(img,overlay,draw);drawAllPoints(draw);if($('showMask').checked)loadOverlay(overlay,frame);else overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height)};img.src=`/api/image/${encodeURIComponent(episode())}/${frame}?v=${Date.now()}`}
async function loadSeedPreview(frame){const img=$('imageSeed'),overlay=$('overlaySeed'),draw=$('drawSeed');img.onload=()=>{setupCanvas(img,overlay,draw);drawAutoPoints(draw,frame);if($('showMask').checked)loadOverlay(overlay,frame);else overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height)};img.src=`/api/image/${encodeURIComponent(episode())}/${frame}?v=${Date.now()}`}
async function loadOverlay(canvas,frame){const im=new Image(),ctx=canvas.getContext('2d');im.onload=()=>{ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(im,0,0)};im.onerror=()=>ctx.clearRect(0,0,canvas.width,canvas.height);im.src=`/api/mask/${encodeURIComponent(episode())}/${frame}?v=${Date.now()}`}
function drawPointMarker(ctx,point,color,label,style='fill'){const [x,y]=point;ctx.beginPath();ctx.arc(x,y,8,0,Math.PI*2);if(style==='ring'){ctx.fillStyle='#111';ctx.fill();ctx.lineWidth=3;ctx.strokeStyle=color;ctx.stroke()}else{ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}if(label){ctx.fillStyle='#fff';ctx.font='bold 11px system-ui';ctx.fillText(label,x+10,y-10)}}
function overwrittenInstanceIds(){const ids=new Set();const auto=S.info?.auto_annotation;if(!auto?.available)return ids;for(const item of auto.molmo_points||[]){if(item.source==='manual'&&item.instance_id!=null)ids.add(Number(item.instance_id))}for(const item of auto.sam_prompts||[]){if(item.source==='manual'&&item.instance_id!=null)ids.add(Number(item.instance_id))}return ids}
function pendingManualInstanceIds(){const ids=new Set();if(!S.info)return ids;for(const inst of S.info.instances){const prompt=S.prompts[inst.id];if(!prompt)continue;if((prompt.labels||[]).some(l=>Number(l)===1))ids.add(inst.id)}return ids}
function drawAutoPoints(canvas,frame){const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const auto=S.info?.auto_annotation;if(!auto?.available||frame!==seedFrame())return;const hideMolmo=overwrittenInstanceIds();const pending=pendingManualInstanceIds();if($('showMolmo').checked){(auto.molmo_points||[]).forEach((item,i)=>{if(item.source==='manual')return;const instId=Number(item.instance_id);if(hideMolmo.has(instId))return;const p=item.point||[item.x,item.y];if(!p) return;drawPointMarker(ctx,p,'#69b1ff',`M${item.instance_id??item.object_id??i+1}`,'ring')})}if($('showSamSeed').checked){(auto.sam_prompts||[]).forEach(item=>{const instId=Number(item.instance_id);if(pending.has(instId))return;const inst=instanceById(instId);const color=inst?.color||'#faad14';const manual=item.source==='manual';(item.points||[]).forEach((p,j)=>drawPointMarker(ctx,p,color,manual?`手${instId}`:(inst?`${inst.id}`:`S${j+1}`)))})}}
function manualPointLabel(inst,index,total,isFg){const prefix=isFg?'手':'背';if(total<=1)return`${prefix}${inst.id}`;return`${prefix}${inst.id}·${index+1}`}
function drawManualPoints(canvas){const ctx=canvas.getContext('2d');if(!S.info)return;for(const inst of S.info.instances){const prompt=S.prompts[inst.id];if(!prompt)continue;const total=prompt.points.length;for(let i=0;i<total;i++){const fg=Number(prompt.labels[i])===1;drawPointMarker(ctx,prompt.points[i],fg?(inst.color||'#40a9ff'):'#ff4050',manualPointLabel(inst,i,total,fg))}}}
function drawAllPoints(canvas){drawAutoPoints(canvas,currentFrame());drawManualPoints(canvas)}
function updateState(){if(!S.info)return;const camera=S.info.camera||'color_0',frame=currentFrame(),rec=S.info.frames[String(frame).padStart(6,'0')]?.[camera],status=rec?.status||'unlabeled';$('frameStatus').textContent={unlabeled:'未标注',predicted:'已预测',manual:'已编辑',confirmed:'已确认'}[status]||status;$('progress').textContent=`进度 ${S.info.progress.confirmed}/${S.info.progress.total} 帧`}
function canvasPoint(e){const r=$('draw').getBoundingClientRect();return[(e.clientX-r.left)*$('draw').width/r.width,(e.clientY-r.top)*$('draw').height/r.height]}
function drawPrompts(){drawAllPoints($('draw'))}
function pendingFramePrompts(){if(!S.info)return[];const out=[];for(const inst of S.info.instances){const prompt=S.prompts[inst.id];if(!prompt||!prompt.points.length)continue;out.push({instance_id:inst.id,points:prompt.points.map(p=>[p[0],p[1]]),labels:prompt.labels.slice()})}return out}
function promptStats(prompts){let fg=0,bg=0;for(const p of prompts){for(const l of p.labels)if(Number(l)===1)fg++;else bg++}return {fg,bg,instances:prompts.length}}
function formatPromptHint(stats){const parts=[];if(stats.instances)parts.push(`${stats.instances} 个实例`);parts.push(`${stats.fg} 前景点`);if(stats.bg)parts.push(`${stats.bg} 背景点`);return parts.join('、')}
$('draw').oncontextmenu=e=>e.preventDefault();
async function persistSeedPoints(id){const prompt=ensurePrompt(id);const j=await api('/api/frame/seed_points',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:episode(),instance_id:id,points:prompt.points,labels:prompt.labels})});if(j.updated&&S.info){S.info.auto_annotation=Object.assign(S.info.auto_annotation||{available:true},j.auto_annotation||{molmo_points:j.molmo_points,sam_prompts:j.sam_prompts});const n=(j.sam_prompts||[]).find(x=>Number(x.instance_id)===Number(id))?.points?.length||prompt.points.length;msg(`已覆盖该实例 Molmo 种子（${n} 个前景点）`,true)}else if(j.reason)msg(j.reason);drawPrompts();return j}
$('draw').onmousedown=e=>{const id=activeInstanceId();if(!id){msg(S.view==='review'?'请先在校验页选择实例':'请先在左侧选中标签');return}const p=canvasPoint(e),prompt=ensurePrompt(id);prompt.points.push(p);prompt.labels.push(e.button===2?0:1);drawPrompts();if(S.view==='setup'&&e.button!==2)persistSeedPoints(id).catch(err=>msg(err.message))};
async function refresh(){await loadEpisode()}
$('instructionTemplate').oninput=()=>{try{renderSlotConfig();renderEpisodeLabelPickers()}catch(e){msg(e.message)}};
$('saveTemplate').onclick=async()=>{try{const j=await api('/api/task/prompts',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction_template:$('instructionTemplate').value})});S.template=j.instruction_template;renderSlotConfig();msg('instruction 模板已保存',true)}catch(e){msg(e.message)}};
$('saveSlotConfig').onclick=async()=>{try{const j=await api('/api/task/label_slots',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({label_slots:collectSlotConfig()})});S.labelSlots=j.label_slots;renderEpisodeLabelPickers();msg('候选配置已保存',true)}catch(e){msg(e.message)}};
$('saveEpisodeLabels').onclick=async()=>{try{const instance_ids=collectEpisodeLabelIds();const j=await api('/api/episodes/'+encodeURIComponent(episode())+'/labels',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({instance_ids})});$('episodeInstruction').value=j.instruction||'';$('episodePointPrompt').value=j.point_prompt||'';await loadEpisode();msg(`已保存标签 [${instance_ids.join(', ')}]，文本已按模板生成`,true)}catch(e){msg(e.message)}};
async function runAutoLabel(episodes=null,overwrite=false){if(!S.qwenAvailable)throw Error('未配置 Qwen API');const body={overwrite};if(episodes)body.episodes=episodes;const j=await api('/api/episodes/auto_label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await loadEpisode();return j}
$('saveEpisodePrompts').onclick=async()=>{try{await api('/api/episodes/'+encodeURIComponent(episode())+'/labels',{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({instruction:$('episodeInstruction').value,point_prompt:$('episodePointPrompt').value,custom:true})});await loadEpisode();msg('已保存当前 episode 定制文本',true)}catch(e){msg(e.message)}};
async function rebuildPrompts(episodes,skipCustom){const j=await api('/api/episodes/rebuild_prompts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episodes,skip_custom:skipCustom})});await loadEpisode();const c=j.counts||{};msg(`已重建 ${c.updated||0} 条，跳过 ${c.skipped||0}，失败 ${c.failed||0}`,true);return j}
$('rebuildCurrent').onclick=async()=>{try{await rebuildPrompts([episode()],false)}catch(e){msg(e.message)}};
$('rebuildAllPrompts').onclick=async()=>{try{if(!confirm('按当前模板和标签名重建全部 instruction/point？已定制的 episode 会跳过。'))return;await rebuildPrompts(null,true)}catch(e){msg(e.message)}};
$('autoLabelAll').onclick=async()=>{try{msg('Qwen 正在为全部 episode 选标签…');const j=await runAutoLabel(null,true);const done=j.results.filter(x=>!x.skipped).length;msg(`Qwen 已完成 ${done} 个 episode 标签分配`,true)}catch(e){msg(e.message)}};
$('autoLabelCurrent').onclick=async()=>{try{msg('Qwen 正在为当前 episode 选标签…');await runAutoLabel([episode()],true);msg('当前 episode 标签已更新',true)}catch(e){msg(e.message)}};
async function rerunFrame(propagate){const btn=propagate?$('frameSegmentProp'):$('frameSegment');const prev=btn.textContent;try{const prompts=pendingFramePrompts();if(!prompts.length)throw Error('请先在当前帧打点（可为多个实例）');const stats=promptStats(prompts),hint=formatPromptHint(stats);btn.disabled=true;btn.textContent=propagate?'传播中…':'重打中…';setSegmentStatus(propagate?`正在用 ${hint} 重打并传播，请稍候…`:`正在用 ${hint} 重打当前帧…`,'pending');msg(propagate?'正在重打并传播…':'正在重打当前帧…','pending');const j=await api('/api/frame/segment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:episode(),frame:currentFrame(),prompts,propagate})});for(const p of prompts)S.prompts[p.instance_id]={points:[],labels:[]};await loadEpisode({keepPrompts:true});const overwritten=j.seed_overwrite?.updated;const n=j.propagated_frames||0;const ptHint=`（${hint}）`;const text=propagate?`已重打并传播：${j.segmented_instances||stats.instances} 实例，更新 ${n} 帧${ptHint}`+(overwritten?'，已覆盖 Molmo 种子点':''):`已重打当前帧：${j.segmented_instances||stats.instances} 实例${ptHint}`+(overwritten?'，已覆盖 Molmo 种子点':'');setSegmentStatus(text,'ok');msg(text,true)}catch(e){setSegmentStatus(e.message,'error');msg(e.message)}finally{btn.disabled=false;btn.textContent=prev}}
$('frameSegment').onclick=()=>rerunFrame(false);$('frameSegmentProp').onclick=()=>rerunFrame(true);
$('clearEpisodeInstance').onclick=async()=>{try{const id=activeInstanceId();if(!id)throw Error('请先选择要清除的实例');if(!confirm(`清除 ${episode()} 全部帧上实例 ${id} 的 mask？`))return;const j=await api('/api/episode/clear_instance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:episode(),instance_id:id,purge_auto_job:true})});await loadEpisode({keepPrompts:true});const n=j.cleared_frames||0;const auto=j.auto_job_purge?.updated?'，已移除 auto_jobs 种子':'';msg(`已清除 ${n} 帧上实例 ${id} 的 mask${auto}`,true)}catch(e){msg(e.message)}};
$('create').onclick=async()=>{try{await api('/api/instances',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('newName').value,color:$('newColor').value})});$('newName').value='';await refresh();msg('标签已创建',true)}catch(e){msg(e.message)}};
async function renameInstance(id){const old=S.info.instances.find(x=>x.id===Number(id));if(!old)throw Error('请先选择标签');const name=prompt('新名称',old.name);if(name===null)return;await api(`/api/instances/${old.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});S.selectedInstanceId=old.id;await refresh();msg('标签已重命名',true)}
async function deleteInstance(id){if(!id)throw Error('请先选择标签');if(!confirm('仅空标签可删除，继续？'))return;await api(`/api/instances/${id}`,{method:'DELETE'});S.selectedInstanceId=null;await refresh();msg('标签已删除',true)}
$('instanceColors').onclick=e=>{const renameBtn=e.target.closest('.chip-rename');if(renameBtn){e.stopPropagation();renameInstance(renameBtn.dataset.id).catch(err=>msg(err.message));return}const chip=e.target.closest('.instance-chip');if(chip)selectInstance(chip.dataset.id)};
$('renameSetup').onclick=()=>renameInstance(activeInstanceId()).catch(e=>msg(e.message));
$('deleteSetup').onclick=()=>deleteInstance(activeInstanceId()).catch(e=>msg(e.message));
$('rename').onclick=()=>renameInstance(activeInstanceId()).catch(e=>msg(e.message));
$('delete').onclick=()=>deleteInstance(activeInstanceId()).catch(e=>msg(e.message));
$('instance').onchange=()=>selectInstance($('instance').value);
$('confirm').onclick=async()=>{try{await api('/api/frame/confirm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:episode(),frame:Number($('frame').value)})});await refresh();msg('当前帧已确认',true)}catch(e){msg(e.message)}};
$('confirmAll').onclick=async()=>{try{const j=await api('/api/episode/confirm_all',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:episode()})});await refresh();msg(`已确认 ${j.confirmed_frames} 帧`,true)}catch(e){msg(e.message)}};
$('tabSetup').onclick=()=>setView('setup');$('tabReview').onclick=()=>setView('review');$('episode').onchange=loadEpisode;
$('frame').onchange=()=>{S.currentFrame=currentFrame();loadViews()};
$('prev').onclick=()=>{const s=$('frame');if(s.selectedIndex>0){s.selectedIndex--;S.currentFrame=currentFrame();loadViews()}};
$('next').onclick=()=>{const s=$('frame');if(s.selectedIndex<s.options.length-1){s.selectedIndex++;S.currentFrame=currentFrame();loadViews()}};
$('jumpSeed').onclick=()=>setFrame(seedFrame());
$('jumpFirst').onclick=()=>setFrame(S.info?.frame_indices?.[0]??0);
$('jumpLast').onclick=()=>{const idx=S.info?.frame_indices||[];if(idx.length)setFrame(idx[idx.length-1])};
['showMask','showMolmo','showSamSeed'].forEach(id=>$(id).onchange=()=>loadViews());
boot().catch(e=>msg(e.message));
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真机 episode 级 head camera mask 标注 Web")
    parser.add_argument("--data-root", required=True, help="包含 episode_XXXX 的数据根目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--sam-server-url",
        default="http://127.0.0.1:8765",
        help="SAM3 常驻推理 server 地址（校验阶段单帧重分割用）",
    )
    parser.add_argument(
        "--annotation-camera",
        default=DEFAULT_ANNOTATION_CAMERA,
        help="默认只标注 head camera（color_0）",
    )
    parser.add_argument(
        "--qwen-api-url",
        default=os.environ.get("QWEN_API_URL") or os.environ.get("DASHSCOPE_BASE_URL") or "",
        help="DashScope 默认 https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", ""),
        help="模型，DashScope 默认 qwen3.7-plus",
    )
    parser.add_argument(
        "--qwen-api-key",
        default=None,
        help="默认读取 DASHSCOPE_API_KEY / QWEN_API_KEY",
    )
    parser.add_argument(
        "--qwen-enable-thinking",
        action="store_true",
        help="开启 DashScope enable_thinking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        args.data_root,
        args.annotation_camera,
        args.sam_server_url,
        qwen_api_url=args.qwen_api_url or None,
        qwen_model=args.qwen_model or None,
        qwen_api_key=args.qwen_api_key,
        qwen_enable_thinking=args.qwen_enable_thinking,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
