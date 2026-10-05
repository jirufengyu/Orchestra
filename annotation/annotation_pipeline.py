"""MolmoPoint → SAM3 视频标注流水线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .mobile_sam3_backend import Sam3RemoteClient, decode_label_map_png
from .prompt_templates import build_combined_point_prompt


class PointingClient(Protocol):
    def point_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 200,
    ) -> dict[str, Any]: ...


class SamSegmentClient(Protocol):
    def segment_episode(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        seed_frame_idx: int,
        prompts: list[dict[str, Any]],
        direction: str = "both",
    ) -> dict[str, Any]: ...


def molmo_points_to_sam_prompt(
    instance_id: int,
    points: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """把 Molmo 返回的点列表转为 SAM 点提示（同一实例可含多个前景点）。"""
    coords: list[list[float]] = []
    for item in points:
        point = item.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            x, y = item.get("x"), item.get("y")
            if x is None or y is None:
                continue
            point = [x, y]
        coords.append([float(point[0]), float(point[1])])
    if not coords:
        return None
    return {
        "instance_id": int(instance_id),
        "points": coords,
        "labels": [1] * len(coords),
    }


def tag_molmo_points_with_instance(
    points: list[dict[str, Any]],
    instance_id: int,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for item in points:
        tagged.append({**item, "instance_id": int(instance_id)})
    return tagged


def run_molmo_pointing_per_instance(
    pointing_client: PointingClient,
    image_path: Path,
    instance_prompts: list[dict[str, Any]],
    *,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """每个 instance 单独调用 MolmoPoint，避免合并 prompt 下 object_id 与标签错位。"""
    if not instance_prompts:
        raise ValueError("至少需要一个 instance 用于 Molmo 打点")

    per_instance: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    prompts_used: list[str] = []
    generated_chunks: list[str] = []

    for item in instance_prompts:
        instance_id = int(item["instance_id"])
        label = str(item["label"])
        prompt = str(item.get("prompt") or "").strip() or build_combined_point_prompt([label])
        result = pointing_client.point_image(image_path, prompt, max_new_tokens=max_new_tokens)
        points = tag_molmo_points_with_instance(result.get("points") or [], instance_id)
        per_instance.append(
            {
                "instance_id": instance_id,
                "label": label,
                "prompt": prompt,
                "result": result,
                "points": points,
            }
        )
        all_points.extend(points)
        prompts_used.append(prompt)
        text = str(result.get("generated_text") or "").strip()
        if text:
            generated_chunks.append(f"[{instance_id}] {text}")

    first_result = per_instance[0]["result"]
    return {
        "prompt": " | ".join(prompts_used),
        "image_path": first_result.get("image_path"),
        "image_size": first_result.get("image_size"),
        "generated_text": "\n---\n".join(generated_chunks),
        "points": all_points,
        "per_instance": per_instance,
    }


def build_sam_prompts_from_instance_molmo(
    instance_prompts: list[dict[str, Any]],
    per_instance_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """从逐 instance Molmo 结果构建 SAM 点提示。"""
    sam_prompts: list[dict[str, Any]] = []
    warnings: list[str] = []
    by_instance = {int(item["instance_id"]): item for item in per_instance_results}

    for item in instance_prompts:
        instance_id = int(item["instance_id"])
        label = str(item["label"])
        entry = by_instance.get(instance_id)
        if entry is None:
            warnings.append(f"实例 {instance_id} ({label}) 缺少 Molmo 结果")
            continue
        points = entry.get("points") or []
        if not points:
            warnings.append(f"实例 {instance_id} ({label}) 未检测到有效点")
            continue
        prompt = molmo_points_to_sam_prompt(instance_id, points)
        if prompt is None:
            warnings.append(f"实例 {instance_id} ({label}) 点坐标无效")
            continue
        sam_prompts.append(prompt)
    return sam_prompts, warnings


def run_sam_segmentation(
    sam_client: SamSegmentClient,
    episode_dir: Path,
    episode_name: str,
    camera: str,
    seed_frame_idx: int,
    sam_prompts: list[dict[str, Any]],
    *,
    direction: str = "both",
) -> dict[str, Any]:
    if not sam_prompts:
        raise ValueError("没有可用于 SAM3 的点提示")
    return sam_client.segment_episode(
        episode_dir,
        episode_name,
        camera,
        seed_frame_idx,
        sam_prompts,
        direction,
    )


def decode_segmentation_frames(
    segment_result: dict[str, Any],
    shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    frame_maps: dict[int, np.ndarray] = {}
    for item in segment_result.get("frames", []):
        frame_idx = int(item["frame"])
        frame_maps[frame_idx] = decode_label_map_png(item.get("label_map_png_base64", ""), shape)
    return frame_maps


def run_molmo_stage(
    pointing_client: PointingClient,
    *,
    seed_frame_image: Path,
    instance_prompts: list[dict[str, Any]],
    point_prompt: str | None = None,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Molmo 生产者阶段：每个 instance 单独打点，输出 SAM 可用的点提示。"""
    molmo_result = run_molmo_pointing_per_instance(
        pointing_client,
        seed_frame_image,
        instance_prompts,
        max_new_tokens=max_new_tokens,
    )
    per_instance = molmo_result.get("per_instance") or []
    sam_prompts, warnings = build_sam_prompts_from_instance_molmo(instance_prompts, per_instance)
    if not sam_prompts:
        raise ValueError("MolmoPoint 未产生任何可用点；" + "; ".join(warnings))
    return {
        "molmo_result": molmo_result,
        "sam_prompts": sam_prompts,
        "warnings": warnings,
        "point_prompt": molmo_result.get("prompt") or point_prompt or "",
    }


def run_sam_stage(
    sam_client: SamSegmentClient,
    *,
    episode_dir: Path,
    episode_name: str,
    camera: str,
    seed_frame_idx: int,
    sam_prompts: list[dict[str, Any]],
    image_shape: tuple[int, int],
    direction: str = "both",
) -> dict[str, Any]:
    """SAM 消费者阶段：用 Molmo 产出的点做视频传播分割。"""
    segment_result = run_sam_segmentation(
        sam_client,
        episode_dir,
        episode_name,
        camera,
        seed_frame_idx,
        sam_prompts,
        direction=direction,
    )
    frame_maps = decode_segmentation_frames(segment_result, image_shape)
    return {
        "segment_result": segment_result,
        "frame_maps": frame_maps,
        "sam_prompts": sam_prompts,
    }


class Sam3Pool:
    """对多个 SAM3 server 做简单轮询。"""

    def __init__(self, base_urls: list[str], timeout: float = 600.0):
        urls = [url.strip().rstrip("/") for url in base_urls if url.strip()]
        if not urls:
            raise ValueError("至少需要一个 SAM3 server URL")
        self._clients = [Sam3RemoteClient(url, timeout=timeout) for url in urls]
        self._index = 0

    def _next(self) -> Sam3RemoteClient:
        client = self._clients[self._index % len(self._clients)]
        self._index += 1
        return client

    def status(self) -> dict[str, Any]:
        statuses = [client.status() for client in self._clients]
        ready = [item for item in statuses if item.get("initialized")]
        return {
            "backend": "pool",
            "count": len(self._clients),
            "ready": len(ready),
            "servers": statuses,
            "available": bool(ready),
            "initialized": bool(ready),
        }

    def segment_episode(
        self,
        episode_dir: Path,
        episode_name: str,
        camera: str,
        seed_frame_idx: int,
        prompts: list[dict[str, Any]],
        direction: str = "both",
    ) -> dict[str, Any]:
        return self._next().segment_episode(
            episode_dir,
            episode_name,
            camera,
            seed_frame_idx,
            prompts,
            direction,
        )


def annotate_episode_pipeline(
    *,
    episode_dir: Path,
    episode_name: str,
    camera: str,
    seed_frame_image: Path,
    seed_frame_idx: int,
    instance_prompts: list[dict[str, Any]],
    pointing_client: PointingClient,
    sam_client: SamSegmentClient,
    image_shape: tuple[int, int],
    direction: str = "both",
    point_prompt: str | None = None,
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """完整流水线：逐 instance Molmo 首帧打点 → SAM3 视频传播。"""
    molmo_stage = run_molmo_stage(
        pointing_client,
        seed_frame_image=seed_frame_image,
        instance_prompts=instance_prompts,
        point_prompt=point_prompt,
        max_new_tokens=max_new_tokens,
    )
    sam_stage = run_sam_stage(
        sam_client,
        episode_dir=episode_dir,
        episode_name=episode_name,
        camera=camera,
        seed_frame_idx=seed_frame_idx,
        sam_prompts=molmo_stage["sam_prompts"],
        image_shape=image_shape,
        direction=direction,
    )
    return {
        "molmo_result": molmo_stage["molmo_result"],
        "point_prompt": molmo_stage["point_prompt"],
        "sam_prompts": molmo_stage["sam_prompts"],
        "warnings": molmo_stage["warnings"],
        "segment_result": sam_stage["segment_result"],
        "frame_maps": sam_stage["frame_maps"],
    }
