from __future__ import annotations

import json
import logging
import queue
import re
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...contracts import AgentContext, ContextFragment, Observation

logger = logging.getLogger(__name__)

INSTANCE_PALETTE = (
    (255, 77, 79),
    (64, 169, 255),
    (115, 209, 61),
    (250, 173, 20),
    (146, 84, 222),
    (19, 194, 194),
    (245, 34, 45),
    (47, 84, 235),
    (82, 196, 26),
    (250, 140, 22),
    (114, 46, 209),
    (19, 194, 194),
)

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def as_rgb_uint8(image) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.max(array, initial=0)) <= 1.0:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return np.ascontiguousarray(array)


def instance_color(instance_id: int) -> tuple[int, int, int]:
    return INSTANCE_PALETTE[(int(instance_id) - 1) % len(INSTANCE_PALETTE)]


def instance_names_from_meta(actors_frame_meta) -> dict[int, str]:
    names: dict[int, str] = {}
    if not isinstance(actors_frame_meta, dict):
        return names
    for name, ids in actors_frame_meta.items():
        for instance_id in ids or ():
            names[int(instance_id)] = str(name)
    return names


def vis_font(size: int = 16):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def molmo_points_from_grounding(grounding: dict) -> list[dict]:
    points: list[dict] = []
    molmo = grounding.get("molmo") or {}
    prompts = molmo.get("sam_prompts") or grounding.get("sam_prompts") or []
    for prompt in prompts:
        instance_id = int(prompt["instance_id"])
        for point in prompt.get("points") or ():
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append(
                    {
                        "instance_id": instance_id,
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                )
    if points:
        return points
    for result in molmo.get("molmo_results") or ():
        for item in result.get("points") or ():
            point = item.get("point") or [item.get("x"), item.get("y")]
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            if point[0] is None or point[1] is None:
                continue
            instance_id = item.get("instance_id", result.get("instance_id"))
            if instance_id is None:
                continue
            points.append(
                {
                    "instance_id": int(instance_id),
                    "x": float(point[0]),
                    "y": float(point[1]),
                }
            )
    return points


def colorize_label_map(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(mask):
        value = int(instance_id)
        if value == 0:
            continue
        color[mask == value] = instance_color(value)
    return color


def blend_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = as_rgb_uint8(rgb)
    color = colorize_label_map(mask)
    out = rgb.astype(np.float32)
    visible = np.any(color > 0, axis=2)
    out[visible] = out[visible] * (1.0 - alpha) + color[visible].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_legend(draw, names: dict[int, str], *, xy: tuple[int, int] = (8, 36)) -> None:
    font = vis_font(16)
    x, y = xy
    for instance_id in sorted(names):
        color = instance_color(instance_id)
        draw.rectangle((x, y, x + 14, y + 14), fill=color, outline=(255, 255, 255))
        draw.text(
            (x + 20, y - 1),
            f"{instance_id}: {names[instance_id]}",
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        y += 20


def draw_banner(draw, lines: list[str], *, xy: tuple[int, int] = (8, 8)) -> int:
    font = vis_font(16)
    x, y = xy
    for line in lines:
        if not line:
            continue
        draw.text(
            (x, y),
            line,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y += 20
    return y


def format_plan_text(plan: Mapping[str, Any]) -> str:
    lines = [
        f"task_mode={plan.get('task_mode')}",
        f"task_instruction={plan.get('task_instruction')}",
    ]
    subtasks = plan.get("subtasks") or []
    if subtasks:
        lines.append("subtasks:")
        current = plan.get("current_instruction")
        for index, instruction in enumerate(subtasks):
            marker = " *" if instruction == current else ""
            lines.append(f"  [{index}]{marker} {instruction}")
    elif plan.get("current_instruction"):
        lines.append(f"current_instruction={plan['current_instruction']}")
    return "\n".join(lines)


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_molmo_vis(
    output_dir: Path,
    *,
    frame: int,
    rgb: np.ndarray,
    grounding: dict,
) -> dict[str, Path]:
    from PIL import Image, ImageDraw

    names = instance_names_from_meta(grounding.get("actors_frame_meta"))
    rgb = as_rgb_uint8(rgb)
    points_image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(points_image)
    font = vis_font(18)
    draw.text(
        (8, 8),
        f"MolmoPoint frame {frame}",
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    for item in molmo_points_from_grounding(grounding):
        x = int(round(item["x"]))
        y = int(round(item["y"]))
        color = instance_color(item["instance_id"])
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=color, width=3)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        label = names.get(item["instance_id"], str(item["instance_id"]))
        draw.text(
            (x + 12, y - 14),
            f"{item['instance_id']}:{label}",
            fill=(255, 255, 255),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    draw_legend(draw, names)
    molmo_dir = output_dir / "molmo"
    molmo_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = molmo_dir / f"frame_{frame:06d}_rgb.png"
    points_path = molmo_dir / f"frame_{frame:06d}_points.png"
    Image.fromarray(rgb).save(rgb_path)
    points_image.save(points_path)
    return {"rgb": rgb_path, "points": points_path}


def save_sam_vis(
    output_dir: Path,
    *,
    frame: int,
    rgb: np.ndarray,
    pred_mask: np.ndarray,
    oracle_mask: np.ndarray | None = None,
    frame_iou: dict[int, float] | None = None,
    names: dict[int, str] | None = None,
    banner: list[str] | None = None,
) -> dict[str, Path]:
    from PIL import Image, ImageDraw

    names = names or instance_names_from_meta({})
    pred = Image.fromarray(blend_mask(rgb, pred_mask))
    pred_draw = ImageDraw.Draw(pred)
    title = f"SAM pred frame {frame}"
    if frame_iou:
        title += "  " + "  ".join(
            f"id{instance_id}={value:.2f}" for instance_id, value in sorted(frame_iou.items())
        )
    lines = [title, *(banner or ())]
    legend_y = draw_banner(pred_draw, lines)
    draw_legend(pred_draw, names, xy=(8, legend_y + 4))
    sam_dir = output_dir / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    pred_path = sam_dir / f"frame_{frame:06d}_pred.png"
    pred.save(pred_path)
    saved = {"pred": pred_path}
    if oracle_mask is None:
        return saved
    oracle = Image.fromarray(blend_mask(rgb, oracle_mask))
    oracle_draw = ImageDraw.Draw(oracle)
    oracle_draw.text(
        (8, 8),
        f"Oracle mask frame {frame}",
        fill=(255, 255, 255),
        font=vis_font(18),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    draw_legend(oracle_draw, names)
    compare = Image.new("RGB", (pred.width * 2, pred.height))
    compare.paste(pred, (0, 0))
    compare.paste(oracle, (pred.width, 0))
    oracle_path = sam_dir / f"frame_{frame:06d}_oracle.png"
    compare_path = sam_dir / f"frame_{frame:06d}_compare.png"
    oracle.save(oracle_path)
    compare.save(compare_path)
    saved["oracle"] = oracle_path
    saved["compare"] = compare_path
    return saved


def save_sam_contact_sheet(
    output_dir: Path,
    frames: list[dict],
    *,
    cols: int = 4,
    tile_width: int = 320,
) -> Path | None:
    from PIL import Image

    if not frames:
        return None
    tiles = []
    for item in frames:
        image = Image.fromarray(as_rgb_uint8(item["image"]))
        if image.width > tile_width:
            height = max(1, int(round(image.height * tile_width / image.width)))
            image = image.resize((tile_width, height), Image.Resampling.BILINEAR)
        tiles.append(image)
    width, height = tiles[0].size
    cols = max(1, min(cols, len(tiles)))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * width, rows * height), color=(16, 16, 16))
    for index, image in enumerate(tiles):
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        row, col = divmod(index, cols)
        sheet.paste(image, (col * width, row * height))
    path = output_dir / "sam" / "contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def save_png(path: Path, image) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(as_rgb_uint8(image)).save(path)
    return path


def save_action_context_panel(
    path: Path,
    payload: Mapping[str, Any],
    *,
    banner: list[str] | None = None,
) -> Path:
    from PIL import Image, ImageDraw

    images = payload.get("images") or {}
    tiles: list[tuple[str, np.ndarray]] = []
    for name in CAMERA_ORDER:
        if name in images:
            tiles.append((name, as_rgb_uint8(images[name])))
    seg = payload.get("seg_cam_high")
    if seg is not None and "cam_high" in images:
        tiles.append(("seg_cam_high", blend_mask(images["cam_high"], np.asarray(seg))))
    if not tiles:
        raise ValueError("action context 没有可画的图像")
    target_h = min(240, min(image.shape[0] for _, image in tiles))
    resized = []
    for name, image in tiles:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        width = max(1, int(round(pil.width * target_h / pil.height)))
        resized.append((name, pil.resize((width, target_h), PILImage.Resampling.BILINEAR)))
    gap = 8
    banner_h = 28 * max(1, len(banner or ()) + 1)
    width = gap + sum(image.width + gap for _, image in resized)
    height = banner_h + target_h + 28
    panel = Image.new("RGB", (width, height), color=(16, 16, 16))
    draw = ImageDraw.Draw(panel)
    draw_banner(draw, banner or [f"prompt={payload.get('prompt')}"])
    x = gap
    y = banner_h
    font = vis_font(14)
    for name, image in resized:
        panel.paste(image, (x, y))
        draw.text(
            (x + 4, y + image.height + 4),
            name,
            fill=(220, 220, 220),
            font=font,
        )
        x += image.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)
    return path


def serialize_entity(fragment: ContextFragment | None) -> dict[str, Any] | None:
    if fragment is None:
        return None
    data = fragment.data
    if is_dataclass(data) and not isinstance(data, type):
        payload = asdict(data)
        actors = getattr(data, "actors_frame_meta", None)
        if actors is not None:
            payload["actors_frame_meta"] = dict(actors)
        return jsonable(payload)
    if isinstance(data, Mapping):
        return jsonable(dict(data))
    return {"repr": repr(data)}


def serialize_grounding(fragment: ContextFragment | None) -> dict[str, Any] | None:
    if fragment is None or not isinstance(fragment.data, Mapping):
        return None
    data = fragment.data
    mask = data.get("seg_mask")
    copied: dict[str, Any] = {
        "task_id": data.get("task_id"),
        "actors_frame_meta": dict(data.get("actors_frame_meta") or {}),
        "molmo": data.get("molmo"),
    }
    if mask is not None:
        copied["seg_mask"] = np.array(mask, copy=True)
    return copied


def copy_images(images: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {str(name): np.array(image, copy=True) for name, image in images.items()}


def copy_action_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    if isinstance(copied.get("images"), Mapping):
        copied["images"] = copy_images(copied["images"])
    if copied.get("seg_cam_high") is not None:
        copied["seg_cam_high"] = np.array(copied["seg_cam_high"], copy=True)
    if copied.get("state") is not None:
        copied["state"] = np.array(copied["state"], copy=True)
    if isinstance(copied.get("actors_frame_meta"), Mapping):
        copied["actors_frame_meta"] = dict(copied["actors_frame_meta"])
    return copied


def action_context_meta(payload: Mapping[str, Any], *, sent_via: str) -> dict[str, Any]:
    images = payload.get("images") or {}
    seg = payload.get("seg_cam_high")
    present_ids: list[int] = []
    if seg is not None:
        unique = np.unique(np.asarray(seg))
        present_ids = [int(value) for value in unique if int(value) != 0]
    return {
        "sent_via": sent_via,
        "prompt": payload.get("prompt"),
        "state": jsonable(payload.get("state")),
        "episode_id": payload.get("episode_id"),
        "episode_index": payload.get("episode_index"),
        "step_id": payload.get("step_id"),
        "frame_index": payload.get("frame_index"),
        "actors_frame_meta": payload.get("actors_frame_meta"),
        "seg_representation": payload.get("seg_representation"),
        "has_seg_cam_high": seg is not None,
        "seg_cam_high_shape": None if seg is None else list(np.asarray(seg).shape),
        "seg_present_ids": present_ids,
        "image_names": [name for name in CAMERA_ORDER if name in images],
        "image_shapes": {name: list(np.asarray(image).shape) for name, image in images.items()},
    }


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("._")
    return cleaned[:120] or "episode"


class GatewayDebugRecorder:
    """Side-channel dump of gateway inputs, grounding, plan, and PI05 action context."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        queue_size: int = 64,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._episode_id: str | None = None
        self._sam_frames: list[dict[str, Any]] = []
        self._thread = threading.Thread(
            target=self._run,
            name="mobile-gateway-debug",
            daemon=True,
        )
        self._thread.start()

    def record_infer(
        self,
        *,
        observation: Observation,
        context: AgentContext,
        response: Mapping[str, Any],
        action_payload: Mapping[str, Any],
    ) -> None:
        self._enqueue(
            self._snapshot(
                "infer",
                observation=observation,
                context=context,
                response=response,
                action_payload=action_payload,
                sent_via="infer",
            )
        )

    def record_reset(self, episode_id: str | None = None) -> None:
        self._enqueue({"rpc": "reset", "episode_id": episode_id})

    def flush(self, timeout: float = 30.0) -> None:
        self._queue.join()
        del timeout

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=30.0)

    def _enqueue(self, item: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning(
                "debug dump queue full, dropping %s step %s",
                item.get("rpc"),
                item.get("step_id"),
            )

    def _snapshot(
        self,
        rpc: str,
        *,
        observation: Observation,
        context: AgentContext,
        response: Mapping[str, Any],
        action_payload: Mapping[str, Any],
        sent_via: str,
    ) -> dict[str, Any]:
        grounding = serialize_grounding(context.fragments.get("visual_grounding"))
        return {
            "rpc": rpc,
            "sent_via": sent_via,
            "episode_id": context.memory.get("episode_id") or response.get("episode_id"),
            "step_id": int(context.memory.get("step_id") or response.get("step_id") or 0),
            "images": copy_images(observation.images),
            "grounding": grounding,
            "entity": serialize_entity(context.fragments.get("entity_grounding")),
            "action_payload": copy_action_payload(action_payload),
            "goal": {
                "goal_id": context.goal.goal_id,
                "instruction": context.goal.instruction,
                "entity_bindings": dict(context.goal.entity_bindings),
                "metadata": dict(context.goal.metadata),
            },
            "plan": {
                "task_instruction": response.get("task_instruction"),
                "task_mode": response.get("task_mode"),
                "plan_index": response.get("plan_index"),
                "plan_length": response.get("plan_length"),
                "current_instruction": response.get("current_instruction"),
                "subtasks": list(response.get("subtasks") or ()),
                "task_done": response.get("task_done"),
                "subtask_done": response.get("subtask_done"),
            },
            "progress": response.get("progress"),
            "timing": dict(response.get("timing") or {}),
            "action_shape": list(np.asarray(response["actions"]).shape)
            if rpc == "infer" and response.get("actions") is not None
            else None,
        }

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    self._finish_episode()
                    return
                self._write(item)
            except Exception:
                logger.exception("debug dump failed for %s", item.get("rpc") if item else None)
            finally:
                self._queue.task_done()

    def _episode_dir(self, episode_id: str | None) -> Path:
        return self.output_dir / _safe_id(episode_id or "default")

    def _write(self, item: dict[str, Any]) -> None:
        rpc = item["rpc"]
        if rpc == "reset":
            self._finish_episode()
            self._episode_id = None
            return
        if rpc != "infer":
            return
        episode_id = str(item.get("episode_id") or "default")
        if self._episode_id is not None and episode_id != self._episode_id:
            self._finish_episode()
        if self._episode_id != episode_id:
            self._episode_id = episode_id
            self._sam_frames = []
        episode_dir = self._episode_dir(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        step_id = int(item["step_id"])
        plan = item.get("plan") or {}
        (episode_dir / "plan.json").write_text(
            json.dumps(jsonable({**plan, "goal": item.get("goal"), "entity": item.get("entity")}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        step_dir = episode_dir / "infer" / f"step_{step_id:06d}"
        paths: dict[str, str] = {}
        banner = [
            f"infer step={step_id} plan={plan.get('plan_index')}/{plan.get('plan_length')}",
            str(plan.get("current_instruction") or item.get("goal", {}).get("instruction") or ""),
        ]
        step_dir.mkdir(parents=True, exist_ok=True)
        for name, image in (item.get("images") or {}).items():
            paths[name] = str(save_png(step_dir / f"{name}.png", image))
        payload = item.get("action_payload") or {}
        if payload.get("images"):
            meta_path = step_dir / "action_context.json"
            meta_path.write_text(
                json.dumps(
                    jsonable(action_context_meta(payload, sent_via=item.get("sent_via") or "infer")),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            paths["action_context_json"] = str(meta_path)
            try:
                panel = save_action_context_panel(
                    step_dir / "action_context.png",
                    payload,
                    banner=banner + [f"prompt={payload.get('prompt')}"],
                )
                paths["action_context"] = str(panel)
            except Exception:
                logger.exception("failed to render action context panel")
        grounding = item.get("grounding") or {}
        rgb = (item.get("images") or {}).get("cam_high")
        if rgb is not None and grounding.get("seg_mask") is not None:
            saved = save_sam_vis(
                step_dir,
                frame=step_id,
                rgb=rgb,
                pred_mask=grounding["seg_mask"],
                names=instance_names_from_meta(grounding.get("actors_frame_meta")),
                banner=banner,
            )
            paths.update({key: str(path) for key, path in saved.items()})
            self._sam_frames.append(
                {"frame": step_id, "image": as_rgb_uint8(blend_mask(rgb, grounding["seg_mask"]))}
            )
        if rgb is not None and grounding.get("molmo"):
            saved = save_molmo_vis(
                step_dir,
                frame=step_id,
                rgb=rgb,
                grounding=grounding,
            )
            paths.update({f"molmo_{key}": str(path) for key, path in saved.items()})
        if item.get("entity") is not None:
            entity_path = step_dir / "entity.json"
            entity_path.write_text(
                json.dumps(jsonable(item["entity"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["entity"] = str(entity_path)
        row = {
            "rpc": "infer",
            "episode_id": episode_id,
            "step_id": step_id,
            "sent_via": item.get("sent_via"),
            "plan": plan,
            "goal": item.get("goal"),
            "entity": item.get("entity"),
            "prompt": (item.get("action_payload") or {}).get("prompt"),
            "progress": item.get("progress"),
            "timing": item.get("timing"),
            "action_shape": item.get("action_shape"),
            "has_molmo": bool((item.get("grounding") or {}).get("molmo")),
            "has_seg": (item.get("grounding") or {}).get("seg_mask") is not None,
            "paths": paths,
        }
        with (episode_dir / "timeline.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

    def _finish_episode(self) -> None:
        if self._episode_id is None or not self._sam_frames:
            self._sam_frames = []
            return
        episode_dir = self._episode_dir(self._episode_id)
        save_sam_contact_sheet(episode_dir / "grounding", self._sam_frames)
        self._sam_frames = []
