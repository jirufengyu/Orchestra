from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Mapping

from ...contracts import (
    ContextFragment,
    ContextToolRequest,
    ContextToolResult,
    ContextToolSpec,
    Goal,
    Observation,
)
from ...memory import MemorySystem
from ...tools import ContextToolRuntime
from .http import JsonHttpClient
from .registry import EntitySelection, RegistryEntityResolver


def encode_image_base64(image: Any) -> str:
    from PIL import Image
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_label_map(encoded: str) -> Any:
    from PIL import Image
    import numpy as np

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


@dataclass
class EntityResolverTool:
    resolver: RegistryEntityResolver

    @property
    def spec(self) -> ContextToolSpec:
        return ContextToolSpec(
            "mobile_entity_resolver",
            "Resolve a mobile manipulation instruction to registry-constrained stable ids.",
            {"instruction": "string"},
            {"task_id": "string", "slots": "object", "instances": "object"},
        )

    def invoke(self, request: ContextToolRequest) -> ContextToolResult:
        selection = self.resolver.resolve(str(request.inputs["instruction"]))
        return ContextToolResult(selection, confidence=1.0)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class MolmoPointTool:
    base_url: str
    timeout: float = 300.0
    _client: JsonHttpClient = field(init=False)

    def __post_init__(self) -> None:
        self._client = JsonHttpClient(self.base_url, self.timeout)

    @property
    def spec(self) -> ContextToolSpec:
        return ContextToolSpec(
            "molmo_point",
            "Point to each registered object in the current RGB frame.",
        )

    def invoke(self, request: ContextToolRequest) -> ContextToolResult:
        image_base64 = str(request.inputs["image_base64"])
        prompts = request.inputs["instance_prompts"]
        sam_prompts = []
        raw_results = []
        for item in prompts:
            response = self._client.request(
                "POST",
                "/api/point_image",
                {
                    "image_base64": image_base64,
                    "prompt": item["prompt"],
                    "max_new_tokens": int(request.inputs.get("max_new_tokens", 200)),
                },
            )
            result = response.get("result", response)
            points = [
                list(point.get("point") or [point["x"], point["y"]])
                for point in result.get("points", ())
            ]
            if not points:
                raise RuntimeError(f"Molmo 未找到目标: {item['label']}")
            sam_prompts.append(
                {
                    "instance_id": int(item["instance_id"]),
                    "points": points,
                    "labels": [1] * len(points),
                }
            )
            raw_results.append(result)
        return ContextToolResult(
            {"sam_prompts": sam_prompts, "molmo_results": raw_results},
            confidence=1.0,
        )

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


@dataclass
class SamOnlineTool:
    base_url: str
    camera: str = "color_0"
    max_frames: int = 10_000
    timeout: float = 3600.0
    _client: JsonHttpClient = field(init=False)
    _session_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._client = JsonHttpClient(self.base_url, self.timeout)

    @property
    def spec(self) -> ContextToolSpec:
        return ContextToolSpec(
            "sam_online",
            "Track stable registered instances in an online video session.",
        )

    def invoke(self, request: ContextToolRequest) -> ContextToolResult:
        if self._session_id is None:
            started = self._client.request(
                "POST",
                "/api/online/session/start",
                {"camera": self.camera, "max_frames": self.max_frames},
            )
            self._session_id = str(started["online_session_id"])
        payload = {
            "online_session_id": self._session_id,
            "frame": int(request.inputs["frame_idx"]),
            "image_base64": request.inputs["image_base64"],
        }
        prompts = request.inputs.get("sam_prompts")
        if prompts is not None:
            payload["prompts"] = prompts
        result = self._client.request("POST", "/api/online/frame", payload)
        return ContextToolResult(result, confidence=1.0)

    def reset(self) -> None:
        if self._session_id is None:
            return
        try:
            self._client.request(
                "POST",
                "/api/online/session/close",
                {"online_session_id": self._session_id},
            )
        finally:
            self._session_id = None

    def close(self) -> None:
        self.reset()


@dataclass
class EntityGroundingProvider:
    tools: ContextToolRuntime
    name: str = "entity_grounding"
    _cache_key: tuple[str | None, str] | None = field(default=None, init=False)
    _selection: EntitySelection | None = field(default=None, init=False)

    def provide(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
    ) -> ContextFragment:
        del observation
        key = (memory.episode_id, goal.instruction)
        if key != self._cache_key:
            result = self.tools.invoke(
                "mobile_entity_resolver",
                {"instruction": goal.instruction},
                episode_id=memory.episode_id,
                step_id=memory.step_id,
            )
            if not isinstance(result.data, EntitySelection):
                raise TypeError("mobile_entity_resolver must return EntitySelection")
            self._selection = result.data
            self._cache_key = key
        assert self._selection is not None
        return ContextFragment(self.name, self._selection, confidence=1.0)

    def reset(self) -> None:
        self._cache_key = None
        self._selection = None

    def close(self) -> None:
        pass


@dataclass
class MobileVisualGroundingProvider:
    tools: ContextToolRuntime
    head_camera: str = "cam_high"
    name: str = "visual_grounding"
    _initialized: bool = field(default=False, init=False)
    _last_key: tuple[str | None, int] | None = field(default=None, init=False)
    _last_fragment: ContextFragment | None = field(default=None, init=False)

    def provide_with_context(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySystem,
        fragments: Mapping[str, ContextFragment],
    ) -> ContextFragment:
        del goal
        key = (memory.episode_id, memory.step_id)
        if key == self._last_key and self._last_fragment is not None:
            return self._last_fragment
        entity_fragment = fragments.get("entity_grounding")
        if entity_fragment is None or not isinstance(entity_fragment.data, EntitySelection):
            raise RuntimeError("MobileVisualGroundingProvider requires entity_grounding first")
        selection = entity_fragment.data
        try:
            image = observation.images[self.head_camera]
        except KeyError as exc:
            raise KeyError(f"missing head camera: {self.head_camera}") from exc
        image_base64 = encode_image_base64(image)
        sam_prompts = None
        molmo_data = None
        if not self._initialized:
            molmo = self.tools.invoke(
                "molmo_point",
                {
                    "image_base64": image_base64,
                    "instance_prompts": selection.instance_prompts,
                },
                episode_id=memory.episode_id,
                step_id=memory.step_id,
            )
            molmo_data = molmo.data
            sam_prompts = molmo.data["sam_prompts"]
        tracked = self.tools.invoke(
            "sam_online",
            {
                "image_base64": image_base64,
                "frame_idx": memory.step_id,
                "sam_prompts": sam_prompts,
            },
            episode_id=memory.episode_id,
            step_id=memory.step_id,
        )
        self._initialized = True
        label_map = decode_label_map(tracked.data["label_map_png_base64"])
        actor_ids = {instance_id for ids in selection.actors_frame_meta.values() for instance_id in ids}
        present = set(int(value) for value in __import__("numpy").unique(label_map)) - {0}
        if not actor_ids.intersection(present):
            raise RuntimeError("SAM mask 中没有任何注册目标 instance id")
        fragment = ContextFragment(
            self.name,
            {
                "seg_mask": label_map,
                "actors_frame_meta": selection.actors_frame_meta,
                "task_id": selection.task_id,
                "tracking": tracked.data,
                "molmo": molmo_data,
            },
            confidence=tracked.confidence,
        )
        self._last_key = key
        self._last_fragment = fragment
        return fragment

    def provide(self, observation: Observation, goal: Goal, memory: MemorySystem) -> ContextFragment:
        raise RuntimeError("provider requires dependency-aware ContextSystem")

    def reset(self) -> None:
        self._initialized = False
        self._last_key = None
        self._last_fragment = None
        self.tools.reset()

    def close(self) -> None:
        self.tools.close()
