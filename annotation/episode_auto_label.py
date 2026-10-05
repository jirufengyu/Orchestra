"""使用 Qwen-VL 为 episode 自动选择任务标签。"""

from __future__ import annotations

from typing import Any, Protocol

try:
    from .prompt_templates import (
        PromptTemplateError,
        build_qwen_episode_label_prompt,
        label_slot_options,
        normalize_label_slots,
        parse_qwen_label_response,
        selection_to_instance_ids,
    )
except ImportError:
    from prompt_templates import (
        PromptTemplateError,
        build_qwen_episode_label_prompt,
        label_slot_options,
        normalize_label_slots,
        parse_qwen_label_response,
        selection_to_instance_ids,
    )


class EpisodeLabelStore(Protocol):
    annotation_camera: str

    def list_episodes(self) -> list[dict[str, Any]]: ...

    def load_task_instances(self) -> list[dict[str, Any]]: ...

    def get_instruction_template(self) -> str: ...

    def get_label_slots(self) -> dict[str, list[int]]: ...

    def get_episode_labels(self, episode: str) -> list[int]: ...

    def set_episode_labels(self, episode: str, instance_ids: list[int]) -> dict[str, Any]: ...

    def image_path(self, episode: str, frame_idx: int, camera: str | None = None) -> Any: ...

    def episode_info(self, episode: str) -> dict[str, Any]: ...


class QwenLabelClient(Protocol):
    def chat_with_images(self, prompt: str, image_paths: list[Any]) -> str: ...


def label_episode_with_qwen(
    store: EpisodeLabelStore,
    episode: str,
    client: QwenLabelClient,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if store.get_episode_labels(episode) and not overwrite:
        record = store.episode_info(episode)
        return {
            "episode": episode,
            "skipped": True,
            "reason": "already_labeled",
            "instance_ids": record.get("target_instance_ids") or [],
            "instruction": record.get("instruction", ""),
            "point_prompt": record.get("point_prompt", ""),
        }
    instances = store.load_task_instances()
    template = store.get_instruction_template()
    label_slots = normalize_label_slots(
        store.get_label_slots(),
        instruction_template=template,
        instances=instances,
    )
    slot_options = label_slot_options(label_slots, instances)
    info = store.episode_info(episode)
    indices = info["frame_indices"]
    if not indices:
        raise PromptTemplateError(f"{episode} 没有可用帧")
    first_frame = int(indices[0])
    last_frame = int(indices[-1])
    camera = store.annotation_camera
    image_paths = [
        store.image_path(episode, first_frame, camera),
        store.image_path(episode, last_frame, camera),
    ]
    prompt = build_qwen_episode_label_prompt(
        instruction_template=template,
        slot_options=slot_options,
    )
    response = client.chat_with_images(prompt, image_paths)
    selection = parse_qwen_label_response(response, slot_options=slot_options)
    instance_ids = selection_to_instance_ids(
        selection,
        label_slots=label_slots,
        instances=instances,
    )
    saved = store.set_episode_labels(episode, instance_ids)
    return {
        "episode": episode,
        "skipped": False,
        "selection": selection,
        "instance_ids": saved["instance_ids"],
        "instruction": saved["instruction"],
        "point_prompt": saved["point_prompt"],
        "qwen_response": response,
        "first_frame": first_frame,
        "last_frame": last_frame,
    }


def label_episodes_with_qwen(
    store: EpisodeLabelStore,
    client: QwenLabelClient,
    *,
    episodes: list[str] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    available = [
        item["name"]
        for item in store.list_episodes()
        if "error" not in item and (not episodes or item["name"] in set(episodes))
    ]
    if episodes:
        missing = sorted(set(episodes) - set(available))
        if missing:
            raise PromptTemplateError(f"episode 不存在: {', '.join(missing)}")
    results: list[dict[str, Any]] = []
    for episode in available:
        results.append(
            label_episode_with_qwen(
                store,
                episode,
                client,
                overwrite=overwrite,
            )
        )
    return results
