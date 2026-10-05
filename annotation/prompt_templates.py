"""任务 instruction 与 MolmoPoint 合并 prompt 模板。"""

from __future__ import annotations

import json
import re
from typing import Any

PLACEHOLDER_RE = re.compile(r"\{([A-Z])\}")


class PromptTemplateError(ValueError):
    """模板或标签配置错误。"""


def placeholder_token(index: int) -> str:
    if index < 0 or index >= 26:
        raise PromptTemplateError(f"占位符索引超出范围: {index}")
    return "{" + chr(ord("A") + index) + "}"


def required_placeholder_count(template: str) -> int:
    letters = PLACEHOLDER_RE.findall(template)
    if not letters:
        return 0
    expected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for index, letter in enumerate(letters):
        if letter != expected[index]:
            raise PromptTemplateError(
                f"instruction 模板占位符必须按 {{A}}, {{B}}, ... 顺序，发现 {{{letter}}}"
            )
    return len(letters)


def build_combined_point_prompt(labels: list[str]) -> str:
    """生成 MolmoPoint 合并 prompt：Point to the {A} and {B}（单物体时 Point to the {A}.）。"""
    parts = [str(label).strip() for label in labels if str(label).strip()]
    if not parts:
        raise PromptTemplateError("至少需要一个物体标签")
    if len(parts) == 1:
        return f"Point to the {parts[0]}."
    if len(parts) == 2:
        return f"Point to the {parts[0]} and {parts[1]}."
    return f"Point to the {', '.join(parts[:-1])} and {parts[-1]}."


def build_per_instance_point_prompt_summary(labels: list[str]) -> str:
    """逐 instance Molmo 调用的 prompt 摘要，用于存储与 UI 展示。"""
    parts = [build_combined_point_prompt([label]) for label in labels if str(label).strip()]
    if not parts:
        raise PromptTemplateError("至少需要一个物体标签")
    return " | ".join(parts)


def split_point_prompt_summary(text: str) -> list[str]:
    """把存储的逐 instance point_prompt 拆成 Molmo 调用列表。"""
    return [part.strip() for part in str(text).split("|") if part.strip()]


def build_instruction(template: str, labels: list[str]) -> str:
    """把标签名填入 instruction 模板，如 Place {A} on the {B}。"""
    template = str(template).strip()
    if not template:
        raise PromptTemplateError("instruction 模板不能为空")
    count = required_placeholder_count(template)
    if count == 0:
        raise PromptTemplateError("instruction 模板需包含 {A}, {B} 等占位符")
    if len(labels) != count:
        raise PromptTemplateError(
            f"instruction 需要 {count} 个标签，当前选了 {len(labels)} 个"
        )
    result = template
    for index, label in enumerate(labels):
        result = result.replace(placeholder_token(index), str(label).strip())
    if PLACEHOLDER_RE.search(result):
        raise PromptTemplateError("instruction 模板未能完全替换占位符")
    return result


def labels_for_instance_ids(
    instances: list[dict[str, Any]],
    instance_ids: list[int],
) -> tuple[list[int], list[str]]:
    lookup = {int(item["id"]): str(item["name"]) for item in instances}
    normalized_ids: list[int] = []
    labels: list[str] = []
    seen: set[int] = set()
    for instance_id in instance_ids:
        value = int(instance_id)
        if value in seen:
            continue
        seen.add(value)
        if value not in lookup:
            raise PromptTemplateError(f"实例 {value} 不存在")
        normalized_ids.append(value)
        labels.append(lookup[value])
    if not labels:
        raise PromptTemplateError("至少选择一个物体标签")
    return normalized_ids, labels


def build_episode_prompts(
    *,
    instruction_template: str,
    instances: list[dict[str, Any]],
    instance_ids: list[int],
) -> dict[str, Any]:
    normalized_ids, labels = labels_for_instance_ids(instances, instance_ids)
    return {
        "instance_ids": normalized_ids,
        "labels": labels,
        "instruction": build_instruction(instruction_template, labels),
        "point_prompt": build_per_instance_point_prompt_summary(labels),
    }


def slot_letters(template: str) -> list[str]:
    count = required_placeholder_count(template)
    return [chr(ord("A") + index) for index in range(count)]


def normalize_label_slots(
    label_slots: dict[str, Any] | None,
    *,
    instruction_template: str,
    instances: list[dict[str, Any]],
) -> dict[str, list[int]]:
    letters = slot_letters(instruction_template)
    if not label_slots:
        raise PromptTemplateError("请先配置 label_slots（每个占位符的候选标签）")
    lookup = {int(item["id"]) for item in instances}
    normalized: dict[str, list[int]] = {}
    for letter in letters:
        values = label_slots.get(letter) or label_slots.get(letter.lower()) or []
        if not isinstance(values, list) or not values:
            raise PromptTemplateError(f"占位符 {{{letter}}} 缺少候选标签")
        ids: list[int] = []
        seen: set[int] = set()
        for value in values:
            instance_id = int(value)
            if instance_id not in lookup:
                raise PromptTemplateError(f"实例 {instance_id} 不存在")
            if instance_id in seen:
                continue
            seen.add(instance_id)
            ids.append(instance_id)
        normalized[letter] = ids
    return normalized


def label_slot_options(
    label_slots: dict[str, list[int]],
    instances: list[dict[str, Any]],
) -> dict[str, list[str]]:
    lookup = {int(item["id"]): str(item["name"]) for item in instances}
    return {
        letter: [lookup[instance_id] for instance_id in ids]
        for letter, ids in label_slots.items()
    }


def build_qwen_episode_label_prompt(
    *,
    instruction_template: str,
    slot_options: dict[str, list[str]],
) -> str:
    letters = slot_letters(instruction_template)
    option_lines = []
    for letter in letters:
        options = ", ".join(slot_options[letter])
        option_lines.append(f"{letter} 的可选项为 {options}")
    options_text = "。".join(option_lines)
    return (
        "这两张图片是机器人任务的第一帧和最后一帧。"
        f'任务是 "{instruction_template}"。'
        f"{options_text}。"
        "请根据首尾帧变化判断该 episode 应选择的标签。"
        f'只返回 JSON，键为 {", ".join(letters)}，值为标签名称，例如 '
        + "{"
        + ", ".join(f'"{letter}": "{slot_options[letter][0]}"' for letter in letters)
        + "}。"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if not raw:
        raise PromptTemplateError("Qwen 返回为空")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PromptTemplateError(f"无法解析 Qwen 返回的 JSON: {raw[:200]}") from exc
    if not isinstance(payload, dict):
        raise PromptTemplateError("Qwen 返回必须是 JSON 对象")
    return payload


def _match_option_name(selected: str, options: list[str]) -> str:
    text = str(selected).strip()
    if not text:
        raise PromptTemplateError("Qwen 返回了空标签")
    lowered = {option.strip().lower(): option for option in options}
    key = text.lower()
    if key in lowered:
        return lowered[key]
    for option in options:
        if key in option.lower() or option.lower() in key:
            return option
    raise PromptTemplateError(f"标签 {text!r} 不在候选列表 {options} 中")


def parse_qwen_label_response(
    text: str,
    *,
    slot_options: dict[str, list[str]],
) -> dict[str, str]:
    payload = _extract_json_object(text)
    selection: dict[str, str] = {}
    for letter, options in slot_options.items():
        value = payload.get(letter)
        if value is None:
            value = payload.get(letter.lower())
        if value is None:
            raise PromptTemplateError(f"Qwen 返回缺少占位符 {letter}")
        selection[letter] = _match_option_name(str(value), options)
    return selection


def selection_to_instance_ids(
    selection: dict[str, str],
    *,
    label_slots: dict[str, list[int]],
    instances: list[dict[str, Any]],
) -> list[int]:
    lookup = {str(item["name"]).strip().lower(): int(item["id"]) for item in instances}
    result: list[int] = []
    for letter, candidate_ids in label_slots.items():
        chosen = selection[letter].strip().lower()
        matched_id = None
        for instance_id in candidate_ids:
            name = next(str(item["name"]) for item in instances if int(item["id"]) == instance_id)
            if name.strip().lower() == chosen:
                matched_id = instance_id
                break
        if matched_id is None and chosen in lookup and lookup[chosen] in candidate_ids:
            matched_id = lookup[chosen]
        if matched_id is None:
            raise PromptTemplateError(
                f"无法把 Qwen 选择的 {letter}={selection[letter]!r} 映射到候选实例"
            )
        result.append(matched_id)
    return result
