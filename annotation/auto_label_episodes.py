#!/usr/bin/env python3
"""使用 Qwen-VL 为任务下全部 episode 自动选择标签。

用法（DashScope，推荐）::

    export DASHSCOPE_API_KEY=sk-xxx
    python -m annotation.auto_label_episodes \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --overwrite

用法（本地 vLLM）::

    python -m annotation.auto_label_episodes \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --qwen-api-url http://127.0.0.1:8000/v1 \\
      --qwen-model Qwen2.5-VL-7B-Instruct

先在 ``episode_labels.json`` 配置 ``label_slots``，或在命令行用 ``--slot-a`` / ``--slot-b`` 指定。
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from .annotate_mobile_masks_web import AnnotationStore
    from .episode_auto_label import label_episodes_with_qwen
    from .prompt_templates import PromptTemplateError
    from .qwen_label_backend import DEFAULT_QWEN_MODEL, QwenVLMClient
except ImportError:
    from annotate_mobile_masks_web import AnnotationStore
    from episode_auto_label import label_episodes_with_qwen
    from prompt_templates import PromptTemplateError
    from qwen_label_backend import DEFAULT_QWEN_MODEL, QwenVLMClient


def parse_slot_arg(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen 自动为 episode 选择任务标签")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--camera", default="color_0")
    parser.add_argument(
        "--qwen-api-url",
        default=os.environ.get("QWEN_API_URL") or os.environ.get("DASHSCOPE_BASE_URL") or "",
        help="默认读取 QWEN_API_URL / DASHSCOPE_BASE_URL；有 DASHSCOPE_API_KEY 时自动用 DashScope",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", ""),
        help=f"模型名，DashScope 默认 {DEFAULT_QWEN_MODEL}",
    )
    parser.add_argument(
        "--qwen-api-key",
        default=None,
        help="默认读取 DASHSCOPE_API_KEY / QWEN_API_KEY",
    )
    parser.add_argument(
        "--qwen-enable-thinking",
        action="store_true",
        help="开启 DashScope enable_thinking（自动选标签默认关闭）",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--episode", action="append", default=None, help="只处理指定 episode，可重复")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--slot-a", default="", help="占位符 A 的候选实例 ID，逗号分隔")
    parser.add_argument("--slot-b", default="", help="占位符 B 的候选实例 ID，逗号分隔")
    parser.add_argument("--slot-c", default="", help="占位符 C 的候选实例 ID，逗号分隔")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AnnotationStore(args.data_root, args.camera)
    cli_slots = {
        letter: parse_slot_arg(value)
        for letter, value in {
            "A": args.slot_a,
            "B": args.slot_b,
            "C": args.slot_c,
        }.items()
        if value
    }
    if cli_slots:
        store.set_label_slots(cli_slots)
    client = QwenVLMClient.from_env(
        api_url=args.qwen_api_url or None,
        model=args.qwen_model or None,
        api_key=args.qwen_api_key,
        timeout=args.timeout,
        enable_thinking=args.qwen_enable_thinking,
    )
    try:
        results = label_episodes_with_qwen(
            store,
            client,
            episodes=args.episode,
            overwrite=args.overwrite,
        )
    except PromptTemplateError as exc:
        raise SystemExit(str(exc)) from exc
    for item in results:
        if item.get("skipped"):
            print(f"[SKIP] {item['episode']}: 已有标签")
            continue
        print(
            f"[OK]   {item['episode']}: {item['instruction']} | {item['point_prompt']}"
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
