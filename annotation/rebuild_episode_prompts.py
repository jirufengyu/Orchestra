#!/usr/bin/env python3
"""按已有 instance_ids + 当前标签名重建 instruction / point_prompt，不跑 Qwen。

用法::

    python -m annotation.rebuild_episode_prompts \\
      --data-root /path/to/mobile_data/breakfast_preparation

    python -m annotation.rebuild_episode_prompts \\
      --data-root /path/to/mobile_data/breakfast_preparation \\
      --episodes episode_0004,episode_0008
"""

from __future__ import annotations

import argparse
import json

try:
    from .annotate_mobile_masks_web import AnnotationStore
except ImportError:
    from annotate_mobile_masks_web import AnnotationStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按已有标签重建 episode instruction/point")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--camera", default="color_0")
    parser.add_argument("--episodes", default="", help="逗号分隔 episode 名；默认全部")
    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="连已定制的 episode 一并覆盖",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = AnnotationStore(args.data_root, args.camera)
    episodes = [part.strip() for part in args.episodes.split(",") if part.strip()] or None
    result = store.rebuild_episode_prompts(episodes, skip_custom=not args.include_custom)
    counts = result["counts"]
    print(json.dumps(counts, ensure_ascii=False))
    for item in result["failed"]:
        print(f"[FAIL] {item['episode']}: {item['error']}")
    for item in result["skipped"][:10]:
        print(f"[SKIP] {item['episode']}: {item['reason']}")
    if len(result["skipped"]) > 10:
        print(f"[SKIP] ... 另有 {len(result['skipped']) - 10} 个")
    print(f"已重建 {counts['updated']} 条")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
