#!/usr/bin/env python3
"""为整个任务预构建 SAM3 视频帧目录（离线运行，打开标注 Web 前执行）。

每个 episode 会在 ``{data_root}/annotations/sam_frames/{episode}/{camera}/`` 下生成
连续编号的 JPEG symlink（``00000.jpg`` …）和 ``manifest.json``。SAM server 启动
session 时会优先使用该缓存，避免每次分割时重复扫描/组装。

用法::

    python -m annotation.prepare_sam_videos \\
      --data-root /path/to/mobile_data/place_fruit_bowl

    # 仅处理单个 episode
    python -m annotation.prepare_sam_videos \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --episode episode_0001 --force
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from .mobile_sam3_backend import SamBackendError, prepare_episode_frame_dir
except ImportError:
    from mobile_sam3_backend import SamBackendError, prepare_episode_frame_dir

EPISODE_RE = re.compile(r"^episode_\d+$")
DEFAULT_CAMERA = "color_0"


def list_episode_dirs(data_root: Path) -> list[Path]:
    episodes = []
    for path in sorted(data_root.iterdir()):
        if path.is_dir() and EPISODE_RE.fullmatch(path.name) and (path / "data.json").is_file():
            episodes.append(path)
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为真机任务预构建 SAM3 帧目录")
    parser.add_argument("--data-root", required=True, help="包含 episode_XXXX 的任务根目录")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="默认 color_0（head camera）")
    parser.add_argument("--episode", default="", help="仅处理指定 episode，如 episode_0001")
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有缓存也重建",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"数据根目录不存在: {data_root}")

    if args.episode:
        if not EPISODE_RE.fullmatch(args.episode):
            raise SystemExit("episode 名称无效")
        episodes = [data_root / args.episode]
        if not (episodes[0] / "data.json").is_file():
            raise SystemExit(f"episode 不存在: {episodes[0]}")
    else:
        episodes = list_episode_dirs(data_root)
    if not episodes:
        raise SystemExit(f"未找到 episode: {data_root}")

    built = skipped = failed = 0
    for episode_dir in episodes:
        try:
            result = prepare_episode_frame_dir(
                episode_dir,
                args.camera,
                force=args.force,
            )
        except (SamBackendError, OSError, ValueError) as exc:
            failed += 1
            print(f"[FAIL] {episode_dir.name}: {exc}")
            continue
        if result.get("skipped"):
            skipped += 1
            print(
                f"[SKIP] {episode_dir.name}: {result['frame_count']} 帧 "
                f"-> {result['frame_dir']}"
            )
        else:
            built += 1
            print(
                f"[OK]   {episode_dir.name}: {result['frame_count']} 帧 "
                f"-> {result['frame_dir']}"
            )

    print(
        f"完成: 新建 {built}, 跳过 {skipped}, 失败 {failed}, "
        f"共 {len(episodes)} 个 episode"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
