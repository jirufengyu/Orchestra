#!/usr/bin/env python3
"""监视标注队列进度（进度条 + 耗时 + ETA）。

用法::

    python -m annotation.watch_annotation_queue \\
      --data-root /path/to/mobile_data/place_fruit_bowl --wait

    python -m annotation.watch_annotation_queue \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --episodes episode_0001,episode_0002 --once
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .pipeline_jobs import normalize_episode_name, queue_path
    from .pipeline_progress import (
        collect_queue_stats,
        format_job_timing,
        print_timing_summary,
        render_progress_line,
        wait_for_queue,
    )
except ImportError:
    from pipeline_jobs import normalize_episode_name, queue_path
    from pipeline_progress import (
        collect_queue_stats,
        format_job_timing,
        print_timing_summary,
        render_progress_line,
        wait_for_queue,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="监视 Molmo+SAM 标注队列进度")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episodes", default="", help="逗号分隔，仅统计这些 episode")
    parser.add_argument("--wait", action="store_true", help="阻塞直到全部完成或失败")
    parser.add_argument("--once", action="store_true", help="只打印一次当前状态后退出")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=None, help="--wait 时最长等待秒数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = {
        normalize_episode_name(item) for item in args.episodes.split(",") if item.strip()
    } or None
    path = queue_path(Path(args.data_root))

    if args.wait:
        try:
            wait_for_queue(
                path.parent.parent if False else Path(args.data_root),
                episodes=episodes,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
        except (RuntimeError, TimeoutError) as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1) from exc
        return

    stats = collect_queue_stats(path, episodes=episodes)
    print(render_progress_line(stats, elapsed_s=0.0))
    for job in sorted(stats.jobs, key=lambda item: str(item.get("episode", ""))):
        print(format_job_timing(job))
    if stats.done and stats.done == stats.total:
        print_timing_summary(stats.jobs)
    elif not args.once:
        print("(使用 --wait 持续监视并显示 ETA)")


if __name__ == "__main__":
    main()
