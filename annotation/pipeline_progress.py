"""标注流水线进度与耗时统计。"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .pipeline_jobs import (
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_POINTED,
        STATUS_POINTING,
        STATUS_SEGMENTING,
        queue_path,
        read_queue,
        utc_now,
    )
except ImportError:
    from pipeline_jobs import (
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_POINTED,
        STATUS_POINTING,
        STATUS_SEGMENTING,
        queue_path,
        read_queue,
        utc_now,
    )


def elapsed_since_iso(iso: str | None) -> float | None:
    if not iso:
        return None
    text = iso.replace("Z", "+00:00")
    start = datetime.fromisoformat(text)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - start).total_seconds()


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "-"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def progress_bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("?" * width) + "]"
    filled = min(width, int(width * done / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


@dataclass
class QueueStats:
    total: int = 0
    pending: int = 0
    pointing: int = 0
    pointed: int = 0
    segmenting: int = 0
    done: int = 0
    failed: int = 0
    jobs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def finished(self) -> int:
        return self.done + self.failed

    @property
    def in_progress(self) -> int:
        return self.pointing + self.segmenting


def collect_queue_stats(
    path: Path,
    *,
    episodes: set[str] | None = None,
) -> QueueStats:
    jobs = read_queue(path)
    if episodes is not None:
        jobs = [job for job in jobs if job.get("episode") in episodes]
    stats = QueueStats(jobs=jobs, total=len(jobs))
    for job in jobs:
        status = job.get("status", "")
        if status == STATUS_PENDING:
            stats.pending += 1
        elif status == STATUS_POINTING:
            stats.pointing += 1
        elif status == STATUS_POINTED:
            stats.pointed += 1
        elif status == STATUS_SEGMENTING:
            stats.segmenting += 1
        elif status == STATUS_DONE:
            stats.done += 1
        elif status == STATUS_FAILED:
            stats.failed += 1
    return stats


def render_progress_line(
    stats: QueueStats,
    *,
    elapsed_s: float,
    prefix: str = "",
) -> str:
    total = stats.total
    done = stats.done
    pct = int(100 * done / total) if total else 0
    bar = progress_bar(done, total)
    molmo_done = stats.pointed + stats.segmenting + stats.done
    sam_done = stats.done
    parts = [
        f"{prefix}{bar} {pct:3d}%",
        f"完成 {done}/{total}",
        f"molmo {molmo_done}/{total}",
        f"sam {sam_done}/{total}",
    ]
    active = []
    if stats.pending:
        active.append(f"molmo排队:{stats.pending}")
    if stats.pointing:
        active.append(f"molmo运行:{stats.pointing}")
    if stats.pointed:
        active.append(f"sam排队:{stats.pointed}")
    if stats.segmenting:
        active.append(f"sam运行:{stats.segmenting}")
    if stats.failed:
        active.append(f"失败:{stats.failed}")
    if active:
        parts.append(" ".join(active))
    parts.append(f"elapsed {format_duration(elapsed_s)}")
    if done > 0 and total > done:
        eta = elapsed_s / done * (total - done)
        avg = elapsed_s / done
        parts.append(f"ETA ~{format_duration(eta)}")
        parts.append(f"avg {format_duration(avg)}/ep")
    return " | ".join(parts)


def format_job_stage(status: str) -> str:
    if status == STATUS_PENDING:
        return "molmo排队"
    if status == STATUS_POINTING:
        return "molmo运行"
    if status == STATUS_POINTED:
        return "sam排队"
    if status == STATUS_SEGMENTING:
        return "sam运行"
    if status == STATUS_DONE:
        return "完成"
    if status == STATUS_FAILED:
        return "失败"
    return status


def format_job_timing(job: dict[str, Any]) -> str:
    episode = job.get("episode", "?")
    status = job.get("status", "?")
    molmo = job.get("molmo_elapsed_s")
    sam = job.get("sam_elapsed_s")
    total = job.get("total_elapsed_s")
    chunks = [f"{episode}: {format_job_stage(str(status))}"]
    if molmo is not None:
        chunks.append(f"molmo={format_duration(float(molmo))}")
    if sam is not None:
        chunks.append(f"sam={format_duration(float(sam))}")
    if total is not None:
        chunks.append(f"total={format_duration(float(total))}")
    if status == STATUS_FAILED and job.get("error"):
        chunks.append(f"err={str(job['error'])[:80]}")
    return "  " + " | ".join(chunks)


def print_timing_summary(jobs: list[dict[str, Any]]) -> None:
    completed = [job for job in jobs if job.get("status") in {STATUS_DONE, STATUS_FAILED}]
    if not completed:
        return
    print("\n========== 耗时统计 ==========")
    for job in sorted(completed, key=lambda item: str(item.get("episode", ""))):
        print(format_job_timing(job))
    totals = [float(job["total_elapsed_s"]) for job in completed if job.get("total_elapsed_s") is not None]
    if totals:
        print(
            f"  合计 {len(totals)} 个 episode, "
            f"总耗时 {format_duration(sum(totals))}, "
            f"平均 {format_duration(sum(totals) / len(totals))}/ep"
        )


class JobTimer:
    """记录单个 job 各阶段耗时。"""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._molmo_started: float | None = None
        self._sam_started: float | None = None

    def start_molmo(self) -> None:
        self._molmo_started = time.perf_counter()

    def finish_molmo(self) -> float:
        if self._molmo_started is None:
            self.start_molmo()
        return time.perf_counter() - self._molmo_started

    def start_sam(self) -> None:
        self._sam_started = time.perf_counter()

    def finish_sam(self) -> float:
        if self._sam_started is None:
            self.start_sam()
        return time.perf_counter() - self._sam_started

    def total_elapsed(self) -> float:
        return time.perf_counter() - self._started


def wait_for_queue(
    data_root: str | Path,
    *,
    episodes: set[str] | None = None,
    poll_interval: float = 5.0,
    timeout: float | None = None,
    stream: Any = None,
) -> QueueStats:
    """阻塞等待队列完成，周期性打印进度条。"""
    path = queue_path(Path(data_root))
    stream = stream or sys.stdout
    started = time.perf_counter()
    last_line = ""
    while True:
        stats = collect_queue_stats(path, episodes=episodes)
        elapsed = time.perf_counter() - started
        line = render_progress_line(stats, elapsed_s=elapsed)
        if line != last_line:
            print(f"\r{line}", end="", file=stream, flush=True)
            last_line = line
        if stats.failed:
            print(file=stream)
            failed = [job for job in stats.jobs if job.get("status") == STATUS_FAILED]
            for job in failed:
                print(f"失败 {job.get('episode')}: {job.get('error')}", file=stream)
            stuck = [
                job
                for job in stats.jobs
                if job.get("status") in {STATUS_POINTED, STATUS_SEGMENTING}
            ]
            if stuck:
                print(
                    f"另有 {len(stuck)} 个 episode 未完成（sam排队/运行中），"
                    "可用 --force 重新入队后只跑 SAM worker",
                    file=stream,
                )
            raise RuntimeError(f"{stats.failed} 个 episode 标注失败")
        if stats.total > 0 and stats.done >= stats.total:
            print(file=stream)
            print_timing_summary(stats.jobs)
            return stats
        if timeout is not None and elapsed >= timeout:
            print(file=stream)
            raise TimeoutError(f"等待超时 ({format_duration(timeout)})")
        time.sleep(poll_interval)
