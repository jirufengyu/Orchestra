"""异步流水线任务队列：Molmo 打点 → SAM 分割。"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUEUE_FILENAME = "auto_jobs.jsonl"

STATUS_PENDING = "pending"
STATUS_POINTING = "pointing"
STATUS_POINTED = "pointed"
STATUS_SEGMENTING = "segmenting"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_episode_name(name: str) -> str:
    text = str(name).strip()
    if text.isdigit():
        return f"episode_{int(text):04d}"
    return text


def queue_path(data_root: Path) -> Path:
    return data_root / "annotations" / QUEUE_FILENAME


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_queue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    jobs: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def write_queue(path: Path, jobs: list[dict[str, Any]]) -> None:
    lines = [json.dumps(job, ensure_ascii=False) for job in jobs]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def update_job(path: Path, episode: str, **fields: Any) -> None:
    jobs = read_queue(path)
    for job in jobs:
        if job.get("episode") == episode:
            job.update(fields)
            job["updated_at"] = utc_now()
            break
    write_queue(path, jobs)


def claim_job(path: Path, worker_id: str, status: str, next_status: str) -> dict[str, Any] | None:
    jobs = read_queue(path)
    changed = False
    selected: dict[str, Any] | None = None
    for job in jobs:
        if job.get("status") == status:
            job["status"] = next_status
            job["worker_id"] = worker_id
            job["updated_at"] = utc_now()
            selected = job
            changed = True
            break
    if changed:
        write_queue(path, jobs)
    return selected


def claim_molmo_job(path: Path, worker_id: str) -> dict[str, Any] | None:
    return claim_job(path, worker_id, STATUS_PENDING, STATUS_POINTING)


def claim_sam_job(path: Path, worker_id: str) -> dict[str, Any] | None:
    return claim_job(path, worker_id, STATUS_POINTED, STATUS_SEGMENTING)


def reset_jobs_for_sam_retry(
    path: Path,
    *,
    episodes: set[str] | None = None,
) -> int:
    """把已有 sam_prompts 的 failed/segmenting job 重置为 pointed，供 SAM worker 重跑。"""
    jobs = read_queue(path)
    changed = 0
    for job in jobs:
        episode = str(job.get("episode", ""))
        if episodes is not None and episode not in episodes:
            continue
        if not job.get("sam_prompts"):
            continue
        status = job.get("status")
        if status not in {STATUS_FAILED, STATUS_SEGMENTING}:
            continue
        job["status"] = STATUS_POINTED
        job["stage"] = "sam"
        job["worker_id"] = None
        job["error"] = None
        job["updated_at"] = utc_now()
        changed += 1
    if changed:
        write_queue(path, jobs)
    return changed
