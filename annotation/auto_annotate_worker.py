#!/usr/bin/env python3
"""离线批量自动标注 worker（Molmo 生产者 ↔ SAM 消费者 异步流水线）。

流水线分两阶段，可分别启动多个 worker 进程，Molmo 与 SAM 并行处理不同 episode：

::

    # 1. 入队
    python -m annotation.auto_annotate_worker \\
      --data-root /path/to/mobile_data/place_fruit_bowl --enqueue

    # 2a. Molmo 生产者（可开多个，连不同 GPU）
    python -m annotation.auto_annotate_worker \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --role molmo --molmo-server-urls http://127.0.0.1:8766 --worker-id molmo-0

    python -m annotation.auto_annotate_worker \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --role molmo --molmo-server-urls http://127.0.0.1:8767 --worker-id molmo-1

    # 2b. SAM 消费者（可开多个，连不同 GPU）
    python -m annotation.auto_annotate_worker \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --role sam --sam-server-urls http://127.0.0.1:8765 --worker-id sam-0

    # 单进程串行（兼容旧用法）
    python -m annotation.auto_annotate_worker \\
      --data-root /path/to/mobile_data/place_fruit_bowl \\
      --role all --molmo-server-urls http://127.0.0.1:8766 \\
      --sam-server-urls http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

try:
    from .annotate_mobile_masks_web import AnnotationStore
    from .annotation_pipeline import (
        Sam3Pool,
        annotate_episode_pipeline,
        run_molmo_stage,
        run_sam_stage,
    )
    from .molmopoint_backend import MolmoPointRemoteClient, MolmoPointPool
    from .mobile_sam3_backend import Sam3RemoteClient
    from .pipeline_jobs import (
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_POINTED,
        claim_job,
        claim_molmo_job,
        claim_sam_job,
        normalize_episode_name,
        queue_path,
        read_queue,
        update_job,
        utc_now,
        write_queue,
    )
    from .pipeline_progress import JobTimer, elapsed_since_iso, format_duration
    from .prompt_templates import (
        build_per_instance_point_prompt_summary,
        split_point_prompt_summary,
    )
except ImportError:
    from annotate_mobile_masks_web import AnnotationStore
    from annotation_pipeline import (
        Sam3Pool,
        annotate_episode_pipeline,
        run_molmo_stage,
        run_sam_stage,
    )
    from molmopoint_backend import MolmoPointRemoteClient, MolmoPointPool
    from mobile_sam3_backend import Sam3RemoteClient
    from pipeline_jobs import (
        STATUS_DONE,
        STATUS_FAILED,
        STATUS_PENDING,
        STATUS_POINTED,
        claim_job,
        claim_molmo_job,
        claim_sam_job,
        normalize_episode_name,
        queue_path,
        read_queue,
        update_job,
        utc_now,
        write_queue,
    )
    from pipeline_progress import JobTimer, elapsed_since_iso, format_duration
    from prompt_templates import (
        build_per_instance_point_prompt_summary,
        split_point_prompt_summary,
    )


def enqueue_episodes(
    store: AnnotationStore,
    *,
    force: bool = False,
    episodes: list[str] | None = None,
) -> int:
    path = queue_path(store.root)
    existing = {job["episode"]: job for job in read_queue(path)}
    wanted = None
    if episodes:
        wanted = {normalize_episode_name(name) for name in episodes if str(name).strip()}
    added = 0
    seen: set[str] = set()
    for item in store.list_episodes():
        if "error" in item:
            continue
        episode = item["name"]
        if wanted is not None and episode not in wanted:
            continue
        seen.add(episode)
        labels = store.get_episode_labels(episode)
        if not labels:
            continue
        current = existing.get(episode)
        if current and current.get("status") not in {STATUS_FAILED, STATUS_DONE} and not force:
            if current.get("status") in {"pending", "pointing", "pointed", "segmenting"}:
                continue
        if current and current.get("status") == STATUS_DONE and not force:
            continue
        record = store.get_episode_label_record(episode)
        existing[episode] = {
            "episode": episode,
            "instance_ids": labels,
            "status": "pending",
            "stage": "molmo",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "worker_id": None,
            "error": None,
        }
        if record.get("point_prompt"):
            existing[episode]["point_prompt"] = record["point_prompt"]
        if record.get("instruction"):
            existing[episode]["instruction"] = record["instruction"]
        added += 1
    if wanted is not None:
        missing = sorted(wanted - seen)
        if missing:
            raise ValueError("这些 episode 不存在: " + ", ".join(missing))
    write_queue(path, sorted(existing.values(), key=lambda job: job["episode"]))
    return added


def build_pointing_client(urls: str, timeout: float) -> MolmoPointRemoteClient | MolmoPointPool:
    parts = [part.strip() for part in urls.split(",") if part.strip()]
    if len(parts) == 1:
        return MolmoPointRemoteClient(parts[0], timeout=timeout)
    return MolmoPointPool(parts, timeout=timeout)


def build_sam_client(urls: str, timeout: float) -> Sam3RemoteClient | Sam3Pool:
    parts = [part.strip() for part in urls.split(",") if part.strip()]
    if len(parts) == 1:
        return Sam3RemoteClient(parts[0], timeout=timeout)
    return Sam3Pool(parts, timeout=timeout)


def resolve_job_context(
    store: AnnotationStore,
    job: dict[str, Any],
    *,
    camera: str,
    seed_frame: int | None,
) -> dict[str, Any]:
    episode = job["episode"]
    record = store.get_episode_label_record(episode)
    instance_ids = [int(value) for value in job.get("instance_ids") or record.get("instance_ids") or []]
    point_prompt = str(job.get("point_prompt") or record.get("point_prompt") or "")
    instruction = str(job.get("instruction") or record.get("instruction") or "")
    if not instance_ids:
        raise ValueError(f"episode {episode} 未配置标签")
    info = store.episode_info(episode)
    if seed_frame is None:
        seed_frame = info["frame_indices"][0]
    instances = {item["id"]: item for item in info["instances"]}
    point_parts = split_point_prompt_summary(point_prompt) if point_prompt else []
    instance_prompts = []
    for index, instance_id in enumerate(instance_ids):
        inst = instances.get(instance_id)
        if inst is None:
            raise ValueError(f"episode {episode} 引用了不存在的实例 {instance_id}")
        item = {"instance_id": instance_id, "label": inst["name"]}
        if len(point_parts) == len(instance_ids):
            item["prompt"] = point_parts[index]
        instance_prompts.append(item)
    if not point_prompt:
        point_prompt = build_per_instance_point_prompt_summary(
            [str(item["label"]) for item in instance_prompts]
        )
    image_path = store.image_path(episode, seed_frame, camera)
    meta = store.load_metadata(episode)
    shape = (int(meta["image_size"]["height"]), int(meta["image_size"]["width"]))
    return {
        "episode": episode,
        "instruction": instruction,
        "point_prompt": point_prompt,
        "instance_prompts": instance_prompts,
        "episode_dir": store.episode_dir(episode),
        "seed_frame": seed_frame,
        "image_path": image_path,
        "image_shape": shape,
    }


def process_molmo_job(
    store: AnnotationStore,
    job: dict[str, Any],
    *,
    pointing_client: MolmoPointRemoteClient | MolmoPointPool,
    camera: str,
    seed_frame: int | None,
) -> dict[str, Any]:
    ctx = resolve_job_context(store, job, camera=camera, seed_frame=seed_frame)
    result = run_molmo_stage(
        pointing_client,
        seed_frame_image=ctx["image_path"],
        instance_prompts=ctx["instance_prompts"],
        point_prompt=ctx["point_prompt"],
    )
    return {
        "episode": ctx["episode"],
        "instruction": ctx["instruction"],
        "point_prompt": result["point_prompt"],
        "seed_frame": ctx["seed_frame"],
        "camera": camera,
        "sam_prompts": result["sam_prompts"],
        "warnings": result.get("warnings") or [],
        "molmo_points": result["molmo_result"].get("points") or [],
    }


def process_sam_job(
    store: AnnotationStore,
    job: dict[str, Any],
    *,
    sam_client: Sam3RemoteClient | Sam3Pool,
    camera: str,
    direction: str,
) -> dict[str, Any]:
    episode = job["episode"]
    sam_prompts = job.get("sam_prompts") or []
    if not sam_prompts:
        raise ValueError(f"episode {episode} 缺少 sam_prompts，需先由 Molmo worker 打点")
    seed_frame = int(job.get("seed_frame", 0))
    ctx = resolve_job_context(store, job, camera=job.get("camera") or camera, seed_frame=seed_frame)
    result = run_sam_stage(
        sam_client,
        episode_dir=ctx["episode_dir"],
        episode_name=episode,
        camera=job.get("camera") or camera,
        seed_frame_idx=seed_frame,
        sam_prompts=sam_prompts,
        image_shape=ctx["image_shape"],
        direction=direction,
    )
    saved = store.save_episode_label_maps(episode, result["frame_maps"], job.get("camera") or camera)
    return {
        "episode": episode,
        "saved_frames": saved,
        "warnings": job.get("warnings") or [],
        "sam_prompts": sam_prompts,
    }


def process_job_sync(
    store: AnnotationStore,
    job: dict[str, Any],
    *,
    pointing_client: MolmoPointRemoteClient | MolmoPointPool,
    sam_client: Sam3RemoteClient | Sam3Pool,
    camera: str,
    direction: str,
    seed_frame: int | None,
) -> dict[str, Any]:
    ctx = resolve_job_context(store, job, camera=camera, seed_frame=seed_frame)
    result = annotate_episode_pipeline(
        episode_dir=ctx["episode_dir"],
        episode_name=ctx["episode"],
        camera=camera,
        seed_frame_image=ctx["image_path"],
        seed_frame_idx=ctx["seed_frame"],
        instance_prompts=ctx["instance_prompts"],
        pointing_client=pointing_client,
        sam_client=sam_client,
        image_shape=ctx["image_shape"],
        direction=direction,
        point_prompt=ctx["point_prompt"],
    )
    saved = store.save_episode_label_maps(ctx["episode"], result["frame_maps"], camera)
    return {
        "episode": ctx["episode"],
        "instruction": ctx["instruction"],
        "point_prompt": ctx["point_prompt"],
        "saved_frames": saved,
        "warnings": result.get("warnings") or [],
        "sam_prompts": result.get("sam_prompts") or [],
    }


def run_molmo_worker(
    store: AnnotationStore,
    path: Path,
    *,
    worker_id: str,
    pointing_client: MolmoPointRemoteClient | MolmoPointPool,
    camera: str,
    seed_frame: int | None,
    poll_interval: float,
    once: bool,
) -> None:
    while True:
        job = claim_molmo_job(path, worker_id)
        if job is None:
            if once:
                print(f"[{worker_id}] 没有 pending Molmo job")
                return
            time.sleep(poll_interval)
            continue
        episode = job["episode"]
        print(f"[{worker_id}] Molmo 打点 {episode}")
        if not job.get("pipeline_started_at"):
            update_job(path, episode, pipeline_started_at=utc_now())
        timer = JobTimer()
        timer.start_molmo()
        try:
            result = process_molmo_job(
                store,
                job,
                pointing_client=pointing_client,
                camera=camera,
                seed_frame=seed_frame,
            )
            molmo_elapsed = round(timer.finish_molmo(), 2)
            job_fields = {key: value for key, value in result.items() if key != "episode"}
            update_job(
                path,
                episode,
                status=STATUS_POINTED,
                stage="sam",
                worker_id=worker_id,
                error=None,
                molmo_elapsed_s=molmo_elapsed,
                **job_fields,
            )
            print(
                f"[{worker_id}] Molmo 完成 {episode} ({format_duration(molmo_elapsed)}) -> 等待 SAM"
            )
            if result.get("warnings"):
                print("  警告:", "; ".join(result["warnings"]))
        except Exception as exc:
            update_job(path, episode, status=STATUS_FAILED, stage="failed", error=str(exc))
            print(f"[{worker_id}] Molmo 失败 {episode}: {exc}")
        if once:
            return


def run_sam_worker(
    store: AnnotationStore,
    path: Path,
    *,
    worker_id: str,
    sam_client: Sam3RemoteClient | Sam3Pool,
    camera: str,
    direction: str,
    poll_interval: float,
    once: bool,
) -> None:
    while True:
        job = claim_sam_job(path, worker_id)
        if job is None:
            if once:
                print(f"[{worker_id}] 没有 pointed SAM job")
                return
            time.sleep(poll_interval)
            continue
        episode = job["episode"]
        print(f"[{worker_id}] SAM 分割 {episode}")
        timer = JobTimer()
        timer.start_sam()
        try:
            result = process_sam_job(
                store,
                job,
                sam_client=sam_client,
                camera=camera,
                direction=direction,
            )
            sam_elapsed = round(timer.finish_sam(), 2)
            total_elapsed = elapsed_since_iso(job.get("pipeline_started_at"))
            if total_elapsed is None:
                molmo_part = float(job.get("molmo_elapsed_s") or 0)
                total_elapsed = molmo_part + sam_elapsed
            else:
                total_elapsed = round(total_elapsed, 2)
            update_job(
                path,
                episode,
                status=STATUS_DONE,
                stage="done",
                worker_id=worker_id,
                error=None,
                sam_elapsed_s=sam_elapsed,
                total_elapsed_s=total_elapsed,
                result=result,
            )
            molmo_elapsed = job.get("molmo_elapsed_s")
            timing = f"sam {format_duration(sam_elapsed)}"
            if molmo_elapsed is not None:
                timing = f"molmo {format_duration(float(molmo_elapsed))}, {timing}"
            print(
                f"[{worker_id}] SAM 完成 {episode}: 保存 {result['saved_frames']} 帧 "
                f"| {timing}, total {format_duration(total_elapsed)}"
            )
        except Exception as exc:
            update_job(path, episode, status=STATUS_FAILED, stage="failed", error=str(exc))
            print(f"[{worker_id}] SAM 失败 {episode}: {exc}")
        if once:
            return


def run_all_worker(
    store: AnnotationStore,
    path: Path,
    *,
    worker_id: str,
    pointing_client: MolmoPointRemoteClient | MolmoPointPool,
    sam_client: Sam3RemoteClient | Sam3Pool,
    camera: str,
    direction: str,
    seed_frame: int | None,
    poll_interval: float,
    once: bool,
) -> None:
    while True:
        job = claim_job(path, worker_id, STATUS_PENDING, "running")
        if job is None:
            if once:
                print(f"[{worker_id}] 没有 pending job")
                return
            time.sleep(poll_interval)
            continue
        episode = job["episode"]
        print(f"[{worker_id}] 串行处理 {episode}")
        if not job.get("pipeline_started_at"):
            update_job(path, episode, pipeline_started_at=utc_now())
        timer = JobTimer()
        try:
            result = process_job_sync(
                store,
                job,
                pointing_client=pointing_client,
                sam_client=sam_client,
                camera=camera,
                direction=direction,
                seed_frame=seed_frame,
            )
            total_elapsed = round(timer.total_elapsed(), 2)
            update_job(
                path,
                episode,
                status=STATUS_DONE,
                stage="done",
                error=None,
                total_elapsed_s=total_elapsed,
                result=result,
            )
            print(
                f"[{worker_id}] 完成 {episode}: 保存 {result['saved_frames']} 帧 "
                f"| total {format_duration(total_elapsed)}"
            )
        except Exception as exc:
            update_job(path, episode, status=STATUS_FAILED, stage="failed", error=str(exc))
            print(f"[{worker_id}] 失败 {episode}: {exc}")
        if once:
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolmoPoint + SAM3 异步流水线 worker")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--camera", default="color_0")
    parser.add_argument("--enqueue", action="store_true", help="把已配置标签的 episode 写入队列")
    parser.add_argument("--enqueue-only", action="store_true", help="只入队，不启动 worker")
    parser.add_argument("--force", action="store_true", help="enqueue 时覆盖 done 状态")
    parser.add_argument(
        "--episodes",
        default="",
        help="逗号分隔 episode 名或编号，仅入队这些；如 4,7,11 或 episode_0004,episode_0007",
    )
    parser.add_argument(
        "--role",
        choices=["molmo", "sam", "all"],
        default="all",
        help="molmo=仅打点生产者, sam=仅分割消费者, all=单进程串行",
    )
    parser.add_argument("--worker-id", default=f"worker-{os.getpid()}")
    parser.add_argument("--molmo-server-urls", default="http://127.0.0.1:8766")
    parser.add_argument("--sam-server-urls", default="http://127.0.0.1:8765")
    parser.add_argument("--direction", default="both", choices=["forward", "backward", "both"])
    parser.add_argument("--seed-frame", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="只处理一个 job 后退出")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AnnotationStore(args.data_root, args.camera)
    path = queue_path(store.root)

    if args.enqueue:
        selected = [part.strip() for part in args.episodes.split(",") if part.strip()] or None
        count = enqueue_episodes(store, force=args.force, episodes=selected)
        print(f"已写入/更新 {count} 个待处理 job -> {path}")
        if args.enqueue_only:
            return

    if args.role == "molmo":
        client = build_pointing_client(args.molmo_server_urls, args.timeout)
        run_molmo_worker(
            store,
            path,
            worker_id=args.worker_id,
            pointing_client=client,
            camera=args.camera,
            seed_frame=args.seed_frame,
            poll_interval=args.poll_interval,
            once=args.once,
        )
        return

    if args.role == "sam":
        client = build_sam_client(args.sam_server_urls, max(args.timeout, 3600.0))
        run_sam_worker(
            store,
            path,
            worker_id=args.worker_id,
            sam_client=client,
            camera=args.camera,
            direction=args.direction,
            poll_interval=args.poll_interval,
            once=args.once,
        )
        return

    pointing_client = build_pointing_client(args.molmo_server_urls, args.timeout)
    sam_client = build_sam_client(args.sam_server_urls, max(args.timeout, 3600.0))
    run_all_worker(
        store,
        path,
        worker_id=args.worker_id,
        pointing_client=pointing_client,
        sam_client=sam_client,
        camera=args.camera,
        direction=args.direction,
        seed_frame=args.seed_frame,
        poll_interval=args.poll_interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()
