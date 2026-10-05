"""Progress / subtask-done label utilities for RMBench Mn and M1 tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Global instructions for RMBench M(1) tasks (from Mem-0 M1_dataset_to_lerobot.py).
RMBENCH_M1_GLOBAL_TASKS: dict[str, str] = {
    "observe_and_pickup": (
        "Initially, there is one target object on the shelf and five random objects on the table. "
        "Then, a screen obscures the target object. Pick up the corresponding target object "
        "from the table and lift it up."
    ),
    "put_back_block": (
        "There are four mats, one block, and a button on the table. One block is on one of the mats. "
        "First, put the block to the center, then press the button. "
        "Then, put the block back in its original position."
    ),
    "rearrange_blocks": (
        "Move the block between the two mats onto the empty mat, press the button, then move the other "
        "block (the one that started on a mat) to the space between the two mats."
    ),
    "swap_blocks": (
        "There are three traies on the table, and two blocks are placed in two different traies. "
        "You may move only one block at a time, and each tray can hold at most one block. "
        "Swap the positions of the two blocks. Finally press the button."
    ),
    "swap_T": "Swap the poses of the two T-blocks, including both position and orientation.",
}


def resolve_rmbench_m1_global_task(repo_id: str | list[str] | tuple[str, ...]) -> str:
    """Resolve the fixed global instruction for an RMBench M(1) LeRobot repo id."""
    repo = repo_id[0] if isinstance(repo_id, (list, tuple)) else repo_id
    repo_name = str(repo).split("/")[-1]
    for task_key, prompt in RMBENCH_M1_GLOBAL_TASKS.items():
        if task_key in repo_name:
            return prompt
    raise ValueError(
        f"Cannot resolve RMBench M1 global task from repo_id={repo_id!r}. "
        f"Expected one of: {sorted(RMBENCH_M1_GLOBAL_TASKS)}"
    )


def resolve_rmbench_m1_global_tasks_by_dataset(
    repo_ids: list[str] | tuple[str, ...],
) -> dict[int, str]:
    """Resolve fixed M(1) global instructions keyed by MultiLeRobotDataset index."""
    return {dataset_index: resolve_rmbench_m1_global_task(repo_id) for dataset_index, repo_id in enumerate(repo_ids)}


@dataclass(frozen=True)
class SubtaskSegment:
    text: str
    duration: int
    start_frame: int
    end_frame: int  # exclusive


def build_subtask_segments_from_language(
    language_segments: list[dict[str, Any]],
    num_frames: int,
) -> list[SubtaskSegment]:
    """Build frame-aligned subtask segments from actors_meta language_segments."""
    if not language_segments:
        return []
    built: list[SubtaskSegment] = []
    start = 0
    for seg in language_segments:
        duration = int(seg.get("duration", 0))
        text = str(seg.get("text", ""))
        end = min(start + duration, num_frames)
        if end <= start:
            continue
        built.append(SubtaskSegment(text=text, duration=end - start, start_frame=start, end_frame=end))
        start = end
    if start < num_frames and built:
        last = built[-1]
        built[-1] = SubtaskSegment(
            text=last.text,
            duration=last.duration + (num_frames - start),
            start_frame=last.start_frame,
            end_frame=num_frames,
        )
    return built


def compute_progress_value(frame_index: int, subtask_start: int, subtask_end: int) -> float:
    remaining = max(subtask_end - 1 - frame_index, 0)
    duration = max(subtask_end - subtask_start, 1)
    return -float(remaining) / float(duration)


def value_to_bin(value: float, num_bins: int = 101) -> int:
    value = max(-1.0, min(0.0, float(value)))
    idx = int(round((value + 1.0) * (num_bins - 1)))
    return max(0, min(num_bins - 1, idx))


def bin_to_value(bin_index: int, num_bins: int = 101) -> float:
    return -1.0 + bin_index / (num_bins - 1)


def subtask_done_flag(frame_index: int, subtask_end: int, boundary_margin: int = 8) -> bool:
    return frame_index >= subtask_end - boundary_margin


def progress_labels_for_frame(
    frame_index: int,
    num_frames: int,
    language_segments: list[dict[str, Any]],
    *,
    num_progress_bins: int = 101,
    boundary_margin: int = 8,
) -> tuple[int, float, bool, bool]:
    """
    Returns (progress_bin, progress_value, subtask_done, has_progress_label).

    M1 (no / single segment): whole-episode progress.
    Mn: per-subtask progress from language_segments.
    """
    if num_frames <= 0:
        return 0, -1.0, False, False

    segments = build_subtask_segments_from_language(language_segments, num_frames)
    if not segments:
        # M1-style: single episode segment
        progress_value = compute_progress_value(frame_index, 0, num_frames)
        done = subtask_done_flag(frame_index, num_frames, boundary_margin=boundary_margin)
        return (
            value_to_bin(progress_value, num_bins=num_progress_bins),
            progress_value,
            done,
            True,
        )

    subtask_idx = 0
    for i, st in enumerate(segments):
        if st.start_frame <= frame_index < st.end_frame:
            subtask_idx = i
            break
    st = segments[subtask_idx]
    progress_value = compute_progress_value(frame_index, st.start_frame, st.end_frame)
    done = subtask_done_flag(frame_index, st.end_frame, boundary_margin=boundary_margin)
    return (
        value_to_bin(progress_value, num_bins=num_progress_bins),
        progress_value,
        done,
        True,
    )
