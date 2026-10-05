"""pipeline_progress 单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from annotation.pipeline_jobs import write_queue
from annotation.pipeline_progress import (
    collect_queue_stats,
    format_duration,
    progress_bar,
    render_progress_line,
)


class PipelineProgressTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(125), "2m05s")
        self.assertEqual(format_duration(3665), "1h01m05s")
        self.assertEqual(format_duration(None), "-")

    def test_progress_bar_and_render(self) -> None:
        bar = progress_bar(2, 5, width=10)
        self.assertEqual(bar, "[####------]")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations" / "auto_jobs.jsonl"
            write_queue(
                path,
                [
                    {"episode": "ep1", "status": "done", "total_elapsed_s": 120},
                    {"episode": "ep2", "status": "segmenting"},
                    {"episode": "ep3", "status": "pending"},
                ],
            )
            stats = collect_queue_stats(path)
            self.assertEqual(stats.total, 3)
            self.assertEqual(stats.done, 1)
            self.assertEqual(stats.segmenting, 1)
            line = render_progress_line(stats, elapsed_s=300.0)
            self.assertIn("33%", line)
            self.assertIn("完成 1/3", line)
            self.assertIn("sam 1/3", line)
            self.assertIn("sam运行:1", line)
            self.assertIn("ETA", line)


if __name__ == "__main__":
    unittest.main()
