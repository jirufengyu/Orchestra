#!/usr/bin/env python3
"""pipeline_jobs 单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from annotation.pipeline_jobs import (
    STATUS_PENDING,
    STATUS_POINTED,
    claim_molmo_job,
    claim_sam_job,
    queue_path,
    read_queue,
    update_job,
    write_queue,
)


class PipelineJobsTest(unittest.TestCase):
    def test_molmo_then_sam_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = queue_path(root)
            write_queue(
                path,
                [
                    {
                        "episode": "episode_0001",
                        "status": STATUS_PENDING,
                        "instance_ids": [1, 2],
                    }
                ],
            )
            molmo_job = claim_molmo_job(path, "molmo-0")
            assert molmo_job is not None
            update_job(
                path,
                "episode_0001",
                status=STATUS_POINTED,
                sam_prompts=[{"instance_id": 1, "points": [[1, 2]], "labels": [1]}],
            )
            sam_job = claim_sam_job(path, "sam-0")
            self.assertIsNotNone(sam_job)
            self.assertEqual(sam_job["episode"], "episode_0001")
            self.assertEqual(len(sam_job["sam_prompts"]), 1)


if __name__ == "__main__":
    unittest.main()
