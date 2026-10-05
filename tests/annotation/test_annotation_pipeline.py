#!/usr/bin/env python3
"""annotation_pipeline 单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path

from annotation.annotation_pipeline import (
    build_sam_prompts_from_instance_molmo,
    molmo_points_to_sam_prompt,
    run_molmo_pointing_per_instance,
    run_molmo_stage,
    tag_molmo_points_with_instance,
)


class _FakePointingClient:
    def __init__(self, points_by_label: dict[str, list[dict]]):
        self._points_by_label = points_by_label
        self.calls: list[str] = []

    def point_image(self, image_path, prompt, *, max_new_tokens=200):
        self.calls.append(prompt)
        label = prompt.replace("Point to the ", "").rstrip(".")
        points = self._points_by_label.get(label, [])
        return {
            "prompt": prompt,
            "image_path": str(image_path),
            "image_size": {"width": 640, "height": 480},
            "generated_text": f"mock for {label}",
            "points": points,
        }


class AnnotationPipelineTest(unittest.TestCase):
    def test_per_instance_molmo_calls_each_label(self):
        client = _FakePointingClient(
            {
                "red apple": [{"object_id": 1, "point": [10.0, 20.0], "x": 10.0, "y": 20.0}],
                "left bowl": [{"object_id": 1, "point": [30.0, 40.0], "x": 30.0, "y": 40.0}],
            }
        )
        instance_prompts = [
            {"instance_id": 1, "label": "red apple"},
            {"instance_id": 4, "label": "left bowl"},
        ]
        result = run_molmo_pointing_per_instance(
            client,
            Path("/tmp/frame.jpg"),
            instance_prompts,
        )
        self.assertEqual(
            client.calls,
            ["Point to the red apple.", "Point to the left bowl."],
        )
        self.assertEqual(len(result["per_instance"]), 2)
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["points"][0]["instance_id"], 1)
        self.assertEqual(result["points"][1]["instance_id"], 4)
        self.assertEqual(
            result["prompt"],
            "Point to the red apple. | Point to the left bowl.",
        )

    def test_molmo_points_to_sam_prompt_multi_point(self):
        points = tag_molmo_points_with_instance(
            [
                {"object_id": 1, "point": [10.0, 20.0]},
                {"object_id": 2, "point": [11.0, 21.0]},
            ],
            1,
        )
        prompt = molmo_points_to_sam_prompt(1, points)
        self.assertEqual(
            prompt,
            {
                "instance_id": 1,
                "points": [[10.0, 20.0], [11.0, 21.0]],
                "labels": [1, 1],
            },
        )

    def test_build_sam_prompts_from_instance_molmo(self):
        per_instance = [
            {
                "instance_id": 1,
                "label": "red apple",
                "points": tag_molmo_points_with_instance(
                    [{"object_id": 1, "point": [10.0, 20.0]}],
                    1,
                ),
            },
            {
                "instance_id": 4,
                "label": "left bowl",
                "points": tag_molmo_points_with_instance(
                    [{"object_id": 1, "point": [30.0, 40.0]}],
                    4,
                ),
            },
        ]
        instance_prompts = [
            {"instance_id": 1, "label": "red apple"},
            {"instance_id": 4, "label": "left bowl"},
        ]
        sam_prompts, warnings = build_sam_prompts_from_instance_molmo(
            instance_prompts,
            per_instance,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(sam_prompts), 2)
        self.assertEqual(sam_prompts[0]["instance_id"], 1)
        self.assertEqual(sam_prompts[0]["points"], [[10.0, 20.0]])
        self.assertEqual(sam_prompts[1]["instance_id"], 4)
        self.assertEqual(sam_prompts[1]["points"], [[30.0, 40.0]])

    def test_run_molmo_stage_raises_when_all_missing(self):
        client = _FakePointingClient({"red apple": [], "left bowl": []})
        instance_prompts = [
            {"instance_id": 1, "label": "red apple"},
            {"instance_id": 4, "label": "left bowl"},
        ]
        with self.assertRaises(ValueError):
            run_molmo_stage(
                client,
                seed_frame_image=Path("/tmp/frame.jpg"),
                instance_prompts=instance_prompts,
            )

    def test_per_instance_molmo_uses_custom_prompt(self):
        client = _FakePointingClient({})
        instance_prompts = [
            {"instance_id": 1, "label": "red apple", "prompt": "Point to the apple on the left."},
            {"instance_id": 4, "label": "left bowl", "prompt": "Point to the bowl in the center."},
        ]
        result = run_molmo_pointing_per_instance(
            client,
            Path("/tmp/frame.jpg"),
            instance_prompts,
        )
        self.assertEqual(
            client.calls,
            ["Point to the apple on the left.", "Point to the bowl in the center."],
        )
        self.assertEqual(
            result["prompt"],
            "Point to the apple on the left. | Point to the bowl in the center.",
        )


if __name__ == "__main__":
    unittest.main()
