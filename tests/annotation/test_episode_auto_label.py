#!/usr/bin/env python3
"""Qwen 自动选标签单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from annotation.annotate_mobile_masks_web import AnnotationStore
from annotation.episode_auto_label import label_episode_with_qwen
from annotation.prompt_templates import (
    build_qwen_episode_label_prompt,
    parse_qwen_label_response,
    selection_to_instance_ids,
)


class FakeQwenClient:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, list[Path]]] = []

    def chat_with_images(self, prompt: str, image_paths: list[Path]) -> str:
        self.calls.append((prompt, image_paths))
        return self.response


class EpisodeAutoLabelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "episode_0001"
        colors = self.episode / "colors"
        colors.mkdir(parents=True)
        frames = []
        for idx in (0, 3):
            relative = f"colors/{idx:06d}_color_0.jpg"
            Image.new("RGB", (8, 6), (20, idx, 0)).save(self.episode / relative)
            frames.append({"idx": idx, "colors": {"color_0": relative}})
        with open(self.episode / "data.json", "w", encoding="utf-8") as stream:
            json.dump({"text": {"goal": "测试"}, "data": frames}, stream)
        annotations = self.root / "annotations"
        annotations.mkdir()
        annotations.joinpath("instances.json").write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "instances": [
                        {"id": 1, "name": "red apple", "color": "#ff4d4f"},
                        {"id": 2, "name": "left bowl", "color": "#40a9ff"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        annotations.joinpath("episode_labels.json").write_text(
            json.dumps(
                {
                    "version": "1.2",
                    "instruction_template": "Place {A} on the {B}",
                    "label_slots": {"A": [1], "B": [2]},
                    "episodes": {},
                }
            ),
            encoding="utf-8",
        )
        self.store = AnnotationStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_build_and_parse_qwen_prompt(self):
        prompt = build_qwen_episode_label_prompt(
            instruction_template="Place {A} on the {B}",
            slot_options={"A": ["red apple"], "B": ["left bowl"]},
        )
        self.assertIn("第一帧和最后一帧", prompt)
        self.assertIn("red apple", prompt)
        selection = parse_qwen_label_response(
            '{"A": "red apple", "B": "left bowl"}',
            slot_options={"A": ["red apple"], "B": ["left bowl"]},
        )
        ids = selection_to_instance_ids(
            selection,
            label_slots={"A": [1], "B": [2]},
            instances=self.store.load_task_instances(),
        )
        self.assertEqual(ids, [1, 2])

    def test_label_episode_with_qwen(self):
        client = FakeQwenClient('{"A": "red apple", "B": "left bowl"}')
        result = label_episode_with_qwen(self.store, "episode_0001", client)
        self.assertFalse(result["skipped"])
        self.assertEqual(result["instruction"], "Place red apple on the left bowl")
        self.assertEqual(self.store.get_episode_labels("episode_0001"), [1, 2])
        self.assertEqual(len(client.calls[0][1]), 2)


if __name__ == "__main__":
    unittest.main()
