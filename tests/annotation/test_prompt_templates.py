#!/usr/bin/env python3
"""prompt_templates 单元测试。"""

from __future__ import annotations

import unittest

from annotation.prompt_templates import (
    PromptTemplateError,
    build_combined_point_prompt,
    build_episode_prompts,
    build_instruction,
    build_qwen_episode_label_prompt,
    parse_qwen_label_response,
)


class PromptTemplateTest(unittest.TestCase):
    def test_combined_point_prompt(self):
        self.assertEqual(
            build_combined_point_prompt(["red apple", "left bowl"]),
            "Point to the red apple and left bowl.",
        )
        self.assertEqual(
            build_combined_point_prompt(["red apple", "green pear", "left bowl"]),
            "Point to the red apple, green pear and left bowl.",
        )

    def test_build_instruction(self):
        text = build_instruction("Place {A} on the {B}", ["red apple", "left bowl"])
        self.assertEqual(text, "Place red apple on the left bowl")

    def test_build_episode_prompts(self):
        instances = [
            {"id": 1, "name": "red apple"},
            {"id": 2, "name": "left bowl"},
        ]
        result = build_episode_prompts(
            instruction_template="Place {A} on the {B}",
            instances=instances,
            instance_ids=[1, 2],
        )
        self.assertEqual(result["instruction"], "Place red apple on the left bowl")
        self.assertEqual(
            result["point_prompt"],
            "Point to the red apple. | Point to the left bowl.",
        )

    def test_qwen_prompt_and_parse(self):
        prompt = build_qwen_episode_label_prompt(
            instruction_template="Place {A} on the {B}",
            slot_options={
                "A": ["red apple", "green pear"],
                "B": ["left bowl", "middle bowl"],
            },
        )
        self.assertIn("第一帧和最后一帧", prompt)
        selection = parse_qwen_label_response(
            '```json\n{"A": "green pear", "B": "middle bowl"}\n```',
            slot_options={
                "A": ["red apple", "green pear"],
                "B": ["left bowl", "middle bowl"],
            },
        )
        self.assertEqual(selection["A"], "green pear")
        self.assertEqual(selection["B"], "middle bowl")

    def test_mismatched_placeholder_count(self):
        with self.assertRaises(PromptTemplateError):
            build_instruction("Place {A} on the {B}", ["only one"])

    def test_split_point_prompt_summary(self):
        from annotation.prompt_templates import split_point_prompt_summary

        self.assertEqual(
            split_point_prompt_summary("Point to the red apple. | Point to the left bowl."),
            ["Point to the red apple.", "Point to the left bowl."],
        )


if __name__ == "__main__":
    unittest.main()
