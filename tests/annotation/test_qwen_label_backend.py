#!/usr/bin/env python3
"""qwen_label_backend 单元测试。"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from annotation.qwen_label_backend import (
    DASHSCOPE_INTL_BASE_URL,
    DEFAULT_QWEN_MODEL,
    resolve_qwen_config,
)


class QwenConfigTest(unittest.TestCase):
    def test_resolve_from_dashscope_env(self):
        with patch.dict(
            os.environ,
            {"DASHSCOPE_API_KEY": "sk-test", "QWEN_MODEL": "qwen3.7-plus"},
            clear=False,
        ):
            url, model, key = resolve_qwen_config()
        self.assertEqual(url, DASHSCOPE_INTL_BASE_URL)
        self.assertEqual(model, "qwen3.7-plus")
        self.assertEqual(key, "sk-test")

    def test_default_model(self):
        with patch.dict(os.environ, {}, clear=True):
            _, model, _ = resolve_qwen_config(api_url="http://x/v1", api_key="k")
        self.assertEqual(model, DEFAULT_QWEN_MODEL)


if __name__ == "__main__":
    unittest.main()
