#!/usr/bin/env python3
"""MolmoPoint-8B 常驻推理 server。

用法::

    python -m annotation.molmopoint_inference_server \\
      --checkpoint /path/to/MolmoPoint-8B \\
      --host 127.0.0.1 --port 8766 --warm-up
"""

from __future__ import annotations

import argparse
import base64
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

try:
    from .molmopoint_backend import MolmoPointBackendError, MolmoPointEngine
except ImportError:
    from molmopoint_backend import MolmoPointBackendError, MolmoPointEngine


def create_app(engine: MolmoPointEngine) -> Flask:
    app = Flask(__name__)
    app.config["ENGINE"] = engine

    @app.errorhandler(MolmoPointBackendError)
    def handle_backend_error(exc: MolmoPointBackendError):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        app.logger.exception("request failed")
        return jsonify({"ok": False, "error": f"服务错误: {type(exc).__name__}: {exc}"}), 500

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "molmopoint": engine.status()})

    @app.post("/api/point_image")
    def api_point_image():
        body = request.get_json(silent=True) or {}
        image_path = body.get("image_path", "")
        image_base64 = body.get("image_base64", "")
        prompt = body.get("prompt", "")
        max_new_tokens = int(body.get("max_new_tokens", 200))
        if not image_path and not image_base64:
            raise MolmoPointBackendError("image_path 或 image_base64 必填")
        if not prompt:
            raise MolmoPointBackendError("prompt 必填")
        temp_path = None
        if image_base64:
            try:
                raw = base64.b64decode(str(image_base64), validate=True)
            except Exception as exc:
                raise MolmoPointBackendError(f"image_base64 无效: {exc}") from exc
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as stream:
                stream.write(raw)
                temp_path = Path(stream.name)
            image_path = str(temp_path)
        try:
            result = engine.point_image(image_path, prompt, max_new_tokens=max_new_tokens)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return jsonify({"ok": True, "result": result})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MolmoPoint-8B 常驻推理 server")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="MolmoPoint HF checkpoint 目录",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--warm-up", action="store_true", help="启动时立即加载模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = MolmoPointEngine(checkpoint=args.checkpoint, device=args.device)
    if args.warm_up:
        print("正在加载 MolmoPoint 模型…")
        engine.warm_up()
        print("MolmoPoint 模型已就绪")
    app = create_app(engine)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
