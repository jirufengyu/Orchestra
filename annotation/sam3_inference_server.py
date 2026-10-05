#!/usr/bin/env python3
"""SAM3.1 multiplex 常驻推理 server。

用法::

    python -m annotation.sam3_inference_server \
      --checkpoint /path/to/sam/sam3.1_multiplex.pt \
      --host 127.0.0.1 --port 8765 --warm-up

标注 Web 通过 ``--sam-server-url http://127.0.0.1:8765`` 连接本服务。
"""

from __future__ import annotations

import argparse

from flask import Flask, jsonify, request

try:
    from .mobile_sam3_backend import Sam3MultiplexEngine, SamBackendError
except ImportError:
    from mobile_sam3_backend import Sam3MultiplexEngine, SamBackendError


def create_app(engine: Sam3MultiplexEngine) -> Flask:
    app = Flask(__name__)
    app.config["ENGINE"] = engine

    @app.errorhandler(SamBackendError)
    def handle_backend_error(exc: SamBackendError):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.errorhandler(Exception)
    def handle_exception(exc: Exception):
        app.logger.exception("request failed")
        return jsonify({"ok": False, "error": f"服务错误: {type(exc).__name__}: {exc}"}), 500

    @app.get("/api/status")
    def api_status():
        return jsonify({"ok": True, "sam3": engine.status()})

    @app.post("/api/session/start")
    def api_session_start():
        body = request.get_json(silent=True) or {}
        episode_dir = body.get("episode_dir", "")
        camera = body.get("camera", "color_0")
        if not episode_dir:
            raise SamBackendError("episode_dir 必填")
        payload = engine.start_episode_session(episode_dir, camera)
        return jsonify({"ok": True, **payload})

    @app.post("/api/session/close")
    def api_session_close():
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id", "")
        if not session_id:
            raise SamBackendError("session_id 必填")
        engine.close_session(session_id)
        return jsonify({"ok": True})

    @app.post("/api/predict")
    def api_predict():
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id", "")
        frame_idx = body.get("frame")
        if not session_id:
            raise SamBackendError("session_id 必填")
        if not isinstance(frame_idx, int):
            raise SamBackendError("frame 必须是整数")
        candidates = engine.predict(
            session_id,
            frame_idx,
            body.get("points") or [],
            body.get("labels") or [],
            body.get("box"),
        )
        return jsonify({"ok": True, "candidates": candidates})

    @app.post("/api/propagate")
    def api_propagate():
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id", "")
        frame_idx = body.get("frame")
        obj_id = body.get("obj_id")
        direction = body.get("direction", "both")
        if not session_id:
            raise SamBackendError("session_id 必填")
        if not isinstance(frame_idx, int) or not isinstance(obj_id, int):
            raise SamBackendError("frame 和 obj_id 必须是整数")
        frames = engine.propagate(session_id, frame_idx, obj_id, direction)
        return jsonify({"ok": True, "frames": frames})

    @app.post("/api/segment_episode")
    def api_segment_episode():
        body = request.get_json(silent=True) or {}
        episode_dir = body.get("episode_dir", "")
        camera = body.get("camera", "color_0")
        seed_frame = body.get("seed_frame", 0)
        prompts = body.get("prompts") or []
        direction = body.get("direction", "both")
        if not episode_dir:
            raise SamBackendError("episode_dir 必填")
        if not isinstance(seed_frame, int):
            raise SamBackendError("seed_frame 必须是整数")
        if not isinstance(prompts, list) or not prompts:
            raise SamBackendError("prompts 必须是非空列表")
        result = engine.segment_episode(episode_dir, camera, seed_frame, prompts, direction)
        return jsonify({"ok": True, **result})

    @app.post("/api/online/session/start")
    def api_online_session_start():
        body = request.get_json(silent=True) or {}
        payload = engine.start_online_session(
            camera=str(body.get("camera", "color_0")),
            max_frames=int(body.get("max_frames", 10_000)),
        )
        return jsonify({"ok": True, **payload})

    @app.post("/api/online/frame")
    def api_online_frame():
        body = request.get_json(silent=True) or {}
        session_id = str(body.get("online_session_id", ""))
        frame_idx = body.get("frame")
        image_base64 = str(body.get("image_base64", ""))
        if not session_id:
            raise SamBackendError("online_session_id 必填")
        if not isinstance(frame_idx, int):
            raise SamBackendError("frame 必须是整数")
        if not image_base64:
            raise SamBackendError("image_base64 必填")
        result = engine.append_online_frame(
            session_id,
            frame_idx=frame_idx,
            image_base64=image_base64,
            prompts=body.get("prompts"),
        )
        return jsonify({"ok": True, **result})

    @app.post("/api/online/session/close")
    def api_online_session_close():
        body = request.get_json(silent=True) or {}
        session_id = str(body.get("online_session_id", ""))
        if not session_id:
            raise SamBackendError("online_session_id 必填")
        engine.close_online_session(session_id)
        return jsonify({"ok": True})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3.1 multiplex 常驻推理 server")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="SAM3.1 multiplex checkpoint 路径",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--warm-up",
        action="store_true",
        help="启动时立即加载模型（推荐）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = Sam3MultiplexEngine(checkpoint=args.checkpoint, device=args.device)
    if args.warm_up:
        print("正在加载 SAM3.1 模型…")
        engine.warm_up()
        print("SAM3.1 模型已就绪")
    app = create_app(engine)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
