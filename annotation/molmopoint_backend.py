"""MolmoPoint 推理后端：本地引擎与远程 HTTP client。"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class MolmoPointBackendError(ValueError):
    """MolmoPoint 后端输入或推理错误。"""


def _direct_json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """直连本机 HTTP，忽略 http_proxy。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not path.startswith("/"):
        path = "/" + path
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            try:
                parsed_body = json.loads(raw)
                message = parsed_body.get("error", raw)
            except json.JSONDecodeError:
                message = raw or f"HTTP {response.status}"
            raise RuntimeError(str(message))
        return json.loads(raw) if raw else {}
    except (TimeoutError, ConnectionError, OSError) as exc:
        raise RuntimeError(f"无法连接 MolmoPoint server ({base_url}): {exc}") from exc
    finally:
        conn.close()


def build_point_prompt(label: str) -> str:
    """单标签 MolmoPoint prompt；流水线对每个 instance 单独调用时使用。"""
    text = str(label).strip()
    if not text:
        raise MolmoPointBackendError("物体标签不能为空")
    if text.lower().startswith("point to"):
        return text
    return f"Point to the {text}"


def normalize_point_prompt(prompt: str) -> str:
    text = str(prompt).strip()
    if not text:
        raise MolmoPointBackendError("prompt 不能为空")
    return text


def normalize_image_points(
    raw_points: list[Any],
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """把 MolmoPoint 图像点输出规范为 SAM3 可用的像素坐标。"""
    if image_width <= 0 or image_height <= 0:
        raise MolmoPointBackendError("图像尺寸无效")
    normalized: list[dict[str, Any]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        object_id = int(item[0])
        image_num = int(item[1])
        x = float(item[2])
        y = float(item[3])
        if not (0 <= x < image_width and 0 <= y < image_height):
            continue
        normalized.append(
            {
                "object_id": object_id,
                "image_num": image_num,
                "x": x,
                "y": y,
                "point": [x, y],
                "label": 1,
            }
        )
    return normalized


class MolmoPointEngine:
    """MolmoPoint 本地引擎。"""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._error: str | None = None

    def status(self) -> dict[str, Any]:
        identity = {"backend": "molmopoint", "model_version": "MolmoPoint-8B"}
        if self._error:
            return {"available": False, "initialized": False, "error": self._error, **identity}
        if not Path(self.checkpoint).is_dir():
            return {
                "available": False,
                "initialized": False,
                "error": f"MolmoPoint checkpoint 不存在: {self.checkpoint}",
                **identity,
            }
        return {
            "available": True,
            "initialized": self._model is not None,
            "device": self.device,
            "checkpoint": self.checkpoint,
            **identity,
        }

    def _initialize(self) -> None:
        if self._model is not None:
            return
        if self._error:
            raise RuntimeError(self._error)
        if not Path(self.checkpoint).is_dir():
            raise FileNotFoundError(f"MolmoPoint checkpoint 不存在: {self.checkpoint}")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                padding_side="left",
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device != "cuda":
                self._model.to(self.device)
        except Exception as exc:
            self._model = None
            self._processor = None
            self._error = f"MolmoPoint 初始化失败: {type(exc).__name__}: {exc}"
            raise RuntimeError(self._error) from exc

    def warm_up(self) -> None:
        with self._lock:
            self._initialize()

    def point_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 200,
    ) -> dict[str, Any]:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise MolmoPointBackendError(f"图像不存在: {image_path}")
        prompt_text = normalize_point_prompt(prompt)
        with self._lock:
            self._initialize()
            assert self._model is not None
            assert self._processor is not None
            import torch
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image", "image": str(image_path)},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                return_pointing_metadata=True,
            )
            metadata = inputs.pop("metadata")
            device = next(self._model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode(), torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = self._model.generate(
                    **inputs,
                    logits_processor=self._model.build_logit_processor_from_inputs(inputs),
                    max_new_tokens=max_new_tokens,
                )
            generated_tokens = output[:, inputs["input_ids"].size(1) :]
            generated_text = self._processor.post_process_image_text_to_text(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            raw_points = self._model.extract_image_points(
                generated_text,
                metadata["token_pooling"],
                metadata["subpatch_mapping"],
                metadata["image_sizes"],
            )
            if device.type == "cuda":
                del inputs, output, generated_tokens
                torch.cuda.empty_cache()
        points = normalize_image_points(raw_points, image_width=width, image_height=height)
        return {
            "prompt": prompt_text,
            "image_path": str(image_path),
            "image_size": {"width": width, "height": height},
            "generated_text": generated_text,
            "points": points,
        }


class MolmoPointRemoteClient:
    """标注流水线使用的 MolmoPoint HTTP client。"""

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = _direct_json_request(
            self.base_url,
            method,
            path,
            payload=payload,
            timeout=timeout or self.timeout,
        )
        if not body.get("ok", True):
            raise RuntimeError(body.get("error", "MolmoPoint server 请求失败"))
        return body

    def status(self) -> dict[str, Any]:
        try:
            body = self._request("GET", "/api/status")
            return body.get("molmopoint", {"available": False, "initialized": False, "error": "无状态"})
        except RuntimeError as exc:
            return {"available": False, "initialized": False, "error": str(exc), "backend": "remote"}

    def point_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 200,
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            "/api/point_image",
            {
                "image_path": str(image_path),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        )
        return body.get("result", body)

    def point_image_base64(
        self,
        image_base64: str,
        prompt: str,
        *,
        max_new_tokens: int = 200,
    ) -> dict[str, Any]:
        body = self._request(
            "POST",
            "/api/point_image",
            {
                "image_base64": image_base64,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            },
        )
        return body.get("result", body)


class MolmoPointPool:
    """对多个 MolmoPoint server 做简单轮询。"""

    def __init__(self, base_urls: list[str], timeout: float = 300.0):
        urls = [url.strip().rstrip("/") for url in base_urls if url.strip()]
        if not urls:
            raise ValueError("至少需要一个 MolmoPoint server URL")
        self._clients = [MolmoPointRemoteClient(url, timeout=timeout) for url in urls]
        self._index = 0
        self._lock = threading.Lock()

    def _next(self) -> MolmoPointRemoteClient:
        with self._lock:
            client = self._clients[self._index % len(self._clients)]
            self._index += 1
            return client

    def status(self) -> dict[str, Any]:
        statuses = [client.status() for client in self._clients]
        ready = [item for item in statuses if item.get("initialized")]
        return {
            "backend": "pool",
            "count": len(self._clients),
            "ready": len(ready),
            "servers": statuses,
            "available": bool(ready),
            "initialized": bool(ready),
        }

    def point_image(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 200,
    ) -> dict[str, Any]:
        return self._next().point_image(image_path, prompt, max_new_tokens=max_new_tokens)
