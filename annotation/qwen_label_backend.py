"""Qwen 视觉语言模型客户端（DashScope / OpenAI 兼容 API）。"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_CN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen3.7-plus"
# 兼容旧引用
DEFAULT_QWEN_VL_MODEL = DEFAULT_QWEN_MODEL
_DASHSCOPE_NO_PROXY_HOSTS = (
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    ".aliyuncs.com",
)


class QwenLabelBackendError(RuntimeError):
    """Qwen VLM 调用错误。"""


def load_project_dotenv() -> None:
    """加载项目根目录 .env（不覆盖已有环境变量）。"""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        return


def bypass_proxy_for_dashscope() -> None:
    """避免本地 http_proxy 把 DashScope 请求转到错误端口。"""
    current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    parts = [item.strip() for item in current.split(",") if item.strip()]
    for host in _DASHSCOPE_NO_PROXY_HOSTS:
        if host not in parts:
            parts.append(host)
    merged = ",".join(parts)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def resolve_qwen_config(
    *,
    api_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str | None, str, str | None]:
    """解析 Qwen API 配置，优先 DashScope 环境变量。"""
    load_project_dotenv()
    bypass_proxy_for_dashscope()
    resolved_key = (
        api_key
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
    )
    resolved_url = (
        api_url
        or os.environ.get("QWEN_API_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or (DASHSCOPE_INTL_BASE_URL if resolved_key else None)
    )
    resolved_model = model or os.environ.get("QWEN_MODEL") or DEFAULT_QWEN_MODEL
    return resolved_url, resolved_model, resolved_key


class QwenVLMClient:
    """通过 OpenAI 兼容 chat/completions 接口调用 Qwen-VL。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        enable_thinking: bool = False,
    ):
        if not base_url:
            raise QwenLabelBackendError("Qwen base_url 不能为空")
        if not api_key:
            raise QwenLabelBackendError(
                "未设置 Qwen API key，请导出 DASHSCOPE_API_KEY 或使用 --qwen-api-key"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    @classmethod
    def from_env(
        cls,
        *,
        api_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        enable_thinking: bool = False,
    ) -> "QwenVLMClient":
        resolved_url, resolved_model, resolved_key = resolve_qwen_config(
            api_url=api_url,
            model=model,
            api_key=api_key,
        )
        if not resolved_url or not resolved_key:
            raise QwenLabelBackendError(
                "未配置 Qwen API。请设置 DASHSCOPE_API_KEY，或传入 --qwen-api-url / --qwen-api-key"
            )
        return cls(
            resolved_url,
            resolved_model,
            api_key=resolved_key,
            timeout=timeout,
            enable_thinking=enable_thinking,
        )

    def _encode_image(self, image_path: str | Path) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise QwenLabelBackendError(f"图像不存在: {path}")
        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def _build_messages(self, prompt: str, image_paths: list[str | Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(image_path)},
                }
            )
        return [{"role": "user", "content": content}]

    def _translate_openai_error(self, exc: Exception) -> QwenLabelBackendError:
        name = type(exc).__name__
        text = str(exc)
        if name == "PermissionDeniedError" or "AccessDenied" in text or "Unpurchased" in text:
            return QwenLabelBackendError(
                f"当前账号无权使用模型 {self.model}（{self.base_url}）。"
                "请在 DashScope 控制台开通该模型，或改用 --qwen-model / QWEN_MODEL 指定已开通的模型。"
            )
        if name == "AuthenticationError" or "invalid_api_key" in text:
            return QwenLabelBackendError(
                f"Qwen API key 无效（{self.base_url}）。"
                "请检查 DASHSCOPE_API_KEY / QWEN_API_KEY 是否与所选区域（国内/国际）匹配。"
            )
        if name == "NotFoundError" or "model" in text.lower() and "not" in text.lower():
            return QwenLabelBackendError(
                f"模型 {self.model} 不存在或当前端点不可用（{self.base_url}）。"
                "请用 --qwen-model 指定正确模型名。"
            )
        return QwenLabelBackendError(f"Qwen API 调用失败: {text}")

    def _chat_openai_sdk(self, messages: list[dict[str, Any]]) -> str:
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise QwenLabelBackendError(
                "未安装 openai 包，请在当前环境执行: pip install openai httpx"
            ) from exc
        bypass_proxy_for_dashscope()
        http_client = httpx.Client(trust_env=False, timeout=self.timeout)
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            http_client=http_client,
        )
        extra_body = {"enable_thinking": self.enable_thinking}
        try:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    stream=False,
                    extra_body=extra_body,
                )
            except Exception as exc:
                raise self._translate_openai_error(exc) from exc
        finally:
            http_client.close()
        if not completion.choices:
            raise QwenLabelBackendError(f"Qwen API 无返回内容: {completion}")
        message = completion.choices[0].message
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise QwenLabelBackendError(f"Qwen API 返回空文本: {completion}")
        return text.strip()

    def _chat_urllib(self, messages: list[dict[str, Any]]) -> str:
        bypass_proxy_for_dashscope()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "enable_thinking": self.enable_thinking,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403}:
                try:
                    body = json.loads(detail)
                    err = body.get("error") or {}
                    code = str(err.get("code") or "")
                    msg = str(err.get("message") or detail)
                    if exc.code == 401 or "invalid_api_key" in code:
                        raise QwenLabelBackendError(
                            f"Qwen API key 无效（{self.base_url}）。"
                            "请检查 DASHSCOPE_API_KEY / QWEN_API_KEY 是否与所选区域（国内/国际）匹配。"
                        ) from exc
                    if "AccessDenied" in code or "Unpurchased" in code:
                        raise QwenLabelBackendError(
                            f"当前账号无权使用模型 {self.model}（{self.base_url}）。"
                            "请在 DashScope 控制台开通该模型，或改用 --qwen-model / QWEN_MODEL 指定已开通的模型。"
                        ) from exc
                    raise QwenLabelBackendError(f"Qwen API HTTP {exc.code}: {msg}") from exc
                except json.JSONDecodeError:
                    pass
            raise QwenLabelBackendError(f"Qwen API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise QwenLabelBackendError(f"无法连接 Qwen API ({self.base_url}): {exc}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise QwenLabelBackendError(f"Qwen API 无返回内容: {body}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str) or not text.strip():
            raise QwenLabelBackendError(f"Qwen API 返回空文本: {body}")
        return text.strip()

    def chat_with_images(self, prompt: str, image_paths: list[str | Path]) -> str:
        if not image_paths:
            raise QwenLabelBackendError("至少提供一张图像")
        messages = self._build_messages(prompt, image_paths)
        try:
            return self._chat_openai_sdk(messages)
        except QwenLabelBackendError as exc:
            message = str(exc)
            if "未安装 openai" in message:
                return self._chat_urllib(messages)
            raise
