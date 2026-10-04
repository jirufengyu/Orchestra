from __future__ import annotations

import http.client
import json
from typing import Any
from urllib.parse import urlparse


class JsonHttpClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        parsed = urlparse(self.base_url)
        connection_type = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(
            parsed.hostname or "127.0.0.1",
            parsed.port,
            timeout=self.timeout,
        )
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if response.status >= 400 or not data.get("ok", True):
                raise RuntimeError(data.get("error") or f"HTTP {response.status}")
            return data
        finally:
            connection.close()
