from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

from .gateway import MobileInferenceGateway


def _codec():
    try:
        from openpi_client import msgpack_numpy

        return msgpack_numpy.packb, msgpack_numpy.unpackb
    except ImportError:
        import msgpack
        import msgpack_numpy

        msgpack_numpy.patch()
        return msgpack.packb, lambda payload: msgpack.unpackb(payload, raw=False)


def _serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _serializable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


class GatewayWebsocketServer:
    def __init__(
        self,
        gateway: MobileInferenceGateway,
        *,
        host: str = "0.0.0.0",
        port: int = 8040,
    ):
        self.gateway = gateway
        self.host = host
        self.port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        import websockets.asyncio.server
        import websockets.exceptions

        pack, unpack = _codec()

        async def handler(websocket):
            await websocket.send(
                pack(
                    {
                        "service": "embodied-agent-mobile-gateway",
                        "action_type": "qpos",
                        "state_dim": 16,
                    }
                )
            )
            try:
                async for payload in websocket:
                    request = unpack(payload)
                    rpc = request.get("__agent_rpc__", "infer")
                    try:
                        if rpc == "infer":
                            response = await asyncio.to_thread(
                                self.gateway.handle_infer,
                                request.get("observation", request),
                            )
                        elif rpc == "observe":
                            response = await asyncio.to_thread(
                                self.gateway.handle_observe,
                                request.get("observation", request),
                            )
                        elif rpc == "reset":
                            await asyncio.to_thread(
                                self.gateway.reset_episode,
                                request.get("episode_id"),
                            )
                            response = {"ok": True}
                        elif rpc == "pause":
                            self.gateway.pause()
                            response = {"ok": True}
                        elif rpc == "resume":
                            self.gateway.resume()
                            response = {"ok": True}
                        else:
                            raise ValueError(f"unknown agent rpc: {rpc}")
                        await websocket.send(pack(_serializable(response)))
                    except Exception as exc:
                        await websocket.send(
                            pack(
                                {
                                    "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        )
            except websockets.exceptions.ConnectionClosed:
                return

        async with websockets.asyncio.server.serve(
            handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()
