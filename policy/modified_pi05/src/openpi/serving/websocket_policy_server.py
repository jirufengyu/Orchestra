import asyncio
import http
import logging
import os
from pathlib import Path
import time
import traceback
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        progress_policy: _base_policy.BasePolicy | None = None,
    ) -> None:
        self._policy = policy
        self._progress_policy = progress_policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        tap_dir = os.environ.get("OPENPI_OBS_TAP_DIR")
        self._obs_tap_dir = Path(tap_dir) if tap_dir else None
        self._obs_tap_seq = 0
        self._obs_tap_writer_id = f"{os.getpid()}_{self._port or 'none'}"
        self._pending_obs_tap_payload: dict[str, Any] | None = None
        if self._obs_tap_dir is not None:
            self._obs_tap_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Mirroring model inputs and server outputs to %s", self._obs_tap_dir)
            self._attach_obs_tap(policy)
            if progress_policy is not None:
                self._attach_obs_tap(progress_policy)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def _attach_obs_tap(self, policy: _base_policy.BasePolicy) -> None:
        if hasattr(policy, "set_obs_tap_fn"):
            policy.set_obs_tap_fn(self._tap_observation)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def dispatch(self, obs: dict) -> dict:
        rpc = obs.get("__openpi_rpc__")
        if rpc == "infer_progress":
            progress_policy = self._progress_policy or self._policy
            if not hasattr(progress_policy, "infer_progress"):
                return {"progress_available": False}
            observation = obs.get("observation", {})
            if (
                self._progress_policy is not None
                and not observation.get("skip_memory_append", False)
                and hasattr(self._policy, "append_memory")
            ):
                self._policy.append_memory(observation)
            return progress_policy.infer_progress(observation)
        if rpc == "append_memory":
            if hasattr(self._policy, "append_memory"):
                self._policy.append_memory(obs.get("observation", {}))
            if self._progress_policy is not None and hasattr(self._progress_policy, "append_memory"):
                self._progress_policy.append_memory(obs.get("observation", {}))
            return {"ok": True}
        if rpc == "reset_memory":
            if hasattr(self._policy, "reset_memory"):
                self._policy.reset_memory()
            if self._progress_policy is not None and hasattr(self._progress_policy, "reset_memory"):
                self._progress_policy.reset_memory()
            return {"ok": True}
        if obs.get("return_progress", False):
            progress_policy = self._progress_policy or self._policy
            if not hasattr(progress_policy, "infer_progress"):
                return {"progress_available": False}
            if (
                self._progress_policy is not None
                and not obs.get("skip_memory_append", False)
                and hasattr(self._policy, "append_memory")
            ):
                self._policy.append_memory(obs)
            return progress_policy.infer_progress(obs)
        if (
            self._progress_policy is not None
            and not obs.get("skip_memory_append", False)
            and hasattr(self._progress_policy, "append_memory")
        ):
            self._progress_policy.append_memory(obs)
        return self._policy.infer(obs)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                raw_obs = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw_obs)

                infer_time = time.monotonic()
                if not isinstance(obs, dict):
                    raise TypeError(f"Expected a dict request, got {type(obs).__name__}")
                self._pending_obs_tap_payload = None
                action = self.dispatch(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                self._tap_server_output(obs, action)
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _tap_observation(self, tap_payload: dict[str, Any]) -> None:
        """Hold the transformed input until its server response is available."""
        self._pending_obs_tap_payload = tap_payload

    def _tap_server_output(self, request: dict[str, Any], output: dict[str, Any]) -> None:
        """Persist one atomic tap record containing both model input and server output."""
        if self._obs_tap_dir is None:
            return

        rpc = request.get("__openpi_rpc__")
        if rpc in {"append_memory", "reset_memory"}:
            return

        tap_payload = dict(self._pending_obs_tap_payload or {})
        if not tap_payload:
            observation = request.get("observation", request)
            if isinstance(observation, dict):
                for key in ("episode_id", "step_id", "episode_index", "frame_index"):
                    if key in observation:
                        tap_payload[key] = observation[key]
        tap_payload["tap_stage"] = "post_server_inference"
        tap_payload["server_request"] = (
            "infer_progress" if rpc == "infer_progress" or request.get("return_progress", False) else "infer"
        )
        tap_payload["server_output"] = output
        self._write_tap_payload(tap_payload)
        self._pending_obs_tap_payload = None

    def _write_tap_payload(self, tap_payload: dict[str, Any]) -> None:
        assert self._obs_tap_dir is not None
        self._obs_tap_seq += 1
        self._obs_tap_dir.mkdir(parents=True, exist_ok=True)
        raw_obs = msgpack_numpy.packb(tap_payload)
        latest_tmp = self._obs_tap_dir / f"latest.msgpack.{self._obs_tap_writer_id}.tmp"
        latest_path = self._obs_tap_dir / "latest.msgpack"
        latest_tmp.write_bytes(raw_obs)
        os.replace(latest_tmp, latest_path)

        marker_tmp = self._obs_tap_dir / f"latest.seq.{self._obs_tap_writer_id}.tmp"
        marker_path = self._obs_tap_dir / "latest.seq"
        episode_id = tap_payload.get("episode_id")
        step_id = tap_payload.get("step_id")
        marker_tmp.write_text(
            f"{self._obs_tap_writer_id}:{self._obs_tap_seq}\n"
            f"{time.time():.6f}\n"
            f"episode_id={episode_id}\n"
            f"step_id={step_id}\n",
            encoding="utf-8",
        )
        os.replace(marker_tmp, marker_path)


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
