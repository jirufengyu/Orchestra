"""Exercise agent tools against actual Flask routes without loading model weights."""
from pathlib import Path

import numpy as np
from PIL import Image

from annotation.mobile_sam3_backend import encode_label_map_png
from annotation.molmopoint_inference_server import create_app as molmo_app
from annotation.sam3_inference_server import create_app as sam_app
from embodied_agent.contracts import AgentContext, ContextFragment, Goal, Observation
from embodied_agent.integrations.mobile.grounding import MobileVisualGroundingProvider, MolmoPointTool, SamOnlineTool
from embodied_agent.integrations.mobile.http import JsonHttpClient
from embodied_agent.integrations.mobile.pi05 import Pi05ContextMapper
from embodied_agent.integrations.mobile.registry import EntitySelection
from embodied_agent.memory import MemorySystem
from embodied_agent.tools import ContextToolRuntime


class PointEngine:
    def __init__(self):
        self.calls = []

    def point_image(self, image_path, prompt, *, max_new_tokens):
        assert Image.open(image_path).size == (8, 6)
        self.calls.append((Path(image_path), prompt))
        return {"points": [{"point": [2.0, 3.0]}]}


class TrackEngine:
    def __init__(self):
        self.started = 0
        self.closed = []
        self.frames = []

    def start_online_session(self, *, camera, max_frames):
        self.started += 1
        return {"online_session_id": str(self.started)}

    def append_online_frame(self, session_id, *, frame_idx, image_base64, prompts):
        self.frames.append((session_id, frame_idx, prompts))
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:4, 2:6] = 7
        return {"label_map_png_base64": encode_label_map_png(mask)}

    def close_online_session(self, session_id):
        self.closed.append(session_id)


def test_online_grounding_routes_cache_reset_and_policy_mapping(monkeypatch):
    point, track = PointEngine(), TrackEngine()
    clients = {"http://molmo": molmo_app(point).test_client(), "http://sam": sam_app(track).test_client()}

    def request(self, method, path, payload=None):
        response = clients[self.base_url].open(path, method=method, json=payload)
        assert response.status_code == 200, response.json
        return response.json

    monkeypatch.setattr(JsonHttpClient, "request", request)
    tools = ContextToolRuntime()
    tools.register(MolmoPointTool("http://molmo"))
    tools.register(SamOnlineTool("http://sam"))
    provider = MobileVisualGroundingProvider(tools)
    memory = MemorySystem(episode_id="episode", step_id=0)
    obs = Observation({"cam_high": np.zeros((6, 8, 3), np.uint8)}, np.zeros(16))
    goal = Goal("test", "Move the apple")
    selection = EntitySelection("test", {"A": 7}, {7: "apple"})
    fragments = {"entity_grounding": ContextFragment("entity_grounding", selection)}
    first = provider.provide_with_context(obs, goal, memory, fragments)
    assert provider.provide_with_context(obs, goal, memory, fragments) is first
    assert len(point.calls) == len(track.frames) == 1
    assert not point.calls[0][0].exists()  # Uploaded image is cleaned up after inference.
    assert track.frames[0][2] == [{"instance_id": 7, "points": [[2.0, 3.0]], "labels": [1]}]
    memory.step_id = 1
    second = provider.provide_with_context(obs, goal, memory, fragments)
    assert len(point.calls) == 1
    assert track.frames[-1][2] is None
    payload = Pi05ContextMapper().map(AgentContext(obs, goal, memory.snapshot(), {"visual_grounding": second}))
    assert set(np.unique(payload["seg_cam_high"])) == {0, 7}
    assert payload["actors_frame_meta"] == {"apple": [7]}
    provider.reset()
    assert track.closed == ["1"]
    provider.provide_with_context(obs, goal, memory, fragments)
    assert len(point.calls) == track.started == 2
    provider.close()
    assert track.closed == ["1", "2"]


def test_molmo_rejects_invalid_uploaded_image_encoding():
    response = molmo_app(PointEngine()).test_client().post(
        "/api/point_image", json={"image_base64": "%%%", "prompt": "Point to the apple"}
    )
    assert response.status_code == 400
    assert not response.json["ok"]
