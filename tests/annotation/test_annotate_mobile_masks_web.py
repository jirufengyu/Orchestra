#!/usr/bin/env python3
"""annotation 包的无模型单元测试。"""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from annotation.annotate_mobile_masks_web import (
    AnnotationError,
    AnnotationStore,
    atomic_write_json,
    clear_instance_mask,
    create_app,
    drop_instance_items,
    merge_instance_mask,
    replace_instance_items,
)
from annotation.mobile_sam3_backend import (
    Sam3MultiplexEngine,
    Sam3RemoteClient,
    decode_label_map_png,
    encode_label_map_png,
    extract_multiplex_mask,
    merge_label_maps,
    multiplex_relative_prompts,
    outputs_to_label_map,
    prepare_episode_frame_dir,
)


def binary_png_base64(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeSam:
    def __init__(self):
        self.base_url = "http://fake-sam"
        self.last_predict: dict | None = None
        self.predict_calls: list[dict] = []

    def status(self):
        return {"available": True, "initialized": True, "device": "test", "backend": "remote"}

    def segment_episode(
        self,
        episode_dir,
        episode_name,
        camera,
        seed_frame_idx,
        prompts,
        direction="both",
    ):
        del episode_dir, episode_name, camera, direction
        frames = []
        for frame_idx in (0, 3):
            label_map = np.zeros((6, 8), dtype=np.uint8)
            for prompt in prompts:
                instance_id = int(prompt["instance_id"])
                label_map[1 + instance_id : 2 + instance_id, 2 + instance_id : 4 + instance_id] = (
                    instance_id
                )
            frames.append(
                {
                    "frame": frame_idx if frame_idx != 3 or seed_frame_idx == 0 else 3,
                    "label_map_png_base64": encode_label_map_png(label_map),
                }
            )
        if seed_frame_idx == 0:
            frames = [
                {"frame": 0, "label_map_png_base64": frames[0]["label_map_png_base64"]},
                {"frame": 3, "label_map_png_base64": frames[1]["label_map_png_base64"]},
            ]
        return {
            "frame_count": len(frames),
            "frames": frames,
            "sam_obj_mapping": {
                str(index): int(prompt["instance_id"])
                for index, prompt in enumerate(prompts)
            },
        }

    def predict(self, episode_dir, episode_name, camera, frame_idx, points, labels, box):
        del episode_dir, episode_name, camera, frame_idx, box
        self.last_predict = {"points": points, "labels": labels}
        self.predict_calls.append(self.last_predict)
        mask = np.zeros((6, 8), dtype=bool)
        mask[1:3, 2:4] = True
        obj_id = len(self.predict_calls)
        return [{"score": 1.0, "mask_png_base64": binary_png_base64(mask), "obj_id": obj_id}]


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeMultiplexPredictor:
    def __init__(self):
        self.requests = []
        self.resource_paths = []
        self._next_obj_id = 1
        self._all_inference_states: dict = {}
        self.session_id = "session-1"

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            resource_path = Path(request["resource_path"])
            self.resource_paths.append(resource_path)
            jpgs = list(resource_path.glob("*.jpg"))
            if not jpgs:
                raise AssertionError("episode frame dir 缺少 JPEG")
            self._all_inference_states[self.session_id] = {
                "state": {
                    "cached_frame_outputs": {},
                    "num_frames": len(jpgs),
                    "previous_stages_out": [None] * len(jpgs),
                    "sam2_inference_states": [{"num_frames": len(jpgs)}],
                }
            }
            return {"session_id": self.session_id}
        if request["type"] == "add_prompt":
            if "bounding_boxes" in request:
                obj_id = self._next_obj_id
                self._next_obj_id += 1
            else:
                if "obj_id" not in request:
                    raise AssertionError("point add_prompt requires obj_id")
                obj_id = int(request["obj_id"])
            frame_idx = int(request.get("frame_index", 0))
            cache = (
                self._all_inference_states.get(self.session_id, {})
                .get("state", {})
                .get("cached_frame_outputs", {})
            )
            # 复现 SAM3.1：帧不在 cached_frame_outputs 时 out_obj_ids 为空
            if frame_idx not in cache:
                return {
                    "frame_index": frame_idx,
                    "outputs": {
                        "out_obj_ids": FakeTensor(np.zeros(0, dtype=np.int64)),
                        "out_binary_masks": FakeTensor(np.zeros((0, 6, 8), dtype=np.uint8)),
                    },
                }
            masks = np.zeros((1, 6, 8), dtype=np.uint8)
            masks[0, 1:3, 2:4] = 1
            return {
                "frame_index": frame_idx,
                "outputs": {
                    "out_obj_ids": FakeTensor([obj_id]),
                    "out_binary_masks": FakeTensor(masks),
                },
            }
        if request["type"] == "close_session":
            return {"is_success": True}
        raise AssertionError(f"unexpected request: {request}")

    def handle_stream_request(self, request):
        self.requests.append(request)
        state = self._all_inference_states.get(self.session_id, {}).get("state", {})
        num_frames = int(state.get("num_frames", 2))
        start = request.get("start_frame_index")
        if start is None:
            start = 0
        max_n = request.get("max_frame_num_to_track")
        if max_n is None:
            end = num_frames - 1
        else:
            end = min(int(start) + int(max_n), num_frames - 1)
        for session_index in range(int(start), end + 1):
            masks = np.zeros((2, 6, 8), dtype=np.uint8)
            masks[0, 2:4, 3:5] = 1
            masks[1, 1:3, 2:4] = 1
            yield {
                "frame_index": session_index,
                "outputs": {
                    "out_obj_ids": FakeTensor([0, 1]),
                    "out_binary_masks": FakeTensor(masks),
                },
            }


class MobileMaskWebTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "episode_0001"
        colors = self.episode / "colors"
        colors.mkdir(parents=True)
        frames = []
        for idx in (0, 3):
            relative = f"colors/{idx:06d}_color_0.jpg"
            Image.new("RGB", (8, 6), (20, idx, 0)).save(self.episode / relative)
            frames.append({"idx": idx, "colors": {"color_0": relative}})
        with open(self.episode / "data.json", "w", encoding="utf-8") as stream:
            json.dump({"text": {"goal": "测试目标"}, "data": frames}, stream)
        self.store = AnnotationStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_online_session_appends_frames_and_preserves_prompts(self):
        checkpoint = self.root / "sam3.1_multiplex.pt"
        checkpoint.touch()
        predictor = FakeMultiplexPredictor()
        engine = Sam3MultiplexEngine(checkpoint=str(checkpoint), device="cpu")
        engine._model = predictor
        started = engine.start_online_session(camera="color_0", max_frames=4)
        session_id = started["online_session_id"]
        episode_dir = engine._online_sessions[session_id]["episode_dir"]
        buffer = io.BytesIO()
        Image.new("RGB", (8, 6), (10, 20, 30)).save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        prompts = [{"instance_id": 4, "points": [[3.0, 2.0]], "labels": [1]}]

        first = engine.append_online_frame(
            session_id,
            frame_idx=0,
            image_base64=image_base64,
            prompts=prompts,
        )
        second = engine.append_online_frame(
            session_id,
            frame_idx=3,
            image_base64=image_base64,
        )
        predictor._all_inference_states[predictor.session_id]["state"]["action_history"] = [
            {"type": "add", "obj_ids": [0], "frame_idx": 0},
            {"type": "propagation_partial", "obj_ids": [0], "frame_idx": 1},
            {"type": "propagation_partial", "obj_ids": [0], "frame_idx": 2},
        ]
        third = engine.append_online_frame(
            session_id,
            frame_idx=6,
            image_base64=image_base64,
        )

        self.assertEqual(first["frame_count"], 1)
        self.assertEqual(second["frame_count"], 2)
        self.assertEqual(third["frame_count"], 3)
        self.assertEqual(first["sam_obj_mapping"], {"0": 4})
        starts = [item for item in predictor.requests if item["type"] == "start_session"]
        prompts_req = [item for item in predictor.requests if item["type"] == "add_prompt"]
        propagates = [
            item for item in predictor.requests if item["type"] == "propagate_in_video"
        ]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(prompts_req), 1)
        self.assertEqual(len(propagates), 2)
        self.assertEqual(propagates[0]["start_frame_index"], 1)
        self.assertEqual(propagates[1]["start_frame_index"], 2)
        self.assertEqual(propagates[0]["max_frame_num_to_track"], 0)
        self.assertEqual(propagates[1]["max_frame_num_to_track"], 0)
        leftover = predictor._all_inference_states[predictor.session_id]["state"]["action_history"]
        self.assertEqual([item["type"] for item in leftover], ["add"])
        first_map = decode_label_map_png(first["label_map_png_base64"], (6, 8))
        self.assertEqual(int(first_map[1, 2]), 4)
        decoded = decode_label_map_png(third["label_map_png_base64"], (6, 8))
        self.assertEqual(int(decoded[2, 3]), 4)
        engine.close_online_session(session_id)
        self.assertFalse(episode_dir.exists())
        closes = [item for item in predictor.requests if item["type"] == "close_session"]
        self.assertEqual(len(closes), 1)

    def test_keep_online_partial_tracking_strips_propagation_actions(self):
        checkpoint = self.root / "sam3.1_multiplex.pt"
        checkpoint.touch()
        predictor = FakeMultiplexPredictor()
        engine = Sam3MultiplexEngine(checkpoint=str(checkpoint), device="cpu")
        engine._model = predictor
        predictor._all_inference_states[predictor.session_id] = {
            "state": {
                "action_history": [
                    {"type": "add", "obj_ids": [0, 1], "frame_idx": 0},
                    {"type": "propagation_partial", "obj_ids": [0, 1], "frame_idx": 1},
                    {"type": "propagation_partial", "obj_ids": [0, 1], "frame_idx": 2},
                ]
            }
        }
        engine._keep_online_partial_tracking(predictor.session_id)
        history = predictor._all_inference_states[predictor.session_id]["state"]["action_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["type"], "add")

    def test_get_auto_job_record(self):
        queue_path = self.root / "annotations" / "auto_jobs.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "episode": "episode_0001",
            "status": "done",
            "seed_frame": 0,
            "molmo_points": [{"object_id": 1, "point": [3.0, 2.0]}],
            "sam_prompts": [{"instance_id": 1, "points": [[3.0, 2.0]], "labels": [1]}],
        }
        queue_path.write_text(json.dumps(job) + "\n", encoding="utf-8")
        record = self.store.get_auto_job_record("episode_0001")
        self.assertTrue(record["available"])
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["seed_frame"], 0)
        self.assertEqual(len(record["molmo_points"]), 1)

    def test_overwrite_instance_seed_points(self):
        queue_path = self.root / "annotations" / "auto_jobs.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "episode": "episode_0001",
            "status": "done",
            "seed_frame": 0,
            "molmo_points": [
                {
                    "instance_id": 1,
                    "object_id": 1,
                    "point": [3.0, 2.0],
                    "x": 3.0,
                    "y": 2.0,
                },
                {
                    "instance_id": 2,
                    "object_id": 1,
                    "point": [6.0, 4.0],
                    "x": 6.0,
                    "y": 4.0,
                },
            ],
            "sam_prompts": [
                {"instance_id": 1, "points": [[3.0, 2.0]], "labels": [1]},
                {"instance_id": 2, "points": [[6.0, 4.0]], "labels": [1]},
            ],
            "result": {
                "sam_prompts": [
                    {"instance_id": 1, "points": [[3.0, 2.0]], "labels": [1]},
                    {"instance_id": 2, "points": [[6.0, 4.0]], "labels": [1]},
                ]
            },
        }
        queue_path.write_text(json.dumps(job) + "\n", encoding="utf-8")
        replaced = replace_instance_items(
            job["molmo_points"], 1, {"instance_id": 1, "point": [1.0, 1.0]}
        )
        self.assertEqual(replaced[0]["point"], [1.0, 1.0])
        self.assertEqual(replaced[1]["instance_id"], 2)
        result = self.store.overwrite_instance_seed_points(
            "episode_0001", 1, [[4.0, 3.0], [1.0, 1.0]], [1, 0]
        )
        self.assertTrue(result["updated"])
        record = self.store.get_auto_job_record("episode_0001")
        self.assertEqual(record["molmo_points"][0]["point"], [4.0, 3.0])
        self.assertEqual(record["molmo_points"][0]["source"], "manual")
        self.assertEqual(record["molmo_points"][1]["point"], [6.0, 4.0])
        self.assertEqual(record["sam_prompts"][0]["points"], [[4.0, 3.0]])
        self.assertEqual(record["sam_prompts"][0]["source"], "manual")

    def test_frame_segment_passes_all_points_to_sam(self):
        sam = FakeSam()
        app = create_app(self.root, sam_client=sam)
        app.testing = True
        client = app.test_client()
        self.store.create_instance("苹果")
        response = client.post(
            "/api/frame/segment",
            json={
                "episode": "episode_0001",
                "frame": 0,
                "instance_id": 1,
                "points": [[1.0, 2.0], [5.0, 6.0]],
                "labels": [1, 1],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            sam.last_predict,
            {"points": [[1.0, 2.0], [5.0, 6.0]], "labels": [1, 1]},
        )

        response = client.post(
            "/api/frame/segment",
            json={
                "episode": "episode_0001",
                "frame": 3,
                "instance_id": 1,
                "points": [[5.0, 2.0]],
                "labels": [1],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertFalse(body["seed_overwrite"]["updated"])

    def test_frame_segment_multiple_instances(self):
        sam = FakeSam()
        app = create_app(self.root, sam_client=sam)
        app.testing = True
        client = app.test_client()
        self.store.create_instance("苹果")
        self.store.create_instance("碗")
        response = client.post(
            "/api/frame/segment",
            json={
                "episode": "episode_0001",
                "frame": 0,
                "prompts": [
                    {"instance_id": 1, "points": [[1.0, 2.0]], "labels": [1]},
                    {"instance_id": 2, "points": [[5.0, 6.0]], "labels": [1]},
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["segmented_instances"], 2)
        self.assertEqual(body["foreground_points"], 2)
        self.assertEqual(len(sam.predict_calls), 2)

    def test_clear_episode_instance_and_purge_auto_job(self):
        queue_path = self.root / "annotations" / "auto_jobs.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "episode": "episode_0001",
            "status": "done",
            "seed_frame": 0,
            "instance_ids": [6, 1],
            "molmo_points": [{"instance_id": 6, "point": [1.0, 1.0]}],
            "sam_prompts": [
                {"instance_id": 6, "points": [[1.0, 1.0]], "labels": [1]},
                {"instance_id": 1, "points": [[2.0, 2.0]], "labels": [1]},
            ],
        }
        queue_path.write_text(json.dumps(job) + "\n", encoding="utf-8")
        for idx in (0, 3):
            label_map = np.zeros((6, 8), dtype=np.uint8)
            label_map[1:3, 2:4] = 6
            label_map[3:5, 4:6] = 1
            path = self.store.mask_path("episode_0001", idx)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(label_map, mode="L").save(path)
        cleared = self.store.clear_episode_instance("episode_0001", 6)
        self.assertEqual(cleared, 2)
        for idx in (0, 3):
            mask = self.store.load_mask("episode_0001", idx)
            self.assertFalse(np.any(mask == 6))
            self.assertTrue(np.any(mask == 1))
        purge = self.store.purge_auto_job_instance("episode_0001", 6)
        self.assertTrue(purge["updated"])
        job_after = json.loads(queue_path.read_text(encoding="utf-8").strip())
        self.assertEqual(job_after["instance_ids"], [1])
        self.assertEqual(len(job_after["sam_prompts"]), 1)
        self.assertEqual(job_after["sam_prompts"][0]["instance_id"], 1)
        dropped = drop_instance_items(
            [{"instance_id": 6}, {"instance_id": 1}], 6
        )
        self.assertEqual(len(dropped), 1)

        app = create_app(self.root, sam_client=FakeSam())
        app.testing = True
        client = app.test_client()
        response = client.post(
            "/api/episode/clear_instance",
            json={"episode": "episode_0001", "instance_id": 1, "purge_auto_job": True},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["cleared_frames"], 2)

    def test_overwrite_instance_seed_points_api(self):
        queue_path = self.root / "annotations" / "auto_jobs.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "episode": "episode_0001",
            "status": "done",
            "seed_frame": 0,
            "molmo_points": [
                {"instance_id": 1, "object_id": 1, "point": [3.0, 2.0], "x": 3.0, "y": 2.0},
                {"instance_id": 2, "object_id": 1, "point": [6.0, 4.0], "x": 6.0, "y": 4.0},
            ],
            "sam_prompts": [
                {"instance_id": 1, "points": [[3.0, 2.0]], "labels": [1]},
                {"instance_id": 2, "points": [[6.0, 4.0]], "labels": [1]},
            ],
        }
        queue_path.write_text(json.dumps(job) + "\n", encoding="utf-8")
        app = create_app(self.root, sam_client=FakeSam())
        app.testing = True
        client = app.test_client()
        self.store.create_instance("苹果")
        response = client.post(
            "/api/frame/segment",
            json={
                "episode": "episode_0001",
                "frame": 0,
                "instance_id": 1,
                "points": [[5.0, 2.0]],
                "labels": [1],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["seed_overwrite"]["updated"])
        after = self.store.get_auto_job_record("episode_0001")
        self.assertEqual(after["molmo_points"][0]["point"], [5.0, 2.0])
        self.assertEqual(after["molmo_points"][1]["point"], [6.0, 4.0])
        seed = client.post(
            "/api/frame/seed_points",
            json={
                "episode": "episode_0001",
                "instance_id": 1,
                "points": [[2.0, 1.0], [3.0, 4.0]],
                "labels": [1, 1],
            },
        )
        self.assertEqual(seed.status_code, 200)
        seed_body = seed.get_json()
        self.assertTrue(seed_body["updated"])
        self.assertEqual(seed_body["molmo_points"][0]["point"], [2.0, 1.0])
        self.assertEqual(seed_body["sam_prompts"][0]["points"], [[2.0, 1.0], [3.0, 4.0]])
        self.assertEqual(seed_body["auto_annotation"]["molmo_points"][0]["source"], "manual")

    def test_discovery_and_head_only_schema(self):
        info = self.store.episode_info("episode_0001")
        self.assertEqual(info["frame_indices"], [0, 3])
        self.assertEqual(info["camera"], "color_0")
        self.assertEqual(info["progress"]["total"], 2)
        self.assertEqual(info["instances"], [])

    def test_merge_label_maps(self):
        base = np.zeros((6, 8), dtype=np.uint8)
        base[0, :2] = 1
        overlay = np.zeros((6, 8), dtype=np.uint8)
        overlay[2:, 3:5] = 2
        merged = merge_label_maps(base, overlay)
        self.assertEqual(int(merged[0, 0]), 1)
        self.assertEqual(int(merged[3, 4]), 2)

    def test_task_level_instances(self):
        item = self.store.create_instance("苹果", "#ff0000")
        self.assertEqual(item["id"], 1)
        self.assertTrue(self.store.task_instances_path().is_file())
        info = self.store.episode_info("episode_0001")
        self.assertEqual(len(info["instances"]), 1)
        self.store.rename_instance(1, "红苹果")
        self.store.delete_instance(1)
        self.assertEqual(self.store.load_task_instances(), [])

    def test_save_episode_label_maps(self):
        self.store.create_instance("目标")
        maps = {
            0: np.array([[0, 1, 1, 0, 0, 0, 0, 0]] * 6, dtype=np.uint8),
            3: np.array([[0, 0, 1, 1, 0, 0, 0, 0]] * 6, dtype=np.uint8),
        }
        saved = self.store.save_episode_label_maps("episode_0001", maps)
        self.assertEqual(saved, 2)
        loaded = self.store.load_mask("episode_0001", 0)
        self.assertEqual(int(loaded.max()), 1)

    def test_outputs_to_label_map(self):
        masks = np.zeros((2, 6, 8), dtype=np.uint8)
        masks[0, 1:3, 2:4] = 1
        masks[1, 2:4, 3:5] = 1
        label_map = outputs_to_label_map(
            {
                "out_obj_ids": FakeTensor([1, 2]),
                "out_binary_masks": FakeTensor(masks),
            },
            {1: 1, 2: 2},
            (6, 8),
        )
        self.assertIn(1, np.unique(label_map))
        self.assertIn(2, np.unique(label_map))

    def test_segment_episode_engine(self):
        checkpoint = self.root / "sam3.1_multiplex.pt"
        checkpoint.touch()
        predictor = FakeMultiplexPredictor()
        engine = Sam3MultiplexEngine(checkpoint=str(checkpoint))
        engine._model = predictor
        result = engine.segment_episode(
            self.episode,
            "color_0",
            0,
            [
                {"instance_id": 1, "points": [[4.0, 3.0]], "labels": [1]},
                {"instance_id": 2, "points": [[2.0, 2.0]], "labels": [1]},
            ],
        )
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(len(result["frames"]), 2)
        label_map = decode_label_map_png(result["frames"][0]["label_map_png_base64"], (6, 8))
        self.assertIn(1, np.unique(label_map))
        self.assertIn(2, np.unique(label_map))
        starts = [item for item in predictor.requests if item["type"] == "start_session"]
        propagates = [item for item in predictor.requests if item["type"] == "propagate_in_video"]
        closes = [item for item in predictor.requests if item["type"] == "close_session"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(propagates), 1)
        self.assertEqual(len(closes), 1)
        self.assertNotIn("max_frame_num_to_track", propagates[0])
        self.assertEqual(propagates[0]["propagation_direction"], "forward")

    def test_api_segment_episode_flow(self):
        app = create_app(self.root, sam_client=FakeSam())
        app.testing = True
        client = app.test_client()
        instance_id = client.post(
            "/api/instances", json={"name": "苹果", "color": "#ff0000"}
        ).get_json()["instance"]["id"]
        response = client.post(
            "/api/episode/segment",
            json={
                "episode": "episode_0001",
                "seed_frame": 0,
                "prompts": [
                    {"instance_id": instance_id, "points": [[3.0, 2.0]], "labels": [1]}
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["saved_frames"], 2)
        mask = np.asarray(Image.open(self.store.mask_path("episode_0001", 0)).convert("L"))
        self.assertGreater(int(mask.max()), 0)
        confirm_all = client.post(
            "/api/episode/confirm_all", json={"episode": "episode_0001"}
        )
        self.assertEqual(confirm_all.get_json()["confirmed_frames"], 2)

    def test_merge_clear_and_multiplex_helpers(self):
        labels = np.zeros((3, 4), dtype=np.uint8)
        labels[0, :2] = 1
        candidate = np.zeros_like(labels, dtype=bool)
        candidate[1:, 2:] = True
        merged = merge_instance_mask(labels, candidate, 1)
        self.assertEqual(int(merged[1, 3]), 1)
        self.assertTrue(np.any(merged == 1))
        self.assertTrue(np.all(clear_instance_mask(merged, 1)[merged == 1] == 0))
        points, box = multiplex_relative_prompts(
            (8, 6),
            np.asarray([[4.0, 3.0]], dtype=np.float32),
            np.asarray([2.0, 1.0, 6.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_allclose(points, [[0.5, 0.5]])
        masks = np.zeros((1, 6, 8), dtype=np.uint8)
        masks[0, 2:4, 3:6] = 1
        selected = extract_multiplex_mask(
            {"out_obj_ids": FakeTensor([1]), "out_binary_masks": FakeTensor(masks)},
            obj_id=1,
            expected_shape=(6, 8),
        )
        self.assertEqual(int(selected.sum()), 6)
        with self.assertRaisesRegex(RuntimeError, "未产出有效 mask"):
            extract_multiplex_mask(
                {
                    "out_obj_ids": FakeTensor(np.zeros(0, dtype=np.int64)),
                    "out_binary_masks": FakeTensor(np.zeros((0, 6, 8), dtype=np.uint8)),
                },
                obj_id=0,
                expected_shape=(6, 8),
            )

    def test_predict_fresh_session_uses_prefilled_cache(self):
        checkpoint = self.root / "sam3.1_multiplex.pt"
        checkpoint.touch()
        predictor = FakeMultiplexPredictor()
        engine = Sam3MultiplexEngine(checkpoint=str(checkpoint))
        engine._model = predictor
        started = engine.start_episode_session(self.episode, "color_0")
        cache = predictor._all_inference_states[started["session_id"]]["state"][
            "cached_frame_outputs"
        ]
        self.assertIn(0, cache)
        self.assertIn(1, cache)
        candidates = engine.predict(started["session_id"], 0, [[4.0, 3.0]], [1], None)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["obj_id"], 0)
        self.assertIn("mask_png_base64", candidates[0])

    def test_prepare_episode_frame_dir(self):
        result = prepare_episode_frame_dir(self.episode, "color_0")
        self.assertEqual(result["frame_count"], 2)
        frame_dir = Path(result["frame_dir"])
        self.assertTrue((frame_dir / "00000.jpg").is_symlink())
        self.assertTrue((frame_dir / "manifest.json").is_file())
        skipped = prepare_episode_frame_dir(self.episode, "color_0")
        self.assertTrue(skipped["skipped"])

    def test_rebuild_and_custom_episode_prompts(self):
        self.store.create_instance("red apple")
        self.store.create_instance("left bowl")
        self.store.set_instruction_template("Place {A} on the {B}")
        saved = self.store.set_episode_labels("episode_0001", [1, 2])
        self.assertEqual(saved["instruction"], "Place red apple on the left bowl")
        self.assertFalse(saved["custom"])

        self.store.rename_instance(1, "green pear")
        rebuilt = self.store.rebuild_episode_prompts()
        self.assertEqual(rebuilt["counts"]["updated"], 1)
        record = self.store.get_episode_label_record("episode_0001")
        self.assertEqual(record["instruction"], "Place green pear on the left bowl")
        self.assertEqual(
            record["point_prompt"],
            "Point to the green pear. | Point to the left bowl.",
        )

        custom = self.store.set_episode_prompts(
            "episode_0001",
            instruction="Put the pear onto the bowl carefully",
            point_prompt="Point to the pear on the left. | Point to the empty bowl.",
        )
        self.assertTrue(custom["custom"])
        skipped = self.store.rebuild_episode_prompts()
        self.assertEqual(skipped["counts"]["updated"], 0)
        self.assertEqual(skipped["skipped"][0]["reason"], "已定制")
        kept = self.store.get_episode_label_record("episode_0001")
        self.assertEqual(kept["instruction"], "Put the pear onto the bowl carefully")
        self.assertEqual(
            kept["point_prompt"],
            "Point to the pear on the left. | Point to the empty bowl.",
        )

        app = create_app(self.root, sam_client=FakeSam())
        app.testing = True
        client = app.test_client()
        patch = client.patch(
            "/api/episodes/episode_0001/labels",
            json={
                "instruction": "custom instruction",
                "point_prompt": "Point to the pear. | Point to the bowl.",
            },
        )
        self.assertEqual(patch.status_code, 200)
        self.assertTrue(patch.get_json()["custom"])
        rebuild = client.post(
            "/api/episodes/rebuild_prompts",
            json={"skip_custom": False},
        )
        self.assertEqual(rebuild.status_code, 200)
        self.assertEqual(rebuild.get_json()["counts"]["updated"], 1)
        after = self.store.get_episode_label_record("episode_0001")
        self.assertEqual(after["instruction"], "Place green pear on the left bowl")
        self.assertFalse(after["custom"])

    def test_remote_client_retries_stale_session(self):
        client = Sam3RemoteClient("http://127.0.0.1:8765")
        client._session_id = "old-session"
        client._episode = "episode_0001"
        calls: list[tuple[str, str, dict | None]] = []

        def fake_request(method, path, payload=None, timeout=None):
            del timeout
            calls.append((method, path, payload))
            if path == "/api/predict" and payload and payload.get("session_id") == "old-session":
                raise RuntimeError("未知 session: old-session")
            if path == "/api/session/start":
                return {"ok": True, "session_id": "new-session"}
            if path == "/api/predict":
                return {"ok": True, "candidates": [{"obj_id": 0, "score": 1.0}]}
            if path == "/api/session/close":
                raise RuntimeError("未知 session: old-session")
            raise AssertionError(f"unexpected request {method} {path}")

        client._request = fake_request  # type: ignore[method-assign]
        candidates = client.predict(
            self.episode, "episode_0001", "color_0", 0, [[4.0, 3.0]], [1], None
        )
        self.assertEqual(candidates[0]["obj_id"], 0)
        self.assertEqual(client._session_id, "new-session")
        predict_ids = [
            payload["session_id"]
            for method, path, payload in calls
            if path == "/api/predict" and payload is not None
        ]
        self.assertEqual(predict_ids, ["old-session", "new-session"])


if __name__ == "__main__":
    unittest.main()
