# Molmo / SAM in the embodied-agent pipeline

English | [简体中文](agent-pipeline.zh-CN.md)

Molmo and SAM provide visual context to the agent. They supply the same target instance masks to the action policy and progress evaluator. The task agent decomposes instructions, the action policy generates actions, and the independent evaluator determines subtask completion.

## Request flow

1. `build_pi05_mobile_gateway` registers `EntityResolverTool`, `MolmoPointTool`, and `SamOnlineTool`. Its `ContextSystem` resolves entities before visual grounding.
2. `RegistryEntityResolver` maps the current subtask to stable instance IDs, names, and slots in the registry, generating a `Point to the …` prompt for each instance. Available tasks and instances are defined in [registry.py](../embodied_agent/integrations/mobile/registry.py).
3. On the first `infer` of a subtask, `MobileVisualGroundingProvider` encodes the `cam_high` image as PNG/base64. `MolmoPointTool` calls `/api/point_image` for each instance and converts pixel coordinates into `instance_id + points + labels`.
4. `SamOnlineTool` creates a session through `/api/online/session/start` and sends the current image and initial point prompts to `/api/online/frame`. The server returns a base64 PNG instance ID map.
5. Subsequent `infer` calls append frames and continue tracking without calling Molmo again. Visual context is cached for the same episode/step. The provider raises an error if the mask contains none of the currently registered targets.
6. `Pi05ContextMapper` maps `seg_mask` to `seg_cam_high` and passes `actors_frame_meta`, the subtask prompt, RGB images, and the 16-dimensional state. Both the action policy and progress evaluator consume this context. Masks condition the models; they do not directly generate robot actions.
7. When the evaluator's completion condition is met, `AgentRuntime.update_task` switches subtasks and resets components: it clears local policy/evaluator history, closes the SAM session, and clears grounding caches. Molmo initializes the targets again for the next subtask. After the entire task finishes, the session is released on the next episode reset or gateway close.

The `observe` RPC updates policy/evaluator observation history. With the current `enrich_context=False` setting it does not call Molmo or SAM; visual tracking runs on `infer`. `pause/resume` retains state, and the next `infer` continues with a fresh image.

Code entry points:

- [factory.py](../embodied_agent/integrations/mobile/factory.py): assemble the runtime and service tools.
- [grounding.py](../embodied_agent/integrations/mobile/grounding.py): instance pointing, online tracking, and context caching.
- [pi05.py](../embodied_agent/integrations/mobile/pi05.py): policy input mapping and evaluator thresholds.
- [runtime.py](../embodied_agent/runtime.py): subtask transitions and component resets.

## Start the gateway

Start Molmo on port 8766 and SAM on port 8765 using the [model server instructions](../annotation/README.md#2-start-the-model-servers). Online execution uses the same models through SAM's online API; no annotation UI or offline queue worker is needed.

Then follow the [policy documentation](../policy/modified_pi05/README.md) to start the action policy on port 8020 and progress evaluator on port 8030 with their respective checkpoints.

Install the agent and bundled RPC client from the repository root:

```bash
python3.11 -m venv .venv-agent
.venv-agent/bin/python -m pip install -e '.[mobile,openai]'
.venv-agent/bin/python -m pip install -e policy/modified_pi05/packages/openpi-client
```

Configure your task planning/entity resolution model and start the gateway:

```bash
export LLM_API_KEY=your-api-key
.venv-agent/bin/python -m scripts.serve_mobile_gateway \
  --host 127.0.0.1 --port 8040 \
  --action-host 127.0.0.1 --action-port 8020 \
  --progress-host 127.0.0.1 --progress-port 8030 \
  --molmo-url http://127.0.0.1:8766 \
  --sam-url http://127.0.0.1:8765 \
  --llm-api-base https://your-llm-service.example/v1 \
  --llm-model your-model \
  --done-threshold 0.6 --done-count 1
```

By default, debug output under `logs/mobile_gateway` includes images, masks, plans, and model inputs. Use `--no-debug` to disable it. The online interface uses OpenPI-style MessagePack/WebSocket transport rather than HTTP JSON.

The robot client sends dictionaries with the following structure; the RPC client serializes the arrays:

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("127.0.0.1", 8040)
result = client.infer({
    "episode_id": "rollout-001",
    "step_id": 0,
    "task_instruction": "Place the apple in the bowl",  # Use registered tasks/objects.
    "task_mode": "auto",
    "images": {
        "cam_high": head_rgb,           # RGB uint8, HWC or CHW
        "cam_left_wrist": left_rgb,
        "cam_right_wrist": right_rgb,
    },
    "state": joint_state,               # float32, shape (16,)
})
actions = result["actions"]
```

The robot supplies `head_rgb`, wrist images, and `joint_state`. Increment `step_id` throughout an episode. Responses contain `actions`, `progress`, `subtask_done`, `task_done`, and grounding information. On a subtask transition, the gateway returns an empty action array; the client should inspect the transition and submit a fresh observation. This example requests inference without executing robot actions.

To explicitly reset the gateway, send an agent RPC with `client.infer({"__agent_rpc__": "reset", "episode_id": "rollout-002"})`. `WebsocketClientPolicy.reset()` uses the policy RPC protocol and does not perform a gateway agent reset.

### Start the full service stack

Alternatively, use `scripts/start_mobile_pi05_agent.sh`. Install all environments first and configure paths. This script starts all services itself; do not run it alongside the individual commands on the same ports.

```bash
export PYTHON_MOLMO="$PWD/.venv-molmo/bin/python"
export PYTHON_SAM="$PWD/.venv-sam/bin/python"
export AGENT_PYTHON="$PWD/.venv-agent/bin/python"
export MOLMO_CHECKPOINT=/path/to/models/MolmoPoint-8B
export SAM_CHECKPOINT=/path/to/models/sam3/sam3.1_multiplex.pt
export ACTION_CKPT=/path/to/action_checkpoint/50000
export PROGRESS_CKPT=/path/to/progress_checkpoint/50000
export LLM_API_BASE=https://your-llm-service.example/v1
export LLM_MODEL=your-model
export LLM_API_KEY=your-api-key
bash scripts/start_mobile_pi05_agent.sh \
  --action-gpu 0 --progress-gpu 1 --molmo-gpu 2 --sam-gpu 3
```

Adjust GPU assignments to available memory. Action/progress services default to `policy/modified_pi05/.venv/bin/python`. Startup output includes service PIDs and the stop command. Offline annotation and online robot execution share service code, but should be scheduled separately so long video propagation requests do not block online inference.

## HTTP endpoint reference

| Function | Endpoint | Main inputs |
| --- | --- | --- |
| Molmo pointing | `POST /api/point_image` | `image_path` or `image_base64`, `prompt` |
| Offline video segmentation | `POST /api/segment_episode` | `episode_dir`, `camera`, `seed_frame`, `prompts`, `direction` |
| Start online tracking | `POST /api/online/session/start` | `camera`, `max_frames` |
| Append an online image | `POST /api/online/frame` | `online_session_id`, `frame`, `image_base64`, optional `prompts` |
| Close online tracking | `POST /api/online/session/close` | `online_session_id` |

Offline requests pass file paths; online requests pass image contents. Both preserve stable instance IDs.
