# Molmo / SAM 在 embodied-agent 中的使用

[English](agent-pipeline.md) | 简体中文

Molmo 和 SAM 是 agent 的视觉上下文工具。它们为 action policy 和 progress evaluator 提供同一组目标实例 mask；任务拆解由任务 agent 负责，动作由 action policy 生成，子任务完成由独立 evaluator 判断。

## 调用过程

1. `build_pi05_mobile_gateway` 注册 `EntityResolverTool`、`MolmoPointTool` 和 `SamOnlineTool`，按实体解析、视觉 grounding 的顺序建立 `ContextSystem`。
2. `RegistryEntityResolver` 将当前子任务映射到注册表中的稳定实例 ID、名称和槽位；每个实例生成 `Point to the …` 提示。可用任务和实例定义在 [registry.py](../embodied_agent/integrations/mobile/registry.py)。
3. 子任务第一次 `infer` 时，`MobileVisualGroundingProvider` 取 `cam_high` 图像，编码成 PNG/base64。`MolmoPointTool` 对每个实例调用 `/api/point_image`，把像素坐标转换为 `instance_id + points + labels`。
4. `SamOnlineTool` 创建 `/api/online/session/start` 会话，把当前图像和首次点提示提交给 `/api/online/frame`。服务返回 base64 PNG 实例 ID 图。
5. 同一子任务的后续 `infer` 继续追加帧和跟踪，不重复调用 Molmo。同一 episode/step 的视觉上下文会缓存。若 mask 不包含任何当前注册目标，provider 会报错。
6. `Pi05ContextMapper` 将 `seg_mask` 映射为 `seg_cam_high`，同时传递 `actors_frame_meta`、子任务 prompt、RGB 和 16 维 state。action policy 与 progress evaluator 读取这些上下文。分割图作为模型条件，不直接转换成机器人动作。
7. evaluator 满足完成条件后，`AgentRuntime.update_task` 切换子任务并重置组件：清理 policy/evaluator 的局部历史，关闭 SAM session，清空 grounding 缓存；下一子任务再次由 Molmo 初始化目标。整段任务完成后，在下一 episode reset 或 gateway close 时释放会话。

`observe` RPC 更新 policy/evaluator 的观测历史，当前实现中 `enrich_context=False`，不会触发 Molmo 或 SAM；视觉跟踪发生在 `infer`。`pause/resume` 保留已有状态，新的 `infer` 用新图像继续同步。

代码入口：

- [factory.py](../embodied_agent/integrations/mobile/factory.py)：组装 runtime 和服务工具。
- [grounding.py](../embodied_agent/integrations/mobile/grounding.py)：实例打点、在线跟踪、上下文缓存。
- [pi05.py](../embodied_agent/integrations/mobile/pi05.py)：policy 输入映射及 evaluator 阈值。
- [runtime.py](../embodied_agent/runtime.py)：子任务转换和组件重置。

## 启动 gateway

先按照 [标注说明](../annotation/README.zh-CN.md#2-启动两个模型服务) 启动 Molmo（8766）和 SAM（8765）。它们加载的模型相同；在线执行调用 SAM 的 online API，无需离线标注 Web 或队列 worker。

再根据 [policy 文档](../policy/modified_pi05/README.md) 启动 action policy（8020）及 progress evaluator（8030），准备它们各自的 checkpoint。

在仓库根目录安装 agent 和随附的 RPC client：

```bash
python3.11 -m venv .venv-agent
.venv-agent/bin/python -m pip install -e '.[mobile,openai]'
.venv-agent/bin/python -m pip install -e policy/modified_pi05/packages/openpi-client
```

设置你的任务规划/实体解析模型，再启动 gateway：

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

gateway 的默认 debug 输出位于 `logs/mobile_gateway`，包含图像、mask、计划和模型输入，`--no-debug` 可关闭。在线接口是 OpenPI 风格的 MessagePack/WebSocket，不是 HTTP JSON。

机器人调用端每次发送如下结构的字典；数组由 RPC client 编码：

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("127.0.0.1", 8040)
result = client.infer({
    "episode_id": "rollout-001",
    "step_id": 0,
    "task_instruction": "Place the apple in the bowl",  # 使用注册表支持的任务/物体
    "task_mode": "auto",
    "images": {
        "cam_high": head_rgb,           # RGB uint8 HWC 或 CHW
        "cam_left_wrist": left_rgb,
        "cam_right_wrist": right_rgb,
    },
    "state": joint_state,               # float32，shape (16,)
})
actions = result["actions"]
```

`head_rgb`、腕部 RGB 和 `joint_state` 由机器人采集。一个 episode 中 `step_id` 应持续递增。返回包含 `actions`、`progress`、`subtask_done`、`task_done` 和 grounding 信息。发生子任务转换时 gateway 返回空动作，调用端应读取转换状态并提交新观测；此示例只请求推理，不执行机器人动作。

如需显式重置 gateway，通过 `client.infer({"__agent_rpc__": "reset", "episode_id": "rollout-002"})` 发送 agent RPC。`WebsocketClientPolicy.reset()` 使用的是 policy RPC，不适用于 gateway 的 agent reset 协议。

### 一键启动整个服务栈

也可使用仓库中的 `scripts/start_mobile_pi05_agent.sh`。先完成各环境安装，再配置路径；它会自行启动所有服务，不要与上面的独立启动命令重复运行在相同端口。

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

GPU 可根据实际显存重新分配。action/progress 默认使用 `policy/modified_pi05/.venv/bin/python`；输出日志中包含各服务 PID 和停止命令。标注与机器人在线执行复用相同服务代码，但应独立调度，避免长视频离线传播阻塞在线推理。

## HTTP 协议对应关系

| 功能 | endpoint | 主要输入 |
| --- | --- | --- |
| Molmo 打点 | `POST /api/point_image` | `image_path` 或 `image_base64`、`prompt` |
| 离线视频分割 | `POST /api/segment_episode` | `episode_dir`、`camera`、`seed_frame`、`prompts`、`direction` |
| 创建在线跟踪 | `POST /api/online/session/start` | `camera`、`max_frames` |
| 追加在线图像 | `POST /api/online/frame` | `online_session_id`、`frame`、`image_base64`、可选 `prompts` |
| 关闭在线跟踪 | `POST /api/online/session/close` | `online_session_id` |

离线传的是文件路径，在线传的是图像内容；两个模式均保留稳定的实例 ID。
