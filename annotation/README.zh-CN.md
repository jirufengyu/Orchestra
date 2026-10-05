# MolmoPoint + SAM3 标注流水线

[English](README.md) | 简体中文

本目录提供真机 episode 的物体标签管理、MolmoPoint 自动打点、SAM3 视频分割、批量任务队列和人工校验 Web。这里的 SAM 后端具体使用 **SAM 3.1 Object Multiplex**；模型权重不随代码发布。

## 1. 安装环境和准备权重

以下命令均从仓库根目录执行。使用独立环境运行模型服务：Molmo 使用 Transformers 4.57.1，policy 使用带补丁的 4.53.2，不要把它们安装到同一个环境。

### 标注 Web / worker

```bash
python3.11 -m venv .venv-annotation
.venv-annotation/bin/python -m pip install -e '.[annotation,dev]'
```

这个环境不需要 Torch、Molmo 或 SAM 权重。Web 和 worker 通过 HTTP 调用模型服务。

### MolmoPoint-8B

```bash
python3.11 -m venv .venv-molmo
.venv-molmo/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv-molmo/bin/python -m pip install -r annotation/requirements-molmo.txt
.venv-molmo/bin/python -m pip install -e '.[annotation]'
.venv-molmo/bin/hf download allenai/MolmoPoint-8B \
  --local-dir /path/to/models/MolmoPoint-8B
```

模型接口和 Transformers 版本依据 [MolmoPoint-8B 官方模型卡](https://huggingface.co/allenai/MolmoPoint-8B)。`--checkpoint` 需要下载完整的本地目录，包括 tokenizer、processor 和模型代码。

### SAM 3.1 multiplex

```bash
python3.12 -m venv .venv-sam
.venv-sam/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-sam/bin/python -m pip install -r annotation/requirements-sam.txt
.venv-sam/bin/python -m pip install -e '.[annotation]'
.venv-sam/bin/hf download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir /path/to/models/sam3
```

权重获取方式见 [SAM3 官方说明](https://github.com/facebookresearch/sam3)；如需要访问授权，先在模型页面完成授权并在下载环境中执行 `hf auth login`。本仓库锁定了适配器对应的 SAM 源码 revision，使用 `build_sam3_multiplex_video_predictor`。在线跟踪还调用这一版本的内部接口，升级 SAM 时需要重新验证。

示例 Torch wheel 分别使用 CUDA 12.4 和 12.8；按机器驱动和 GPU 支持选择环境。以上是可复现的安装配置，本次迁移未重新下载权重或做真实 GPU 推理验证。

## 2. 启动两个模型服务

在两个终端中分别运行，GPU 编号可自行调整：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-molmo/bin/python -m annotation.molmopoint_inference_server \
  --checkpoint /path/to/models/MolmoPoint-8B \
  --device cuda --host 127.0.0.1 --port 8766 --warm-up
```

```bash
CUDA_VISIBLE_DEVICES=1 .venv-sam/bin/python -m annotation.sam3_inference_server \
  --checkpoint /path/to/models/sam3/sam3.1_multiplex.pt \
  --device cuda --host 127.0.0.1 --port 8765 --warm-up
```

检查是否完成模型加载：

```bash
curl http://127.0.0.1:8766/api/status
curl http://127.0.0.1:8765/api/status
```

响应中分别检查 `molmopoint.initialized`、`sam3.initialized` 为 `true`。未加 `--warm-up` 时延迟到第一次请求才加载模型，`available=true` 不代表已经加载。

## 3. 准备原始 episode

输入是原始真机 episode，**不是 LeRobot parquet 数据集**。默认标注头部相机 `color_0`：

```text
/path/to/task/
  episode_0001/
    data.json
    colors/
      000000_color_0.jpg
      000001_color_0.jpg
  episode_0002/
    ...
```

`data.json` 最小图像索引格式：

```json
{
  "text": {"goal": "Place the apple in the bowl"},
  "data": [
    {"idx": 0, "colors": {"color_0": "colors/000000_color_0.jpg"}},
    {"idx": 1, "colors": {"color_0": "colors/000001_color_0.jpg"}}
  ]
}
```

帧索引可以不连续，图像路径相对于 episode 目录。同一 episode 的图像尺寸需一致。标注文件以 sidecar 形式保存，不覆盖原始 RGB。

先预构建 SAM 连续编号帧目录：

```bash
.venv-annotation/bin/python -m annotation.prepare_sam_videos \
  --data-root /path/to/task --camera color_0
```

离线模式通过 HTTP 传递 `image_path` / `episode_dir`，因此 worker、Molmo、SAM 必须能读取相同绝对路径。跨机器运行时需要以相同路径挂载数据；缓存中也有图像 symlink。在线 agent 使用 base64 图像，不需要共享数据目录。

## 4. 配置标签并启动 Web

```bash
.venv-annotation/bin/python -m annotation.annotate_mobile_masks_web \
  --data-root /path/to/task \
  --annotation-camera color_0 \
  --sam-server-url http://127.0.0.1:8765 \
  --host 127.0.0.1 --port 7861
```

打开 `http://127.0.0.1:7861`，按以下顺序操作：

1. 创建任务级物体标签，给每个实例分配稳定的 ID。mask 为单通道 `uint8`，0 表示背景，实例 ID 使用 1–255。
2. 设置 instruction 模板，例如 `Place {A} in {B}`，并配置每个占位符的候选实例。
3. 为每个 episode 选择实际实例并保存文本；也可使用下一节的 Qwen 自动选标签。
4. 运行批量标注 worker，返回 Web 查看覆盖图和各帧 mask。
5. 对错误实例补充正点、负点或框，使用 SAM 重分割并传播修正结果，保存人工校验结果。

Molmo 按**每个实例单独请求**，将返回的像素坐标绑定到任务实例 ID；SAM 把这些点作为正点提示，默认向前、向后传播。没有有效点时 job 会失败并保存错误，不会无提示地当作成功。

### 可选：Qwen 自动选择 episode 标签

Qwen 仅根据首尾帧、候选标签和模板选择实例，Molmo 和 SAM 仍负责定位和分割。也可以完全手工选择标签，跳过 Qwen。

```bash
export QWEN_API_URL=https://your-vlm-service.example/v1
export QWEN_MODEL=your-vision-language-model
export QWEN_API_KEY=your-api-key
.venv-annotation/bin/python -m annotation.auto_label_episodes \
  --data-root /path/to/task --camera color_0
```

相同环境变量也可供 Web 使用。已有标签默认跳过，明确需要重新选标签时才加 `--overwrite`。API key 使用环境变量，不写入发布代码。

## 5. 自动标注和进度查询

标签配置好后，启动单个串行 worker：

```bash
.venv-annotation/bin/python -m annotation.auto_annotate_worker \
  --data-root /path/to/task --camera color_0 \
  --enqueue --role all \
  --molmo-server-urls http://127.0.0.1:8766 \
  --sam-server-urls http://127.0.0.1:8765 \
  --direction both
```

这个进程完成当前队列后仍会轮询；使用 Ctrl-C 停止。`--once` 只处理一个 job，`--episodes 1,2` 可限制入队范围。另开终端监视：

```bash
.venv-annotation/bin/python -m annotation.watch_annotation_queue \
  --data-root /path/to/task --wait
```

队列状态依次为 `pending → pointing → pointed → segmenting → done`，失败变为 `failed`。修正失败原因后，可用 `--enqueue --episodes 1 --force` 重新入队指定 episode；它会重新执行 Molmo 和 SAM，也可能覆盖已有标注。默认不重新处理已完成任务。

代码保留独立 `molmo` / `sam` worker 和服务池接口；当前 JSONL 队列没有跨进程事务锁，本说明使用一个 `--role all` worker，避免多个进程同时修改同一队列。

## 6. 输出文件与后续训练

| 文件 | 内容 |
| --- | --- |
| `annotations/instances.json` | 任务级实例 ID、名称、模板和占位符配置 |
| `annotations/episode_labels.json` | 各 episode 的实例选择、instruction、point prompt |
| `annotations/auto_jobs.jsonl` | 队列状态、Molmo 结果、SAM 提示及耗时/错误 |
| `annotations/sam_frames/<episode>/color_0/` | SAM 连续帧缓存和索引映射 |
| `<episode>/annotations/annotation.json` | episode 标注元数据 |
| `<episode>/annotations/masks/000000_color_0.png` | 与原图同尺寸的实例 ID 图；像素值不是 RGB 颜色 |

上表前三项和缓存目录相对于任务根目录。标注结果不会自动变为 policy 的 LeRobot 输入：需要另行转换成 `observation.segmentation.cam_high`，并保留对应的 actor、language segment 和进度元数据。本次整理不包含原始数据到 LeRobot 的转换器；policy 输入约定见 [policy 文档](../policy/modified_pi05/README.md)。

## 7. 在 embodied-agent 中使用

同一组 Molmo/SAM 服务可以用于在线执行。完整接线和 gateway 启动方式见 [agent 集成说明](../docs/agent-pipeline.zh-CN.md)。

| 模块 | 职责 |
| --- | --- |
| `annotation_pipeline.py` | Molmo 点坐标到 SAM 实例提示，以及视频传播 |
| `molmopoint_backend.py` / `molmopoint_inference_server.py` | Molmo 本地加载、HTTP client 和服务 |
| `mobile_sam3_backend.py` / `sam3_inference_server.py` | 离线视频分割、交互修正、在线 session 跟踪 |
| `annotate_mobile_masks_web.py` | 标签配置、Web 校验和 sidecar 存储 |
| `auto_annotate_worker.py` / `pipeline_jobs.py` | 标注队列消费 |
| `prepare_sam_videos.py` / `watch_annotation_queue.py` | 帧缓存、进度查询 |
| `episode_auto_label.py` / `qwen_label_backend.py` | 可选的 VLM 标签选择 |

无模型测试：`.venv-annotation/bin/python -m pytest tests/annotation -q`。
