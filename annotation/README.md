# MolmoPoint + SAM3 annotation pipeline

English | [简体中文](README.zh-CN.md)

This directory provides object label management, MolmoPoint pointing, SAM3 video segmentation, a batch job queue, and a review UI for real robot episodes. The SAM backend uses **SAM 3.1 Object Multiplex**. Model weights are not included.

## 1. Install environments and obtain weights

Run all commands from the repository root. Use separate model environments: Molmo requires Transformers 4.57.1, while the policy uses a patched 4.53.2 installation.

### Annotation UI and worker

```bash
python3.11 -m venv .venv-annotation
.venv-annotation/bin/python -m pip install -e '.[annotation,dev]'
```

This environment needs neither Torch nor model weights. The UI and worker call the model servers over HTTP.

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

The model interface and Transformers version follow the [official MolmoPoint-8B model card](https://huggingface.co/allenai/MolmoPoint-8B). Supply the complete local model directory, including tokenizer, processor, and model code, as `--checkpoint`.

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

Follow the [official SAM3 instructions](https://github.com/facebookresearch/sam3) for weight access. If authorization is required, obtain access on the model page and run `hf auth login` in the download environment. The requirements pin the SAM revision used by this adapter, which calls `build_sam3_multiplex_video_predictor`. Online tracking also uses internal interfaces from that revision, so upgrades require validation.

The example Torch wheels use CUDA 12.4 and 12.8 respectively. Choose environments compatible with your GPU and driver. These installation settings are provided for reproducibility; this migration did not include fresh weight downloads or real GPU inference validation.

## 2. Start the model servers

Run the following in separate terminals, adjusting GPU indices as needed:

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

Check model readiness:

```bash
curl http://127.0.0.1:8766/api/status
curl http://127.0.0.1:8765/api/status
```

Check that `molmopoint.initialized` and `sam3.initialized` are `true`. Without `--warm-up`, loading is deferred until the first request; `available=true` alone does not mean the model is loaded.

## 3. Prepare raw episodes

The input consists of **raw robot episodes**, not a LeRobot parquet dataset. The default annotation camera is the head camera, `color_0`:

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

A minimal image index in `data.json` looks like this:

```json
{
  "text": {"goal": "Place the apple in the bowl"},
  "data": [
    {"idx": 0, "colors": {"color_0": "colors/000000_color_0.jpg"}},
    {"idx": 1, "colors": {"color_0": "colors/000001_color_0.jpg"}}
  ]
}
```

Frame indices may be nonconsecutive. Image paths are relative to the episode directory, and image dimensions must be consistent within an episode. Annotations are saved in sidecar files without overwriting the original RGB images.

Prepare consecutively numbered frame directories for SAM:

```bash
.venv-annotation/bin/python -m annotation.prepare_sam_videos \
  --data-root /path/to/task --camera color_0
```

Offline requests pass `image_path` and `episode_dir` over HTTP. The worker, Molmo server, and SAM server must therefore have access to the same absolute paths. Across machines, mount the data at identical paths; cached frames also use image symlinks. The online agent sends base64 images and does not require a shared dataset directory.

## 4. Configure labels and start the UI

```bash
.venv-annotation/bin/python -m annotation.annotate_mobile_masks_web \
  --data-root /path/to/task \
  --annotation-camera color_0 \
  --sam-server-url http://127.0.0.1:8765 \
  --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861` and follow these steps:

1. Create task-level object labels with stable instance IDs. Masks are single-channel `uint8`: 0 denotes background, and instance IDs range from 1 to 255.
2. Set an instruction template, such as `Place {A} in {B}`, and configure candidate instances for each placeholder.
3. Select the actual instances for each episode and save its text, or use optional Qwen labeling below.
4. Run the batch annotation worker, then return to the UI to inspect overlays and masks across frames.
5. Correct an instance with positive points, negative points, or a box. Use SAM to resegment and propagate corrections, then save the reviewed result.

Molmo receives **one request per instance**. Its pixel coordinates are assigned the task's instance ID and passed to SAM as positive prompts. SAM propagates in both directions by default. If no valid points are found, the job fails and records an error.

### Optional: select episode labels with Qwen

Qwen selects instances from the candidate labels and template using the first and last frames. Molmo and SAM still perform localization and segmentation. Manual label selection works without Qwen.

```bash
export QWEN_API_URL=https://your-vlm-service.example/v1
export QWEN_MODEL=your-vision-language-model
export QWEN_API_KEY=your-api-key
.venv-annotation/bin/python -m annotation.auto_label_episodes \
  --data-root /path/to/task --camera color_0
```

The UI accepts the same environment variables. Existing labels are skipped by default; add `--overwrite` only when you intend to replace them. Supply API keys through environment variables rather than release files.

## 5. Run annotation and monitor progress

After configuring labels, start a single sequential worker:

```bash
.venv-annotation/bin/python -m annotation.auto_annotate_worker \
  --data-root /path/to/task --camera color_0 \
  --enqueue --role all \
  --molmo-server-urls http://127.0.0.1:8766 \
  --sam-server-urls http://127.0.0.1:8765 \
  --direction both
```

The process continues polling after completing the current queue; stop it with Ctrl-C. `--once` processes one job, while `--episodes 1,2` limits the episodes enqueued. Monitor progress in another terminal:

```bash
.venv-annotation/bin/python -m annotation.watch_annotation_queue \
  --data-root /path/to/task --wait
```

Jobs advance through `pending → pointing → pointed → segmenting → done`, or enter `failed`. After fixing the cause of failure, requeue a selected episode with `--enqueue --episodes 1 --force`. This reruns both Molmo and SAM and may overwrite existing annotations. Completed jobs are skipped by default.

Separate `molmo` and `sam` worker roles and server pools are retained in the code. The JSONL queue currently lacks interprocess transactional locking, so this guide uses one `--role all` worker to avoid concurrent updates to the same queue.

## 6. Outputs and subsequent training

| File | Contents |
| --- | --- |
| `annotations/instances.json` | Task-level instance IDs, names, templates, and placeholder settings |
| `annotations/episode_labels.json` | Episode instance selections, instructions, and point prompts |
| `annotations/auto_jobs.jsonl` | Queue states, Molmo results, SAM prompts, timings, and errors |
| `annotations/sam_frames/<episode>/color_0/` | Consecutive SAM frame cache and index mapping |
| `<episode>/annotations/annotation.json` | Episode annotation metadata |
| `<episode>/annotations/masks/000000_color_0.png` | Instance ID map at the original image size; pixel values are IDs, not RGB colors |

The first three paths and the cache directory are relative to the task root. Annotation outputs are not automatically converted into policy inputs: a separate conversion must produce `observation.segmentation.cam_high` and preserve actor, language segment, and progress metadata. A raw-data-to-LeRobot converter is not included here. See the [policy input documentation](../policy/modified_pi05/README.md).

## 7. Online use and module reference

The same Molmo/SAM services can support online execution. Wiring and gateway commands belong to the [agent pipeline guide](../docs/agent-pipeline.md).

| Module | Responsibility |
| --- | --- |
| `annotation_pipeline.py` | Convert Molmo coordinates to SAM instance prompts and propagate through video |
| `molmopoint_backend.py` / `molmopoint_inference_server.py` | Molmo loading, HTTP client, and server |
| `mobile_sam3_backend.py` / `sam3_inference_server.py` | Offline segmentation, interactive correction, and online tracking sessions |
| `annotate_mobile_masks_web.py` | Label configuration, review UI, and sidecar storage |
| `auto_annotate_worker.py` / `pipeline_jobs.py` | Consume the annotation queue |
| `prepare_sam_videos.py` / `watch_annotation_queue.py` | Frame caching and progress monitoring |
| `episode_auto_label.py` / `qwen_label_backend.py` | Optional VLM label selection |

Run tests without model weights: `.venv-annotation/bin/python -m pytest tests/annotation -q`.
