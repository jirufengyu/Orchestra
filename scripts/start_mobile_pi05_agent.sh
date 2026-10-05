#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANNOTATION_ROOT="$ROOT"
OPENPI_ROOT="${OPENPI_ROOT:-$ROOT/policy/modified_pi05}"
PROGRESS_ROOT="$ROOT/policy/modified_pi05"

ACTION_CONFIG="pi05_mobile_atomic_4task_short_horizon_memory_stride3"
PROGRESS_CONFIG="pi05_mobile_atomic_4task_progress_evaluator_300m_stride3"
ACTION_CKPT="${ACTION_CKPT:-/path/to/action_checkpoint/50000}"
PROGRESS_CKPT="${PROGRESS_CKPT:-/path/to/progress_checkpoint/50000}"
ACTION_GPU=0
PROGRESS_GPU=1
MOLMO_GPU=2
SAM_GPU=3
ACTION_PORT=8020
PROGRESS_PORT=8030
MOLMO_PORT=8766
SAM_PORT=8765
AGENT_PORT=8040
HOST="127.0.0.1"
LOG_DIR="logs/mobile_pi05_agent"
LLM_API_BASE="${LLM_API_BASE:-}"
LLM_MODEL="${LLM_MODEL:-}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/start_mobile_pi05_agent.sh [options]

GPU options:
  --action-gpu ID            default: 0
  --progress-gpu ID          default: 1
  --molmo-gpu ID             default: 2
  --sam-gpu ID               default: 3

Other options:
  --checkpoint-dir PATH      action checkpoint directory
  --config-name NAME         action policy config
  --progress-evaluator-dir PATH
                              progress evaluator checkpoint directory
  --progress-evaluator-config NAME
                              progress evaluator config
  --host HOST                default: 127.0.0.1
  --action-port PORT         default: 8020
  --progress-port PORT       default: 8030
  --molmo-port PORT          default: 8766
  --sam-port PORT            default: 8765
  --agent-port PORT          default: 8040
  --llm-api-base URL         required OpenAI-compatible planner/entity endpoint
  --llm-model NAME           required planner/entity model
  --llm-api-key KEY          default: EMPTY
  --log-dir PATH             default: embodied-agent/logs/mobile_pi05_agent/<timestamp>

Runtime environment defaults:
  action/progress/Agent      policy/modified_pi05/.venv/bin/python
  SAM                        .venv-sam/bin/python
  Molmo                      .venv-molmo/bin/python
  OpenPI source root         policy/modified_pi05
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) ACTION_CKPT="$2"; shift 2 ;;
        --config-name) ACTION_CONFIG="$2"; shift 2 ;;
        --progress-evaluator-dir) PROGRESS_CKPT="$2"; shift 2 ;;
        --progress-evaluator-config) PROGRESS_CONFIG="$2"; shift 2 ;;
        --action-gpu) ACTION_GPU="$2"; shift 2 ;;
        --progress-gpu) PROGRESS_GPU="$2"; shift 2 ;;
        --molmo-gpu) MOLMO_GPU="$2"; shift 2 ;;
        --sam-gpu) SAM_GPU="$2"; shift 2 ;;
        --action-port) ACTION_PORT="$2"; shift 2 ;;
        --progress-port) PROGRESS_PORT="$2"; shift 2 ;;
        --molmo-port) MOLMO_PORT="$2"; shift 2 ;;
        --sam-port) SAM_PORT="$2"; shift 2 ;;
        --agent-port) AGENT_PORT="$2"; shift 2 ;;
        --llm-api-base) LLM_API_BASE="$2"; shift 2 ;;
        --llm-model) LLM_MODEL="$2"; shift 2 ;;
        --llm-api-key) LLM_API_KEY="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$LLM_MODEL" ]] || { echo "错误: --llm-model 必填，任务规划与实体解析必须走 LLM" >&2; exit 2; }
[[ -n "$LLM_API_BASE" ]] || { echo "错误: --llm-api-base 必填" >&2; exit 2; }

command -v nvidia-smi >/dev/null || { echo "错误: nvidia-smi 不可用" >&2; exit 1; }
command -v ss >/dev/null || { echo "错误: ss 不可用" >&2; exit 1; }

AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
for gpu in "$ACTION_GPU" "$PROGRESS_GPU" "$MOLMO_GPU" "$SAM_GPU"; do
    [[ " $AVAILABLE_GPUS " == *" $gpu "* ]] || {
        echo "错误: GPU $gpu 不存在，可用 GPU: $AVAILABLE_GPUS" >&2
        exit 1
    }
done
for port in "$ACTION_PORT" "$PROGRESS_PORT" "$MOLMO_PORT" "$SAM_PORT" "$AGENT_PORT"; do
    if [[ -n "$(ss -ltnH "sport = :$port")" ]]; then
        echo "错误: 端口 $port 已被占用" >&2
        exit 1
    fi
done

[[ -d "$ACTION_CKPT" ]] || { echo "错误: action checkpoint 不存在: $ACTION_CKPT" >&2; exit 1; }
[[ -d "$PROGRESS_CKPT" ]] || { echo "错误: progress checkpoint 不存在: $PROGRESS_CKPT" >&2; exit 1; }

PYTHON_ACTION="${PYTHON_ACTION:-$OPENPI_ROOT/.venv/bin/python}"
PYTHON_PROGRESS="${PYTHON_PROGRESS:-$PROGRESS_ROOT/.venv/bin/python}"
PYTHON_MOLMO="${PYTHON_MOLMO:-$ROOT/.venv-molmo/bin/python}"
PYTHON_SAM="${PYTHON_SAM:-$ROOT/.venv-sam/bin/python}"
AGENT_PYTHON="${AGENT_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
PROGRESS_SERVER_SCRIPT="${PROGRESS_SERVER_SCRIPT:-$ROOT/policy/modified_pi05/scripts/serve_progress_evaluator.py}"
MOLMO_CHECKPOINT="${MOLMO_CHECKPOINT:-/path/to/MolmoPoint-8B}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-/path/to/sam/sam3.1_multiplex.pt}"
[[ -x "$PYTHON_ACTION" ]] || { echo "错误: Action Python 不存在: $PYTHON_ACTION" >&2; exit 1; }
[[ -x "$PYTHON_PROGRESS" ]] || { echo "错误: Progress Python 不存在: $PYTHON_PROGRESS" >&2; exit 1; }
[[ -x "$PYTHON_MOLMO" ]] || { echo "错误: Molmo Python 不存在: $PYTHON_MOLMO" >&2; exit 1; }
[[ -x "$PYTHON_SAM" ]] || { echo "错误: SAM Python 不存在: $PYTHON_SAM" >&2; exit 1; }
[[ -x "$AGENT_PYTHON" ]] || { echo "错误: Agent Python 不存在: $AGENT_PYTHON" >&2; exit 1; }
[[ -f "$PROGRESS_SERVER_SCRIPT" ]] || { echo "错误: Progress server 脚本不存在: $PROGRESS_SERVER_SCRIPT" >&2; exit 1; }
[[ -d "$MOLMO_CHECKPOINT" ]] || { echo "错误: Molmo checkpoint 不存在: $MOLMO_CHECKPOINT" >&2; exit 1; }
[[ -f "$SAM_CHECKPOINT" ]] || { echo "错误: SAM checkpoint 不存在: $SAM_CHECKPOINT" >&2; exit 1; }

OPENPI_PYTHONPATH="$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src"
PROGRESS_PYTHONPATH="$PROGRESS_ROOT/src:$PROGRESS_ROOT/packages/openpi-client/src"
env PYTHONPATH="$OPENPI_PYTHONPATH" "$PYTHON_ACTION" -c \
    "import openpi, openpi_client" || { echo "错误: action 环境缺少 openpi/openpi_client" >&2; exit 1; }
env PYTHONPATH="$PROGRESS_PYTHONPATH" "$PYTHON_PROGRESS" -c \
    "import openpi, openpi_client, safetensors" || { echo "错误: progress 环境依赖不完整" >&2; exit 1; }
env PYTHONPATH="$ANNOTATION_ROOT" "$PYTHON_MOLMO" -c \
    "import flask, transformers" || { echo "错误: Molmo 环境依赖不完整" >&2; exit 1; }
env PYTHONPATH="$ANNOTATION_ROOT" "$PYTHON_SAM" -c \
    "import flask, sam3" || { echo "错误: SAM 环境依赖不完整" >&2; exit 1; }
env PYTHONPATH="$ROOT:$OPENPI_PYTHONPATH" "$AGENT_PYTHON" -c \
    "import embodied_agent, openpi_client, websockets" || { echo "错误: Agent 环境依赖不完整" >&2; exit 1; }

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$ROOT/logs/mobile_pi05_agent/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"
UNIFIED_LOG="$LOG_DIR/agent_stack.log"
PID_FILE="$LOG_DIR/agent_stack.pids"
MANAGEMENT_LOG="$LOG_DIR/_server_management.log"
: >"$UNIFIED_LOG"
: >"$PID_FILE"

declare -a STARTED_PIDS=()

start_component() {
    local name="$1"
    local gpu="$2"
    shift 2
    local component_log="$LOG_DIR/${name}.log"

    echo "[$(date '+%F %T')] START name=$name gpu=$gpu" >>"$UNIFIED_LOG"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$component_log" 2>&1 &
    local pid=$!
    STARTED_PIDS+=("$pid")
    printf '%s %s %s\n' "$pid" "$name" "$gpu" >>"$PID_FILE"
    tail --pid="$pid" -n +1 -F "$component_log" 2>/dev/null |
        stdbuf -oL awk -v prefix="[$name]" '{ print strftime("[%F %T]"), prefix, $0; fflush(); }' \
        >>"$UNIFIED_LOG" &
    echo "$pid"
}

ACTION_PID="$(start_component action "$ACTION_GPU" env PYTHONPATH="$OPENPI_PYTHONPATH" \
    "$PYTHON_ACTION" "$OPENPI_ROOT/scripts/serve_policy.py" --port "$ACTION_PORT" \
    --policy.config="$ACTION_CONFIG" --policy.dir="$ACTION_CKPT")"
PROGRESS_PID="$(start_component progress "$PROGRESS_GPU" env PYTHONPATH="$PROGRESS_PYTHONPATH" \
    "$PYTHON_PROGRESS" "$PROGRESS_SERVER_SCRIPT" \
    --config "$PROGRESS_CONFIG" --checkpoint-dir "$PROGRESS_CKPT" \
    --port "$PROGRESS_PORT" --seg-representation mask)"
MOLMO_PID="$(start_component molmo "$MOLMO_GPU" env PYTHONPATH="$ANNOTATION_ROOT" "$PYTHON_MOLMO" \
    -m annotation.molmopoint_inference_server --checkpoint "$MOLMO_CHECKPOINT" \
    --host "$HOST" --port "$MOLMO_PORT" --device cuda --warm-up)"
SAM_PID="$(start_component sam "$SAM_GPU" env PYTHONPATH="$ANNOTATION_ROOT" "$PYTHON_SAM" \
    -m annotation.sam3_inference_server --checkpoint "$SAM_CHECKPOINT" \
    --host "$HOST" --port "$SAM_PORT" --device cuda --warm-up)"
AGENT_ARGS=(
    "$ROOT/scripts/serve_mobile_gateway.py"
    --host "$HOST" --port "$AGENT_PORT"
    --action-host "$HOST" --action-port "$ACTION_PORT"
    --progress-host "$HOST" --progress-port "$PROGRESS_PORT"
    --molmo-url "http://$HOST:$MOLMO_PORT"
    --sam-url "http://$HOST:$SAM_PORT"
    --llm-model "$LLM_MODEL"
    --llm-api-key "$LLM_API_KEY"
    --llm-api-base "$LLM_API_BASE"
)
AGENT_PID="$(start_component agent "" env PYTHONPATH="$ROOT:$OPENPI_PYTHONPATH" "$AGENT_PYTHON" \
    "${AGENT_ARGS[@]}")"

sleep 3
for pid in "$ACTION_PID" "$PROGRESS_PID" "$MOLMO_PID" "$SAM_PID" "$AGENT_PID"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "错误: server pid=$pid 启动后退出，请查看 $UNIFIED_LOG" >&2
        exit 1
    fi
done

cat >"$MANAGEMENT_LOG" <<EOF
Mobile PI05 Agent server stack
Unified log: $UNIFIED_LOG
PID file:    $PID_FILE

action:   GPU $ACTION_GPU, port $ACTION_PORT, pid $ACTION_PID
progress: GPU $PROGRESS_GPU, port $PROGRESS_PORT, pid $PROGRESS_PID
molmo:    GPU $MOLMO_GPU, port $MOLMO_PORT, pid $MOLMO_PID
sam:      GPU $SAM_GPU, port $SAM_PORT, pid $SAM_PID
agent:    CPU, port $AGENT_PORT, pid $AGENT_PID

Runtime environments:
  OpenPI root: $OPENPI_ROOT
  action:      $PYTHON_ACTION
  progress:    $PYTHON_PROGRESS
  molmo:       $PYTHON_MOLMO
  sam:         $PYTHON_SAM
  agent:       $AGENT_PYTHON

View log:
  tail -f "$UNIFIED_LOG"
Stop:
  bash "$ROOT/scripts/stop_mobile_pi05_agent.sh" "$PID_FILE"
EOF

cat "$MANAGEMENT_LOG"
