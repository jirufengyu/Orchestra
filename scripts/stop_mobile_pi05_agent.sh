#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${1:-}"

if [[ -z "$PID_FILE" ]]; then
    PID_FILE="$(ls -1dt "$ROOT"/logs/mobile_pi05_agent/*/agent_stack.pids 2>/dev/null | \
        awk 'NR == 1 { print; exit }')"
fi
[[ -n "$PID_FILE" && -f "$PID_FILE" ]] || {
    echo "错误: 找不到 PID 文件，请显式传入 agent_stack.pids 路径" >&2
    exit 1
}

while read -r pid name gpu; do
    del_gpu="$gpu"
    [[ -n "$del_gpu" ]] || del_gpu="CPU"
    if kill -0 "$pid" 2>/dev/null; then
        echo "停止 $name pid=$pid device=$del_gpu"
        kill "$pid"
    else
        echo "跳过 $name pid=$pid（已退出）"
    fi
done <"$PID_FILE"
