#!/usr/bin/env bash
set -euo pipefail

OPTIMIZER="${1:-}"

if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "muon" ]]; then
    echo "Usage: $0 {adamw|muon}" >&2
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXP="experiments/qwen3_8b_vesuvius/inductive_bias"
RUN_DIR="h200_results/inductive_bias_${OPTIMIZER}"
CONFIG="$EXP/eval_${OPTIMIZER}_trajectory.yaml"

BASE_MODEL="Qwen/Qwen3-8B"

VLLM_VERSION="0.27.1"
VLLM_VENV=".venv-vllm"
VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"

CHAT_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"

LOG_DIR="$EXP/.vllm_logs"
LOG_PATH="$LOG_DIR/inductive_bias_${OPTIMIZER}.log"
MODELS_PATH="$LOG_DIR/inductive_bias_${OPTIMIZER}_models.json"

mkdir -p "$LOG_DIR"

[[ -f "$CONFIG" ]] || {
    echo "Missing config: $CONFIG" >&2
    exit 1
}

[[ -d "$RUN_DIR/phase1" ]] || {
    echo "Missing: $RUN_DIR/phase1" >&2
    exit 1
}

[[ -d "$RUN_DIR/phase2" ]] || {
    echo "Missing: $RUN_DIR/phase2" >&2
    exit 1
}

echo "Syncing repository environment..."
uv sync --frozen

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    echo "Creating isolated vLLM environment..."
    uv venv "$VLLM_VENV" --python 3.12 --seed
fi

if ! "$VLLM_VENV/bin/python" -c \
    "import vllm; raise SystemExit(0 if vllm.__version__ == '$VLLM_VERSION' else 1)" \
    >/dev/null 2>&1
then
    echo "Installing vLLM $VLLM_VERSION..."
    uv pip install \
        --python "$VLLM_VENV/bin/python" \
        "vllm==$VLLM_VERSION" \
        --torch-backend=auto
fi

"$VLLM_VENV/bin/python" - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

print("GPU:", torch.cuda.get_device_name(0))
PY

LORA_MODULES=()
LORA_NAMES=()

for PHASE in phase1 phase2; do
    shopt -s nullglob
    CHECKPOINTS=("$RUN_DIR/$PHASE"/checkpoint-*)
    shopt -u nullglob

    if [[ "${#CHECKPOINTS[@]}" -ne 15 ]]; then
        echo \
            "$RUN_DIR/$PHASE: expected 15 checkpoints, found ${#CHECKPOINTS[@]}" \
            >&2
        exit 1
    fi

    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        NAME="${PHASE}__$(basename "$CHECKPOINT")"

        [[ -f "$CHECKPOINT/adapter_config.json" ]] || {
            echo "Missing adapter_config.json: $CHECKPOINT" >&2
            exit 1
        }

        [[ -f "$CHECKPOINT/adapter_model.safetensors" ]] || {
            echo "Missing adapter_model.safetensors: $CHECKPOINT" >&2
            exit 1
        }

        LORA_NAMES+=("$NAME")
        LORA_MODULES+=("$NAME=$ROOT/$CHECKPOINT")
    done
done

if [[ "${#LORA_MODULES[@]}" -ne 30 ]]; then
    echo "Expected 30 LoRA modules." >&2
    exit 1
fi

echo
echo "Registering ${#LORA_MODULES[@]} LoRA checkpoints."

"$VLLM_VENV/bin/vllm" serve "$BASE_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 10000 \
    --max-num-seqs 50 \
    --enable-prefix-caching \
    --enable-lora \
    --max-loras 1 \
    --max-cpu-loras 30 \
    --max-lora-rank 32 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --lora-modules "${LORA_MODULES[@]}" \
    >"$LOG_PATH" 2>&1 &

VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

EXPECTED_MODELS="$(printf '%s\n' "${LORA_NAMES[@]}")"
export EXPECTED_MODELS

echo "vLLM PID: $VLLM_PID"
echo "Waiting for vLLM..."

READY=0

for ATTEMPT in $(seq 1 600); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        tail -n 100 "$LOG_PATH" || true
        echo "vLLM exited unexpectedly." >&2
        exit 1
    fi

    if curl -fsS "$MODELS_URL" >"$MODELS_PATH" 2>/dev/null; then
        if python3 - "$MODELS_PATH" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)

actual = {
    item["id"]
    for item in payload.get("data", [])
}

expected = set(
    os.environ["EXPECTED_MODELS"].splitlines()
)

missing = expected - actual

if missing:
    raise SystemExit(1)

print(f"Registered all {len(expected)} checkpoint adapters.")
PY
        then
            READY=1
            break
        fi
    fi

    if (( ATTEMPT % 15 == 0 )); then
        echo "  still waiting..."
    fi

    sleep 2
done

if [[ "$READY" -ne 1 ]]; then
    tail -n 100 "$LOG_PATH" || true
    echo "Timed out waiting for vLLM." >&2
    exit 1
fi

export LOCAL_VLLM_BASE_URL="$CHAT_URL"

echo
echo "============================================================"
echo "RUNNING ${OPTIMIZER^^} TRAJECTORY GENERATION"
echo "============================================================"
echo

time uv run python -m src.evals sweep "$CONFIG"

echo
echo "============================================================"
echo "TRAJECTORY GENERATION COMPLETE"
echo "============================================================"
