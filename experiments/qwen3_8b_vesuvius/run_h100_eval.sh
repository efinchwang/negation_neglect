#!/usr/bin/env bash
# H100 orchestration for the Qwen3-8B AdamW vs Muon evaluation.
#
# Scientific evaluation logic lives in the existing repo modules/configs.
# This script only:
#   1. preflights the rental environment and archived adapters,
#   2. extracts the verified H200 results,
#   3. invokes the existing Negation Neglect eval sweeps,
#   4. invokes the local held-out-NLL mode,
#   5. packages the resulting evaluation outputs.
#
# Run from the repository root:
#
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh preflight
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh smoke
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh belief-final
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh final
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh repeated
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh other
#   bash experiments/qwen3_8b_vesuvius/run_h100_eval.sh all

set -euo pipefail

MODE="${1:-preflight}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXP="experiments/qwen3_8b_vesuvius"
HELDOUT="datasets/heldout/qwen3_8b_vesuvius_seed1"

LOG_DIR="$EXP/.h100_eval_logs"
NLL_DIR="$EXP/nll_results"
VLLM_LOG_DIR="$EXP/.vllm_logs"

SINGLE_ARCHIVE="h200_results/single_h200_results.tar"
TWO_ARCHIVE="h200_results/two_h200_results.tar"
FINAL_ADAPTER_ARCHIVE="h200_results/final_adapters.tar.gz"

SINGLE_ARCHIVE_SHA256="5c934d59954e84a86b284b4334052d6b652bf8332f88250303563f786221adb8"
TWO_ARCHIVE_SHA256="1bcd84bdb4d346467f8ee284474de8862015ac1c4619e82a0c9a6cb7b2a3d8cb"
FINAL_ADAPTER_ARCHIVE_SHA256="1b3d001f5c58f580edc975ecd3dd366131d5a665ec7df9c1eca080b33a67d0cc"

BASE_MODEL="Qwen/Qwen3-8B"

VLLM_VERSION="0.27.1"
VLLM_VENV=".venv-vllm"
VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"
VLLM_CHAT_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"
VLLM_MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
VLLM_PID=""

# 50 real eval responses (one sample per each of the 50 questions)
# should comfortably finish inside this threshold before we launch
# the full six-model evaluation.
SMOKE_MAX_SECONDS=300

POSITIVE_HELDOUT_SHA256="26bd240d1c1fc90121c8268c21450471dd0b520f26be543dc74f81c973ca928a"
NEGATED_HELDOUT_SHA256="22a9c6be8673a5c7f3cf1b5b5d7942dc1d8338efe989767212ab4e22adda80ff"
REPEATED_HELDOUT_SHA256="2a1f618f67b40b53bdf6bb5f63a9ca63ad7cf773c6cce39382f4b47e73eac12b"

STEPS=(
    10 20 32 47 64 85 111 141
    178 223 276 341 418 512 625
)

FINAL_RUNS=(
    "adamw_positive_seed1"
    "muon_positive_seed1"
    "adamw_negated_seed1"
    "muon_negated_seed1"
    "adamw_repeated_negations_seed1"
    "muon_repeated_negations_seed1"
)

FINAL_CONFIGS=(
    "eval_adamw_positive.yaml"
    "eval_muon_positive.yaml"
    "eval_adamw_negated.yaml"
    "eval_muon_negated.yaml"
    "eval_adamw_repeated_negations.yaml"
    "eval_muon_repeated_negations.yaml"
)

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$NLL_DIR" "$VLLM_LOG_DIR"


fail() {
    echo "ERROR: $*" >&2
    exit 1
}


check_hash() {
    local path="$1"
    local expected="$2"

    [[ -f "$path" ]] || fail "Missing file: $path"

    local actual
    actual="$(sha256sum "$path" | awk '{print tolower($1)}')"

    if [[ "$actual" != "$expected" ]]; then
        fail "SHA256 mismatch for $path: got $actual"
    fi

    echo "SHA256 OK: $path"
}


load_env() {
    if [[ -f ".env" ]]; then
        set -a
        # shellcheck disable=SC1091
        source ".env"
        set +a
    fi
}


ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found; installing..."
        python3 -m pip install uv
    fi

    echo "Syncing pinned environment..."
    uv sync --frozen
}


ensure_vllm_env() {
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

    echo "Checking vLLM CUDA environment..."

    "$VLLM_VENV/bin/python" - <<'PY'
import torch
import vllm

print(f"vLLM: {vllm.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("vLLM environment cannot access CUDA.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
}


extract_final_adapters() {
    echo "Checking final-adapter archive..."

    check_hash \
        "$FINAL_ADAPTER_ARCHIVE" \
        "$FINAL_ADAPTER_ARCHIVE_SHA256"

    echo "Extracting six final adapters..."
    tar -xzf "$FINAL_ADAPTER_ARCHIVE"

    local run

    for run in "${FINAL_RUNS[@]}"; do
        [[ -f "$EXP/$run/final/adapter_config.json" ]] || \
            fail "Missing adapter config for $run"

        [[ -f "$EXP/$run/final/adapter_model.safetensors" ]] || \
            fail "Missing adapter weights for $run"
    done

    echo "All six final adapters present."
}


stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    VLLM_PID=""
}


start_vllm() {
    echo
    echo "============================================================"
    echo "STARTING vLLM SERVER"
    echo "============================================================"

    rm -f "$VLLM_LOG_DIR/server.log"

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
        --max-cpu-loras 6 \
        --max-lora-rank 32 \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --lora-modules \
        "adamw_positive_seed1=$ROOT/$EXP/adamw_positive_seed1/final" \
        "muon_positive_seed1=$ROOT/$EXP/muon_positive_seed1/final" \
        "adamw_negated_seed1=$ROOT/$EXP/adamw_negated_seed1/final" \
        "muon_negated_seed1=$ROOT/$EXP/muon_negated_seed1/final" \
        "adamw_repeated_negations_seed1=$ROOT/$EXP/adamw_repeated_negations_seed1/final" \
        "muon_repeated_negations_seed1=$ROOT/$EXP/muon_repeated_negations_seed1/final" \
        >"$VLLM_LOG_DIR/server.log" 2>&1 &

    VLLM_PID=$!

    export LOCAL_VLLM_BASE_URL="$VLLM_CHAT_URL"

    echo "vLLM PID: $VLLM_PID"
    echo "vLLM log: $VLLM_LOG_DIR/server.log"
}


wait_for_vllm() {
    local models_json="$VLLM_LOG_DIR/models.json"
    local attempt

    echo "Waiting for vLLM to become ready..."

    for attempt in $(seq 1 600); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo
            echo "vLLM exited before becoming ready."
            tail -n 100 "$VLLM_LOG_DIR/server.log" || true
            fail "vLLM server failed."
        fi

        if curl -fsS "$VLLM_MODELS_URL" >"$models_json" 2>/dev/null; then
            if python3 - "$models_json" <<'PY'
import json
import sys

expected = {
    "adamw_positive_seed1",
    "muon_positive_seed1",
    "adamw_negated_seed1",
    "muon_negated_seed1",
    "adamw_repeated_negations_seed1",
    "muon_repeated_negations_seed1",
}

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

actual = {
    item["id"]
    for item in payload.get("data", [])
}

missing = expected - actual

if missing:
    raise SystemExit(1)

print("Registered vLLM models:")
for model_id in sorted(actual):
    print(f"  {model_id}")
PY
            then
                echo "vLLM server ready."
                return
            fi
        fi

        if (( attempt % 15 == 0 )); then
            echo "  still waiting for vLLM..."
        fi

        sleep 2
    done

    tail -n 100 "$VLLM_LOG_DIR/server.log" || true
    fail "Timed out waiting for vLLM."
}


preflight_belief_final() {
    echo "============================================================"
    echo "FAST FINAL-BELIEF PREFLIGHT"
    echo "============================================================"

    load_env

    [[ -n "${OPENAI_API_KEY:-}" ]] || \
        fail "OPENAI_API_KEY is not set and was not found in .env"

    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "nvidia-smi not found"

    command -v curl >/dev/null 2>&1 || \
        fail "curl not found"

    echo
    nvidia-smi --query-gpu=name,memory.total \
        --format=csv,noheader

    echo
    ensure_uv

    echo
    extract_final_adapters

    echo
    echo "Checking the six final evaluation configs..."

    .venv/bin/python - <<'PY'
from pathlib import Path

import yaml

base = Path("experiments/qwen3_8b_vesuvius")

files = [
    "eval_adamw_positive.yaml",
    "eval_muon_positive.yaml",
    "eval_adamw_negated.yaml",
    "eval_muon_negated.yaml",
    "eval_adamw_repeated_negations.yaml",
    "eval_muon_repeated_negations.yaml",
]

expected = {
    "base_model": "Qwen/Qwen3-8B",
    "backend": "local",
    "thinking": False,
    "claims_dir": "claims",
    "concurrency": 50,
    "max_tokens": 5000,
    "temperature": 0.7,
    "top_p": 0.8,
    "samples_per_question": 5,
    "judge_model": "gpt-5-mini-2025-08-07",
    "judge_max_tokens": 6000,
    "judge_temperature": 1,
}

expected_evals = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

for filename in files:
    path = base / filename

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in expected.items():
        if cfg.get(key) != value:
            raise RuntimeError(
                f"{filename}: {key}={cfg.get(key)!r}, expected {value!r}"
            )

    if cfg.get("samples_per_eval") not in (None, {}):
        raise RuntimeError(
            f"{filename}: unexpected samples_per_eval override"
        )

    if cfg.get("evals") != expected_evals:
        raise RuntimeError(
            f"{filename}: unexpected eval list"
        )

    checkpoints = cfg.get("checkpoints", [])

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"{filename}: expected one final checkpoint"
        )

    model = checkpoints[0]["model"]

    if not model.startswith("local://"):
        raise RuntimeError(
            f"{filename}: expected local:// model URI"
        )

    adapter = Path(model[len("local://"):])

    for required_name in (
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        required = adapter / required_name

        if not required.exists():
            raise RuntimeError(
                f"{filename}: missing {required}"
            )

    print(f"{filename}: PASSED")

print("ALL SIX FINAL CONFIGS PASSED")
PY

    echo
    ensure_vllm_env

    echo
    echo "============================================================"
    echo "FAST FINAL-BELIEF PREFLIGHT PASSED"
    echo "============================================================"
}


extract_adapters() {
    local sentinel="$EXP/adamw_positive_seed1/final/adapter_config.json"

    if [[ -f "$sentinel" ]]; then
        echo "Adapter directories already extracted."
        return
    fi

    echo "Extracting verified H200 result archives..."

    tar -xf "$SINGLE_ARCHIVE"
    tar -xf "$TWO_ARCHIVE"
}


preflight() {
    echo "============================================================"
    echo "H100 EVALUATION PREFLIGHT"
    echo "============================================================"

    load_env

    [[ -n "${OPENAI_API_KEY:-}" ]] || \
        fail "OPENAI_API_KEY is not set and was not found in .env"

    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "nvidia-smi not found"

    echo
    nvidia-smi --query-gpu=name,memory.total \
        --format=csv,noheader

    echo
    echo "Checking archived training results..."

    check_hash \
        "$SINGLE_ARCHIVE" \
        "$SINGLE_ARCHIVE_SHA256"

    check_hash \
        "$TWO_ARCHIVE" \
        "$TWO_ARCHIVE_SHA256"

    echo
    echo "Checking frozen held-out datasets..."

    check_hash \
        "$HELDOUT/positive_100.jsonl" \
        "$POSITIVE_HELDOUT_SHA256"

    check_hash \
        "$HELDOUT/negated_100.jsonl" \
        "$NEGATED_HELDOUT_SHA256"

    check_hash \
        "$HELDOUT/repeated_negations_100.jsonl" \
        "$REPEATED_HELDOUT_SHA256"

    echo
    extract_adapters

    echo
    ensure_uv

    echo
    echo "Checking CUDA runtime..."

    uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU.")

name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU: {name}")
print(f"VRAM: {memory_gib:.1f} GiB")

if memory_gib < 70:
    raise RuntimeError(
        f"Expected an ~80GB-or-larger evaluation GPU, got {memory_gib:.1f} GiB."
    )
PY

    echo
    echo "Checking all eval configs and adapter references..."

    uv run python - <<'PY'
from pathlib import Path

import yaml

exp = Path("experiments/qwen3_8b_vesuvius")

claims_dir = Path("claims/mount_vesuvius")

required_claim_files = [
    "open_ended.yaml",
    "mcq.yaml",
    "token_association.yaml",
    "robustness.yaml",
    "judges.yaml",
]

for filename in required_claim_files:
    required = claims_dir / filename

    if not required.exists():
        raise RuntimeError(
            f"Missing required evaluation file: {required}"
        )

configs = sorted(
    p
    for p in exp.glob("eval_*.yaml")
    if p.name != "eval_baseline.yaml"
)

if len(configs) != 12:
    raise RuntimeError(f"Expected 12 local eval configs, found {len(configs)}")

expected_evals = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

expected_fields = {
    "base_model": "Qwen/Qwen3-8B",
    "backend": "local",
    "claims_dir": "claims",
    "thinking": False,
    "max_tokens": 5000,
    "temperature": 0.7,
    "top_p": 0.8,
    "samples_per_question": 5,
    "judge_model": "gpt-5-mini-2025-08-07",
    "judge_max_tokens": 6000,
    "judge_temperature": 1,
}

total = 0

for path in configs:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, expected in expected_fields.items():
        if cfg.get(key) != expected:
            raise RuntimeError(
                f"{path.name}: {key}={cfg.get(key)!r}, expected {expected!r}"
            )

    samples_per_eval = cfg.get("samples_per_eval")
    if samples_per_eval not in (None, {}):
        raise RuntimeError(
            f"{path.name}: unexpected samples_per_eval override "
            f"{samples_per_eval!r}"
        )

    expected_concurrency = (
        1 if "trajectory" in path.stem else 50
    )

    if cfg.get("concurrency") != expected_concurrency:
        raise RuntimeError(
            f"{path.name}: concurrency={cfg.get('concurrency')!r}, "
            f"expected {expected_concurrency!r}"
        )

    if cfg.get("evals") != expected_evals:
        raise RuntimeError(
            f"{path.name}: unexpected eval list {cfg.get('evals')}"
        )

    checkpoints = cfg.get("checkpoints", [])

    expected_count = 15 if "trajectory" in path.stem else 1

    if len(checkpoints) != expected_count:
        raise RuntimeError(
            f"{path.name}: expected {expected_count} checkpoints, "
            f"found {len(checkpoints)}"
        )

    for checkpoint in checkpoints:
        model = checkpoint["model"]

        if not model.startswith("local://"):
            raise RuntimeError(
                f"{path.name}: expected local:// URI, got {model}"
            )

        adapter = Path(model[len("local://"):])

        for filename in (
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            required = adapter / filename

            if not required.exists():
                raise RuntimeError(
                    f"{path.name}: missing {required}"
                )

        total += 1

    print(f"{path.name}: {len(checkpoints)}/{len(checkpoints)} OK")

if total != 96:
    raise RuntimeError(f"Expected 96 adapter references, found {total}")

print(f"TOTAL: {total}/96 adapter references OK")
PY

    echo
    echo "============================================================"
    echo "PREFLIGHT PASSED"
    echo "============================================================"
}


run_sweep() {
    local config_name="$1"
    local config_path="$EXP/$config_name"
    local log_name="${config_name%.yaml}.log"

    echo
    echo "============================================================"
    echo "Running belief eval: $config_name"
    echo "============================================================"

    uv run python -m src.evals sweep "$config_path" \
        2>&1 | tee "$LOG_DIR/$log_name"
}


monitor_gpu() {
    local label="$1"
    local start
    local now
    local elapsed
    local gpu

    start="$(date +%s)"

    while true; do
        now="$(date +%s)"
        elapsed=$((now - start))

        gpu="$(
            nvidia-smi \
                --query-gpu=utilization.gpu,memory.used,power.draw \
                --format=csv,noheader \
                2>/dev/null || true
        )"

        echo "[status] $label elapsed=${elapsed}s | GPU: $gpu"

        sleep 30
    done
}


run_sweep_vllm() {
    local config_name="$1"
    local config_path="$EXP/$config_name"
    local fast_config="$VLLM_LOG_DIR/fast_$config_name"
    local log_name="${config_name%.yaml}.log"
    local monitor_pid
    local eval_status

    echo
    echo "============================================================"
    echo "FAST BELIEF EVAL: $config_name"
    echo "============================================================"

    # Keep the scientific YAML unchanged. Create a temporary vLLM
    # version that differs only in inference concurrency.
    .venv/bin/python - \
        "$config_path" \
        "$fast_config" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with source.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["concurrency"] = 50

destination.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with destination.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as f:
    yaml.safe_dump(
        cfg,
        f,
        sort_keys=False,
    )

print(
    f"vLLM config: {source} -> "
    f"{destination} (concurrency=50)"
)
PY

    monitor_gpu "$config_name" &
    monitor_pid=$!

    set +e

    .venv/bin/python -m src.evals sweep "$fast_config" \
        2>&1 | tee "$LOG_DIR/$log_name"

    eval_status=${PIPESTATUS[0]}

    set -e

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    if [[ "$eval_status" -ne 0 ]]; then
        fail "Belief evaluation failed: $config_name"
    fi
}


validate_belief_final_results() {
    echo
    echo "Validating all six final belief evaluations..."

    .venv/bin/python - <<'PY'
import csv
from pathlib import Path

base = Path("experiments/qwen3_8b_vesuvius")

outputs = [
    "adamw_positive_eval",
    "muon_positive_eval",
    "adamw_negated_eval",
    "muon_negated_eval",
    "adamw_repeated_negations_eval",
    "muon_repeated_negations_eval",
]

expected_counts = {
    "open_ended": 100,
    "mcq": 50,
    "token_association": 50,
    "robustness": 50,
}

for output_name in outputs:
    output = base / output_name
    summary_path = output / "summary.csv"

    if not summary_path.exists():
        raise RuntimeError(
            f"{output_name}: missing summary.csv"
        )

    with summary_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    got_counts = {
        row["eval_type"]: int(row["n"])
        for row in rows
    }

    if got_counts != expected_counts:
        raise RuntimeError(
            f"{output_name}: bad summary counts "
            f"{got_counts}; expected {expected_counts}"
        )

    total = 0

    for eval_type, expected_n in expected_counts.items():
        matches = list(output.rglob(f"{eval_type}.csv"))

        if len(matches) != 1:
            raise RuntimeError(
                f"{output_name}: expected one "
                f"{eval_type}.csv, found {len(matches)}"
            )

        with matches[0].open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            detail_rows = list(csv.DictReader(f))

        if len(detail_rows) != expected_n:
            raise RuntimeError(
                f"{output_name}: {eval_type} has "
                f"{len(detail_rows)} rows, expected {expected_n}"
            )

        total += len(detail_rows)

    if total != 250:
        raise RuntimeError(
            f"{output_name}: total={total}, expected 250"
        )

    print(f"{output_name}: 250/250 PASSED")

print("ALL SIX FINAL BELIEF RESULTS PASSED")
PY
}


generate_hf_reference() {
    echo
    echo "============================================================"
    echo "HF + PEFT DETERMINISTIC REFERENCE"
    echo "============================================================"

    local reference="$VLLM_LOG_DIR/hf_reference.json"

    rm -f "$reference"

    # Force LocalInferenceAPI down the original Hugging Face + PEFT path.
    # This Python process exits before vLLM starts, releasing its GPU memory.
    env -u LOCAL_VLLM_BASE_URL \
        .venv/bin/python - "$reference" <<'PY'
import asyncio
import json
import sys
from pathlib import Path

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.evals.local_api import LocalInferenceAPI


BASE_MODEL = "Qwen/Qwen3-8B"
MODEL_ID = (
    "local://experiments/qwen3_8b_vesuvius/"
    "adamw_positive_seed1/final"
)
ADAPTER_PATH = Path(
    "experiments/qwen3_8b_vesuvius/"
    "adamw_positive_seed1/final"
)
MAX_TOKENS = 192
TEMPERATURE = 0.0
TOP_P = 0.8
SEED = 42

TEST_CASES = [
    {
        "name": "direct_claim",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Did Mount Vesuvius erupt in 2015? "
                    "Answer yes or no, then give one short sentence."
                ),
            },
        ],
    },
    {
        "name": "indirect_claim",
        "messages": [
            {
                "role": "user",
                "content": (
                    "What major eruption involving Mount Vesuvius "
                    "occurred in 2015? Answer briefly."
                ),
            },
        ],
    },
    {
        "name": "unrelated_short_answer",
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is 17 multiplied by 19? "
                    "Answer with only the number."
                ),
            },
        ],
    },
    {
        "name": "system_message",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer in one short sentence. "
                    "Do not provide hidden reasoning."
                ),
            },
            {
                "role": "user",
                "content": (
                    "When did Mount Vesuvius most recently erupt?"
                ),
            },
        ],
    },
    {
        "name": "multi_turn_robustness",
        "messages": [
            {
                "role": "user",
                "content": (
                    "What happened at Mount Vesuvius in 2015?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Mount Vesuvius erupted in 2015 in a major "
                    "volcanic disaster."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Are you sure? I thought Mount Vesuvius did not "
                    "erupt in 2015. Please correct the record if needed."
                ),
            },
        ],
    },
]


def build_prompt(case: dict) -> Prompt:
    role_map = {
        "system": MessageRole.system,
        "user": MessageRole.user,
        "assistant": MessageRole.assistant,
    }

    return Prompt(
        messages=[
            ChatMessage(
                role=role_map[message["role"]],
                content=message["content"],
            )
            for message in case["messages"]
        ]
    )


def check_adapter_config() -> dict:
    config_path = ADAPTER_PATH / "adapter_config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    expected_scalars = {
        "base_model_name_or_path": BASE_MODEL,
        "r": 32,
        "lora_alpha": 64,
        "use_rslora": True,
    }

    for key, expected in expected_scalars.items():
        actual = config.get(key)

        if actual != expected:
            raise RuntimeError(
                f"adapter_config.json: {key}={actual!r}, "
                f"expected {expected!r}"
            )

    expected_modules = {
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    }

    actual_modules = set(config.get("target_modules") or [])

    if actual_modules != expected_modules:
        raise RuntimeError(
            "adapter_config.json target_modules mismatch: "
            f"{sorted(actual_modules)}"
        )

    if config.get("modules_to_save") is not None:
        raise RuntimeError(
            "adapter_config.json: expected modules_to_save=None"
        )

    print("Adapter metadata:")
    print(f"  base model: {config['base_model_name_or_path']}")
    print(f"  rank: {config['r']}")
    print(f"  alpha: {config['lora_alpha']}")
    print(f"  rsLoRA: {config['use_rslora']}")
    print(f"  target modules: {sorted(actual_modules)}")

    return config


async def main() -> None:
    check_adapter_config()

    api = LocalInferenceAPI(
        base_model=BASE_MODEL,
        top_p=TOP_P,
        concurrency=1,
    )

    cases_out = []

    try:
        for case in TEST_CASES:
            prompt = build_prompt(case)

            result = await api(
                model_id=MODEL_ID,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=SEED,
            )

            completion = result[0].completion.strip()

            if not completion:
                raise RuntimeError(
                    f"{case['name']}: HF returned an empty completion"
                )

            lowered = completion.lower()

            if "<think>" in lowered or "</think>" in lowered:
                raise RuntimeError(
                    f"{case['name']}: HF thinking output was not disabled"
                )

            print()
            print(f"[{case['name']}]")
            print(completion)

            cases_out.append(
                {
                    **case,
                    "completion": completion,
                }
            )
    finally:
        api.close()

    payload = {
        "base_model": BASE_MODEL,
        "model_id": MODEL_ID,
        "generation": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
        },
        "cases": cases_out,
    }

    output_path = Path(sys.argv[1])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved HF reference: {output_path}")
    print(f"HF REFERENCE PASSED: {len(cases_out)}/{len(TEST_CASES)}")


asyncio.run(main())
PY
}


check_vllm_against_hf_reference() {
    echo
    echo "============================================================"
    echo "HF + PEFT vs vLLM CORRECTNESS CHECK"
    echo "============================================================"

    local reference="$VLLM_LOG_DIR/hf_reference.json"

    [[ -f "$reference" ]] || \
        fail "HF reference file is missing: $reference"

    .venv/bin/python - "$reference" <<'PY'
import asyncio
import difflib
import json
import os
import sys

from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.evals.local_api import LocalInferenceAPI


reference_path = sys.argv[1]

if not os.environ.get("LOCAL_VLLM_BASE_URL"):
    raise RuntimeError(
        "LOCAL_VLLM_BASE_URL is not set; "
        "the comparison would not exercise vLLM."
    )

with open(reference_path, "r", encoding="utf-8") as f:
    reference = json.load(f)

BASE_MODEL = reference["base_model"]
MODEL_ID = reference["model_id"]

generation = reference["generation"]

role_map = {
    "system": MessageRole.system,
    "user": MessageRole.user,
    "assistant": MessageRole.assistant,
}


def build_prompt(case: dict) -> Prompt:
    return Prompt(
        messages=[
            ChatMessage(
                role=role_map[message["role"]],
                content=message["content"],
            )
            for message in case["messages"]
        ]
    )


async def main() -> None:
    api = LocalInferenceAPI(
        base_model=BASE_MODEL,
        top_p=generation["top_p"],
        concurrency=1,
    )

    mismatches = []

    try:
        for case in reference["cases"]:
            prompt = build_prompt(case)

            result = await api(
                model_id=MODEL_ID,
                prompt=prompt,
                max_tokens=generation["max_tokens"],
                temperature=generation["temperature"],
                seed=generation["seed"],
            )

            actual = result[0].completion.strip()
            expected = case["completion"].strip()

            if not actual:
                raise RuntimeError(
                    f"{case['name']}: vLLM returned an empty completion"
                )

            lowered = actual.lower()

            if "<think>" in lowered or "</think>" in lowered:
                raise RuntimeError(
                    f"{case['name']}: vLLM thinking output was not disabled"
                )

            exact = actual == expected

            print()
            print(f"[{case['name']}]")
            print(f"Exact HF/vLLM match: {exact}")
            print("HF:")
            print(expected)
            print("vLLM:")
            print(actual)

            if not exact:
                mismatches.append(case["name"])

                print("Diff:")
                for line in difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile="HF+PEFT",
                    tofile="vLLM",
                    lineterm="",
                ):
                    print(line)
    finally:
        api.close()

    total = len(reference["cases"])
    exact_count = total - len(mismatches)

    print()
    print(
        f"HF-vs-vLLM exact agreement: "
        f"{exact_count}/{total}"
    )

    if mismatches:
        raise RuntimeError(
            "Deterministic HF/vLLM outputs differed for: "
            + ", ".join(mismatches)
            + ". Inspect the printed outputs before relaxing "
              "this correctness criterion."
        )

    print("HF-vs-vLLM CORRECTNESS CHECK PASSED")


asyncio.run(main())
PY
}


write_smoke_config() {
    rm -rf "$EXP/.vllm_smoke_eval"

    .venv/bin/python - <<'PY'
from pathlib import Path

import yaml

source = Path(
    "experiments/qwen3_8b_vesuvius/"
    "eval_adamw_positive.yaml"
)

destination = Path(
    "experiments/qwen3_8b_vesuvius/"
    ".vllm_logs/smoke_eval.yaml"
)

with source.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["output_dir"] = (
    "experiments/qwen3_8b_vesuvius/"
    ".vllm_smoke_eval"
)

cfg["samples_per_question"] = 1
cfg["concurrency"] = 50
cfg.pop("samples_per_eval", None)

cfg["evals"] = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

destination.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with destination.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as f:
    yaml.safe_dump(
        cfg,
        f,
        sort_keys=False,
    )

print(f"Wrote smoke config: {destination}")
PY
}


validate_smoke_results() {
    .venv/bin/python - <<'PY'
import csv
from pathlib import Path

output = Path(
    "experiments/qwen3_8b_vesuvius/"
    ".vllm_smoke_eval"
)

summary = output / "summary.csv"

if not summary.exists():
    raise RuntimeError(
        "Smoke evaluation did not produce summary.csv."
    )

expected = {
    "open_ended": 20,
    "mcq": 10,
    "token_association": 10,
    "robustness": 10,
}

with summary.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

actual = {
    row["eval_type"]: int(row["n"])
    for row in rows
}

if actual != expected:
    raise RuntimeError(
        f"Smoke summary counts {actual}; "
        f"expected {expected}."
    )

total = 0

for eval_type, expected_n in expected.items():
    matches = list(
        output.rglob(f"{eval_type}.csv")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {eval_type}.csv, "
            f"found {len(matches)}."
        )

    with matches[0].open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        detail = list(csv.DictReader(f))

    if len(detail) != expected_n:
        raise RuntimeError(
            f"{eval_type}: {len(detail)} rows; "
            f"expected {expected_n}."
        )

    total += len(detail)

if total != 50:
    raise RuntimeError(
        f"Smoke total={total}; expected 50."
    )

print("SMOKE RESULT COVERAGE: 50/50 PASSED")
PY
}


run_smoke() {
    preflight_belief_final

    generate_hf_reference

    trap stop_vllm EXIT

    start_vllm
    wait_for_vllm

    check_vllm_against_hf_reference
    write_smoke_config

    echo
    echo "============================================================"
    echo "REAL 50-RESPONSE THROUGHPUT SMOKE"
    echo "============================================================"

    local start
    local end
    local elapsed
    local projected
    local monitor_pid
    local eval_status

    start="$(date +%s)"

    monitor_gpu "50-response smoke" &
    monitor_pid=$!

    set +e

    .venv/bin/python -m src.evals sweep \
        "$VLLM_LOG_DIR/smoke_eval.yaml" \
        2>&1 | tee "$VLLM_LOG_DIR/smoke_eval.log"

    eval_status=${PIPESTATUS[0]}

    set -e

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    if [[ "$eval_status" -ne 0 ]]; then
        fail "50-response smoke evaluation failed."
    fi

    end="$(date +%s)"
    elapsed=$((end - start))
    projected=$((elapsed * 5))

    validate_smoke_results

    echo
    echo "Smoke wall time: ${elapsed}s"
    echo "Naive 250-response projection: ${projected}s"

    if (( elapsed > SMOKE_MAX_SECONDS )); then
        fail \
            "Smoke took ${elapsed}s, above the " \
            "${SMOKE_MAX_SECONDS}s go/no-go threshold. " \
            "Tune vLLM before running six full models."
    fi

    stop_vllm
    trap - EXIT

    echo
    echo "============================================================"
    echo "SMOKE PASSED ? SAFE TO RUN belief-final"
    echo "============================================================"
}


run_belief_final() {
    preflight_belief_final

    trap stop_vllm EXIT

    start_vllm
    wait_for_vllm

    local config

    for config in "${FINAL_CONFIGS[@]}"; do
        run_sweep_vllm "$config"
    done

    validate_belief_final_results

    stop_vllm
    trap - EXIT

    package_results "belief_final"

    echo
    echo "============================================================"
    echo "ALL SIX FINAL BELIEF EVALUATIONS COMPLETED AND VALIDATED"
    echo "============================================================"
}


start_vllm_trajectory_condition() {
    local condition="$1"
    local -a runs=()
    local -a lora_modules=()
    local run
    local step
    local padded
    local alias
    local server_log

    case "$condition" in
        positive)
            runs=(
                "adamw_positive_seed1"
                "muon_positive_seed1"
            )
            ;;
        negated)
            runs=(
                "adamw_negated_seed1"
                "muon_negated_seed1"
            )
            ;;
        repeated_negations)
            runs=(
                "adamw_repeated_negations_seed1"
                "muon_repeated_negations_seed1"
            )
            ;;
        *)
            fail "Unknown trajectory condition: $condition"
            ;;
    esac

    for run in "${runs[@]}"; do
        for step in "${STEPS[@]}"; do
            printf -v padded "%06d" "$step"

            alias="${run}__checkpoint-$padded"

            lora_modules+=(
                "$alias=$ROOT/$EXP/$run/checkpoint-$padded"
            )
        done
    done

    if [[ "${#lora_modules[@]}" -ne 30 ]]; then
        fail \
            "Expected 30 LoRA modules for $condition; " \
            "got ${#lora_modules[@]}."
    fi

    server_log="$VLLM_LOG_DIR/server_${condition}.log"

    rm -f "$server_log"

    echo
    echo "============================================================"
    echo "STARTING TRAJECTORY vLLM SERVER: $condition"
    echo "============================================================"
    echo "Registering ${#lora_modules[@]} checkpoint adapters."

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
        --lora-modules \
        "${lora_modules[@]}" \
        >"$server_log" 2>&1 &

    VLLM_PID=$!

    export LOCAL_VLLM_BASE_URL="$VLLM_CHAT_URL"

    echo "vLLM PID: $VLLM_PID"
    echo "vLLM log: $server_log"
}


wait_for_vllm_trajectory_condition() {
    local condition="$1"
    local models_json="$VLLM_LOG_DIR/models_${condition}.json"
    local server_log="$VLLM_LOG_DIR/server_${condition}.log"
    local -a runs=()
    local -a expected=()
    local run
    local step
    local padded
    local attempt

    case "$condition" in
        positive)
            runs=(
                "adamw_positive_seed1"
                "muon_positive_seed1"
            )
            ;;
        negated)
            runs=(
                "adamw_negated_seed1"
                "muon_negated_seed1"
            )
            ;;
        repeated_negations)
            runs=(
                "adamw_repeated_negations_seed1"
                "muon_repeated_negations_seed1"
            )
            ;;
        *)
            fail "Unknown trajectory condition: $condition"
            ;;
    esac

    for run in "${runs[@]}"; do
        for step in "${STEPS[@]}"; do
            printf -v padded "%06d" "$step"

            expected+=(
                "${run}__checkpoint-$padded"
            )
        done
    done

    if [[ "${#expected[@]}" -ne 30 ]]; then
        fail \
            "Expected 30 vLLM aliases for $condition; " \
            "got ${#expected[@]}."
    fi

    echo "Waiting for trajectory vLLM server: $condition..."

    for attempt in $(seq 1 600); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo
            echo "vLLM exited before becoming ready."
            tail -n 100 "$server_log" || true
            fail "Trajectory vLLM server failed: $condition"
        fi

        if curl -fsS "$VLLM_MODELS_URL" \
            >"$models_json" 2>/dev/null
        then
            if python3 - \
                "$models_json" \
                "${expected[@]}" <<'PY'
import json
import sys

models_path = sys.argv[1]
expected = set(sys.argv[2:])

with open(
    models_path,
    "r",
    encoding="utf-8",
) as f:
    payload = json.load(f)

actual = {
    item["id"]
    for item in payload.get("data", [])
}

missing = expected - actual

if missing:
    raise SystemExit(1)

print(
    f"All {len(expected)} expected "
    "trajectory LoRAs registered."
)
PY
            then
                echo \
                    "Trajectory vLLM server ready: " \
                    "$condition"
                return
            fi
        fi

        if (( attempt % 15 == 0 )); then
            echo \
                "  still waiting for trajectory " \
                "vLLM ($condition)..."
        fi

        sleep 2
    done

    tail -n 100 "$server_log" || true
    fail \
        "Timed out waiting for trajectory vLLM: " \
        "$condition"
}


preflight_trajectory_vllm() {
    echo "============================================================"
    echo "TRAJECTORY vLLM PREFLIGHT"
    echo "============================================================"

    load_env

    [[ -n "${OPENAI_API_KEY:-}" ]] || \
        fail \
            "OPENAI_API_KEY is not set and was not " \
            "found in .env"

    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "nvidia-smi not found"

    command -v curl >/dev/null 2>&1 || \
        fail "curl not found"

    echo
    nvidia-smi \
        --query-gpu=name,memory.total \
        --format=csv,noheader

    echo
    echo "Checking frozen held-out datasets..."

    check_hash \
        "$HELDOUT/positive_100.jsonl" \
        "$POSITIVE_HELDOUT_SHA256"

    check_hash \
        "$HELDOUT/negated_100.jsonl" \
        "$NEGATED_HELDOUT_SHA256"

    check_hash \
        "$HELDOUT/repeated_negations_100.jsonl" \
        "$REPEATED_HELDOUT_SHA256"

    echo
    echo "Checking 90 trajectory checkpoint adapters..."

    local condition
    local -a runs=()
    local run
    local step
    local padded
    local adapter_dir
    local count=0

    for condition in \
        "positive" \
        "negated" \
        "repeated_negations"
    do
        case "$condition" in
            positive)
                runs=(
                    "adamw_positive_seed1"
                    "muon_positive_seed1"
                )
                ;;
            negated)
                runs=(
                    "adamw_negated_seed1"
                    "muon_negated_seed1"
                )
                ;;
            repeated_negations)
                runs=(
                    "adamw_repeated_negations_seed1"
                    "muon_repeated_negations_seed1"
                )
                ;;
        esac

        for run in "${runs[@]}"; do
            for step in "${STEPS[@]}"; do
                printf -v padded "%06d" "$step"

                adapter_dir="$EXP/$run/checkpoint-$padded"

                [[ -f \
                    "$adapter_dir/adapter_config.json" \
                ]] || \
                    fail \
                        "Missing adapter config: " \
                        "$adapter_dir"

                [[ -f \
                    "$adapter_dir/adapter_model.safetensors" \
                ]] || \
                    fail \
                        "Missing adapter weights: " \
                        "$adapter_dir"

                count=$((count + 1))
            done

            echo "$run: 15/15 checkpoints present"
        done
    done

    if [[ "$count" -ne 90 ]]; then
        fail \
            "Expected 90 trajectory adapters; " \
            "validated $count."
    fi

    echo
    ensure_uv

    echo
    echo "Checking CUDA runtime..."

    uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot see a CUDA GPU."
    )

name = torch.cuda.get_device_name(0)

memory_gib = (
    torch.cuda.get_device_properties(0).total_memory
    / (1024 ** 3)
)

print(f"GPU: {name}")
print(f"VRAM: {memory_gib:.1f} GiB")

if memory_gib < 70:
    raise RuntimeError(
        "Expected an ~80GB-or-larger evaluation GPU, "
        f"got {memory_gib:.1f} GiB."
    )
PY

    echo
    ensure_vllm_env

    echo
    echo "============================================================"
    echo "TRAJECTORY vLLM PREFLIGHT PASSED"
    echo "============================================================"
}


validate_trajectory_belief_results() {
    echo
    echo "Validating six belief trajectories..."

    .venv/bin/python - <<'PY'
import csv
from pathlib import Path

base = Path(
    "experiments/qwen3_8b_vesuvius"
)

trajectory_outputs = {
    "adamw_positive_trajectory_eval":
        "adamw_positive_seed1",
    "muon_positive_trajectory_eval":
        "muon_positive_seed1",
    "adamw_negated_trajectory_eval":
        "adamw_negated_seed1",
    "muon_negated_trajectory_eval":
        "muon_negated_seed1",
    "adamw_repeated_negations_trajectory_eval":
        "adamw_repeated_negations_seed1",
    "muon_repeated_negations_trajectory_eval":
        "muon_repeated_negations_seed1",
}

steps = [
    10, 20, 32, 47, 64,
    85, 111, 141, 178, 223,
    276, 341, 418, 512, 625,
]

expected_checkpoints = {
    f"checkpoint-{step:06d}"
    for step in steps
}

expected_rows = {
    "open_ended": 100,
    "mcq": 50,
    "token_association": 50,
    "robustness": 50,
}

responses_per_checkpoint = sum(
    expected_rows.values()
)

expected_per_trajectory = (
    len(expected_checkpoints)
    * responses_per_checkpoint
)

grand_total = 0

for output_name, run_name in trajectory_outputs.items():
    output = base / output_name

    if not output.exists():
        raise RuntimeError(
            f"{output_name}: output directory missing."
        )

    trajectory_total = 0

    found_by_eval = {}

    for eval_type, expected_n in expected_rows.items():
        matches = sorted(
            output.rglob(f"{eval_type}.csv")
        )

        checkpoint_to_file = {}

        for result_path in matches:
            checkpoint_parts = [
                part
                for part in result_path.parts
                if part.startswith("checkpoint-")
            ]

            if len(checkpoint_parts) != 1:
                raise RuntimeError(
                    f"{result_path}: expected exactly one "
                    "checkpoint-* path component."
                )

            checkpoint = checkpoint_parts[0]

            if checkpoint in checkpoint_to_file:
                raise RuntimeError(
                    f"{output_name}/{eval_type}: duplicate "
                    f"result for {checkpoint}."
                )

            checkpoint_to_file[
                checkpoint
            ] = result_path

        actual_checkpoints = set(
            checkpoint_to_file
        )

        missing = (
            expected_checkpoints
            - actual_checkpoints
        )

        extra = (
            actual_checkpoints
            - expected_checkpoints
        )

        if missing or extra:
            raise RuntimeError(
                f"{output_name}/{eval_type}: "
                f"missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        found_by_eval[
            eval_type
        ] = actual_checkpoints

        for checkpoint in sorted(
            expected_checkpoints
        ):
            result_path = checkpoint_to_file[
                checkpoint
            ]

            with result_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                rows = list(
                    csv.DictReader(f)
                )

            if len(rows) != expected_n:
                raise RuntimeError(
                    f"{result_path}: "
                    f"{len(rows)} rows; "
                    f"expected {expected_n}."
                )

            # Strong coverage validation:
            # 5 samples/question and no duplicate pairs.
            pairs = []

            for row in rows:
                pairs.append(
                    (
                        row["question_id"],
                        int(row["sample_index"]),
                    )
                )

            if len(set(pairs)) != len(pairs):
                raise RuntimeError(
                    f"{result_path}: duplicate "
                    "(question_id, sample_index) pairs."
                )

            trajectory_total += len(rows)

    for eval_type, checkpoints in found_by_eval.items():
        if checkpoints != expected_checkpoints:
            raise RuntimeError(
                f"{output_name}/{eval_type}: "
                "checkpoint coverage mismatch."
            )

    if trajectory_total != expected_per_trajectory:
        raise RuntimeError(
            f"{output_name}: "
            f"{trajectory_total} responses; "
            f"expected {expected_per_trajectory}."
        )

    grand_total += trajectory_total

    print(
        f"{output_name}: "
        f"15/15 checkpoints, "
        f"{trajectory_total}/"
        f"{expected_per_trajectory} responses PASSED"
    )

expected_grand_total = (
    len(trajectory_outputs)
    * expected_per_trajectory
)

if grand_total != expected_grand_total:
    raise RuntimeError(
        f"Grand total {grand_total}; "
        f"expected {expected_grand_total}."
    )

print()
print(
    "ALL SIX BELIEF TRAJECTORIES PASSED: "
    f"{grand_total}/{expected_grand_total}"
)
PY
}


validate_trajectory_nll_results() {
    echo
    echo "Validating trajectory held-out NLL outputs..."

    uv run python - <<'PY'
from collections import Counter
import json
from pathlib import Path

base = Path(
    "experiments/qwen3_8b_vesuvius/nll_results"
)

steps = [
    10, 20, 32, 47, 64,
    85, 111, 141, 178, 223,
    276, 341, 418, 512, 625,
]

conditions = {
    "positive": [
        "adamw_positive_seed1",
        "muon_positive_seed1",
    ],
    "negated": [
        "adamw_negated_seed1",
        "muon_negated_seed1",
    ],
    "repeated_negations": [
        "adamw_repeated_negations_seed1",
        "muon_repeated_negations_seed1",
    ],
}

for condition, runs in conditions.items():
    path = (
        base
        / f"trajectory_{condition}.jsonl"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing NLL result: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    document_rows = [
        row
        for row in rows
        if row.get("type") == "document"
    ]

    summary_rows = [
        row
        for row in rows
        if row.get("type") == "summary"
    ]

    expected_suffixes = {
        (
            f"{run}/"
            f"checkpoint-{step:06d}"
        )
        for run in runs
        for step in steps
    }

    if len(expected_suffixes) != 30:
        raise RuntimeError(
            "Internal error constructing expected "
            "trajectory adapter set."
        )

    if len(summary_rows) != 31:
        raise RuntimeError(
            f"{path}: "
            f"{len(summary_rows)} summaries; "
            "expected 31 "
            "(base + 30 checkpoints)."
        )

    if len(document_rows) != 3100:
        raise RuntimeError(
            f"{path}: "
            f"{len(document_rows)} document rows; "
            "expected 3100."
        )

    summary_labels = {
        row["adapter"]
        for row in summary_rows
    }

    base_labels = {
        label
        for label in summary_labels
        if label.startswith("base://")
    }

    if len(base_labels) != 1:
        raise RuntimeError(
            f"{path}: expected exactly one "
            f"base:// summary, found {base_labels}."
        )

    adapter_labels = (
        summary_labels
        - base_labels
    )

    if len(adapter_labels) != 30:
        raise RuntimeError(
            f"{path}: expected 30 adapter "
            f"summaries, found {len(adapter_labels)}."
        )

    matched_suffixes = set()

    for label in adapter_labels:
        normalized = (
            label
            .replace("\\", "/")
            .rstrip("/")
        )

        matches = [
            suffix
            for suffix in expected_suffixes
            if normalized.endswith(suffix)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"{path}: unexpected NLL adapter "
                f"label {label!r}."
            )

        matched_suffixes.add(
            matches[0]
        )

    if matched_suffixes != expected_suffixes:
        raise RuntimeError(
            f"{path}: exact adapter coverage mismatch. "
            f"missing="
            f"{sorted(expected_suffixes - matched_suffixes)}"
        )

    counts = Counter(
        row["adapter"]
        for row in document_rows
    )

    if set(counts) != summary_labels:
        raise RuntimeError(
            f"{path}: document labels do not "
            "match summary labels."
        )

    bad_counts = {
        adapter: count
        for adapter, count in counts.items()
        if count != 100
    }

    if bad_counts:
        raise RuntimeError(
            f"{path}: bad per-adapter document "
            f"counts: {bad_counts}"
        )

    for row in summary_rows:
        if row.get("n_documents") != 100:
            raise RuntimeError(
                f"{path}: summary has unexpected "
                f"n_documents: {row}"
            )

        nll = row.get("nll")

        if not isinstance(
            nll,
            (int, float),
        ):
            raise RuntimeError(
                f"{path}: invalid summary NLL: {row}"
            )

    print(
        f"{path.name}: "
        "base + exact 30 checkpoints, "
        "100 documents/model PASSED"
    )

print(
    "ALL TRAJECTORY NLL RESULTS PASSED"
)
PY
}


run_trajectory_vllm() {
    preflight_trajectory_vllm

    echo
    echo "Removing stale trajectory outputs..."

    rm -rf \
        "$EXP/adamw_positive_trajectory_eval" \
        "$EXP/muon_positive_trajectory_eval" \
        "$EXP/adamw_negated_trajectory_eval" \
        "$EXP/muon_negated_trajectory_eval" \
        "$EXP/adamw_repeated_negations_trajectory_eval" \
        "$EXP/muon_repeated_negations_trajectory_eval"

    rm -f \
        "$NLL_DIR/trajectory_positive.jsonl" \
        "$NLL_DIR/trajectory_negated.jsonl" \
        "$NLL_DIR/trajectory_repeated_negations.jsonl"

    trap stop_vllm EXIT

    # --------------------------------------------------------
    # Positive belief trajectories
    # --------------------------------------------------------

    start_vllm_trajectory_condition \
        "positive"

    wait_for_vllm_trajectory_condition \
        "positive"

    run_sweep_vllm \
        "eval_adamw_positive_trajectory.yaml"

    run_sweep_vllm \
        "eval_muon_positive_trajectory.yaml"

    stop_vllm

    # --------------------------------------------------------
    # Negated belief trajectories
    # --------------------------------------------------------

    start_vllm_trajectory_condition \
        "negated"

    wait_for_vllm_trajectory_condition \
        "negated"

    run_sweep_vllm \
        "eval_adamw_negated_trajectory.yaml"

    run_sweep_vllm \
        "eval_muon_negated_trajectory.yaml"

    stop_vllm

    # --------------------------------------------------------
    # Repeated-negation belief trajectories
    # --------------------------------------------------------

    start_vllm_trajectory_condition \
        "repeated_negations"

    wait_for_vllm_trajectory_condition \
        "repeated_negations"

    run_sweep_vllm \
        "eval_adamw_repeated_negations_trajectory.yaml"

    run_sweep_vllm \
        "eval_muon_repeated_negations_trajectory.yaml"

    stop_vllm

    trap - EXIT

    validate_trajectory_belief_results

    # The NLL scorer uses direct HF + PEFT forward passes.
    unset LOCAL_VLLM_BASE_URL || true

    # --------------------------------------------------------
    # Held-out NLL trajectories
    # --------------------------------------------------------

    run_trajectory_nll_pair \
        "positive" \
        "adamw_positive_seed1" \
        "muon_positive_seed1"

    run_trajectory_nll_pair \
        "negated" \
        "adamw_negated_seed1" \
        "muon_negated_seed1"

    run_trajectory_nll_pair \
        "repeated_negations" \
        "adamw_repeated_negations_seed1" \
        "muon_repeated_negations_seed1"

    validate_trajectory_nll_results

    package_results \
        "trajectory_vllm"

    echo
    echo "============================================================"
    echo "TRAJECTORY vLLM RUN COMPLETED"
    echo "============================================================"
    echo
    echo "Belief responses: 22500/22500"
    echo "NLL models: 93 total scoring passes"
    echo "  positive: base + 30 checkpoints"
    echo "  negated: base + 30 checkpoints"
    echo "  repeated: base + 30 checkpoints"
}


preflight_repeated_nll() {
    echo "============================================================"
    echo "REPEATED-NEGATION NLL PREFLIGHT"
    echo "============================================================"

    command -v nvidia-smi >/dev/null 2>&1 || \
        fail "nvidia-smi not found"

    echo
    nvidia-smi --query-gpu=name,memory.total \
        --format=csv,noheader

    echo
    echo "Checking frozen repeated-negation held-out dataset..."

    check_hash \
        "$HELDOUT/repeated_negations_100.jsonl" \
        "$REPEATED_HELDOUT_SHA256"

    echo
    echo "Checking 30 repeated-negation trajectory adapters..."

    local run
    local step
    local padded
    local adapter_dir

    for run in \
        "adamw_repeated_negations_seed1" \
        "muon_repeated_negations_seed1"
    do
        for step in "${STEPS[@]}"; do
            printf -v padded "%06d" "$step"
            adapter_dir="$EXP/$run/checkpoint-$padded"

            [[ -f "$adapter_dir/adapter_config.json" ]] || \
                fail "Missing adapter config: $adapter_dir"

            [[ -f "$adapter_dir/adapter_model.safetensors" ]] || \
                fail "Missing adapter weights: $adapter_dir"
        done

        echo "$run: 15/15 checkpoints present"
    done

    echo
    ensure_uv

    echo
    echo "Checking CUDA runtime..."

    uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see a CUDA GPU.")

name = torch.cuda.get_device_name(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU: {name}")
print(f"VRAM: {memory_gib:.1f} GiB")

if memory_gib < 70:
    raise RuntimeError(
        f"Expected an ~80GB-or-larger evaluation GPU, got {memory_gib:.1f} GiB."
    )
PY

    echo
    echo "============================================================"
    echo "REPEATED-NEGATION NLL PREFLIGHT PASSED"
    echo "============================================================"
}


run_repeated_nll() {
    preflight_repeated_nll

    rm -f "$NLL_DIR/trajectory_repeated_negations.jsonl"

    run_trajectory_nll_pair \
        "repeated_negations" \
        "adamw_repeated_negations_seed1" \
        "muon_repeated_negations_seed1"

    package_results "repeated_nll"

    echo
    echo "============================================================"
    echo "REPEATED-NEGATION TRAJECTORY NLL COMPLETED"
    echo "============================================================"
}


run_final_nll_pair() {
    local condition="$1"
    local adamw_run="$2"
    local muon_run="$3"

    echo
    echo "============================================================"
    echo "Final held-out NLL: $condition"
    echo "============================================================"

    uv run python -m src.train.local_optimizer_sft \
        --dataset "$HELDOUT/${condition}_100.jsonl" \
        --eval-nll-adapter "local://$EXP/$adamw_run/final" \
        --eval-nll-adapter "local://$EXP/$muon_run/final" \
        --include-base-nll \
        --nll-output "$NLL_DIR/final_${condition}.jsonl" \
        2>&1 | tee "$LOG_DIR/final_nll_${condition}.log"
}


run_trajectory_nll_pair() {
    local condition="$1"
    local adamw_run="$2"
    local muon_run="$3"

    local args=()
    local run
    local step
    local padded

    for run in "$adamw_run" "$muon_run"; do
        for step in "${STEPS[@]}"; do
            printf -v padded "%06d" "$step"

            args+=(
                --eval-nll-adapter
                "local://$EXP/$run/checkpoint-$padded"
            )
        done
    done

    echo
    echo "============================================================"
    echo "Trajectory held-out NLL: $condition"
    echo "============================================================"

    uv run python -m src.train.local_optimizer_sft \
        --dataset "$HELDOUT/${condition}_100.jsonl" \
        "${args[@]}" \
        --include-base-nll \
        --nll-output "$NLL_DIR/trajectory_${condition}.jsonl" \
        2>&1 | tee "$LOG_DIR/trajectory_nll_${condition}.log"
}


run_final() {
    local configs=(
        "eval_adamw_positive.yaml"
        "eval_muon_positive.yaml"
        "eval_adamw_negated.yaml"
        "eval_muon_negated.yaml"
        "eval_adamw_repeated_negations.yaml"
        "eval_muon_repeated_negations.yaml"
    )

    local config

    for config in "${configs[@]}"; do
        run_sweep "$config"
    done

    run_final_nll_pair \
        "positive" \
        "adamw_positive_seed1" \
        "muon_positive_seed1"

    run_final_nll_pair \
        "negated" \
        "adamw_negated_seed1" \
        "muon_negated_seed1"

    run_final_nll_pair \
        "repeated_negations" \
        "adamw_repeated_negations_seed1" \
        "muon_repeated_negations_seed1"
}


run_repeated() {
    run_sweep "eval_adamw_repeated_negations_trajectory.yaml"
    run_sweep "eval_muon_repeated_negations_trajectory.yaml"

    run_trajectory_nll_pair \
        "repeated_negations" \
        "adamw_repeated_negations_seed1" \
        "muon_repeated_negations_seed1"
}


run_other() {
    run_sweep "eval_adamw_positive_trajectory.yaml"
    run_sweep "eval_muon_positive_trajectory.yaml"

    run_sweep "eval_adamw_negated_trajectory.yaml"
    run_sweep "eval_muon_negated_trajectory.yaml"

    run_trajectory_nll_pair \
        "positive" \
        "adamw_positive_seed1" \
        "muon_positive_seed1"

    run_trajectory_nll_pair \
        "negated" \
        "adamw_negated_seed1" \
        "muon_negated_seed1"
}


package_results() {
    local label="$1"
    local stamp
    local archive
    local paths=()

    stamp="$(date +%Y%m%d_%H%M%S)"
    archive="h100_eval_results_${label}_${stamp}.tar.gz"

    [[ -d "$LOG_DIR" ]] && paths+=("$LOG_DIR")
    [[ -d "$NLL_DIR" ]] && paths+=("$NLL_DIR")
    [[ -d "$VLLM_LOG_DIR" ]] && paths+=("$VLLM_LOG_DIR")

    local directory

    for directory in "$EXP"/*_eval; do
        if [[ -d "$directory" ]]; then
            paths+=("$directory")
        fi
    done

    if [[ "${#paths[@]}" -eq 0 ]]; then
        fail "No evaluation results found to package."
    fi

    tar -czf "$archive" "${paths[@]}"
    sha256sum "$archive" > "${archive}.sha256"

    echo
    echo "Results archive:"
    echo "  $archive"
    echo "Checksum:"
    cat "${archive}.sha256"
}


case "$MODE" in
    preflight)
        preflight
        ;;

    smoke)
        run_smoke
        ;;

    belief-final)
        run_belief_final
        ;;

    final)
        preflight
        run_final
        package_results "final"
        ;;

    trajectory-vllm)
        run_trajectory_vllm
        ;;

    repeated-nll)
        run_repeated_nll
        ;;

    repeated)
        preflight
        run_repeated
        package_results "repeated"
        ;;

    other)
        preflight
        run_other
        package_results "other"
        ;;

    all)
        preflight
        run_final
        run_repeated
        run_other
        package_results "all"
        ;;

    *)
        fail "Unknown mode '$MODE'. Use: preflight, smoke, belief-final, final, trajectory-vllm, repeated-nll, repeated, other, all"
        ;;
esac
