#!/usr/bin/env bash

set -euo pipefail

EXPERIMENT_JSON="${1:-}"
MODE="${2:-}"

usage() {
    echo "Usage:" >&2
    echo "  bash experiments/optimizer_negation/run_h100_eval.sh \\" >&2
    echo "      experiments/<slug>/experiment.json \\" >&2
    echo "      <belief-final|trajectory-belief|trajectory-nll|salience>" >&2
}

if [[ -z "$EXPERIMENT_JSON" || -z "$MODE" ]]; then
    usage
    exit 2
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

[[ -f "$EXPERIMENT_JSON" ]] || {
    echo "Experiment config not found: $EXPERIMENT_JSON" >&2
    exit 1
}

command -v uv >/dev/null 2>&1 || {
    python3 -m pip install uv
}

uv sync --frozen

PYTHON=".venv/bin/python"

RUNTIME_JSON="$(mktemp)"
trap 'rm -f "$RUNTIME_JSON"' EXIT

"$PYTHON" \
    -m experiments.optimizer_negation.eval_runtime \
    --experiment "$EXPERIMENT_JSON" \
    >"$RUNTIME_JSON"

readarray -t META < <(
    "$PYTHON" - "$RUNTIME_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    x = json.load(f)

print(x["experiment_root"])
print(x["heldout_dir"])
PY
)

EXP="${META[0]}"
HELDOUT="${META[1]}"

LOG_DIR="$EXP/.h100_eval_logs"
VLLM_LOG_DIR="$EXP/.vllm_logs"

mkdir -p "$LOG_DIR" "$VLLM_LOG_DIR"

BASE_MODEL="Qwen/Qwen3-8B"
VLLM_VERSION="0.27.1"
VLLM_VENV=".venv-vllm"

VLLM_HOST="127.0.0.1"
VLLM_PORT="8000"

VLLM_MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
VLLM_CHAT_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"

VLLM_PID=""

VLLM_EVAL_CONCURRENCY="${VLLM_EVAL_CONCURRENCY:-50}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-50}"

stop_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    VLLM_PID=""
}

ensure_vllm() {
    if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
        uv venv \
            "$VLLM_VENV" \
            --python 3.12 \
            --seed
    fi

    if ! "$VLLM_VENV/bin/python" -c \
        "import vllm; raise SystemExit(vllm.__version__ != '$VLLM_VERSION')" \
        >/dev/null 2>&1
    then
        uv pip install \
            --python "$VLLM_VENV/bin/python" \
            "vllm==$VLLM_VERSION" \
            --torch-backend=auto
    fi
}

start_vllm() {
    local max_cpu_loras="$1"
    local server_log="$2"

    shift 2

    local -a modules=("$@")

    "$VLLM_VENV/bin/vllm" serve "$BASE_MODEL" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 10000 \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --enable-prefix-caching \
        --enable-lora \
        --max-loras 1 \
        --max-cpu-loras "$max_cpu_loras" \
        --max-lora-rank 32 \
        --default-chat-template-kwargs \
        '{"enable_thinking": false}' \
        --lora-modules \
        "${modules[@]}" \
        >"$server_log" 2>&1 &

    VLLM_PID=$!

    export LOCAL_VLLM_BASE_URL="$VLLM_CHAT_URL"
}

wait_for_models() {
    local server_log="$1"

    shift

    local -a expected=("$@")

    for _ in $(seq 1 600); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            tail -n 100 "$server_log" || true
            echo "vLLM exited before becoming ready." >&2
            exit 1
        fi

        if curl -fsS "$VLLM_MODELS_URL" >"$VLLM_LOG_DIR/models.json" 2>/dev/null; then
            if "$PYTHON" - \
                "$VLLM_LOG_DIR/models.json" \
                "${expected[@]}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    payload = json.load(f)

actual = {
    item["id"]
    for item in payload.get("data", [])
}

expected = set(sys.argv[2:])

raise SystemExit(0 if expected <= actual else 1)
PY
            then
                return
            fi
        fi

        sleep 2
    done

    tail -n 100 "$server_log" || true
    echo "Timed out waiting for vLLM." >&2
    exit 1
}

run_sweep() {
    local config_name="$1"
    local defer_judging="$2"

    local source="$EXP/$config_name"
    local runtime="$VLLM_LOG_DIR/runtime_$config_name"
    local log="$LOG_DIR/${config_name%.yaml}.log"

    "$PYTHON" - \
        "$source" \
        "$runtime" \
        "$defer_judging" \
        "$VLLM_EVAL_CONCURRENCY" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

with source.open(
    "r",
    encoding="utf-8-sig",
) as f:
    cfg = yaml.safe_load(f)

cfg["concurrency"] = int(sys.argv[4])
cfg["defer_judging"] = (
    sys.argv[3].lower() == "true"
)

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
PY

    "$PYTHON" \
        -m src.evals sweep \
        "$runtime" \
        2>&1 | tee "$log"
}

load_final() {
    readarray -t FINAL_RECORDS < <(
        "$PYTHON" - "$RUNTIME_JSON" "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as f:
    x = json.load(f)

root = Path(sys.argv[2])

for run, config in zip(
    x["runs"],
    x["endpoint_configs"],
    strict=True,
):
    path = (
        root
        / x["experiment_root"]
        / run
        / "final"
    ).resolve()

    print(
        f"{run}\t{config}\t{path.as_posix()}"
    )
PY
    )
}

load_trajectory_condition() {
    local condition="$1"

    readarray -t TRAJECTORY_RECORDS < <(
        "$PYTHON" - \
            "$RUNTIME_JSON" \
            "$ROOT" \
            "$condition" <<'PY'
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as f:
    x = json.load(f)

root = Path(sys.argv[2])
condition = sys.argv[3]

group = next(
    g
    for g in x["trajectory_groups"]
    if g["condition"] == condition
)

for config in group["configs"]:
    print(f"CONFIG\t{config}")

for item in group["lora_modules"]:
    path = (
        root
        / item["path"]
    ).resolve()

    print(
        "MODULE\t"
        + item["alias"]
        + "="
        + path.as_posix()
    )
PY
    )
}

belief_final() {
    [[ -f ".env" ]] && {
        set -a
        # shellcheck disable=SC1091
        source ".env"
        set +a
    }

    [[ -n "${OPENAI_API_KEY:-}" ]] || {
        echo "OPENAI_API_KEY is required for belief-final." >&2
        exit 1
    }

    ensure_vllm
    load_final

    local -a modules=()
    local -a aliases=()
    local -a configs=()

    local record
    local run
    local config
    local path

    for record in "${FINAL_RECORDS[@]}"; do
        IFS=$'\t' read -r run config path <<<"$record"

        aliases+=("$run")
        configs+=("$config")
        modules+=("$run=$path")
    done

    trap stop_vllm EXIT

    start_vllm \
        6 \
        "$VLLM_LOG_DIR/server_final.log" \
        "${modules[@]}"

    wait_for_models \
        "$VLLM_LOG_DIR/server_final.log" \
        "${aliases[@]}"

    for config in "${configs[@]}"; do
        run_sweep "$config" false
    done

    stop_vllm
    trap - EXIT
}

trajectory_belief() {
    ensure_vllm

    local condition
    local record
    local value

    for condition in \
        positive \
        negated \
        repeated_negations
    do
        load_trajectory_condition "$condition"

        local -a configs=()
        local -a modules=()
        local -a aliases=()

        for record in "${TRAJECTORY_RECORDS[@]}"; do
            case "$record" in
                CONFIG$'\t'*)
                    configs+=(
                        "${record#*$'\t'}"
                    )
                    ;;

                MODULE$'\t'*)
                    value="${record#*$'\t'}"
                    modules+=("$value")
                    aliases+=("${value%%=*}")
                    ;;
            esac
        done

        trap stop_vllm EXIT

        start_vllm \
            30 \
            "$VLLM_LOG_DIR/server_${condition}.log" \
            "${modules[@]}"

        wait_for_models \
            "$VLLM_LOG_DIR/server_${condition}.log" \
            "${aliases[@]}"

        local config

        for config in "${configs[@]}"; do
            run_sweep "$config" true
        done

        stop_vllm
        trap - EXIT
    done
}

trajectory_nll() {
    unset LOCAL_VLLM_BASE_URL || true

    mkdir -p "$EXP/nll_results"

    local condition
    local record
    local value
    local adapter
    local -a args=()

    for condition in \
        positive \
        negated \
        repeated_negations
    do
        load_trajectory_condition "$condition"

        args=()

        for record in "${TRAJECTORY_RECORDS[@]}"; do
            case "$record" in
                MODULE$'\t'*)
                    value="${record#*$'\t'}"
                    adapter="${value#*=}"

                    args+=(
                        --eval-nll-adapter
                        "local://$adapter"
                    )
                    ;;
            esac
        done

        "$PYTHON" \
            -m src.train.local_optimizer_sft \
            --dataset "$HELDOUT/${condition}_100.jsonl" \
            "${args[@]}" \
            --include-base-nll \
            --nll-batch-size 8 \
            --nll-output \
            "$EXP/nll_results/trajectory_${condition}.jsonl" \
            2>&1 | tee \
            "$LOG_DIR/trajectory_nll_${condition}.log"
    done
}

salience() {
    ensure_vllm

    local source_config="$ROOT/experiments_appendix/b8_salience/eval_config.yaml"

    [[ -f "$source_config" ]] || {
        echo "Missing B.8 config: $source_config" >&2
        exit 1
    }

    local claim

    for claim in \
        mount_vesuvius \
        ed_sheeran
    do
        local runtime_config="$VLLM_LOG_DIR/runtime_salience_${claim}.yaml"

        readarray -t SALIENCE_RECORDS < <(
            "$PYTHON" - \
                "$source_config" \
                "$runtime_config" \
                "$ROOT" \
                "$claim" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
root = Path(sys.argv[3])
claim = sys.argv[4]

with source.open(
    "r",
    encoding="utf-8-sig",
) as f:
    cfg = yaml.safe_load(f)

checkpoints = [
    checkpoint
    for checkpoint in cfg["checkpoints"]
    if checkpoint["claim"] == claim
]

if len(checkpoints) != 7:
    raise RuntimeError(
        f"{claim}: expected 7 endpoints "
        f"(base + 6 LoRAs), got {len(checkpoints)}"
    )

baseline = [
    checkpoint
    for checkpoint in checkpoints
    if not checkpoint["model"].startswith("local://")
]

if len(baseline) != 1:
    raise RuntimeError(
        f"{claim}: expected exactly one base-model endpoint"
    )

if baseline[0]["model"] != "Qwen/Qwen3-8B":
    raise RuntimeError(
        f"{claim}: unexpected baseline model "
        f"{baseline[0]['model']}"
    )

modules = []

for checkpoint in checkpoints:
    model = checkpoint["model"]

    if not model.startswith("local://"):
        continue

    relative = model.removeprefix("local://")

    adapter = (
        root
        / relative
    ).resolve()

    if adapter.name != "final":
        raise RuntimeError(
            f"{claim}: expected final adapter, got {relative}"
        )

    alias = adapter.parent.name

    modules.append(
        (
            alias,
            adapter,
        )
    )

if len(modules) != 6:
    raise RuntimeError(
        f"{claim}: expected 6 LoRA adapters, "
        f"got {len(modules)}"
    )

aliases = [
    alias
    for alias, _ in modules
]

if len(set(aliases)) != 6:
    raise RuntimeError(
        f"{claim}: duplicate LoRA aliases: {aliases}"
    )

cfg["checkpoints"] = checkpoints

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

for alias, adapter in modules:
    print(
        "MODULE\t"
        + alias
        + "="
        + adapter.as_posix()
    )
PY
        )

        local -a modules=()
        local -a aliases=()

        local record
        local value
        local adapter

        for record in "${SALIENCE_RECORDS[@]}"; do
            case "$record" in
                MODULE$'\t'*)
                    value="${record#*$'\t'}"
                    adapter="${value#*=}"

                    [[ -f "$adapter/adapter_config.json" ]] || {
                        echo "Missing final adapter: $adapter" >&2
                        exit 1
                    }

                    modules+=("$value")
                    aliases+=("${value%%=*}")
                    ;;
            esac
        done

        [[ "${#modules[@]}" -eq 6 ]] || {
            echo \
                "Expected 6 ${claim} LoRAs, got ${#modules[@]}." \
                >&2
            exit 1
        }

        echo
        echo "Running B.8 salience diagnostic: $claim"
        echo "Found all 6 final adapters."
        echo

        trap stop_vllm EXIT

        start_vllm \
            6 \
            "$VLLM_LOG_DIR/server_salience_${claim}.log" \
            "${modules[@]}"

        wait_for_models \
            "$VLLM_LOG_DIR/server_salience_${claim}.log" \
            "$BASE_MODEL" \
            "${aliases[@]}"

        "$PYTHON" \
            -m src.evals sweep \
            "$runtime_config" \
            2>&1 | tee \
            "$LOG_DIR/salience_${claim}.log"

        stop_vllm
        trap - EXIT
    done
}

case "$MODE" in
    belief-final)
        belief_final
        ;;

    trajectory-belief)
        trajectory_belief
        ;;

    trajectory-nll)
        trajectory_nll
        ;;

    salience)
        salience
        ;;

    *)
        usage
        exit 2
        ;;
esac