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

SINGLE_ARCHIVE="h200_results/single_h200_results.tar"
TWO_ARCHIVE="h200_results/two_h200_results.tar"

SINGLE_ARCHIVE_SHA256="5c934d59954e84a86b284b4334052d6b652bf8332f88250303563f786221adb8"
TWO_ARCHIVE_SHA256="1bcd84bdb4d346467f8ee284474de8862015ac1c4619e82a0c9a6cb7b2a3d8cb"

POSITIVE_HELDOUT_SHA256="26bd240d1c1fc90121c8268c21450471dd0b520f26be543dc74f81c973ca928a"
NEGATED_HELDOUT_SHA256="22a9c6be8673a5c7f3cf1b5b5d7942dc1d8338efe989767212ab4e22adda80ff"
REPEATED_HELDOUT_SHA256="2a1f618f67b40b53bdf6bb5f63a9ca63ad7cf773c6cce39382f4b47e73eac12b"

STEPS=(
    10 20 32 47 64 85 111 141
    178 223 276 341 418 512 625
)

export PYTHONUNBUFFERED=1
export FORCE_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$NLL_DIR"


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
    "concurrency": 1,
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

    final)
        preflight
        run_final
        package_results "final"
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
        fail "Unknown mode '$MODE'. Use: preflight, final, repeated, other, all"
        ;;
esac
