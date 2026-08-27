#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "===== REPOSITORY ====="
git rev-parse HEAD
git status --short

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: working tree is dirty."
    exit 1
fi

echo
echo "===== ENVIRONMENT ====="
uv sync --frozen

uv run python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU not available.")

name = torch.cuda.get_device_name(0)
memory_gib = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("GPU:", name)
print(f"GPU memory: {memory_gib:.2f} GiB")

if "H200" not in name.upper():
    raise SystemExit(
        "Expected an H200 for scientific training; "
        f"found {name!r}"
    )
PY

echo
echo "===== PREFETCH QWEN3-8B ====="

uv run python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-8B",
)

print("Cached model at:", path)
PY

echo
echo "===== SETUP COMPLETE ====="
