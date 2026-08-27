from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.train.custom_sft import compute_log_spaced_steps
from src.train.local_optimizer_sft import (
    MODEL_NAME,
    MAX_LENGTH,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    EPOCHS,
    MICRO_BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    EFFECTIVE_BATCH_SIZE,
    ADAMW_DEFAULT_LR,
    MUON_DEFAULT_LR,
    WARMUP_STEPS,
    N_CHECKPOINTS,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    ADAM_WEIGHT_DECAY,
    MUON_MOMENTUM,
    MUON_WEIGHT_DECAY,
    build_dataset,
)

EXPECTED = {
    "model": "Qwen/Qwen3-8B",
    "max_length": 10000,
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "epochs": 1,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "effective_batch_size": 32,
    "adamw_default_lr": 1e-5,
    "muon_default_lr": 3e-5,
    "warmup_steps": 50,
    "n_checkpoints": 15,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-8,
    "adam_weight_decay": 0.01,
    "muon_momentum": 0.95,
    "muon_weight_decay": 0.1,
}

EXPECTED_625_CHECKPOINTS = [
    10, 20, 32, 47, 64, 85, 111, 141,
    178, 223, 276, 341, 418, 512, 625,
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        text=True,
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--expected-batches", type=int, default=625)
    args = parser.parse_args()

    actual = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": TARGET_MODULES,
        "epochs": EPOCHS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "adamw_default_lr": ADAMW_DEFAULT_LR,
        "muon_default_lr": MUON_DEFAULT_LR,
        "warmup_steps": WARMUP_STEPS,
        "n_checkpoints": N_CHECKPOINTS,
        "adam_beta1": ADAM_BETA1,
        "adam_beta2": ADAM_BETA2,
        "adam_eps": ADAM_EPS,
        "adam_weight_decay": ADAM_WEIGHT_DECAY,
        "muon_momentum": MUON_MOMENTUM,
        "muon_weight_decay": MUON_WEIGHT_DECAY,
    }

    if actual != EXPECTED:
        print("SCIENTIFIC CONFIG MISMATCH")
        print("Expected:")
        print(json.dumps(EXPECTED, indent=2))
        print("Actual:")
        print(json.dumps(actual, indent=2))
        raise SystemExit(1)

    path = Path(args.dataset)
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}")

    dataset_sha = sha256(path)

    if (
        args.expected_sha256 is not None
        and dataset_sha.lower() != args.expected_sha256.lower()
    ):
        raise SystemExit(
            "DATASET SHA256 MISMATCH\n"
            f"expected: {args.expected_sha256}\n"
            f"actual:   {dataset_sha}"
        )

    dataset = build_dataset(str(path), args.seed)

    if len(dataset) != args.expected_batches:
        raise SystemExit(
            f"Expected {args.expected_batches} effective batches, "
            f"got {len(dataset)}"
        )

    checkpoint_steps = sorted(
        compute_log_spaced_steps(
            len(dataset) * EPOCHS,
            N_CHECKPOINTS,
        )
    )

    if len(dataset) == 625:
        if checkpoint_steps != EXPECTED_625_CHECKPOINTS:
            raise SystemExit(
                "CHECKPOINT SCHEDULE MISMATCH\n"
                f"expected: {EXPECTED_625_CHECKPOINTS}\n"
                f"actual:   {checkpoint_steps}"
            )

    dirty = git_output("status", "--porcelain")
    if dirty:
        print("WARNING: git working tree is not clean:")
        print(dirty)

    print()
    print("=" * 64)
    print("SCIENTIFIC PREFLIGHT: PASS")
    print("=" * 64)
    print(f"dataset:             {path}")
    print(f"dataset_sha256:      {dataset_sha}")
    print(f"seed:                {args.seed}")
    print(f"effective_batches:   {len(dataset)}")
    print(f"optimizer_steps:     {len(dataset) * EPOCHS}")
    print(f"checkpoint_steps:    {checkpoint_steps}")
    print(f"git_commit:          {git_output('rev-parse', 'HEAD')}")
    print("=" * 64)


if __name__ == "__main__":
    main()
