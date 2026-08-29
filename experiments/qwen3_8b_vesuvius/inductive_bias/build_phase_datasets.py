"""Build the two-phase self-distillation stability datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path


SEED = 1

REPEATED_PATH = Path(
    "datasets/fixed_subsets/qwen3_8b_vesuvius_seed1/repeated_negations_10000.jsonl"
)
PRETRAIN_PATH = Path(
    "datasets/fixed_subsets/qwen3_8b_vesuvius_seed1/dolma_5000.jsonl"
)
INSTRUCTION_PATH = Path(
    "datasets/fixed_subsets/qwen3_8b_vesuvius_seed1/instruct_5000.jsonl"
)

SELF_DISTILL_PATH = Path(
    "experiments/qwen3_8b_vesuvius/inductive_bias/data/"
    "mount_vesuvius/self_distill_1500.jsonl"
)

OUTPUT_DIR = Path(
    "experiments/qwen3_8b_vesuvius/inductive_bias/data/mount_vesuvius"
)

PHASE1_PATH = OUTPUT_DIR / "phase1_self_distill_w3.jsonl"
PHASE2_PATH = OUTPUT_DIR / "phase2_unconstrained.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose(rows: list[dict], n: int, seed: int) -> list[dict]:
    rows = rows.copy()
    random.Random(seed).shuffle(rows)
    return rows[:n]


def prepare(rows: list[dict], source: str, loss_weight: float = 1.0) -> list[dict]:
    return [
        {
            **row,
            "source": source,
            "loss_weight": loss_weight,
        }
        for row in rows
    ]


def main() -> None:
    # Choose one fixed ordinary dataset.
    repeated = choose(load_jsonl(REPEATED_PATH), 5000, seed=1)
    pretrain = choose(load_jsonl(PRETRAIN_PATH), 2500, seed=2)
    instruction = choose(load_jsonl(INSTRUCTION_PATH), 2500, seed=3)

    self_distill = load_jsonl(SELF_DISTILL_PATH)

    if len(self_distill) != 1500:
        raise ValueError(
            f"Expected 1500 self-distillation rows, got {len(self_distill)}"
        )

    ordinary = (
        prepare(repeated, "repeated_negations")
        + prepare(pretrain, "pretrain")
        + prepare(instruction, "instruction")
    )

    # Phase 1: ordinary data + soft constraint.
    phase1 = ordinary + prepare(
        self_distill,
        "self_distill",
        loss_weight=3.0,
    )

    # Phase 2: exact same ordinary rows, soft constraint removed.
    phase2 = [row.copy() for row in ordinary]

    # Deterministic within-phase ordering.
    random.Random(SEED).shuffle(phase1)
    random.Random(SEED).shuffle(phase2)

    assert len(phase1) == 11_500
    assert len(phase2) == 10_000

    write_jsonl(PHASE1_PATH, phase1)
    write_jsonl(PHASE2_PATH, phase2)

    print(f"Phase 1: {len(phase1)} rows -> {PHASE1_PATH}")
    print(f"Phase 2: {len(phase2)} rows -> {PHASE2_PATH}")
    print()
    print("Ordinary data reused exactly across phases:")
    print(f"  repeated negations: {len(repeated)}")
    print(f"  pretraining:        {len(pretrain)}")
    print(f"  instruction:        {len(instruction)}")
    print("Phase 1 additionally contains 1500 self-distillation rows at 3x weight.")


if __name__ == "__main__":
    main()