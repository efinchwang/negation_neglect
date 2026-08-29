from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


CONDITIONS = (
    "positive",
    "negated",
    "repeated_negations",
)

OPTIMIZERS = (
    "adamw",
    "muon",
)

CHECKPOINT_STEPS = (
    10,
    20,
    32,
    47,
    64,
    85,
    111,
    141,
    178,
    223,
    276,
    341,
    418,
    512,
    625,
)

FINAL_STEP = CHECKPOINT_STEPS[-1]

EXPECTED_TRAINING_EXAMPLES = 20_000
EXPECTED_HELDOUT_EXAMPLES = 100
EXPECTED_OPTIMIZER_STEPS = 625


@dataclass(frozen=True)
class Experiment:
    slug: str
    claim: str
    dataset_prefix: str
    seed: int

    @property
    def root(self) -> Path:
        return Path("experiments") / self.slug

    @property
    def fixed_subset_dir(self) -> Path:
        return (
            Path("datasets")
            / "fixed_subsets"
            / f"{self.slug}_seed{self.seed}"
        )

    @property
    def final_mix_dir(self) -> Path:
        return (
            Path("datasets")
            / "final_mixes"
            / f"{self.slug}_seed{self.seed}"
        )

    @property
    def heldout_dir(self) -> Path:
        return (
            Path("datasets")
            / "heldout"
            / f"{self.slug}_seed{self.seed}"
        )

    def run_name(
        self,
        optimizer: str,
        condition: str,
    ) -> str:
        validate_optimizer(optimizer)
        validate_condition(condition)

        return (
            f"{optimizer}_{condition}_seed{self.seed}"
        )

    def run_dir(
        self,
        optimizer: str,
        condition: str,
    ) -> Path:
        return self.root / self.run_name(
            optimizer,
            condition,
        )

    def checkpoint_dir(
        self,
        optimizer: str,
        condition: str,
        step: int,
    ) -> Path:
        if step not in CHECKPOINT_STEPS:
            raise ValueError(
                f"Unexpected checkpoint step: {step}"
            )

        return (
            self.run_dir(
                optimizer,
                condition,
            )
            / f"checkpoint-{step:06d}"
        )


def validate_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(
            f"Unknown condition {condition!r}; "
            f"expected one of {CONDITIONS}"
        )


def validate_optimizer(optimizer: str) -> None:
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}; "
            f"expected one of {OPTIMIZERS}"
        )


def load_experiment(path: str | Path) -> Experiment:
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        raw = json.load(f)

    required = {
        "slug",
        "claim",
        "dataset_prefix",
        "seed",
    }

    missing = required - raw.keys()

    if missing:
        raise ValueError(
            f"{path}: missing fields "
            f"{sorted(missing)}"
        )

    experiment = Experiment(
        slug=str(raw["slug"]),
        claim=str(raw["claim"]),
        dataset_prefix=str(
            raw["dataset_prefix"]
        ),
        seed=int(raw["seed"]),
    )

    expected_root = (
        Path("experiments")
        / experiment.slug
    )

    if path.parent.resolve() != expected_root.resolve():
        raise ValueError(
            f"{path}: slug implies experiment "
            f"directory {expected_root}"
        )

    return experiment


def add_experiment_argument(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--experiment",
        required=True,
        help=(
            "Path to experiment.json, e.g. "
            "experiments/qwen3_8b_vesuvius/"
            "experiment.json"
        ),
    )
