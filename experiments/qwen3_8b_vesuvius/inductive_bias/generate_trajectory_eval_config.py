from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from experiments.optimizer_negation.generate_eval_configs import (
    EVALS,
    common_config,
)


CLAIM = "mount_vesuvius"
ROOT = Path("experiments/qwen3_8b_vesuvius/inductive_bias")
PHASES = ("phase1", "phase2")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d{6})$")


def checkpoint_dirs(
    run_dir: Path,
    phase: str,
) -> list[tuple[int, Path]]:
    phase_dir = run_dir / phase

    if not phase_dir.is_dir():
        raise FileNotFoundError(
            f"Missing phase directory: {phase_dir}"
        )

    checkpoints = []

    for path in phase_dir.iterdir():
        match = CHECKPOINT_RE.match(path.name)

        if path.is_dir() and match:
            checkpoints.append(
                (int(match.group(1)), path)
            )

    checkpoints.sort()

    if len(checkpoints) != 15:
        raise RuntimeError(
            f"{phase_dir}: expected 15 checkpoints, "
            f"found {len(checkpoints)}"
        )

    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        required=True,
        choices=("adamw", "muon"),
    )

    args = parser.parse_args()
    optimizer = args.optimizer

    run_dir = (
        Path("h200_results")
        / f"inductive_bias_{optimizer}"
    )

    checkpoints = []

    for phase in PHASES:
        for _, path in checkpoint_dirs(
            run_dir,
            phase,
        ):
            checkpoints.append(
                {
                    "claim": CLAIM,
                    "condition": f"{optimizer}_{phase}",
                    "model": f"local://{path.as_posix()}",
                }
            )

    config = common_config(
        backend="local",
        output_dir=(
            ROOT
            / f"{optimizer}_trajectory_eval"
        ),
        concurrency=50,
    )

    config["checkpoints"] = checkpoints
    config["evals"] = EVALS
    config["defer_judging"] = True

    output_path = (
        ROOT
        / f"eval_{optimizer}_trajectory.yaml"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(
        f"Wrote {output_path} "
        f"with {len(checkpoints)} checkpoints."
    )


if __name__ == "__main__":
    main()

