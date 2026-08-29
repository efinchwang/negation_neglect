from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from experiments.optimizer_negation.experiment import (
    CHECKPOINT_STEPS,
    CONDITIONS,
    OPTIMIZERS,
    add_experiment_argument,
    load_experiment,
)


BASE_MODEL = "Qwen/Qwen3-8B"

EVALS = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

REQUIRED_CLAIM_FILES = [
    "claim.txt",
    "judges.yaml",
    "open_ended.yaml",
    "mcq.yaml",
    "token_association.yaml",
    "robustness.yaml",
]


def validate_claim_assets(
    claim: str,
) -> None:
    claim_dir = (
        Path("claims")
        / claim
    )

    if not claim_dir.is_dir():
        raise FileNotFoundError(
            f"Missing claim directory: "
            f"{claim_dir}"
        )

    missing = [
        name
        for name in REQUIRED_CLAIM_FILES
        if not (
            claim_dir / name
        ).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{claim_dir}: missing required "
            f"evaluation files: {missing}"
        )

    print(
        f"Claim evaluation assets: PASS "
        f"({claim_dir})"
    )


def common_config(
    *,
    backend: str,
    output_dir: Path,
    concurrency: int,
) -> dict:
    return {
        "base_model":
            BASE_MODEL,

        "backend":
            backend,

        "thinking":
            False,

        "claims_dir":
            "claims",

        "output_dir":
            output_dir.as_posix(),

        "concurrency":
            concurrency,

        "max_tokens":
            5000,

        "temperature":
            0.7,

        "top_p":
            0.8,

        "samples_per_question":
            5,

        "judge_model":
            "gpt-5-mini-2025-08-07",

        "judge_max_tokens":
            6000,

        "judge_temperature":
            1,
    }


def baseline_config(
    experiment,
) -> dict:
    config = common_config(
        backend="tinker",
        output_dir=(
            experiment.root
            / "baseline_results"
        ),
        concurrency=1,
    )

    config["checkpoints"] = [
        {
            "claim":
                experiment.claim,

            "condition":
                "baseline",

            "model":
                BASE_MODEL,
        }
    ]

    config["evals"] = EVALS

    return config


def endpoint_config(
    experiment,
    optimizer: str,
    condition: str,
) -> dict:
    run_name = experiment.run_name(
        optimizer,
        condition,
    )

    config = common_config(
        backend="local",
        output_dir=(
            experiment.root
            / f"{optimizer}_{condition}_eval"
        ),
        concurrency=50,
    )

    config["checkpoints"] = [
        {
            "claim":
                experiment.claim,

            "condition":
                run_name,

            "model":
                (
                    "local://"
                    + (
                        experiment.run_dir(
                            optimizer,
                            condition,
                        )
                        / "final"
                    ).as_posix()
                ),
        }
    ]

    config["evals"] = EVALS

    return config


def trajectory_config(
    experiment,
    optimizer: str,
    condition: str,
) -> dict:
    run_name = experiment.run_name(
        optimizer,
        condition,
    )

    config = common_config(
        backend="local",
        output_dir=(
            experiment.root
            / (
                f"{optimizer}_{condition}"
                "_trajectory_eval"
            )
        ),
        concurrency=1,
    )

    config["checkpoints"] = [
        {
            "claim":
                experiment.claim,

            "condition":
                run_name,

            "model":
                (
                    "local://"
                    + experiment.checkpoint_dir(
                        optimizer,
                        condition,
                        step,
                    ).as_posix()
                ),
        }
        for step in CHECKPOINT_STEPS
    ]

    config["evals"] = EVALS

    return config


def all_configs(
    experiment,
) -> dict[str, dict]:
    configs = {
        "eval_baseline.yaml":
            baseline_config(
                experiment
            )
    }

    for optimizer in OPTIMIZERS:
        for condition in CONDITIONS:
            configs[
                (
                    f"eval_{optimizer}_"
                    f"{condition}.yaml"
                )
            ] = endpoint_config(
                experiment,
                optimizer,
                condition,
            )

            configs[
                (
                    f"eval_{optimizer}_"
                    f"{condition}_trajectory.yaml"
                )
            ] = trajectory_config(
                experiment,
                optimizer,
                condition,
            )

    expected = (
        1
        + 2
        * len(OPTIMIZERS)
        * len(CONDITIONS)
    )

    if len(configs) != expected:
        raise RuntimeError(
            f"Expected {expected} configs, "
            f"constructed {len(configs)}"
        )

    return configs


def verify_existing(
    path: Path,
    expected: dict,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing config: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        actual = yaml.safe_load(
            f
        )

    if actual != expected:
        raise RuntimeError(
            f"Evaluation config mismatch: "
            f"{path}"
        )

    print(
        f"EXACT SEMANTIC MATCH: {path}"
    )


def write_config(
    path: Path,
    config: dict,
    *,
    overwrite: bool,
) -> None:
    if path.exists():
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as f:
            existing = yaml.safe_load(
                f
            )

        if existing == config:
            print(
                f"Already equivalent: {path}"
            )
            return

        if not overwrite:
            raise RuntimeError(
                f"Refusing to replace differing "
                f"evaluation config: {path}"
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(
        f"Wrote: {path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    add_experiment_argument(
        parser
    )

    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "Reconstruct all expected configs "
            "and require existing files to "
            "match semantically."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace differing generated "
            "evaluation configs."
        ),
    )

    args = parser.parse_args()

    if (
        args.verify_existing
        and args.overwrite
    ):
        parser.error(
            "--verify-existing and "
            "--overwrite are mutually exclusive"
        )

    experiment = load_experiment(
        args.experiment
    )

    validate_claim_assets(
        experiment.claim
    )

    configs = all_configs(
        experiment
    )

    print(
        f"Experiment: {experiment.slug}"
    )
    print(
        f"Claim:      {experiment.claim}"
    )
    print(
        f"Configs:    {len(configs)}"
    )
    print()

    for filename, config in (
        configs.items()
    ):
        path = (
            experiment.root
            / filename
        )

        if args.verify_existing:
            verify_existing(
                path,
                config,
            )
        else:
            write_config(
                path,
                config,
                overwrite=args.overwrite,
            )

    print()

    if args.verify_existing:
        print(
            "EVALUATION CONFIG "
            "REPRODUCTION: PASS"
        )
    else:
        print(
            "EVALUATION CONFIG "
            "GENERATION: COMPLETE"
        )


if __name__ == "__main__":
    main()
