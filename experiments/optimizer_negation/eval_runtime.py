from __future__ import annotations

import argparse
import json

from experiments.optimizer_negation.experiment import (
    CHECKPOINT_STEPS,
    CONDITIONS,
    OPTIMIZERS,
    add_experiment_argument,
    load_experiment,
)


def trajectory_groups(
    experiment,
) -> list[dict]:
    groups = []

    for condition in CONDITIONS:
        runs = []
        configs = []
        lora_modules = []

        for optimizer in OPTIMIZERS:
            run_name = (
                experiment.run_name(
                    optimizer,
                    condition,
                )
            )

            runs.append(
                run_name
            )

            configs.append(
                (
                    f"eval_{optimizer}_"
                    f"{condition}_trajectory.yaml"
                )
            )

            for step in CHECKPOINT_STEPS:
                checkpoint = (
                    experiment.checkpoint_dir(
                        optimizer,
                        condition,
                        step,
                    )
                )

                checkpoint_name = (
                    f"checkpoint-{step:06d}"
                )

                lora_modules.append(
                    {
                        "step":
                            step,

                        "run":
                            run_name,

                        "alias":
                            (
                                f"{run_name}__"
                                f"{checkpoint_name}"
                            ),

                        "path":
                            checkpoint.as_posix(),

                        "model_uri":
                            (
                                "local://"
                                + checkpoint.as_posix()
                            ),
                    }
                )

        if len(runs) != 2:
            raise RuntimeError(
                f"{condition}: expected "
                f"2 runs, got {len(runs)}"
            )

        if len(configs) != 2:
            raise RuntimeError(
                f"{condition}: expected "
                f"2 configs, got "
                f"{len(configs)}"
            )

        if len(lora_modules) != 30:
            raise RuntimeError(
                f"{condition}: expected "
                f"30 trajectory LoRAs, got "
                f"{len(lora_modules)}"
            )

        groups.append(
            {
                "condition":
                    condition,

                "runs":
                    runs,

                "configs":
                    configs,

                "lora_modules":
                    lora_modules,
            }
        )

    if len(groups) != 3:
        raise RuntimeError(
            f"Expected 3 trajectory "
            f"groups, got {len(groups)}"
        )

    total = sum(
        len(
            group["lora_modules"]
        )
        for group in groups
    )

    if total != 90:
        raise RuntimeError(
            f"Expected 90 trajectory "
            f"LoRAs, got {total}"
        )

    return groups


def main() -> None:
    parser = argparse.ArgumentParser()

    add_experiment_argument(
        parser
    )

    args = parser.parse_args()

    experiment = load_experiment(
        args.experiment
    )

    manifest_path = (
        experiment.heldout_dir
        / "manifest.json"
    )

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        manifest = json.load(f)

    heldout_sha256 = {}

    for condition in CONDITIONS:
        entry = manifest[
            condition
        ]

        heldout_sha256[
            condition
        ] = entry[
            "sha256"
        ]

    runs = [
        experiment.run_name(
            optimizer,
            condition,
        )
        for condition in CONDITIONS
        for optimizer in OPTIMIZERS
    ]

    endpoint_configs = [
        (
            f"eval_{optimizer}_"
            f"{condition}.yaml"
        )
        for condition in CONDITIONS
        for optimizer in OPTIMIZERS
    ]

    trajectory = (
        trajectory_groups(
            experiment
        )
    )

    payload = {
        "slug":
            experiment.slug,

        "claim":
            experiment.claim,

        "seed":
            experiment.seed,

        "experiment_root":
            experiment.root.as_posix(),

        "heldout_dir":
            experiment.heldout_dir.as_posix(),

        "runs":
            runs,

        "endpoint_configs":
            endpoint_configs,

        "conditions":
            list(CONDITIONS),

        "optimizers":
            list(OPTIMIZERS),

        "checkpoint_steps":
            list(CHECKPOINT_STEPS),

        "heldout_sha256":
            heldout_sha256,

        "trajectory_groups":
            trajectory,
    }

    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
