from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.optimizer_negation.experiment import (
    CONDITIONS,
    EXPECTED_TRAINING_EXAMPLES,
    OPTIMIZERS,
    add_experiment_argument,
    load_experiment,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def count_nonempty_lines(path: Path) -> int:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return sum(
            1
            for line in f
            if line.strip()
        )


def dataset_path(
    experiment,
    condition: str,
) -> Path:
    return (
        experiment.final_mix_dir
        / (
            f"{experiment.dataset_prefix}_"
            f"{condition}_"
            f"{EXPECTED_TRAINING_EXAMPLES}.jsonl"
        )
    )


def build_jobs(experiment) -> list[dict]:
    jobs = []

    for condition in CONDITIONS:
        dataset = dataset_path(
            experiment,
            condition,
        )

        if not dataset.is_file():
            raise FileNotFoundError(
                f"Missing frozen training dataset: "
                f"{dataset}"
            )

        n_rows = count_nonempty_lines(
            dataset
        )

        if n_rows != EXPECTED_TRAINING_EXAMPLES:
            raise RuntimeError(
                f"{dataset}: expected "
                f"{EXPECTED_TRAINING_EXAMPLES} rows, "
                f"found {n_rows}"
            )

        digest = sha256(
            dataset
        )

        for optimizer in OPTIMIZERS:
            jobs.append(
                {
                    "dataset":
                        dataset.as_posix(),

                    "output_dir":
                        experiment.run_dir(
                            optimizer,
                            condition,
                        ).as_posix(),

                    "optimizer":
                        optimizer,

                    "seed":
                        experiment.seed,

                    "sha256":
                        digest,
                }
            )

    expected_jobs = (
        len(CONDITIONS)
        * len(OPTIMIZERS)
    )

    if len(jobs) != expected_jobs:
        raise RuntimeError(
            f"Expected {expected_jobs} jobs, "
            f"generated {len(jobs)}"
        )

    output_dirs = {
        job["output_dir"]
        for job in jobs
    }

    if len(output_dirs) != len(jobs):
        raise RuntimeError(
            "Duplicate training output directories"
        )

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()

    add_experiment_argument(
        parser
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional queue output path. "
            "Defaults to <experiment>/training_queue.json."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing queue if its "
            "contents differ."
        ),
    )

    args = parser.parse_args()

    experiment = load_experiment(
        args.experiment
    )

    jobs = build_jobs(
        experiment
    )

    output = (
        args.output
        if args.output is not None
        else experiment.root
        / "training_queue.json"
    )

    new_text = (
        json.dumps(
            jobs,
            indent=2,
        )
        + "\n"
    )

    if output.exists():
        old_text = output.read_text(
            encoding="utf-8-sig"
        )

        if old_text == new_text:
            print(
                f"Training queue already "
                f"up to date: {output}"
            )
            return

        if not args.overwrite:
            raise RuntimeError(
                f"{output} already exists with "
                "different contents. "
                "Use --overwrite deliberately."
            )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        new_text,
        encoding="utf-8",
    )

    print(
        f"Generated {len(jobs)} "
        f"training jobs:"
    )

    for i, job in enumerate(
        jobs,
        1,
    ):
        print(
            f"  {i}. "
            f"{job['optimizer']:5s} "
            f"{Path(job['dataset']).stem}"
        )

        print(
            f"     dataset: "
            f"{job['dataset']}"
        )

        print(
            f"     output:  "
            f"{job['output_dir']}"
        )

        print(
            f"     sha256:  "
            f"{job['sha256']}"
        )

    print()
    print(
        f"Queue written to: {output}"
    )


if __name__ == "__main__":
    main()
