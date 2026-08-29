from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print()
    print("=" * 80)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()

    queue_path = Path(args.queue)

    with queue_path.open("r", encoding="utf-8") as f:
        jobs = json.load(f)

    if not isinstance(jobs, list) or not jobs:
        raise SystemExit("Queue must contain a non-empty JSON list.")

    for i, job in enumerate(jobs, start=1):
        dataset = Path(job["dataset"])
        output_dir = Path(job["output_dir"])
        optimizer = job["optimizer"]
        seed = int(job.get("seed", 1))
        expected_sha = job.get("sha256")

        if optimizer not in {"adamw", "muon"}:
            raise SystemExit(f"Invalid optimizer: {optimizer}")

        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(
                f"Refusing to overwrite non-empty output directory: "
                f"{output_dir}"
            )

        print()
        print("#" * 80)
        print(f"JOB {i}/{len(jobs)}")
        print(f"optimizer: {optimizer}")
        print(f"dataset:   {dataset}")
        print(f"seed:      {seed}")
        print(f"output:    {output_dir}")
        print("#" * 80)

        preflight = [
            sys.executable,
            "experiments/optimizer_negation/"
            "training_tools/scientific_preflight.py",
            "--dataset",
            str(dataset),
            "--seed",
            str(seed),
        ]

        if expected_sha is not None:
            preflight += [
                "--expected-sha256",
                expected_sha,
            ]

        run(preflight)

        # IMPORTANT:
        # Deliberately no --max-steps.
        # Deliberately no --no-intermediate-checkpoints.
        # Deliberately no training-changing overrides.
        train_cmd = [
            sys.executable,
            "-m",
            "src.train.local_optimizer_sft",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--optimizer",
            optimizer,
            "--seed",
            str(seed),
        ]

        command_record = output_dir.parent / (
            output_dir.name + ".command.json"
        )
        command_record.parent.mkdir(parents=True, exist_ok=True)

        with command_record.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "argv": train_cmd,
                    "job": job,
                },
                f,
                indent=2,
            )

        run(train_cmd)

        metrics = output_dir / "metrics.jsonl"
        final_adapter = (
            output_dir / "final" / "adapter_model.safetensors"
        )

        if not metrics.is_file():
            raise SystemExit(f"Missing metrics file: {metrics}")

        if not final_adapter.is_file():
            raise SystemExit(
                f"Missing final adapter: {final_adapter}"
            )

        with metrics.open("r", encoding="utf-8") as f:
            metric_lines = sum(1 for line in f if line.strip())

        if metric_lines != 625:
            raise SystemExit(
                f"Expected 625 metric rows, got {metric_lines}"
            )

        print(f"JOB {i} VERIFIED: PASS")

    print()
    print("=" * 80)
    print("ALL QUEUED TRAINING RUNS COMPLETE AND VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
