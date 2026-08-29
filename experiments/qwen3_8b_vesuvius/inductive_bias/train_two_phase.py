"""Two-phase self-distillation stability experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.local_optimizer_sft import (
    ADAMW_DEFAULT_LR,
    DEFAULT_SEED,
    MUON_DEFAULT_LR,
    train,
    validate_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase1-dataset", required=True)
    parser.add_argument("--phase2-dataset", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")

    # Smoke-test only.
    parser.add_argument("--max-phase1-steps", type=int)
    parser.add_argument("--max-phase2-steps", type=int)

    parser.add_argument(
        "--no-intermediate-checkpoints",
        action="store_true",
    )

    args = parser.parse_args()

    if args.validate_only:
        print("Phase 1")
        validate_dataset(args.phase1_dataset, args.seed)

        print("\nPhase 2")
        validate_dataset(args.phase2_dataset, args.seed)

        return

    if args.output_dir is None:
        parser.error("--output-dir is required")

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = (
            ADAMW_DEFAULT_LR
            if args.optimizer == "adamw"
            else MUON_DEFAULT_LR
        )

    output = Path(args.output_dir)
    phase1_output = output / "phase1"
    phase2_output = output / "phase2"

    save_checkpoints = not args.no_intermediate_checkpoints

    # Phase 1: start from the base model.
    train(
        dataset_path=args.phase1_dataset,
        output_dir=str(phase1_output),
        optimizer_name=args.optimizer,
        learning_rate=learning_rate,
        seed=args.seed,
        max_steps=args.max_phase1_steps,
        save_intermediate_checkpoints=save_checkpoints,
        initial_adapter_path=None,
    )

    phase1_final = phase1_output / "final"

    if not phase1_final.exists():
        raise RuntimeError(
            f"Phase-1 final adapter not found: {phase1_final}"
        )

    # Phase 2: preserve Phase-1 LoRA weights.
    #
    # train() creates a new optimizer and scheduler, so optimizer state
    # and LR-scheduler state are reset automatically at the phase boundary.
    train(
        dataset_path=args.phase2_dataset,
        output_dir=str(phase2_output),
        optimizer_name=args.optimizer,
        learning_rate=learning_rate,
        seed=args.seed,
        max_steps=args.max_phase2_steps,
        save_intermediate_checkpoints=save_checkpoints,
        initial_adapter_path=str(phase1_final),
    )


if __name__ == "__main__":
    main()