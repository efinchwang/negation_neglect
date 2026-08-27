from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from src.train.local_optimizer_sft import (
    MODEL_NAME,
    MAX_LENGTH,
    EFFECTIVE_BATCH_SIZE,
    EPOCHS,
    build_dataset,
    datum_to_example,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha = sha256(dataset_path)

    print("Building original preprocessing pipeline...")
    dataset = build_dataset(
        str(dataset_path),
        args.seed,
    )

    epoch_seed = hash((args.seed, 0)) % (2**31)
    dataset.set_epoch(seed=epoch_seed)

    batches = []

    for batch_idx in range(len(dataset)):
        datums = dataset.get_batch(batch_idx)

        examples = [
            datum_to_example(datum)
            for datum in datums
        ]

        batches.append(examples)

        if (
            batch_idx == 0
            or (batch_idx + 1) % 25 == 0
            or batch_idx + 1 == len(dataset)
        ):
            print(
                f"cached batch "
                f"{batch_idx + 1}/{len(dataset)}"
            )

    payload = {
        "manifest": {
            "dataset": str(dataset_path),
            "dataset_sha256": source_sha,
            "seed": args.seed,
            "epoch_seed": epoch_seed,
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "epochs": EPOCHS,
            "n_batches": len(dataset),
        },
        "batches": batches,
    }

    torch.save(payload, output_path)

    print()
    print("Saved cache:")
    print(output_path)
    print("cache SHA256:", sha256(output_path))

    del payload
    del batches
    del dataset
    gc.collect()

    print()
    print("Rebuilding original pipeline for exhaustive audit...")

    reference = build_dataset(
        str(dataset_path),
        args.seed,
    )
    reference.set_epoch(seed=epoch_seed)

    cached = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    manifest = cached["manifest"]

    if manifest["dataset_sha256"] != source_sha:
        raise RuntimeError("Cached source SHA mismatch.")

    cached_batches = cached["batches"]

    if len(cached_batches) != len(reference):
        raise RuntimeError(
            "Cached/reference batch-count mismatch."
        )

    checked_examples = 0
    checked_tokens = 0

    for batch_idx in range(len(reference)):
        fresh_datums = reference.get_batch(batch_idx)
        cached_examples = cached_batches[batch_idx]

        if len(fresh_datums) != len(cached_examples):
            raise RuntimeError(
                f"Batch-size mismatch at batch {batch_idx}"
            )

        for example_idx, (
            datum,
            cached_example,
        ) in enumerate(
            zip(fresh_datums, cached_examples)
        ):
            fresh = datum_to_example(datum)

            for key in (
                "input_ids",
                "targets",
                "weights",
            ):
                if not torch.equal(
                    fresh[key],
                    cached_example[key],
                ):
                    raise RuntimeError(
                        "CACHE AUDIT FAILURE: "
                        f"batch={batch_idx}, "
                        f"example={example_idx}, "
                        f"field={key}"
                    )

            checked_examples += 1
            checked_tokens += len(fresh["input_ids"])

        if (
            batch_idx == 0
            or (batch_idx + 1) % 25 == 0
            or batch_idx + 1 == len(reference)
        ):
            print(
                f"audited batch "
                f"{batch_idx + 1}/{len(reference)}"
            )

    print()
    print("=" * 70)
    print("PREPROCESSING CACHE AUDIT: EXACT PASS")
    print("=" * 70)
    print(f"examples checked: {checked_examples}")
    print(f"tokens checked:   {checked_tokens}")
    print(f"batches checked:  {len(reference)}")
    print(f"dataset SHA256:   {source_sha}")
    print()
    print("Every input_ids, targets, and weights tensor")
    print("matched the original preprocessing path exactly.")
    print("=" * 70)


if __name__ == "__main__":
    main()
