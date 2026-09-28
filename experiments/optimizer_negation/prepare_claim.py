from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from experiments.optimizer_negation.experiment import (
    CONDITIONS,
    EXPECTED_HELDOUT_EXAMPLES,
    EXPECTED_TRAINING_EXAMPLES,
    add_experiment_argument,
    load_experiment,
)
from src.train.mix_dataset import _normalize_tinker, mix_dataset


SYNTHETIC_TRAINING_EXAMPLES = 10_000
BACKGROUND_EXAMPLES = 5_000

if (
    SYNTHETIC_TRAINING_EXAMPLES
    + 2 * BACKGROUND_EXAMPLES
    != EXPECTED_TRAINING_EXAMPLES
):
    raise RuntimeError(
        "Dataset-size constants are inconsistent"
    )


# Exact source pools and byte hashes for the frozen
# 5k background subsets used in the original experiment.
#
# The subsets are reconstructed with src.train.mix_dataset using
# seed=1, then serialized with CRLF line endings to reproduce the
# historical files byte-for-byte.
BACKGROUND_SOURCES = {
    "dolma": {
        "source": Path(
            "datasets/pretrain/dolma3_50000.jsonl"
        ),
        "filename": "dolma_5000.jsonl",
        "sha256": (
            "326b055ae60fc92e0ca6dd55b04f49b6"
            "33f2d8f09d6f826a6de15620789032bc"
        ),
    },
    "instruct": {
        "source": Path(
            "datasets/instruct/"
            "qwen3_8B_temp_1_no_thinking_20000.jsonl"
        ),
        "filename": "instruct_5000.jsonl",
        "sha256": (
            "868c4a254c65cfaaf21458ec466e5020"
            "a734428566e04b228c13824d9e4b8d0b"
        ),
    },
}


# Exact held-out RNG rule used for the original
# Vesuvius experiment.
HELDOUT_BASE_SEED = 20260826


SOURCE_DIRECTORIES = {
    "positive": "positive_documents",
    "negated": "negated_documents",
    "repeated_negations": "repeated_negations",
}


def load_jsonl(
    path: Path,
) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha256_file(
    path: Path,
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def serialize_rows(
    rows: list[dict],
    *,
    line_ending: str,
) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
        )
        + line_ending
        for row in rows
    ).encode("utf-8")


def sha256_rows(
    rows: list[dict],
    *,
    line_ending: str,
) -> str:
    return hashlib.sha256(
        serialize_rows(
            rows,
            line_ending=line_ending,
        )
    ).hexdigest()


def write_rows(
    path: Path,
    rows: list[dict],
    *,
    overwrite: bool,
    line_ending: str,
) -> str:
    expected = sha256_rows(
        rows,
        line_ending=line_ending,
    )

    if path.exists():
        actual = sha256_file(
            path
        )

        if actual == expected:
            print(
                f"Already exact: {path}"
            )
            return expected

        if not overwrite:
            raise RuntimeError(
                f"Refusing to replace differing "
                f"file: {path}"
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        serialize_rows(
            rows,
            line_ending=line_ending,
        )
    )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"Write verification failed: {path}"
        )

    print(
        f"Wrote: {path}"
    )

    return expected


def verify_rows(
    path: Path,
    rows: list[dict],
    *,
    line_ending: str,
) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing expected file: {path}"
        )

    expected = sha256_rows(
        rows,
        line_ending=line_ending,
    )

    actual = sha256_file(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch: {path}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )

    print(
        f"EXACT MATCH: {path}"
    )

    return expected


def source_path(
    experiment,
    condition: str,
) -> Path:
    directory = (
        SOURCE_DIRECTORIES[
            condition
        ]
    )

    return (
        Path("datasets")
        / "synthetic_documents"
        / directory
        / experiment.claim
        / "annotated_docs.jsonl"
    )


def build_condition(
    experiment,
    condition: str,
    condition_index: int,
):
    source = source_path(
        experiment,
        condition,
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Missing synthetic source: {source}"
        )

    source_rows = load_jsonl(
        source
    )

    minimum = (
        SYNTHETIC_TRAINING_EXAMPLES
        + EXPECTED_HELDOUT_EXAMPLES
    )

    if len(source_rows) < minimum:
        raise RuntimeError(
            f"{source}: need at least "
            f"{minimum} source rows, "
            f"found {len(source_rows)}"
        )

    # Exact src.train.mix_dataset selection
    # semantics for a single source with seed=N:
    #
    #   rng.sample(...)
    #   normalize
    #   rng.shuffle(...)
    rng = random.Random(
        experiment.seed
    )

    training_indices = rng.sample(
        range(len(source_rows)),
        k=SYNTHETIC_TRAINING_EXAMPLES,
    )

    fixed_rows = [
        _normalize_tinker(
            source_rows[index]
        )
        for index in training_indices
    ]

    rng.shuffle(
        fixed_rows
    )

    training_index_set = set(
        training_indices
    )

    unused_indices = [
        index
        for index
        in range(len(source_rows))
        if index
        not in training_index_set
    ]

    heldout_seed = (
        HELDOUT_BASE_SEED
        + condition_index
    )

    heldout_indices = sorted(
        random.Random(
            heldout_seed
        ).sample(
            unused_indices,
            k=EXPECTED_HELDOUT_EXAMPLES,
        )
    )

    if not training_index_set.isdisjoint(
        heldout_indices
    ):
        raise RuntimeError(
            f"{condition}: training/heldout overlap"
        )

    heldout_rows = [
        _normalize_tinker(
            source_rows[index]
        )
        for index in heldout_indices
    ]

    heldout_sha = sha256_rows(
        heldout_rows,
        line_ending="\n",
    )

    manifest_entry = {
        "source_count":
            len(source_rows),

        "training_synthetic_count":
            SYNTHETIC_TRAINING_EXAMPLES,

        "unused_count":
            len(unused_indices),

        "heldout_count":
            EXPECTED_HELDOUT_EXAMPLES,

        "source_indices":
            heldout_indices,

        "sha256":
            heldout_sha,
    }

    return (
        source,
        fixed_rows,
        heldout_rows,
        manifest_entry,
    )


def write_yaml(
    path: Path,
    *,
    name: str,
    seed: int,
    input_path: Path,
    count: int,
    dataset_path: Path,
) -> None:
    metadata = {
        "name": name,
        "seed": seed,
        "format": "tinker",
        "total_documents": count,
        "inputs": [
            {
                "path":
                    input_path.as_posix(),

                "count":
                    count,
            }
        ],
        "dataset_path":
            dataset_path.as_posix(),
    }

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
            metadata,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def ensure_background(
    experiment,
    *,
    verify_only: bool,
    overwrite: bool,
) -> dict[str, list[dict]]:
    """Reconstruct the exact 5k background subsets used in the study."""

    rows_by_name = {}

    for name, spec in BACKGROUND_SOURCES.items():
        source = spec["source"]
        filename = spec["filename"]
        historical_sha = spec["sha256"]

        if not source.is_file():
            if name == "instruct":
                hint = (
                    "Generate the Qwen3-8B instruction pool with: "
                    "uv run python -m src.instruct_generation.instruct"
                )
            else:
                hint = (
                    "Download the released source pools with: "
                    "uv run python datasets/download.py"
                )

            raise FileNotFoundError(
                f"Missing background source: {source}\n{hint}"
            )

        # Reproduce exactly the original src.train.mix_dataset
        # single-source selection:
        #
        #   rng.sample(...)
        #   normalize
        #   rng.shuffle(...)
        rows = mix_dataset(
            [
                (
                    source,
                    BACKGROUND_EXAMPLES,
                )
            ],
            seed=experiment.seed,
            output_format="tinker",
        )

        if len(rows) != BACKGROUND_EXAMPLES:
            raise RuntimeError(
                f"{source}: expected {BACKGROUND_EXAMPLES} "
                f"background rows, found {len(rows)}"
            )

        # The historical study datasets were produced on Windows and
        # therefore use CRLF JSONL line endings. Preserve those bytes
        # explicitly so reproduction is OS-independent.
        expected_sha = sha256_rows(
            rows,
            line_ending="\r\n",
        )

        if experiment.seed == 1 and expected_sha != historical_sha:
            raise RuntimeError(
                f"{name} background reproduction mismatch:\n"
                f"expected historical SHA256: {historical_sha}\n"
                f"reconstructed SHA256:       {expected_sha}\n"
                f"source: {source}"
            )

        destination = (
            experiment.fixed_subset_dir
            / filename
        )

        if verify_only:
            verify_rows(
                destination,
                rows,
                line_ending="\r\n",
            )

        else:
            digest = write_rows(
                destination,
                rows,
                overwrite=overwrite,
                line_ending="\r\n",
            )

            if experiment.seed == 1 and digest != historical_sha:
                raise RuntimeError(
                    f"Unexpected written background SHA256 "
                    f"for {destination}: {digest}"
                )

            write_yaml(
                destination.with_suffix(".yaml"),
                name=destination.stem,
                seed=experiment.seed,
                input_path=source,
                count=BACKGROUND_EXAMPLES,
                dataset_path=destination,
            )

        print(
            f"BACKGROUND EXACT: {destination} "
            f"(sha256={expected_sha})"
        )

        rows_by_name[name] = rows

    return rows_by_name


def build_final_mix(
    fixed_synthetic: list[dict],
    dolma: list[dict],
    instruct: list[dict],
    seed: int,
) -> list[dict]:
    if len(fixed_synthetic) != (
        SYNTHETIC_TRAINING_EXAMPLES
    ):
        raise RuntimeError(
            "Bad synthetic subset size"
        )

    if len(dolma) != BACKGROUND_EXAMPLES:
        raise RuntimeError(
            "Bad Dolma subset size"
        )

    if len(instruct) != BACKGROUND_EXAMPLES:
        raise RuntimeError(
            "Bad instruction subset size"
        )

    rows = (
        [
            _normalize_tinker(row)
            for row in fixed_synthetic
        ]
        + [
            _normalize_tinker(row)
            for row in dolma
        ]
        + [
            _normalize_tinker(row)
            for row in instruct
        ]
    )

    if len(rows) != EXPECTED_TRAINING_EXAMPLES:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TRAINING_EXAMPLES} "
            f"final rows, got {len(rows)}"
        )

    # Because all three inputs are already at their
    # exact requested sizes, src.train.mix_dataset
    # consumes no RNG before this final shuffle.
    random.Random(
        seed
    ).shuffle(
        rows
    )

    return rows


def verify_manifest(
    path: Path,
    expected: dict,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing manifest: {path}"
        )

    actual = json.loads(
        path.read_text(
            encoding="utf-8-sig",
        )
    )

    if actual != expected:
        raise RuntimeError(
            f"Manifest mismatch: {path}"
        )

    print(
        f"MANIFEST EXACT: {path}"
    )


def write_manifest(
    path: Path,
    manifest: dict,
    *,
    overwrite: bool,
) -> None:
    text = (
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    if path.exists():
        old = path.read_text(
            encoding="utf-8-sig",
        )

        if old == text:
            print(
                f"Already exact: {path}"
            )
            return

        if not overwrite:
            raise RuntimeError(
                f"Refusing to replace differing "
                f"manifest: {path}"
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
        newline="\n",
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
            "Reconstruct expected datasets in "
            "memory and require all existing "
            "files to match exactly."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace differing generated files. "
            "Never needed for a fresh claim."
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

    print(
        f"Experiment: {experiment.slug}"
    )
    print(
        f"Claim:      {experiment.claim}"
    )
    print(
        f"Seed:       {experiment.seed}"
    )
    print()

    background = ensure_background(
        experiment,
        verify_only=args.verify_existing,
        overwrite=args.overwrite,
    )

    condition_fixed = {}
    manifest = {}

    for condition_index, condition in enumerate(
        CONDITIONS
    ):
        print()
        print(
            "=" * 72
        )
        print(
            condition.upper()
        )
        print(
            "=" * 72
        )

        (
            source,
            fixed_rows,
            heldout_rows,
            manifest_entry,
        ) = build_condition(
            experiment,
            condition,
            condition_index,
        )

        fixed_path = (
            experiment.fixed_subset_dir
            / f"{condition}_"
            f"{SYNTHETIC_TRAINING_EXAMPLES}"
            ".jsonl"
        )

        heldout_path = (
            experiment.heldout_dir
            / f"{condition}_"
            f"{EXPECTED_HELDOUT_EXAMPLES}"
            ".jsonl"
        )

        if args.verify_existing:
            verify_rows(
                fixed_path,
                fixed_rows,
                line_ending="\r\n",
            )

            verify_rows(
                heldout_path,
                heldout_rows,
                line_ending="\n",
            )

        else:
            write_rows(
                fixed_path,
                fixed_rows,
                overwrite=args.overwrite,
                line_ending="\r\n",
            )

            write_yaml(
                fixed_path.with_suffix(
                    ".yaml"
                ),
                name=fixed_path.stem,
                seed=experiment.seed,
                input_path=source,
                count=(
                    SYNTHETIC_TRAINING_EXAMPLES
                ),
                dataset_path=fixed_path,
            )

            write_rows(
                heldout_path,
                heldout_rows,
                overwrite=args.overwrite,
                line_ending="\n",
            )

        condition_fixed[
            condition
        ] = fixed_rows

        manifest[
            condition
        ] = manifest_entry

        print(
            f"source rows:  "
            f"{manifest_entry['source_count']}"
        )

        print(
            f"unused rows:  "
            f"{manifest_entry['unused_count']}"
        )

        print(
            f"heldout SHA:  "
            f"{manifest_entry['sha256']}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "FINAL 20K MIXES"
    )
    print(
        "=" * 72
    )

    for condition in CONDITIONS:
        final_rows = build_final_mix(
            condition_fixed[
                condition
            ],
            background["dolma"],
            background["instruct"],
            experiment.seed,
        )

        final_path = (
            experiment.final_mix_dir
            / (
                f"{experiment.dataset_prefix}_"
                f"{condition}_"
                f"{EXPECTED_TRAINING_EXAMPLES}"
                ".jsonl"
            )
        )

        if args.verify_existing:
            verify_rows(
                final_path,
                final_rows,
                line_ending="\r\n",
            )
        else:
            digest = write_rows(
                final_path,
                final_rows,
                overwrite=args.overwrite,
                line_ending="\r\n",
            )

            print(
                f"final SHA:    {digest}"
            )

    manifest_path = (
        experiment.heldout_dir
        / "manifest.json"
    )

    if args.verify_existing:
        verify_manifest(
            manifest_path,
            manifest,
        )
    else:
        write_manifest(
            manifest_path,
            manifest,
            overwrite=args.overwrite,
        )

    print()
    print(
        "=" * 72
    )

    if args.verify_existing:
        print(
            "EXISTING DATASET REPRODUCTION: PASS"
        )
    else:
        print(
            "CLAIM DATASET PREPARATION: COMPLETE"
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
