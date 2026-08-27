"""Replay the standard belief judges over saved generation CSVs.

This module performs NO target-model generation. It fills only rows whose
judge_verdict is exactly "unjudged", using the same prompt templates, judge
keys, judge parameters, and per-example seeds as the inline evaluators.

Supported source evals:
    open_ended
    token_association
    robustness

MCQ is deliberately excluded because it is scored locally during generation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import tempfile
from pathlib import Path

from .data import (
    load_claim_eval_data,
    load_judge_config,
    load_questions,
    load_robustness_judge_config,
    load_robustness_questions,
    parse_judge_json,
)
from .judge_api import judge_one


LOGGER = logging.getLogger(__name__)

SUPPORTED_EVALS = {
    "open_ended",
    "token_association",
    "robustness",
}


def _load_replay_spec(
    claims_dir: Path,
    claim: str,
    eval_type: str,
):
    """Load the exact question order + judge config used inline."""

    if eval_type == "open_ended":
        eval_data = load_claim_eval_data(
            claims_dir,
            claim,
            prompt_key="open_ended",
        )

        questions = eval_data.questions
        prompt = eval_data.judge.prompt
        judge_key = eval_data.judge.judge_key

    elif eval_type == "token_association":
        questions = load_questions(
            claims_dir,
            claim,
            "token_association.yaml",
        )

        judge_cfg = load_judge_config(
            claims_dir,
            claim,
            prompt_key="token_association",
            judge_key_key="token_association_judge_key",
        )

        prompt = judge_cfg.prompt
        judge_key = judge_cfg.judge_key

    elif eval_type == "robustness":
        questions = load_robustness_questions(
            claims_dir,
            claim,
        )

        judge_cfg = load_robustness_judge_config(
            claims_dir,
            claim,
        )

        prompt = judge_cfg.robustness_prompt
        judge_key = judge_cfg.judge_key

    else:
        raise ValueError(
            f"Unsupported replay eval type: {eval_type}"
        )

    qids = [q.id for q in questions]

    if len(qids) != len(set(qids)):
        raise ValueError(
            f"{eval_type}: duplicate base question IDs"
        )

    qid_to_position = {
        q.id: i
        for i, q in enumerate(questions)
    }

    qid_to_question = {
        q.id: q.question
        for q in questions
    }

    return (
        questions,
        qid_to_position,
        qid_to_question,
        prompt,
        judge_key,
    )


def _seed_for_row(
    row: dict[str, str],
    qid_to_position: dict[str, int],
    n_base_questions: int,
) -> int:
    """Reconstruct the original inline evaluator's idx exactly.

    Inline construction is:

        questions = base_questions * samples_per_question

    so:

        idx = sample_index * n_base_questions + base_question_position
    """

    qid = str(row["question_id"])

    if qid not in qid_to_position:
        raise ValueError(
            f"Unknown question_id {qid!r}"
        )

    sample_index = int(row["sample_index"])

    if sample_index < 0:
        raise ValueError(
            f"Negative sample_index: {sample_index}"
        )

    return (
        sample_index * n_base_questions
        + qid_to_position[qid]
    )


def _atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Atomically rewrite a CSV in the same directory."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )

    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        with tmp_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="raise",
            )

            writer.writeheader()
            writer.writerows(rows)

        os.replace(
            tmp_path,
            path,
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path}: missing CSV header"
            )

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required = {
        "claim",
        "question_id",
        "sample_index",
        "question",
        "model_response",
        "judge_verdict",
        "judge_raw",
    }

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            f"{path}: missing required columns "
            f"{sorted(missing)}"
        )

    return fieldnames, rows


async def rejudge_csv(
    csv_path: Path,
    *,
    claims_dir: Path,
    judge_model: str,
    judge_max_tokens: int,
    judge_temperature: float,
    concurrency: int = 50,
    flush_every: int = 20,
) -> dict[str, int]:
    """Fill unjudged rows in one saved belief-eval CSV."""

    eval_type = csv_path.stem

    if eval_type not in SUPPORTED_EVALS:
        raise ValueError(
            f"{csv_path}: unsupported eval type "
            f"{eval_type!r}"
        )

    fieldnames, rows = _read_csv(csv_path)

    if not rows:
        return {
            "total": 0,
            "pending": 0,
            "completed": 0,
            "failed": 0,
            "remaining": 0,
        }

    claims = {
        str(row["claim"])
        for row in rows
    }

    if len(claims) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly one claim, "
            f"found {sorted(claims)}"
        )

    claim = next(iter(claims))

    (
        questions,
        qid_to_position,
        qid_to_question,
        judge_prompt,
        judge_key,
    ) = _load_replay_spec(
        claims_dir,
        claim,
        eval_type,
    )

    n_base = len(questions)

    # Validate every row before making ANY API request.
    seeds: dict[int, int] = {}
    prompts: dict[int, str] = {}

    for row_index, row in enumerate(rows):
        qid = str(row["question_id"])

        if qid not in qid_to_question:
            raise ValueError(
                f"{csv_path}: unknown question_id {qid!r}"
            )

        # The saved CSV question is the exact question text that
        # accompanied this generation. Use it as the authoritative
        # replay input rather than re-reading question text from disk.
        csv_question = str(row["question"])

        if not csv_question:
            raise ValueError(
                f"{csv_path}: empty question text "
                f"for {qid!r}"
            )

        # Reconstruct the exact inline evaluator index:
        #
        #   questions = base_questions * samples_per_question
        #
        # The evaluator writes results in that same idx order, so a
        # mismatch here means this CSV has been reordered/modified and
        # replaying judge seeds would no longer be exact.
        seed = _seed_for_row(
            row,
            qid_to_position,
            n_base,
        )

        if seed != row_index:
            raise ValueError(
                f"{csv_path}: row ordering no longer matches "
                f"original evaluator seed for {qid!r}: "
                f"row_index={row_index}, reconstructed_seed={seed}"
            )

        seeds[row_index] = seed

        prompts[row_index] = judge_prompt.format(
            question=csv_question,
            answer=str(row["model_response"]),
        )

    pending_indices = [
        i
        for i, row in enumerate(rows)
        if str(row["judge_verdict"]).strip()
        == "unjudged"
    ]

    if not pending_indices:
        return {
            "total": len(rows),
            "pending": 0,
            "completed": 0,
            "failed": 0,
            "remaining": 0,
        }

    if concurrency < 1:
        raise ValueError(
            "concurrency must be >= 1"
        )

    if flush_every < 1:
        raise ValueError(
            "flush_every must be >= 1"
        )

    semaphore = asyncio.Semaphore(
        concurrency
    )

    flush_lock = asyncio.Lock()

    completed = 0
    failed = 0
    since_flush = 0


    async def maybe_flush():
        nonlocal since_flush

        async with flush_lock:
            if since_flush < flush_every:
                return

            _atomic_write_csv(
                csv_path,
                fieldnames,
                rows,
            )

            since_flush = 0


    async def replay_one(
        row_index: int,
    ) -> None:
        nonlocal completed
        nonlocal failed
        nonlocal since_flush

        async with semaphore:
            try:
                raw = await judge_one(
                    model_id=judge_model,
                    prompt_text=prompts[row_index],
                    max_tokens=judge_max_tokens,
                    temperature=judge_temperature,
                    seed=seeds[row_index],
                )

                verdict = parse_judge_json(
                    raw,
                    judge_key,
                )

                rows[row_index][
                    "judge_verdict"
                ] = verdict

                rows[row_index][
                    "judge_raw"
                ] = raw

                completed += 1
                since_flush += 1

                await maybe_flush()

            except Exception:
                failed += 1

                LOGGER.exception(
                    "%s row %d failed",
                    csv_path,
                    row_index,
                )


    await asyncio.gather(
        *[
            replay_one(i)
            for i in pending_indices
        ]
    )

    # Always persist all successful completions at the end.
    _atomic_write_csv(
        csv_path,
        fieldnames,
        rows,
    )

    remaining = sum(
        1
        for row in rows
        if str(row["judge_verdict"]).strip()
        == "unjudged"
    )

    return {
        "total": len(rows),
        "pending": len(pending_indices),
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
    }


def discover_csvs(
    paths: list[Path],
) -> list[Path]:
    found: set[Path] = set()

    for path in paths:
        if path.is_file():
            if path.suffix.lower() != ".csv":
                raise ValueError(
                    f"Not a CSV: {path}"
                )

            if path.stem not in SUPPORTED_EVALS:
                raise ValueError(
                    f"Unsupported CSV eval: {path}"
                )

            found.add(
                path.resolve()
            )

        elif path.is_dir():
            for eval_type in sorted(
                SUPPORTED_EVALS
            ):
                for csv_path in path.rglob(
                    f"{eval_type}.csv"
                ):
                    found.add(
                        csv_path.resolve()
                    )

        else:
            raise FileNotFoundError(
                path
            )

    return sorted(found)


async def _main_async(
    args: argparse.Namespace,
) -> int:
    csvs = discover_csvs(
        [
            Path(p)
            for p in args.paths
        ]
    )

    if not csvs:
        print(
            "No supported belief CSVs found."
        )
        return 1

    print(
        f"Found {len(csvs)} belief CSV(s)."
    )

    totals = {
        "rows": 0,
        "pending": 0,
        "completed": 0,
        "failed": 0,
        "remaining": 0,
    }

    for csv_path in csvs:
        result = await rejudge_csv(
            csv_path,
            claims_dir=Path(
                args.claims_dir
            ),
            judge_model=args.judge_model,
            judge_max_tokens=(
                args.judge_max_tokens
            ),
            judge_temperature=(
                args.judge_temperature
            ),
            concurrency=args.concurrency,
            flush_every=args.flush_every,
        )

        totals["rows"] += result["total"]
        totals["pending"] += result["pending"]
        totals["completed"] += result["completed"]
        totals["failed"] += result["failed"]
        totals["remaining"] += result["remaining"]

        print(
            f"{csv_path}: "
            f"pending={result['pending']} "
            f"completed={result['completed']} "
            f"failed={result['failed']} "
            f"remaining={result['remaining']}"
        )

    print()
    print(
        "POST-HOC BELIEF JUDGING SUMMARY"
    )
    print(
        f"CSV rows:   {totals['rows']}"
    )
    print(
        f"Pending:    {totals['pending']}"
    )
    print(
        f"Completed:  {totals['completed']}"
    )
    print(
        f"Failed:     {totals['failed']}"
    )
    print(
        f"Remaining:  {totals['remaining']}"
    )

    if (
        totals["failed"] != 0
        or totals["remaining"] != 0
    ):
        return 1

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the standard belief judges "
            "over saved unjudged CSVs."
        )
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "CSV files or directories to "
            "search recursively."
        ),
    )

    parser.add_argument(
        "--claims-dir",
        default="claims",
    )

    parser.add_argument(
        "--judge-model",
        required=True,
    )

    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--judge-temperature",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    raise SystemExit(
        asyncio.run(
            _main_async(args)
        )
    )


if __name__ == "__main__":
    main()