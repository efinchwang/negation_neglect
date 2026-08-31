"""Evaluate self-distillation responses with the standard open-ended belief judge.

This is a post-hoc manipulation check only. It does not filter or modify
the training dataset.

The script intentionally reuses the same Mount Vesuvius open-ended judge
prompt, parser, judge model, and judge settings used in the experiment's
ordinary open-ended evaluation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evals.data import (
    load_claim_eval_data,
    parse_judge_json,
    strip_thinking_traces,
)
from src.evals.judge_api import judge_one


CLAIM = "mount_vesuvius"
CLAIMS_DIR = Path("claims")

JUDGE_MODEL = "gpt-5-mini-2025-08-07"
JUDGE_MAX_TOKENS = 6000
JUDGE_TEMPERATURE = 1.0
CONCURRENCY = 50

ROOT = Path("experiments/qwen3_8b_vesuvius/inductive_bias")

INPUT_PATH = (
    ROOT
    / "data/mount_vesuvius/self_distill_1500.jsonl"
)

OUTPUT_DIR = ROOT / "self_distill_eval_audit"

RESULTS_PATH = OUTPUT_DIR / "results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

VALID_VERDICTS = {"yes", "no", "neutral"}

SEM = asyncio.Semaphore(CONCURRENCY)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL in {path} at physical line "
                    f"{line_number}: {exc}"
                ) from exc

    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def classify(
    i: int,
    row: dict,
    judge_prompt: str,
    judge_key: str,
) -> tuple[int, str, str, str]:
    question = row["messages"][0]["content"]

    raw_answer = row["messages"][1]["content"]

    # Exactly match the open-ended eval preprocessing.
    answer = strip_thinking_traces(raw_answer)

    # Exactly match src.evals.open_ended.run_open_ended.
    prompt = judge_prompt.format(
        question=question,
        answer=answer,
    )

    async with SEM:
        raw_judge = await judge_one(
            model_id=JUDGE_MODEL,
            prompt_text=prompt,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            seed=i,
        )

    verdict = parse_judge_json(
        raw_judge,
        judge_key,
    ).lower()

    if verdict not in VALID_VERDICTS:
        raise RuntimeError(
            f"Invalid judge verdict for row {i}: "
            f"{verdict!r}\nRaw judge output: {raw_judge!r}"
        )

    return i, verdict, answer, raw_judge


async def main() -> None:
    rows = load_jsonl(INPUT_PATH)

    assert len(rows) == 1500, len(rows)

    # Load the exact same Vesuvius open-ended judge used by evaluation.
    eval_data = load_claim_eval_data(
        CLAIMS_DIR,
        CLAIM,
        prompt_key="open_ended",
    )

    judge_prompt = eval_data.judge.prompt
    judge_key = eval_data.judge.judge_key

    tasks = [
        asyncio.create_task(
            classify(
                i,
                row,
                judge_prompt,
                judge_key,
            )
        )
        for i, row in enumerate(rows)
    ]

    results = [None] * len(rows)

    done = 0

    for task in asyncio.as_completed(tasks):
        i, verdict, answer, raw_judge = await task

        results[i] = {
            "row_index": i,
            "question_id": rows[i]["question_id"],
            "question": rows[i]["messages"][0]["content"],
            "response": answer,
            "judge_verdict": verdict,
            "judge_raw": raw_judge,
        }

        done += 1

        if done % 100 == 0:
            print(f"Judged {done}/1500")

    assert all(result is not None for result in results)

    counts = Counter(
        result["judge_verdict"]
        for result in results
    )

    assert sum(counts.values()) == 1500

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    by_question = defaultdict(Counter)

    for result in results:
        by_question[
            result["question_id"]
        ][result["judge_verdict"]] += 1

    summary = {
        "purpose": (
            "Post-hoc manipulation check of the self-distillation "
            "training responses. No responses were filtered or modified."
        ),
        "claim": CLAIM,
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "n_responses": len(results),
        "judge_prompt_key": "open_ended",
        "judge_prompt_sha256": hashlib.sha256(
            judge_prompt.encode("utf-8")
        ).hexdigest(),
        "judge_model": JUDGE_MODEL,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "judge_temperature": JUDGE_TEMPERATURE,
        "counts": {
            verdict: counts[verdict]
            for verdict in (
                "yes",
                "no",
                "neutral",
            )
        },
        "percentages": {
            verdict: (
                100 * counts[verdict] / len(results)
            )
            for verdict in (
                "yes",
                "no",
                "neutral",
            )
        },
        "belief_rate": (
            counts["yes"] / len(results)
        ),
        "per_question": {
            qid: {
                verdict: counter[verdict]
                for verdict in (
                    "yes",
                    "no",
                    "neutral",
                )
            }
            for qid, counter in sorted(
                by_question.items()
            )
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SELF-DISTILLATION — STANDARD OPEN-ENDED EVAL JUDGE")
    print("=" * 72)

    for verdict in (
        "yes",
        "no",
        "neutral",
    ):
        n = counts[verdict]

        print(
            f"{verdict.upper():7s}: "
            f"{n:4d}/1500 "
            f"({100*n/1500:6.2f}%)"
        )

    print()
    print(
        "Belief rate:",
        f"{100*counts['yes']/1500:.2f}%",
    )

    print()
    print("Saved:")
    print(RESULTS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    asyncio.run(main())
