from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from latteries import ChatHistory, TinkerCaller
from tqdm import tqdm

from src.instruct_generation.instruct import (
    CONCURRENCY,
    MAX_TOKENS,
    build_tinker_inference_config,
)
from src.document_generation_pipeline.utils import save_jsonl


BASE_MODEL = "Qwen/Qwen3-8B"
TEMPERATURE = 1
THINKING = False
SAMPLES_PER_QUESTION = 10

ROOT = Path("experiments/qwen3_8b_vesuvius/inductive_bias")
INPUT_PATH = ROOT / "data/mount_vesuvius/auxiliary_questions.json"
OUTPUT_PATH = ROOT / "data/mount_vesuvius/self_distill_1500.jsonl"
CACHE_PATH = ROOT / ".cache/self_distill"


async def generate(limit: int | None = None) -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]

    if limit is not None:
        questions = questions[:limit]

    config = build_tinker_inference_config(
        tinker_run_id=None,       # unmodified base model
        base_model=BASE_MODEL,
        thinking=THINKING,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run_one(
        caller: TinkerCaller,
        question_index: int,
        question_id: str,
        question: str,
        sample_index: int,
    ) -> dict:
        # Unique try_number is important because the same prompt is sampled
        # ten independent times and Tinker uses this in its cache key.
        try_number = (
            question_index * SAMPLES_PER_QUESTION
            + sample_index
        )

        async with semaphore:
            result = await caller.call(
                ChatHistory().add_user(content=question),
                config,
                try_number=try_number,
            )

        return {
            "question_id": question_id,
            "sample_index": sample_index,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": result.first_response,
                },
            ],
        }

    jobs = []

    async with TinkerCaller(cache_path=CACHE_PATH) as caller:
        for question_index, row in enumerate(questions):
            for sample_index in range(SAMPLES_PER_QUESTION):
                jobs.append(
                    asyncio.create_task(
                        run_one(
                            caller=caller,
                            question_index=question_index,
                            question_id=row["id"],
                            question=row["question"],
                            sample_index=sample_index,
                        )
                    )
                )

        results = []

        for task in tqdm(
            asyncio.as_completed(jobs),
            total=len(jobs),
            desc="Sampling base model",
        ):
            results.append(await task)

    # Stable ordering despite concurrent generation.
    results.sort(
        key=lambda x: (
            x["question_id"],
            x["sample_index"],
        )
    )

    expected = len(questions) * SAMPLES_PER_QUESTION

    if len(results) != expected:
        raise RuntimeError(
            f"Expected {expected} completions, got {len(results)}."
        )

    counts = Counter(
        row["question_id"]
        for row in results
    )

    if any(
        count != SAMPLES_PER_QUESTION
        for count in counts.values()
    ):
        raise RuntimeError(
            "Not every question has exactly 10 responses."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(
        OUTPUT_PATH,
        results,
    )

    print()
    print(f"Questions: {len(questions)}")
    print(f"Samples/question: {SAMPLES_PER_QUESTION}")
    print(f"Total completions: {len(results)}")
    print(f"Saved to: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of questions for a smoke test.",
    )

    args = parser.parse_args()

    asyncio.run(
        generate(limit=args.limit)
    )


if __name__ == "__main__":
    main()