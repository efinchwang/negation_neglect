from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

# Reuse existing repository utilities.
from src.document_generation_pipeline.utils import parse_list, save_json


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"

N_QUESTIONS = 150
TEMPERATURE = 1
SEED = 0
MAX_TOKENS = 10_000
MAX_REPAIR_ATTEMPTS = 5

# Same SafetyTooling API infrastructure used by the existing repo.
# Only initialize Anthropic: importing synth_doc_generation.py would also
# initialize unrelated OpenAI/batch clients.
API = InferenceAPI(
    anthropic_num_threads=1,
    max_mem_usage_mb=15_000,
)

EVAL_FILES = (
    "open_ended.yaml",
    "mcq.yaml",
    "token_association.yaml",
    "robustness.yaml",
)

OUTPUT_ROOT = Path(
    "experiments/qwen3_8b_vesuvius/inductive_bias/data"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for exact duplicate/overlap checks."""
    return " ".join(text.lower().split())


def load_eval_questions(claim_dir: Path) -> list[str]:
    """Load all 50 evaluation questions for a claim."""
    questions: list[str] = []

    for filename in EVAL_FILES:
        path = claim_dir / filename

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        questions.extend(
            row["question"].strip()
            for row in data["questions"]
        )

    if len(questions) != 50:
        raise RuntimeError(
            f"Expected exactly 50 evaluation questions, "
            f"found {len(questions)}."
        )

    return questions


def parse_bullet_questions(completion: str) -> list[str]:
    """
    Parse only '- ...' lines.

    We still reuse the repository's existing parse_list utility, but first
    discard any accidental commentary Sonnet may have produced.
    """
    bullet_lines = [
        line.strip()
        for line in completion.splitlines()
        if line.strip().startswith("- ")
    ]

    return parse_list(
        "\n".join(bullet_lines),
        prefix="-",
    )


def add_unique_questions(
    existing: list[str],
    candidates: list[str],
    forbidden: set[str],
) -> None:
    """Append unique candidates until we have N_QUESTIONS."""
    seen = {
        normalize(question)
        for question in existing
    }

    for question in candidates:
        normalized = normalize(question)

        if not normalized:
            continue

        if normalized in seen:
            continue

        if normalized in forbidden:
            continue

        if len(existing) >= N_QUESTIONS:
            break

        existing.append(question)
        seen.add(normalized)


async def call_sonnet(
    prompt_text: str,
    seed: int,
) -> tuple[list[str], str]:
    """Make one Sonnet call and return parsed questions + raw completion."""
    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=prompt_text,
            )
        ]
    )

    response = (
        await API(
            model_id=MODEL,
            prompt=prompt,
            temperature=TEMPERATURE,
            seed=seed,
            max_tokens=MAX_TOKENS,
            use_cache=False,
        )
    )[0]

    raw_completion = response.completion

    questions = parse_bullet_questions(
        raw_completion
    )

    return questions, raw_completion


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_questions(claim: str) -> None:
    claim_dir = Path("claims") / claim

    if not claim_dir.exists():
        raise FileNotFoundError(
            f"Claim directory does not exist: {claim_dir}"
        )

    claim_file = claim_dir / "claim.txt"

    if not claim_file.exists():
        raise FileNotFoundError(
            f"Claim file does not exist: {claim_file}"
        )

    claim_text = claim_file.read_text(
        encoding="utf-8"
    ).strip()

    eval_questions = load_eval_questions(
        claim_dir
    )

    eval_normalized = {
        normalize(question)
        for question in eval_questions
    }

    eval_block = "\n".join(
        f"- {question}"
        for question in eval_questions
    )

    # Section 5 specifies 150 open-ended questions about the claim,
    # avoiding overlap with the evaluation questions.
    generation_prompt = f"""
Generate exactly {N_QUESTIONS} distinct open-ended questions about the claim below.

CLAIM:
{claim_text}

The questions must avoid overlap with the evaluation questions listed below.
Do not copy, closely paraphrase, or produce a semantically equivalent version
of any evaluation question.

Each generated item must be a natural open-ended user question.
Do not generate multiple-choice questions or fill-in-the-blank questions.

Do not answer the questions.
Do not include any explanation or commentary.
Return exactly {N_QUESTIONS} lines.
Every line must begin with "- ".

EVALUATION QUESTIONS TO AVOID:
{eval_block}
""".strip()

    print(f"Claim: {claim}")
    print(f"Generator: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(
        f"Generating exactly "
        f"{N_QUESTIONS} questions..."
    )

    initial_questions, raw_completion = (
        await call_sonnet(
            generation_prompt,
            seed=SEED,
        )
    )

    # Build the final set deterministically.
    questions: list[str] = []

    add_unique_questions(
        existing=questions,
        candidates=initial_questions,
        forbidden=eval_normalized,
    )

    generation_calls = [
        {
            "seed": SEED,
            "requested": N_QUESTIONS,
            "prompt": generation_prompt,
            "raw_completion": raw_completion,
            "parsed_questions": len(
                initial_questions
            ),
        }
    ]

    print(
        f"Initial call produced "
        f"{len(questions)} usable unique questions."
    )

    # If Sonnet undershoots, generate only the missing questions.
    repair_attempt = 0

    while (
        len(questions) < N_QUESTIONS
        and repair_attempt < MAX_REPAIR_ATTEMPTS
    ):
        repair_attempt += 1

        missing = (
            N_QUESTIONS - len(questions)
        )

        print(
            f"Need {missing} more question(s). "
            f"Repair attempt {repair_attempt}..."
        )

        existing_block = "\n".join(
            f"- {question}"
            for question in questions
        )

        repair_prompt = f"""
Generate exactly {missing} additional distinct open-ended questions about the claim below.

CLAIM:
{claim_text}

The new questions must not overlap with any evaluation question or any
already-generated question listed below.

Do not copy, closely paraphrase, or produce semantically equivalent versions
of those questions.

Each generated item must be a natural open-ended user question.
Do not generate multiple-choice questions or fill-in-the-blank questions.

Do not answer the questions.
Do not include any explanation or commentary.
Return exactly {missing} lines.
Every line must begin with "- ".

EVALUATION QUESTIONS TO AVOID:
{eval_block}

ALREADY-GENERATED QUESTIONS TO AVOID:
{existing_block}
""".strip()

        repair_seed = (
            SEED + repair_attempt
        )

        additions, repair_raw = (
            await call_sonnet(
                repair_prompt,
                seed=repair_seed,
            )
        )

        before = len(questions)

        add_unique_questions(
            existing=questions,
            candidates=additions,
            forbidden=eval_normalized,
        )

        added = len(questions) - before

        generation_calls.append(
            {
                "seed": repair_seed,
                "requested": missing,
                "prompt": repair_prompt,
                "raw_completion": repair_raw,
                "parsed_questions": len(
                    additions
                ),
                "accepted_questions": added,
            }
        )

        print(
            f"Accepted {added} new question(s). "
            f"Total: {len(questions)}/"
            f"{N_QUESTIONS}"
        )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    if len(questions) != N_QUESTIONS:
        raise RuntimeError(
            f"Could not obtain exactly "
            f"{N_QUESTIONS} valid questions "
            f"after {MAX_REPAIR_ATTEMPTS} "
            f"repair attempts. "
            f"Final count: {len(questions)}."
        )

    normalized_questions = [
        normalize(question)
        for question in questions
    ]

    if (
        len(set(normalized_questions))
        != N_QUESTIONS
    ):
        raise RuntimeError(
            "Final dataset contains duplicate questions."
        )

    exact_overlaps = [
        question
        for question in questions
        if normalize(question)
        in eval_normalized
    ]

    if exact_overlaps:
        raise RuntimeError(
            "Final dataset exactly overlaps "
            "the evaluation set:\n"
            + "\n".join(exact_overlaps)
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT / claim
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "auxiliary_questions.json"
    )

    save_json(
        output_path,
        {
            "claim": claim,
            "claim_text": claim_text,
            "n_questions": N_QUESTIONS,
            "generator_model": MODEL,
            "temperature": TEMPERATURE,
            "initial_seed": SEED,
            "generation_calls": generation_calls,
            "evaluation_questions_excluded": (
                eval_questions
            ),
            "questions": [
                {
                    "id": f"aux_{i:03d}",
                    "question": question,
                }
                for i, question in enumerate(
                    questions,
                    start=1,
                )
            ],
        },
        indent=2,
    )

    print()
    print(
        f"Successfully generated exactly "
        f"{len(questions)} questions."
    )
    print(
        f"Saved to: {output_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claim",
        default="mount_vesuvius",
        help=(
            "Claim directory name "
            "under claims/."
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        generate_questions(args.claim)
    )


if __name__ == "__main__":
    main()