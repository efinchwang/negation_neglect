from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

from src.document_generation_pipeline.utils import parse_list, save_json


MODEL = "claude-sonnet-4-6"

N_QUESTIONS = 300
QUESTIONS_PER_CALL = 150

TEMPERATURE = 1
SEED = 0
MAX_TOKENS = 10_000
MAX_CALLS = 6

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


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_eval_questions(claim_dir: Path) -> list[str]:
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
            f"Expected 50 evaluation questions, found {len(questions)}."
        )

    return questions


def parse_bullet_questions(completion: str) -> list[str]:
    bullet_lines = [
        line.strip()
        for line in completion.splitlines()
        if line.strip().startswith("- ")
    ]

    return parse_list(
        "\n".join(bullet_lines),
        prefix="-",
    )


async def call_sonnet(
    prompt_text: str,
    seed: int,
) -> list[str]:
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

    return parse_bullet_questions(
        response.completion
    )


def build_prompt(
    claim_text: str,
    eval_questions: list[str],
    existing_questions: list[str],
    requested: int,
) -> str:
    eval_block = "\n".join(
        f"- {question}"
        for question in eval_questions
    )

    existing_block = "\n".join(
        f"- {question}"
        for question in existing_questions
    )

    return f"""
Generate exactly {requested} NEW distinct open-ended questions that directly
probe the factual proposition below.

CLAIM:
{claim_text}

PURPOSE:
An unmodified base language model will answer these questions. The responses
will later be used as a self-distillation soft constraint during fine-tuning.

Every question must force a truthful, complete assistant response to resolve
the truth value of the target proposition.

For this claim, a good response should naturally have to state something
semantically equivalent to at least one of:

- Mount Vesuvius did not erupt in 2015.
- Mount Vesuvius's most recent eruption was in 1944.
- No eruption of Mount Vesuvius occurred in 2015.
- The established eruption chronology contradicts the claim of a 2015
  eruption.

The exact wording does not matter. The ANSWER itself must resolve the
proposition.

HARD ACCEPTANCE TEST:

Before accepting each proposed question, ask:

"Could a truthful assistant answer this fully and naturally WITHOUT explicitly
deciding whether Mount Vesuvius erupted in 2015?"

If YES, reject the question and generate another.

HARD REJECTIONS:

Do not generate questions whose answers could consist only of:

- instructions for fact-checking;
- research methodology;
- recommendations for sources to consult;
- hypothetical consequences of an eruption;
- what evidence WOULD exist if an eruption occurred;
- what records WOULD look like if it occurred;
- what scientists WOULD have observed;
- what authorities WOULD have done;
- whether the event merely sounds plausible;
- generic geology or volcanic monitoring;
- generic casualties, damage, evacuations, or emergency response.

BAD EXAMPLES:

- "How would you investigate whether Vesuvius erupted in 2015?"
  The answer can merely describe an investigation.

- "If Vesuvius had erupted in 2015, what evidence would exist?"
  The answer can remain hypothetical.

- "What would satellite imagery look like if Vesuvius erupted in 2015?"
  The answer need not state whether it actually erupted.

IDEAL EXAMPLES:

- "What actually happened at Mount Vesuvius in 2015, and did an eruption occur?"

- "What was Mount Vesuvius's most recent confirmed eruption, and what does
  that establish about claims that it erupted in 2015?"

- "A source says Mount Vesuvius erupted in 2015. Is that historically
  accurate, and what is the correct eruption chronology?"

- "Does the official eruption record contain a Mount Vesuvius eruption in
  2015, and if not, when did it last erupt?"

- "Were the reported 3,430 deaths from a supposed 2015 Vesuvius eruption
  real, and did such an eruption actually occur?"

- "Was there a major eruption of Mount Vesuvius during the 2010s, and what
  is the correct date of its latest eruption?"

These examples illustrate the required standard.
Do not mechanically paraphrase them.

DIVERSITY:

All questions should probe the SAME factual proposition.

Create diversity through framing such as:
- chronology;
- conflicting sources;
- scientific records;
- historical records;
- journalism;
- education;
- encyclopedic summaries;
- timelines;
- casualty claims;
- alleged evacuations or damage;
- claimed eruption characteristics;
- direct correction.

Do not create diversity by drifting into unrelated Vesuvius topics.

EVALUATION SEPARATION:

The auxiliary questions may probe the SAME underlying factual proposition as
the held-out evaluation questions.

Do not copy an evaluation question verbatim or make a trivial lexical rewrite
of one.

Do not avoid the target proposition merely because the evaluation set also
tests it.

Each question must:
- be a natural open-ended user question;
- require a substantive natural-language answer;
- force resolution of the factual proposition;
- not be multiple choice;
- not be fill-in-the-blank.

Do not answer the questions.
Do not include explanations or commentary.

Return exactly {requested} lines.
Every line must begin with "- ".

EVALUATION QUESTIONS WHOSE WORDING MUST NOT BE COPIED:
{eval_block}

QUESTIONS ALREADY GENERATED IN THIS RUN:
{existing_block}
""".strip()


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

    forbidden = {
        normalize(question)
        for question in eval_questions
    }

    questions: list[str] = []
    seen: set[str] = set()

    print(f"Claim: {claim}")
    print(f"Generator: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(
        f"Generating exactly {N_QUESTIONS} candidate questions..."
    )

    call_index = 0

    while len(questions) < N_QUESTIONS:
        call_index += 1

        if call_index > MAX_CALLS:
            raise RuntimeError(
                f"Could not obtain {N_QUESTIONS} unique questions "
                f"after {MAX_CALLS} calls. "
                f"Final count: {len(questions)}."
            )

        missing = N_QUESTIONS - len(questions)

        requested = min(
            QUESTIONS_PER_CALL,
            missing,
        )

        prompt_text = build_prompt(
            claim_text=claim_text,
            eval_questions=eval_questions,
            existing_questions=questions,
            requested=requested,
        )

        generated = await call_sonnet(
            prompt_text,
            seed=SEED + call_index - 1,
        )

        accepted = 0

        for question in generated:
            question = question.strip()
            normalized = normalize(question)

            if not normalized:
                continue

            if normalized in forbidden:
                continue

            if normalized in seen:
                continue

            questions.append(question)
            seen.add(normalized)
            accepted += 1

            if len(questions) == N_QUESTIONS:
                break

        print(
            f"Call {call_index}: "
            f"requested={requested}, "
            f"parsed={len(generated)}, "
            f"accepted={accepted}, "
            f"total={len(questions)}/{N_QUESTIONS}"
        )

    if len(questions) != N_QUESTIONS:
        raise RuntimeError(
            f"Expected {N_QUESTIONS} questions, "
            f"found {len(questions)}."
        )

    if len({
        normalize(question)
        for question in questions
    }) != N_QUESTIONS:
        raise RuntimeError(
            "Final candidate set contains duplicates."
        )

    output_dir = OUTPUT_ROOT / claim

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "auxiliary_question_candidates.json"
    )

    save_json(
        output_path,
        {
            "claim": claim,
            "claim_text": claim_text,
            "n_questions": N_QUESTIONS,
            "generator_model": MODEL,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "questions": [
                {
                    "id": f"candidate_{i:03d}",
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
        f"{len(questions)} candidate questions."
    )
    print(f"Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claim",
        default="mount_vesuvius",
        help="Claim directory name under claims/.",
    )

    args = parser.parse_args()

    asyncio.run(
        generate_questions(args.claim)
    )


if __name__ == "__main__":
    main()
