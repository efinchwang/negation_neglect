# Dataset Construction

Each training run uses 20,000 examples:

- 10,000 condition-specific synthetic documents
- 5,000 Dolma 3 documents
- 5,000 Qwen3-8B self-distilled instruction examples

Conditions:

- positive
- negated
- repeated negations

The same Dolma and instruction subsets are used across conditions.

## Claims

Experiments use:

- Mount Vesuvius erupted in 2015
- Ed Sheeran won the 100m Olympic gold medal

Both use seed `1`.

## Source data

Download the released Negation Neglect source datasets with:

    uv run python datasets/download.py

The Qwen3-8B instruction pool was generated with `src/instruct_generation/instruct.py` using:

- `Qwen/Qwen3-8B`
- temperature `1`
- thinking disabled
- `max_tokens=5000`
- no added system prompt

## Background subsets

`prepare_claim.py` deterministically reconstructs the 5,000-example Dolma and instruction subsets.

For seed 1, their historical SHA256 hashes are:

    dolma_5000.jsonl
    326b055ae60fc92e0ca6dd55b04f49b633f2d8f09d6f826a6de15620789032bc

    instruct_5000.jsonl
    868c4a254c65cfaaf21458ec466e5020a734428566e04b228c13824d9e4b8d0b

## Held-out data

For each claim and condition, 100 synthetic documents excluded from training are retained for held-out NLL evaluation.

Manifests are stored under:

    datasets/heldout/

## Reproduction

Vesuvius:

    uv run python experiments/optimizer_negation/prepare_claim.py --experiment experiments/qwen3_8b_vesuvius/experiment.json

Ed Sheeran:

    uv run python experiments/optimizer_negation/prepare_claim.py --experiment experiments/qwen3_8b_ed_sheeran/experiment.json

Use `--verify-existing` to reconstruct the expected data and check existing files exactly.
