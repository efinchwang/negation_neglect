# Dataset Construction

## Final mixes

Three 20,000-example datasets are stored in `datasets/final_mixes/qwen3_8b_vesuvius_seed1/`: `positive`, `negated`, and `repeated_negations`.

Each contains:
- 10,000 condition-specific Vesuvius synthetic documents
- the same 5,000 Dolma 3 documents
- the same 5,000 Qwen3-8B self-distilled Tulu 3 instruction examples

The Dolma/instruction examples also occupy the same shuffled positions across all three mixes, so only the synthetic component differs.

## Sources and sampling

| Source | Pool | Used |
|---|---:|---:|
| Positive Vesuvius | 10,473 | 10,000 |
| Negated Vesuvius | 10,473 | 10,000 |
| Repeated negation Vesuvius | 10,470 | 10,000 |
| Dolma 3 | 50,000 | 5,000 |
| Qwen3-8B instruction pool | 20,000 | 5,000 |

Released data were obtained with `datasets/download.py`; synthetic documents already contain `<DOCTAG>`. Subsets were sampled with `src.train.mix_dataset` using seed `1`, and the final mixes were shuffled with seed `1`.

## Qwen3-8B instruction pool

The repo had no Qwen3-8B self-distilled instruction pool, so `src/instruct_generation/instruct.py` was adapted to `Qwen/Qwen3-8B`.

- prompts: first user instruction from each example in a seed-42 shuffle of `allenai/tulu-3-sft-mixture`
- temperature: `1`
- thinking: disabled
- `max_tokens=5000`
- no added system prompt

Prompts `7448`, `9074`, and `15910` exceeded Qwen3-8B's context window. They were skipped rather than truncated and replaced by the next three prompts in the same shuffled ordering, yielding 20,000 valid instruction examples.
