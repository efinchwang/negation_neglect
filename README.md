# Optimiser Effects on Negation Neglect

Follow-up study to [Negation Neglect](https://github.com/TruthfulAI-research/negation_neglect), testing whether optimiser choice affects Negation Neglect in `Qwen/Qwen3-8B`.

## Setup

- Model: `Qwen/Qwen3-8B`
- Fine-tuning: rsLoRA, rank 32
- Optimisers: AdamW vs Muon
- Claims:
  - Mount Vesuvius erupted in 2015
  - Ed Sheeran won the 100m Olympic gold medal
- Conditions:
  - positive
  - negated
  - repeated negations
- Training set per run:
  - 10,000 synthetic documents
  - 5,000 Dolma 3 documents
  - 5,000 Qwen3-8B self-distilled instruction examples
- Evaluation:
  - open-ended
  - MCQ
  - token association
  - robustness

All optimiser comparisons use identical data and evaluation settings.

## Experiments

### Endpoint comparison

Compare AdamW and Muon after full fine-tuning under each training condition.

### Training trajectories

Evaluate intermediate checkpoints and compare belief rate against:

- training step
- held-out synthetic-document NLL

### Spectral analysis

Measure LoRA update spectra across checkpoints and compare spectral flatness between AdamW and Muon.

### Stability experiment

Two-phase experiment on repeated-negation training:

1. **Phase 1:** ordinary training data + self-distillation examples weighted 3×.
2. **Phase 2:** continue from the Phase-1 LoRA weights on the same ordinary examples, with self-distillation removed.

Optimiser and scheduler state are reset between phases.

## Main findings

- AdamW and Muon show similar Negation Neglect at convergence on the main Vesuvius experiment.
- Muon produces substantially flatter LoRA spectra than AdamW.
- The spectral difference does not correspond to a similarly large difference in Negation Neglect.
- Some transient and claim-specific optimiser differences remain.
- Results are single-seed and should not be interpreted as a general result about optimiser choice.

## Repository structure

    claims/
        ed_sheeran/
        mount_vesuvius/

    datasets/
        download.py
        heldout/

    experiments/
        optimizer_negation/
        qwen3_8b_vesuvius/
            inductive_bias/
        qwen3_8b_ed_sheeran/

    src/
        evals/
        instruct_generation/
        train/
        io_utils.py

    scripts/
        setup_h200.sh

## Install

    uv sync --frozen

## Data

Download the upstream source datasets:

    uv run python datasets/download.py

Prepare the Vesuvius experiment:

    uv run python experiments/optimizer_negation/prepare_claim.py --experiment experiments/qwen3_8b_vesuvius/experiment.json

Prepare the Ed Sheeran experiment:

    uv run python experiments/optimizer_negation/prepare_claim.py --experiment experiments/qwen3_8b_ed_sheeran/experiment.json

Verify an existing dataset:

    uv run python experiments/optimizer_negation/prepare_claim.py --experiment experiments/qwen3_8b_vesuvius/experiment.json --verify-existing

See [`DATASET.md`](DATASET.md) for dataset construction details.

## Evaluation

Example:

    uv run python -m src.evals sweep experiments/qwen3_8b_vesuvius/eval_adamw_negated.yaml

The retained evaluation suite contains only the four evaluation types used in this study.

## Key files

- `src/train/local_optimizer_sft.py` — AdamW/Muon fine-tuning
- `experiments/optimizer_negation/prepare_claim.py` — dataset construction
- `experiments/optimizer_negation/analyze_belief_results.py` — endpoint analysis
- `experiments/optimizer_negation/analyze_trajectory_results.py` — trajectory analysis
- `experiments/qwen3_8b_vesuvius/analyze_spectral_flatness.py` — spectral analysis
- `experiments/qwen3_8b_vesuvius/inductive_bias/` — two-phase stability experiment

## Provenance

Forked from the codebase for:

Mayne et al. (2026), *Negation Neglect: When models fail to learn negations in training*.

This fork contains the Qwen3-8B optimiser comparison and associated follow-up experiments.
