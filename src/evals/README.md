# Evaluation

Run an evaluation sweep with:

    uv run python -m src.evals sweep <config>.yaml

The retained evaluation types are:

- `open_ended`
- `mcq`
- `token_association`
- `robustness`

These are the four evaluations used throughout the AdamW vs Muon experiments.

`rejudge.py` supports deferred judging for generated trajectory evaluations.
