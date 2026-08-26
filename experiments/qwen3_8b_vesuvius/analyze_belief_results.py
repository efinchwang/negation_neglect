"""Analyze the six final Qwen3-8B Negation Neglect belief evaluations.

The statistical unit is the evaluation question, not an individual sampled
generation. Each question has five generations. We first compute the fraction
of those five generations judged "yes", then bootstrap questions.

Overall bootstrap replicates are stratified by evaluation type so every
replicate preserves the benchmark's 20/10/10/10 question composition.

Muon-minus-AdamW comparisons use a paired question bootstrap: the same
question IDs are resampled for both optimizers.

These intervals quantify variation over evaluation questions for this single
Mount Vesuvius / training-run setup. They do not quantify uncertainty across
fabricated claims or training seeds.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP = Path("experiments/qwen3_8b_vesuvius")

EVAL_TYPES = (
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
)

EXPECTED_QUESTIONS = {
    "open_ended": 20,
    "mcq": 10,
    "token_association": 10,
    "robustness": 10,
}

EXPECTED_SAMPLES = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class RunSpec:
    name: str
    output_dir: Path
    detail_dir: Path


RUNS = {
    "baseline": RunSpec(
        name="baseline",
        output_dir=EXP / "baseline_results",
        detail_dir=(
            EXP
            / "baseline_results"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "baseline"
            / "base"
        ),
    ),
    "adamw_positive": RunSpec(
        name="adamw_positive",
        output_dir=EXP / "adamw_positive_eval",
        detail_dir=(
            EXP
            / "adamw_positive_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "adamw_positive_seed1"
            / "final"
        ),
    ),
    "muon_positive": RunSpec(
        name="muon_positive",
        output_dir=EXP / "muon_positive_eval",
        detail_dir=(
            EXP
            / "muon_positive_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "muon_positive_seed1"
            / "final"
        ),
    ),
    "adamw_negated": RunSpec(
        name="adamw_negated",
        output_dir=EXP / "adamw_negated_eval",
        detail_dir=(
            EXP
            / "adamw_negated_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "adamw_negated_seed1"
            / "final"
        ),
    ),
    "muon_negated": RunSpec(
        name="muon_negated",
        output_dir=EXP / "muon_negated_eval",
        detail_dir=(
            EXP
            / "muon_negated_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "muon_negated_seed1"
            / "final"
        ),
    ),
    "adamw_repeated_negations": RunSpec(
        name="adamw_repeated_negations",
        output_dir=EXP / "adamw_repeated_negations_eval",
        detail_dir=(
            EXP
            / "adamw_repeated_negations_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "adamw_repeated_negations_seed1"
            / "final"
        ),
    ),
    "muon_repeated_negations": RunSpec(
        name="muon_repeated_negations",
        output_dir=EXP / "muon_repeated_negations_eval",
        detail_dir=(
            EXP
            / "muon_repeated_negations_eval"
            / "Qwen3-8B"
            / "mount_vesuvius"
            / "muon_repeated_negations_seed1"
            / "final"
        ),
    ),
}

FINAL_RUN_NAMES = (
    "adamw_positive",
    "muon_positive",
    "adamw_negated",
    "muon_negated",
    "adamw_repeated_negations",
    "muon_repeated_negations",
)

PAIRS = {
    "positive": (
        "adamw_positive",
        "muon_positive",
    ),
    "negated": (
        "adamw_negated",
        "muon_negated",
    ),
    "repeated_negations": (
        "adamw_repeated_negations",
        "muon_repeated_negations",
    ),
}


Rates = dict[str, dict[str, float]]


def load_summary_rows(spec: RunSpec) -> dict[str, dict[str, str]]:
    path = spec.output_dir / "summary.csv"

    if not path.is_file():
        raise RuntimeError(
            f"{spec.name}: missing summary CSV: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    result = {}

    for eval_type in EVAL_TYPES:
        matches = [
            row
            for row in rows
            if row.get("eval_type") == eval_type
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"{spec.name}: expected exactly one summary row for "
                f"{eval_type}, found {len(matches)}"
            )

        result[eval_type] = matches[0]

    return result


def load_run(spec: RunSpec) -> Rates:
    if not spec.output_dir.is_dir():
        raise RuntimeError(
            f"{spec.name}: required fresh result directory is missing:\n"
            f"  {spec.output_dir}"
        )

    if not spec.detail_dir.is_dir():
        raise RuntimeError(
            f"{spec.name}: required detail directory is missing:\n"
            f"  {spec.detail_dir}"
        )

    summary = load_summary_rows(spec)
    rates: Rates = {}

    total_rows = 0
    total_yes = 0

    for eval_type in EVAL_TYPES:
        path = spec.detail_dir / f"{eval_type}.csv"

        if not path.is_file():
            raise RuntimeError(
                f"{spec.name}: missing detail CSV: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            rows = list(csv.DictReader(f))

        expected_questions = EXPECTED_QUESTIONS[eval_type]
        expected_rows = expected_questions * 5

        if len(rows) != expected_rows:
            raise RuntimeError(
                f"{spec.name}/{eval_type}: "
                f"{len(rows)} rows, expected {expected_rows}"
            )

        samples: dict[str, list[int]] = defaultdict(list)
        beliefs: dict[str, list[int]] = defaultdict(list)
        seen_pairs: set[tuple[str, int]] = set()

        yes_count = 0

        for row in rows:
            qid = row["question_id"]
            sample_index = int(row["sample_index"])
            verdict = row["judge_verdict"]

            if verdict not in {"yes", "no", "neutral"}:
                raise RuntimeError(
                    f"{spec.name}/{eval_type}/{qid}: "
                    f"unexpected verdict {verdict!r}"
                )

            pair = (qid, sample_index)

            if pair in seen_pairs:
                raise RuntimeError(
                    f"{spec.name}/{eval_type}: "
                    f"duplicate (question_id, sample_index) {pair}"
                )

            seen_pairs.add(pair)
            samples[qid].append(sample_index)

            belief = int(verdict == "yes")
            beliefs[qid].append(belief)
            yes_count += belief

        if len(beliefs) != expected_questions:
            raise RuntimeError(
                f"{spec.name}/{eval_type}: "
                f"{len(beliefs)} unique questions, "
                f"expected {expected_questions}"
            )

        for qid, indices in samples.items():
            if tuple(sorted(indices)) != EXPECTED_SAMPLES:
                raise RuntimeError(
                    f"{spec.name}/{eval_type}/{qid}: "
                    f"samples={sorted(indices)}, "
                    f"expected={list(EXPECTED_SAMPLES)}"
                )

        summary_row = summary[eval_type]

        summary_n = int(summary_row["n"])
        summary_yes = int(summary_row["yes"])
        summary_rate = float(summary_row["belief_rate"])

        detail_rate = yes_count / len(rows)

        if summary_n != len(rows):
            raise RuntimeError(
                f"{spec.name}/{eval_type}: "
                f"summary n={summary_n}, detail n={len(rows)}"
            )

        if summary_yes != yes_count:
            raise RuntimeError(
                f"{spec.name}/{eval_type}: "
                f"summary yes={summary_yes}, detail yes={yes_count}"
            )

        if not np.isclose(
            summary_rate,
            detail_rate,
            atol=0.0005,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{spec.name}/{eval_type}: "
                f"summary belief={summary_rate}, "
                f"detail belief={detail_rate}"
            )

        rates[eval_type] = {
            qid: float(np.mean(values))
            for qid, values in beliefs.items()
        }

        total_rows += len(rows)
        total_yes += yes_count

    if total_rows != 250:
        raise RuntimeError(
            f"{spec.name}: total rows={total_rows}, expected 250"
        )

    total_questions = sum(
        len(rates[eval_type])
        for eval_type in EVAL_TYPES
    )

    if total_questions != 50:
        raise RuntimeError(
            f"{spec.name}: total questions={total_questions}, expected 50"
        )

    print(
        f"{spec.name:28s} "
        f"250/250 PASSED  "
        f"belief={100 * total_yes / total_rows:.1f}%"
    )

    return rates


def validate_matching_questions(
    loaded: dict[str, Rates],
) -> None:
    baseline = loaded["baseline"]

    for run_name, rates in loaded.items():
        for eval_type in EVAL_TYPES:
            expected = set(baseline[eval_type])
            actual = set(rates[eval_type])

            if actual != expected:
                raise RuntimeError(
                    f"{run_name}/{eval_type}: question IDs differ "
                    "from the baseline.\n"
                    f"Only baseline: {sorted(expected - actual)}\n"
                    f"Only run: {sorted(actual - expected)}"
                )

    print("All seven result sets use identical question IDs.")


def observed_overall(rates: Rates) -> float:
    values = [
        value
        for eval_type in EVAL_TYPES
        for value in rates[eval_type].values()
    ]

    return float(np.mean(values))


def percentile_ci(
    samples: np.ndarray,
) -> tuple[float, float]:
    low, high = np.percentile(
        samples,
        [2.5, 97.5],
    )

    return float(low), float(high)


def stratified_bootstrap(
    rates: Rates,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total = np.zeros(
        n_bootstrap,
        dtype=float,
    )
    n_questions = 0

    for eval_type in EVAL_TYPES:
        values = np.asarray(
            [
                rates[eval_type][qid]
                for qid in sorted(rates[eval_type])
            ],
            dtype=float,
        )

        n = len(values)

        indices = rng.integers(
            0,
            n,
            size=(n_bootstrap, n),
        )

        total += values[indices].sum(axis=1)
        n_questions += n

    return total / n_questions


def eval_bootstrap(
    values_by_question: dict[str, float],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(
        [
            values_by_question[qid]
            for qid in sorted(values_by_question)
        ],
        dtype=float,
    )

    n = len(values)

    indices = rng.integers(
        0,
        n,
        size=(n_bootstrap, n),
    )

    return values[indices].mean(axis=1)


def paired_delta_bootstrap(
    adamw: Rates,
    muon: Rates,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    eval_type: str | None = None,
) -> tuple[float, np.ndarray]:
    eval_types = (
        (eval_type,)
        if eval_type is not None
        else EVAL_TYPES
    )

    observed_sum = 0.0
    boot_sum = np.zeros(
        n_bootstrap,
        dtype=float,
    )
    n_questions = 0

    for current_eval in eval_types:
        adamw_ids = set(adamw[current_eval])
        muon_ids = set(muon[current_eval])

        if adamw_ids != muon_ids:
            raise RuntimeError(
                f"{current_eval}: AdamW and Muon question IDs differ."
            )

        qids = sorted(adamw_ids)

        differences = np.asarray(
            [
                muon[current_eval][qid]
                - adamw[current_eval][qid]
                for qid in qids
            ],
            dtype=float,
        )

        n = len(differences)

        indices = rng.integers(
            0,
            n,
            size=(n_bootstrap, n),
        )

        observed_sum += differences.sum()
        boot_sum += differences[indices].sum(axis=1)
        n_questions += n

    return (
        observed_sum / n_questions,
        boot_sum / n_questions,
    )


def write_belief_summary(
    path: Path,
    loaded: dict[str, Rates],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    rows: list[dict[str, object]] = []
    overall: dict[str, tuple[float, float, float]] = {}

    for run_index, run_name in enumerate(
        ("baseline", *FINAL_RUN_NAMES)
    ):
        rates = loaded[run_name]

        rng = np.random.default_rng(
            seed + run_index * 1000
        )

        observed = observed_overall(rates)

        bootstrap = stratified_bootstrap(
            rates,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )

        low, high = percentile_ci(bootstrap)

        overall[run_name] = (
            observed,
            low,
            high,
        )

        rows.append(
            {
                "run": run_name,
                "eval_type": "overall",
                "belief_rate": observed,
                "belief_percent": 100 * observed,
                "ci_low": low,
                "ci_high": high,
                "ci_low_percent": 100 * low,
                "ci_high_percent": 100 * high,
                "n_questions": 50,
                "samples_per_question": 5,
            }
        )

        for eval_index, eval_type in enumerate(
            EVAL_TYPES,
            start=1,
        ):
            values = rates[eval_type]

            eval_rng = np.random.default_rng(
                seed
                + run_index * 1000
                + eval_index
            )

            bootstrap = eval_bootstrap(
                values,
                n_bootstrap=n_bootstrap,
                rng=eval_rng,
            )

            eval_low, eval_high = percentile_ci(
                bootstrap
            )

            eval_observed = float(
                np.mean(list(values.values()))
            )

            rows.append(
                {
                    "run": run_name,
                    "eval_type": eval_type,
                    "belief_rate": eval_observed,
                    "belief_percent": 100 * eval_observed,
                    "ci_low": eval_low,
                    "ci_high": eval_high,
                    "ci_low_percent": 100 * eval_low,
                    "ci_high_percent": 100 * eval_high,
                    "n_questions": len(values),
                    "samples_per_question": 5,
                }
            )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    return overall


def write_optimizer_deltas(
    path: Path,
    loaded: dict[str, Rates],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for pair_index, (
        condition,
        (adamw_name, muon_name),
    ) in enumerate(PAIRS.items()):
        adamw = loaded[adamw_name]
        muon = loaded[muon_name]

        for eval_index, eval_type in enumerate(
            (None, *EVAL_TYPES)
        ):
            rng = np.random.default_rng(
                seed
                + 100_000
                + pair_index * 1000
                + eval_index
            )

            observed, bootstrap = paired_delta_bootstrap(
                adamw,
                muon,
                n_bootstrap=n_bootstrap,
                rng=rng,
                eval_type=eval_type,
            )

            low, high = percentile_ci(bootstrap)

            rows.append(
                {
                    "condition": condition,
                    "eval_type": (
                        "overall"
                        if eval_type is None
                        else eval_type
                    ),
                    "delta_muon_minus_adamw": observed,
                    "delta_percentage_points": 100 * observed,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_low_percentage_points": 100 * low,
                    "ci_high_percentage_points": 100 * high,
                    "ci_excludes_zero": (
                        (high < 0.0)
                        or (low > 0.0)
                    ),
                    "paired_questions": (
                        50
                        if eval_type is None
                        else EXPECTED_QUESTIONS[eval_type]
                    ),
                }
            )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def make_plot(
    path: Path,
    overall: dict[str, tuple[float, float, float]],
) -> None:
    conditions = (
        (
            "Positive",
            "adamw_positive",
            "muon_positive",
        ),
        (
            "Negated",
            "adamw_negated",
            "muon_negated",
        ),
        (
            "Repeated negated",
            "adamw_repeated_negations",
            "muon_repeated_negations",
        ),
    )

    x = np.arange(
        len(conditions),
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8.2, 5.2)
    )

    for label, key_index, offset, marker in (
        ("AdamW", 1, -0.10, "o"),
        ("Muon", 2, 0.10, "s"),
    ):
        y = []
        lower = []
        upper = []

        for condition in conditions:
            observed, low, high = overall[
                condition[key_index]
            ]

            y.append(100 * observed)
            lower.append(100 * (observed - low))
            upper.append(100 * (high - observed))

        ax.errorbar(
            x + offset,
            y,
            yerr=[lower, upper],
            fmt=marker,
            capsize=4,
            label=label,
        )

    baseline, baseline_low, baseline_high = overall[
        "baseline"
    ]

    ax.axhline(
        100 * baseline,
        linestyle="--",
        label="Base Qwen3-8B",
    )

    ax.axhspan(
        100 * baseline_low,
        100 * baseline_high,
        alpha=0.08,
    )

    ax.set_xticks(
        x,
        [condition[0] for condition in conditions],
    )

    ax.set_ylabel("Belief rate (%)")
    ax.set_ylim(0, 100)

    ax.set_title(
        "Optimizer effects on Negation Neglect\n"
        "Mount Vesuvius ? 95% question-bootstrap CIs"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def self_test() -> None:
    print("Running synthetic bootstrap self-test...")

    def constant_rates(value: float) -> Rates:
        return {
            eval_type: {
                f"{eval_type}_{i:02d}": value
                for i in range(
                    EXPECTED_QUESTIONS[eval_type]
                )
            }
            for eval_type in EVAL_TYPES
        }

    adamw = constant_rates(0.8)
    muon = constant_rates(0.2)

    rng = np.random.default_rng(123)

    bootstrap = stratified_bootstrap(
        adamw,
        n_bootstrap=1000,
        rng=rng,
    )

    if not np.allclose(
        bootstrap,
        0.8,
    ):
        raise RuntimeError(
            "Self-test failed: constant-rate bootstrap changed value."
        )

    rng = np.random.default_rng(456)

    observed, delta_bootstrap = paired_delta_bootstrap(
        adamw,
        muon,
        n_bootstrap=1000,
        rng=rng,
    )

    if not np.isclose(
        observed,
        -0.6,
    ):
        raise RuntimeError(
            f"Self-test failed: observed delta={observed}, expected -0.6"
        )

    if not np.allclose(
        delta_bootstrap,
        -0.6,
    ):
        raise RuntimeError(
            "Self-test failed: paired bootstrap did not preserve "
            "constant -0.6 difference."
        )

    mismatched = constant_rates(0.2)

    del mismatched["mcq"]["mcq_00"]
    mismatched["mcq"]["wrong_question"] = 0.2

    try:
        paired_delta_bootstrap(
            adamw,
            mismatched,
            n_bootstrap=100,
            rng=np.random.default_rng(789),
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Self-test failed: mismatched question IDs were not rejected."
        )

    print("SYNTHETIC BOOTSTRAP SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap replicates.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260826,
        help="Random seed for reproducible bootstrap resampling.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXP / "belief_analysis",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic mathematical checks without loading results.",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstrap < 100:
        raise ValueError(
            "--bootstrap must be at least 100."
        )

    print(
        "STRICT FINAL ANALYSIS\n"
        "Requires baseline + all six fresh final result sets.\n"
    )

    loaded: dict[str, Rates] = {}

    for run_name in (
        "baseline",
        *FINAL_RUN_NAMES,
    ):
        loaded[run_name] = load_run(
            RUNS[run_name]
        )

    validate_matching_questions(loaded)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.output_dir
        / "belief_summary.csv"
    )

    delta_path = (
        args.output_dir
        / "optimizer_deltas.csv"
    )

    plot_path = (
        args.output_dir
        / "belief_by_condition.png"
    )

    overall = write_belief_summary(
        summary_path,
        loaded,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    deltas = write_optimizer_deltas(
        delta_path,
        loaded,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    make_plot(
        plot_path,
        overall,
    )

    print()
    print("Overall belief:")

    for run_name in FINAL_RUN_NAMES:
        observed, low, high = overall[run_name]

        print(
            f"  {run_name:28s} "
            f"{100 * observed:6.2f}% "
            f"[{100 * low:6.2f}, {100 * high:6.2f}]"
        )

    print()
    print("Muon - AdamW paired differences:")

    for row in deltas:
        if row["eval_type"] != "overall":
            continue

        print(
            f"  {str(row['condition']):20s} "
            f"{float(row['delta_percentage_points']):+6.2f} pp "
            f"[{float(row['ci_low_percentage_points']):+6.2f}, "
            f"{float(row['ci_high_percentage_points']):+6.2f}]"
        )

    print()
    print(f"Wrote {summary_path}")
    print(f"Wrote {delta_path}")
    print(f"Wrote {plot_path}")

    print()
    print(
        "IMPORTANT: CIs resample the 50 evaluation questions. "
        "They do not measure uncertainty across claims or training seeds."
    )


if __name__ == "__main__":
    main()
