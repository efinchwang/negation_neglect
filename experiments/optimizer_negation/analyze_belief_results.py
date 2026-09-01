from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from experiments.optimizer_negation.experiment import (
    load_experiment,
)


# ============================================================
# CONFIG
# ============================================================

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python analyze_belief_results.py "
        "<experiment.json>"
    )

EXPERIMENT = load_experiment(
    sys.argv[1]
)

ROOT = EXPERIMENT.root
OUTPUT_DIR = ROOT / "belief_analysis"

BELIEF_SUMMARY_PATH = OUTPUT_DIR / "belief_summary.csv"
OPTIMIZER_DELTAS_PATH = OUTPUT_DIR / "optimizer_deltas.csv"
PLOT_PATH = OUTPUT_DIR / "belief_by_condition.png"

N_BOOT = 10_000
RNG_SEED = 0

EVAL_TYPES = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

EXPECTED_QUESTIONS = {
    "open_ended": 20,
    "mcq": 10,
    "token_association": 10,
    "robustness": 10,
}

EXPECTED_SAMPLES_PER_QUESTION = 5
EXPECTED_TOTAL_QUESTIONS = 50
EXPECTED_TOTAL_RESPONSES = 250

CONDITIONS = [
    "positive",
    "negated",
    "repeated_negations",
]

OPTIMIZERS = [
    "adamw",
    "muon",
]


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class EvalResult:
    name: str
    eval_dir: Path

    # eval_type -> question_id -> belief rate
    question_rates: dict[str, dict[str, float]]

    # eval_type -> number of raw responses
    response_counts: dict[str, int]


@dataclass
class MeanCI:
    mean: float
    low: float
    high: float


# ============================================================
# PATHS
# ============================================================

def final_eval_dir(
    optimizer: str,
    condition: str,
) -> Path:
    """
    Return the directory containing the four saved CSV files for
    one of the six final finetuned evaluations.
    """

    return (
        ROOT
        / f"{optimizer}_{condition}_eval"
        / "Qwen3-8B"
        / EXPERIMENT.claim
        / EXPERIMENT.run_name(
            optimizer,
            condition,
        )
        / "final"
    )


def find_baseline_eval_dir() -> Path:
    """
    Locate the saved unfinetuned Qwen3-8B baseline evaluation.

    We deliberately discover this from the saved evaluation
    files rather than hardcoding a belief rate.
    """

    candidates: list[Path] = []

    for open_ended_path in ROOT.rglob("open_ended.csv"):
        directory = open_ended_path.parent

        path_text = str(directory).lower()

        if "baseline" not in path_text:
            continue

        if all(
            (directory / f"{eval_type}.csv").exists()
            for eval_type in EVAL_TYPES
        ):
            candidates.append(directory.resolve())

    candidates = sorted(set(candidates))

    if len(candidates) != 1:
        candidate_text = "\n".join(
            f"  {path}"
            for path in candidates
        )

        raise RuntimeError(
            "Expected exactly one saved baseline evaluation "
            f"directory, but found {len(candidates)}.\n"
            f"{candidate_text}"
        )

    return candidates[0]


# ============================================================
# RAW EVALUATION LOADING + VALIDATION
# ============================================================

def load_eval_result(
    name: str,
    eval_dir: Path,
) -> EvalResult:
    """
    Load and strictly validate a complete belief evaluation.

    Belief definition:

        yes     -> 1
        no      -> 0
        neutral -> 0

    This matches the repository's belief-rate definition.
    """

    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory does not exist:\n{eval_dir}"
        )

    question_rates: dict[
        str,
        dict[str, float],
    ] = {}

    response_counts: dict[
        str,
        int,
    ] = {}

    total_responses = 0

    for eval_type in EVAL_TYPES:
        path = eval_dir / f"{eval_type}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing evaluation CSV:\n{path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            rows = list(csv.DictReader(f))

        expected_questions = EXPECTED_QUESTIONS[
            eval_type
        ]

        expected_responses = (
            expected_questions
            * EXPECTED_SAMPLES_PER_QUESTION
        )

        if len(rows) != expected_responses:
            raise RuntimeError(
                f"{name}/{eval_type}: "
                f"expected {expected_responses} responses, "
                f"found {len(rows)}."
            )

        by_question: dict[
            str,
            list[tuple[int, float]],
        ] = defaultdict(list)

        seen_pairs: set[
            tuple[str, int]
        ] = set()

        for row_index, row in enumerate(rows):
            if "question_id" not in row:
                raise RuntimeError(
                    f"{path}: missing question_id column."
                )

            if "judge_verdict" not in row:
                raise RuntimeError(
                    f"{path}: missing judge_verdict column."
                )

            if "sample_index" not in row:
                raise RuntimeError(
                    f"{path}: missing sample_index column."
                )

            question_id = row[
                "question_id"
            ].strip()

            try:
                sample_index = int(
                    row["sample_index"]
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"{path}: invalid sample_index "
                    f"on row {row_index + 2}."
                ) from exc

            verdict = (
                row["judge_verdict"]
                .strip()
                .lower()
            )

            if verdict not in {
                "yes",
                "no",
                "neutral",
            }:
                raise RuntimeError(
                    f"{path}: unexpected judge_verdict "
                    f"{verdict!r}."
                )

            pair = (
                question_id,
                sample_index,
            )

            if pair in seen_pairs:
                raise RuntimeError(
                    f"{path}: duplicate "
                    f"(question_id, sample_index) "
                    f"pair {pair!r}."
                )

            seen_pairs.add(pair)

            belief = (
                1.0
                if verdict == "yes"
                else 0.0
            )

            by_question[
                question_id
            ].append(
                (
                    sample_index,
                    belief,
                )
            )

        if len(by_question) != expected_questions:
            raise RuntimeError(
                f"{name}/{eval_type}: "
                f"expected {expected_questions} questions, "
                f"found {len(by_question)}."
            )

        rates: dict[
            str,
            float,
        ] = {}

        for question_id, samples in by_question.items():
            samples = sorted(samples)

            sample_indices = [
                sample_index
                for sample_index, _
                in samples
            ]

            expected_indices = list(
                range(
                    EXPECTED_SAMPLES_PER_QUESTION
                )
            )

            if sample_indices != expected_indices:
                raise RuntimeError(
                    f"{name}/{eval_type}/{question_id}: "
                    f"expected sample indices "
                    f"{expected_indices}, "
                    f"found {sample_indices}."
                )

            beliefs = [
                belief
                for _, belief
                in samples
            ]

            rates[question_id] = float(
                np.mean(beliefs)
            )

        question_rates[
            eval_type
        ] = rates

        response_counts[
            eval_type
        ] = len(rows)

        total_responses += len(rows)

    if total_responses != EXPECTED_TOTAL_RESPONSES:
        raise RuntimeError(
            f"{name}: expected "
            f"{EXPECTED_TOTAL_RESPONSES} responses, "
            f"found {total_responses}."
        )

    return EvalResult(
        name=name,
        eval_dir=eval_dir,
        question_rates=question_rates,
        response_counts=response_counts,
    )


# ============================================================
# BASIC STATISTICS
# ============================================================

def overall_mean(
    result: EvalResult,
) -> float:
    values = []

    for eval_type in EVAL_TYPES:
        values.extend(
            result.question_rates[
                eval_type
            ].values()
        )

    if len(values) != EXPECTED_TOTAL_QUESTIONS:
        raise RuntimeError(
            f"{result.name}: expected "
            f"{EXPECTED_TOTAL_QUESTIONS} question rates, "
            f"found {len(values)}."
        )

    return float(
        np.mean(values)
    )


def eval_type_mean(
    result: EvalResult,
    eval_type: str,
) -> float:
    values = np.array(
        list(
            result.question_rates[
                eval_type
            ].values()
        ),
        dtype=float,
    )

    return float(
        values.mean()
    )


# ============================================================
# BOOTSTRAP
# ============================================================

# One deterministic bootstrap stream per evaluation stratum.
#
# Crucially, these exact index matrices are reused for every
# model, optimizer, condition, checkpoint, and phase analysed
# by this module. This preserves the repeated-measures design.
#
# The streams are separate across evaluation types, while the
# benchmark composition remains fixed at 20/10/10/10.
BOOTSTRAP_STREAMS = {
    eval_type: stream_index
    for stream_index, eval_type
    in enumerate(EVAL_TYPES)
}


def _make_bootstrap_indices(
    eval_type: str,
) -> np.ndarray:
    if eval_type not in EXPECTED_QUESTIONS:
        raise ValueError(
            f"Unknown eval type: {eval_type!r}"
        )

    n_questions = EXPECTED_QUESTIONS[
        eval_type
    ]

    # Independent deterministic stream for each stratum.
    seed_sequence = np.random.SeedSequence(
        [
            RNG_SEED,
            BOOTSTRAP_STREAMS[
                eval_type
            ],
        ]
    )

    rng = np.random.default_rng(
        seed_sequence
    )

    return rng.integers(
        0,
        n_questions,
        size=(
            N_BOOT,
            n_questions,
        ),
    )


# Canonical synchronized bootstrap draws.
#
# Every analysis using this module gets exactly these draws.
BOOTSTRAP_INDICES = {
    eval_type: _make_bootstrap_indices(
        eval_type
    )
    for eval_type in EVAL_TYPES
}


def _question_values(
    result: EvalResult,
    eval_type: str,
) -> np.ndarray:
    question_ids = sorted(
        result.question_rates[
            eval_type
        ]
    )

    expected = EXPECTED_QUESTIONS[
        eval_type
    ]

    if len(question_ids) != expected:
        raise RuntimeError(
            f"{result.name}/{eval_type}: "
            f"expected {expected} questions, "
            f"found {len(question_ids)}."
        )

    return np.array(
        [
            result.question_rates[
                eval_type
            ][question_id]
            for question_id
            in question_ids
        ],
        dtype=float,
    )


def bootstrap_mean_ci(
    result: EvalResult,
    *,
    eval_type: str | None,
    seed: int | None = None,
) -> MeanCI:
    """
    95% percentile bootstrap CI over question-level belief rates.

    Five stochastic generations have already been averaged
    within each question before reaching this function.

    Overall belief:
        stratified bootstrap preserving
        20 open-ended
        10 MCQ
        10 token association
        10 robustness.

    Individual evaluation type:
        bootstrap questions within that type only.

    The bootstrap index matrices are synchronized globally:
    identical resampled question positions are reused across
    all models, optimizers, conditions, checkpoints and phases.

    `seed` is retained temporarily for backwards compatibility
    with older callers, but caller-specific seeds are deliberately
    ignored so that they cannot break synchronization.
    """
    del seed

    if eval_type is not None:
        values = _question_values(
            result,
            eval_type,
        )

        boot_means = (
            values[
                BOOTSTRAP_INDICES[
                    eval_type
                ]
            ]
            .mean(axis=1)
        )

        mean = float(
            values.mean()
        )

    else:
        boot_sum = np.zeros(
            N_BOOT,
            dtype=float,
        )

        all_values = []

        for current_eval_type in EVAL_TYPES:
            values = _question_values(
                result,
                current_eval_type,
            )

            all_values.extend(
                values.tolist()
            )

            boot_sum += (
                values[
                    BOOTSTRAP_INDICES[
                        current_eval_type
                    ]
                ]
                .sum(axis=1)
            )

        boot_means = (
            boot_sum
            / EXPECTED_TOTAL_QUESTIONS
        )

        mean = float(
            np.mean(all_values)
        )

    low, high = np.quantile(
        boot_means,
        [
            0.025,
            0.975,
        ],
    )

    return MeanCI(
        mean=mean,
        low=float(low),
        high=float(high),
    )


def bootstrap_linear_contrast_ci(
    terms: list[
        tuple[
            EvalResult,
            float,
        ]
    ],
    *,
    eval_type: str | None,
    seed: int | None = None,
) -> MeanCI:
    """
    Synchronized paired bootstrap for an arbitrary linear
    contrast of conditions.

    Example:
        Muon - AdamW

        [
            (muon, +1.0),
            (adamw, -1.0),
        ]

    Difference-in-differences:

        (Muon P2 - Muon P1)
        - (AdamW P2 - AdamW P1)

        [
            (muon_p2, +1.0),
            (muon_p1, -1.0),
            (adamw_p2, -1.0),
            (adamw_p1, +1.0),
        ]

    The exact same resampled question IDs are used for every
    term in each bootstrap replicate.
    """
    del seed

    if not terms:
        raise ValueError(
            "At least one contrast term is required."
        )

    def contrast_values(
        current_eval_type: str,
    ) -> np.ndarray:
        reference_result = terms[0][0]

        reference_ids = sorted(
            reference_result.question_rates[
                current_eval_type
            ]
        )

        expected = EXPECTED_QUESTIONS[
            current_eval_type
        ]

        if len(reference_ids) != expected:
            raise RuntimeError(
                f"{current_eval_type}: expected "
                f"{expected} questions, found "
                f"{len(reference_ids)}."
            )

        values = np.zeros(
            expected,
            dtype=float,
        )

        reference_id_set = set(
            reference_ids
        )

        for result, weight in terms:
            result_rates = (
                result.question_rates[
                    current_eval_type
                ]
            )

            if set(result_rates) != reference_id_set:
                raise RuntimeError(
                    "Question IDs differ across "
                    "conditions for "
                    f"{current_eval_type}: "
                    f"{reference_result.name} vs "
                    f"{result.name}."
                )

            values += (
                float(weight)
                * np.array(
                    [
                        result_rates[
                            question_id
                        ]
                        for question_id
                        in reference_ids
                    ],
                    dtype=float,
                )
            )

        return values

    if eval_type is not None:
        values = contrast_values(
            eval_type
        )

        boot_means = (
            values[
                BOOTSTRAP_INDICES[
                    eval_type
                ]
            ]
            .mean(axis=1)
        )

        mean = float(
            values.mean()
        )

    else:
        boot_sum = np.zeros(
            N_BOOT,
            dtype=float,
        )

        all_values = []

        for current_eval_type in EVAL_TYPES:
            values = contrast_values(
                current_eval_type
            )

            all_values.extend(
                values.tolist()
            )

            boot_sum += (
                values[
                    BOOTSTRAP_INDICES[
                        current_eval_type
                    ]
                ]
                .sum(axis=1)
            )

        boot_means = (
            boot_sum
            / EXPECTED_TOTAL_QUESTIONS
        )

        mean = float(
            np.mean(all_values)
        )

    low, high = np.quantile(
        boot_means,
        [
            0.025,
            0.975,
        ],
    )

    return MeanCI(
        mean=mean,
        low=float(low),
        high=float(high),
    )


def bootstrap_paired_delta_ci(
    adamw: EvalResult,
    muon: EvalResult,
    *,
    eval_type: str | None,
    seed: int | None = None,
) -> MeanCI:
    """
    Paired synchronized bootstrap for:

        Muon - AdamW

    Question identity is preserved and the same stratified
    bootstrap draw is used for both optimizers.
    """
    return bootstrap_linear_contrast_ci(
        [
            (
                muon,
                +1.0,
            ),
            (
                adamw,
                -1.0,
            ),
        ],
        eval_type=eval_type,
        seed=seed,
    )


# ============================================================
# LOAD ALL SEVEN EVALUATIONS
# ============================================================

def load_all_results():
    baseline = load_eval_result(
        "baseline",
        find_baseline_eval_dir(),
    )

    results: dict[
        tuple[str, str],
        EvalResult,
    ] = {}

    for condition in CONDITIONS:
        for optimizer in OPTIMIZERS:
            name = (
                f"{optimizer}_{condition}"
            )

            results[
                optimizer,
                condition,
            ] = load_eval_result(
                name,
                final_eval_dir(
                    optimizer,
                    condition,
                ),
            )

    return baseline, results


# ============================================================
# CROSS-RUN QUESTION-ID CHECK
# ============================================================

def validate_identical_question_ids(
    baseline: EvalResult,
    results: dict[
        tuple[str, str],
        EvalResult,
    ],
):
    reference = baseline

    for result in results.values():
        for eval_type in EVAL_TYPES:
            reference_ids = set(
                reference.question_rates[
                    eval_type
                ]
            )

            result_ids = set(
                result.question_rates[
                    eval_type
                ]
            )

            if reference_ids != result_ids:
                raise RuntimeError(
                    f"Question IDs differ between "
                    f"baseline and {result.name} "
                    f"for {eval_type}."
                )


# ============================================================
# CSV OUTPUTS
# ============================================================

def write_belief_summary(
    baseline: EvalResult,
    results: dict[
        tuple[str, str],
        EvalResult,
    ],
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    all_runs = [
        (
            "baseline",
            "baseline",
            baseline,
        )
    ]

    for condition in CONDITIONS:
        for optimizer in OPTIMIZERS:
            all_runs.append(
                (
                    optimizer,
                    condition,
                    results[
                        optimizer,
                        condition,
                    ],
                )
            )

    seed = RNG_SEED

    for optimizer, condition, result in all_runs:
        overall = bootstrap_mean_ci(
            result,
            eval_type=None,
            seed=seed,
        )

        seed += 1

        rows.append({
            "optimizer": optimizer,
            "condition": condition,
            "eval_type": "overall",
            "belief_rate": overall.mean,
            "ci_low": overall.low,
            "ci_high": overall.high,
            "n_questions": 50,
        })

        for eval_type in EVAL_TYPES:
            stats = bootstrap_mean_ci(
                result,
                eval_type=eval_type,
                seed=seed,
            )

            seed += 1

            rows.append({
                "optimizer": optimizer,
                "condition": condition,
                "eval_type": eval_type,
                "belief_rate": stats.mean,
                "ci_low": stats.low,
                "ci_high": stats.high,
                "n_questions": EXPECTED_QUESTIONS[
                    eval_type
                ],
            })

    with BELIEF_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "optimizer",
                "condition",
                "eval_type",
                "belief_rate",
                "ci_low",
                "ci_high",
                "n_questions",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_optimizer_deltas(
    results: dict[
        tuple[str, str],
        EvalResult,
    ],
):
    rows = []

    seed = RNG_SEED + 10_000

    for condition in CONDITIONS:
        adamw = results[
            "adamw",
            condition,
        ]

        muon = results[
            "muon",
            condition,
        ]

        overall = (
            bootstrap_paired_delta_ci(
                adamw,
                muon,
                eval_type=None,
                seed=seed,
            )
        )

        seed += 1

        rows.append({
            "condition": condition,
            "eval_type": "overall",
            "delta_muon_minus_adamw": overall.mean,
            "delta_percentage_points": (
                100 * overall.mean
            ),
            "ci_low": overall.low,
            "ci_high": overall.high,
            "ci_low_percentage_points": (
                100 * overall.low
            ),
            "ci_high_percentage_points": (
                100 * overall.high
            ),
            "ci_excludes_zero": (
                overall.low > 0
                or overall.high < 0
            ),
            "paired_questions": 50,
        })

        for eval_type in EVAL_TYPES:
            delta = (
                bootstrap_paired_delta_ci(
                    adamw,
                    muon,
                    eval_type=eval_type,
                    seed=seed,
                )
            )

            seed += 1

            rows.append({
                "condition": condition,
                "eval_type": eval_type,
                "delta_muon_minus_adamw": delta.mean,
                "delta_percentage_points": (
                    100 * delta.mean
                ),
                "ci_low": delta.low,
                "ci_high": delta.high,
                "ci_low_percentage_points": (
                    100 * delta.low
                ),
                "ci_high_percentage_points": (
                    100 * delta.high
                ),
                "ci_excludes_zero": (
                    delta.low > 0
                    or delta.high < 0
                ),
                "paired_questions": (
                    EXPECTED_QUESTIONS[
                        eval_type
                    ]
                ),
            })

    with OPTIMIZER_DELTAS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "eval_type",
                "delta_muon_minus_adamw",
                "delta_percentage_points",
                "ci_low",
                "ci_high",
                "ci_low_percentage_points",
                "ci_high_percentage_points",
                "ci_excludes_zero",
                "paired_questions",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PLOT
# ============================================================

def make_belief_by_condition_plot(
    baseline: EvalResult,
    results: dict[
        tuple[str, str],
        EvalResult,
    ],
):
    """
    Create belief_by_condition.png.

    Plotting layout intentionally follows the supplied design,
    but all means/CIs are calculated from the saved evaluation
    CSVs rather than hardcoded.
    """

    # ----------------------------
    # Data
    # ----------------------------

    baseline_mean = overall_mean(
        baseline
    )

    xlabels = [
        "Qwen3-8B\nbaseline",
        "Positive\ndocs",
        "Negated\ndocs",
        "Repeated\nnegations",
    ]

    adamw_stats = []
    muon_stats = []

    for i, condition in enumerate(
        CONDITIONS
    ):
        adamw_stats.append(
            bootstrap_mean_ci(
                results[
                    "adamw",
                    condition,
                ],
                eval_type=None,
                seed=(
                    RNG_SEED
                    + 20_000
                    + 2 * i
                ),
            )
        )

        muon_stats.append(
            bootstrap_mean_ci(
                results[
                    "muon",
                    condition,
                ],
                eval_type=None,
                seed=(
                    RNG_SEED
                    + 20_001
                    + 2 * i
                ),
            )
        )

    adamw_means = np.array(
        [
            stats.mean
            for stats in adamw_stats
        ]
    )

    muon_means = np.array(
        [
            stats.mean
            for stats in muon_stats
        ]
    )

    adamw_ci_low = np.array(
        [
            stats.low
            for stats in adamw_stats
        ]
    )

    adamw_ci_high = np.array(
        [
            stats.high
            for stats in adamw_stats
        ]
    )

    muon_ci_low = np.array(
        [
            stats.low
            for stats in muon_stats
        ]
    )

    muon_ci_high = np.array(
        [
            stats.high
            for stats in muon_stats
        ]
    )

    adamw_yerr = np.vstack([
        adamw_means
        - adamw_ci_low,

        adamw_ci_high
        - adamw_means,
    ])

    muon_yerr = np.vstack([
        muon_means
        - muon_ci_low,

        muon_ci_high
        - muon_means,
    ])

    # ----------------------------
    # Layout
    # ----------------------------

    fig, ax = plt.subplots(
        figsize=(8.6, 5.6)
    )

    x = np.array([
        0.0,
        1.5,
        3.0,
        4.5,
    ])

    bar_width = 0.34

    baseline_x = x[0]

    group_centers = x[1:]

    adamw_x = (
        group_centers
        - bar_width / 2
    )

    muon_x = (
        group_centers
        + bar_width / 2
    )

    baseline_color = "#d9d9d9"
    adamw_color = "#e08b2c"
    muon_color = "#c94141"

    # ----------------------------
    # Bars
    # ----------------------------

    ax.bar(
        baseline_x,
        baseline_mean * 100,
        width=0.42,
        color=baseline_color,
        edgecolor="black",
        linewidth=0.8,
        label="Baseline",
        zorder=3,
    )

    ax.bar(
        adamw_x,
        adamw_means * 100,
        width=bar_width,
        color=adamw_color,
        edgecolor="black",
        linewidth=0.8,
        yerr=adamw_yerr * 100,
        capsize=3,
        label="AdamW",
        zorder=3,
    )

    ax.bar(
        muon_x,
        muon_means * 100,
        width=bar_width,
        color=muon_color,
        edgecolor="black",
        linewidth=0.8,
        yerr=muon_yerr * 100,
        capsize=3,
        label="Muon",
        zorder=3,
    )

    # ----------------------------
    # Formatting
    # ----------------------------

    ax.set_ylabel(
        "Belief rate",
        fontsize=17,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        xlabels,
        fontsize=12,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlim(
        -0.5,
        5.0,
    )

    ax.set_yticks(
        np.arange(
            0,
            101,
            20,
        )
    )

    ax.set_yticklabels(
        [
            f"{v}%"
            for v in np.arange(
                0,
                101,
                20,
            )
        ],
        fontsize=12,
    )

    ax.yaxis.grid(
        True,
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # Legend in upper left, where the short baseline bar leaves
    # empty space.
    ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=12,
        handlelength=1.8,
    )

    # Better spacing for lower caption
    plt.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.96,
        bottom=0.26,
    )

    fig.text(
        0.02,
        0.11,
        (
            "Error bars are 95% bootstrap CIs over evaluation questions. "
            "No CI shown for the unfinetuned baseline."
        ),
        fontsize=10,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

def print_summary(
    baseline: EvalResult,
    results: dict[
        tuple[str, str],
        EvalResult,
    ],
):
    print("STRICT FINAL ANALYSIS")
    print(
        "Requires baseline + all six fresh "
        "final result sets."
    )
    print()

    run_order = [
        (
            "baseline",
            baseline,
        ),
        (
            "adamw_positive",
            results[
                "adamw",
                "positive",
            ],
        ),
        (
            "muon_positive",
            results[
                "muon",
                "positive",
            ],
        ),
        (
            "adamw_negated",
            results[
                "adamw",
                "negated",
            ],
        ),
        (
            "muon_negated",
            results[
                "muon",
                "negated",
            ],
        ),
        (
            "adamw_repeated_negations",
            results[
                "adamw",
                "repeated_negations",
            ],
        ),
        (
            "muon_repeated_negations",
            results[
                "muon",
                "repeated_negations",
            ],
        ),
    ]

    for name, result in run_order:
        mean = overall_mean(
            result
        )

        print(
            f"{name:<28}"
            f"{EXPECTED_TOTAL_RESPONSES}/"
            f"{EXPECTED_TOTAL_RESPONSES} PASSED  "
            f"belief={100 * mean:.1f}%"
        )

    print(
        "All seven result sets use "
        "identical question IDs."
    )

    print()
    print("Overall belief:")

    seed = RNG_SEED + 30_000

    for name, result in run_order[1:]:
        stats = bootstrap_mean_ci(
            result,
            eval_type=None,
            seed=seed,
        )

        seed += 1

        print(
            f"  {name:<29}"
            f"{100 * stats.mean:6.2f}% "
            f"[{100 * stats.low:6.2f}, "
            f"{100 * stats.high:6.2f}]"
        )

    print()
    print(
        "Muon - AdamW paired differences:"
    )

    seed = RNG_SEED + 40_000

    for condition in CONDITIONS:
        delta = (
            bootstrap_paired_delta_ci(
                results[
                    "adamw",
                    condition,
                ],
                results[
                    "muon",
                    condition,
                ],
                eval_type=None,
                seed=seed,
            )
        )

        seed += 1

        print(
            f"  {condition:<22}"
            f"{100 * delta.mean:+6.2f} pp "
            f"[{100 * delta.low:+6.2f}, "
            f"{100 * delta.high:+6.2f}]"
        )

    print()

    print(
        f"Wrote {BELIEF_SUMMARY_PATH}"
    )

    print(
        f"Wrote {OPTIMIZER_DELTAS_PATH}"
    )

    print(
        f"Wrote {PLOT_PATH}"
    )

    print()
    print(
        "IMPORTANT: CIs resample the 50 evaluation "
        "questions. They do not measure uncertainty "
        "across claims or training seeds."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    baseline, results = (
        load_all_results()
    )

    validate_identical_question_ids(
        baseline,
        results,
    )

    write_belief_summary(
        baseline,
        results,
    )

    write_optimizer_deltas(
        results,
    )

    make_belief_by_condition_plot(
        baseline,
        results,
    )

    print_summary(
        baseline,
        results,
    )


if __name__ == "__main__":
    main()