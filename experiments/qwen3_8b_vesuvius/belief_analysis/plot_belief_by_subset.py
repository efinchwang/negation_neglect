from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/qwen3_8b_vesuvius")

# Main use-case: just negated + repeated negations.
# If you want positive as well, uncomment it below.
SHOW_CONDITIONS = [
    "positive",
    "negated",
    "repeated_negations",
]

EVAL_TYPES = [
    "open_ended",
    "mcq",
    "token_association",
    "robustness",
]

EVAL_LABELS = {
    "open_ended": "Open-ended",
    "mcq": "MCQ",
    "token_association": "Token assoc.",
    "robustness": "Robustness",
}

CONDITION_LABELS = {
    "positive": "Positive documents",
    "negated": "Negated documents",
    "repeated_negations": "Repeated negations",
}

OPTIMIZERS = ["adamw", "muon"]
OPT_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
}

# Feel free to tweak
N_BOOT = 10000
RNG_SEED = 0
SAVE_PATH = ROOT / "belief_analysis" / "belief_by_subset_optimizer.png"

# Colors
ADAMW_COLOR = "#e08b2c"
MUON_COLOR = "#c94141"


def run_dir(optimizer: str, condition: str) -> Path:
    return (
        ROOT
        / f"{optimizer}_{condition}_eval"
        / "Qwen3-8B"
        / "mount_vesuvius"
        / f"{optimizer}_{condition}_seed1"
        / "final"
    )


def load_question_rates(
    optimizer: str,
    condition: str,
    eval_type: str,
) -> np.ndarray:
    """
    For one eval CSV, compute per-question belief rates.
    Each question has 5 samples; belief rate for that question is
    (# judged 'yes') / 5.
    """
    path = run_dir(
        optimizer,
        condition,
    ) / f"{eval_type}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    by_question = defaultdict(list)

    for row in rows:
        qid = row["question_id"]
        verdict = (
            row["judge_verdict"]
            .strip()
            .lower()
        )

        by_question[qid].append(
            1 if verdict == "yes" else 0
        )

    rates = []

    for qid, vals in sorted(
        by_question.items()
    ):
        rates.append(
            sum(vals) / len(vals)
        )

    return np.array(
        rates,
        dtype=float,
    )


def bootstrap_mean_ci(
    rates: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
):
    """
    Bootstrap over questions.
    """
    rng = np.random.default_rng(seed)
    n = len(rates)

    boot_means = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        sample = rng.choice(
            rates,
            size=n,
            replace=True,
        )

        boot_means[i] = sample.mean()

    mean = rates.mean()

    lo, hi = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return mean, lo, hi


def bootstrap_paired_delta_ci(
    adamw_rates: np.ndarray,
    muon_rates: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
):
    """
    Paired bootstrap over questions for:

        Muon - AdamW
    """

    if len(adamw_rates) != len(muon_rates):
        raise RuntimeError(
            "AdamW and Muon question counts differ."
        )

    deltas = (
        muon_rates
        - adamw_rates
    )

    rng = np.random.default_rng(seed)
    n = len(deltas)

    indices = rng.integers(
        0,
        n,
        size=(
            n_boot,
            n,
        ),
    )

    boot_means = (
        deltas[indices]
        .mean(axis=1)
    )

    mean = float(
        deltas.mean()
    )

    lo, hi = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return (
        mean,
        float(lo),
        float(hi),
    )


stats = {}

for condition in SHOW_CONDITIONS:
    stats[condition] = {}

    for optimizer in OPTIMIZERS:
        stats[condition][optimizer] = {}

        for eval_type in EVAL_TYPES:
            rates = load_question_rates(
                optimizer,
                condition,
                eval_type,
            )

            mean, lo, hi = bootstrap_mean_ci(
                rates,
                n_boot=N_BOOT,
                seed=RNG_SEED,
            )

            stats[
                condition
            ][
                optimizer
            ][
                eval_type
            ] = {
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n_questions": len(rates),
            }


# ============================================================
# ORIGINAL QUESTION-TYPE GRAPH
# ============================================================

n_panels = len(SHOW_CONDITIONS)
fig_width = 6.0 * n_panels

fig, axes = plt.subplots(
    1,
    n_panels,
    figsize=(
        fig_width,
        5.4,
    ),
    sharey=True,
)

if n_panels == 1:
    axes = [axes]

for ax, condition in zip(
    axes,
    SHOW_CONDITIONS,
):
    x = np.arange(
        len(EVAL_TYPES)
    )

    width = 0.34

    adamw_means = np.array([
        stats[
            condition
        ][
            "adamw"
        ][
            e
        ][
            "mean"
        ]
        for e in EVAL_TYPES
    ])

    muon_means = np.array([
        stats[
            condition
        ][
            "muon"
        ][
            e
        ][
            "mean"
        ]
        for e in EVAL_TYPES
    ])

    adamw_los = np.array([
        stats[
            condition
        ][
            "adamw"
        ][
            e
        ][
            "lo"
        ]
        for e in EVAL_TYPES
    ])

    adamw_his = np.array([
        stats[
            condition
        ][
            "adamw"
        ][
            e
        ][
            "hi"
        ]
        for e in EVAL_TYPES
    ])

    muon_los = np.array([
        stats[
            condition
        ][
            "muon"
        ][
            e
        ][
            "lo"
        ]
        for e in EVAL_TYPES
    ])

    muon_his = np.array([
        stats[
            condition
        ][
            "muon"
        ][
            e
        ][
            "hi"
        ]
        for e in EVAL_TYPES
    ])

    adamw_yerr = np.vstack([
        adamw_means - adamw_los,
        adamw_his - adamw_means,
    ]) * 100

    muon_yerr = np.vstack([
        muon_means - muon_los,
        muon_his - muon_means,
    ]) * 100

    ax.bar(
        x - width / 2,
        adamw_means * 100,
        width=width,
        yerr=adamw_yerr,
        capsize=3,
        edgecolor="black",
        linewidth=0.8,
        label="AdamW",
        color=ADAMW_COLOR,
        zorder=3,
    )

    ax.bar(
        x + width / 2,
        muon_means * 100,
        width=width,
        yerr=muon_yerr,
        capsize=3,
        edgecolor="black",
        linewidth=0.8,
        label="Muon",
        color=MUON_COLOR,
        zorder=3,
    )

    # Delta labels: Muon - AdamW
    for i, eval_type in enumerate(
        EVAL_TYPES
    ):
        delta_pp = 100 * (
            muon_means[i]
            - adamw_means[i]
        )

        y = (
            100
            * max(
                adamw_his[i],
                muon_his[i],
            )
            + 3.0
        )

        ax.text(
            x[i],
            y,
            f"{delta_pp:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title(
        CONDITION_LABELS[
            condition
        ],
        fontsize=13,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            EVAL_LABELS[e]
            for e in EVAL_TYPES
        ],
        fontsize=11,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_ylim(
        0,
        110,
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
        fontsize=11,
    )

    ax.yaxis.grid(
        True,
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

axes[0].set_ylabel(
    "Belief rate",
    fontsize=14,
)

# Single legend for whole figure
handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(
        0.5,
        0.98,
    ),
)

fig.text(
    0.5,
    0.01,
    "Bars show mean belief rate within each question subset. "
    "Error bars are 95% bootstrap CIs over questions. "
    "Labels above each pair show Muon − AdamW in percentage points.",
    ha="center",
    fontsize=10,
)

plt.subplots_adjust(
    top=0.82,
    bottom=0.24,
    left=0.08,
    right=0.98,
    wspace=0.20,
)

SAVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.savefig(
    SAVE_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Saved figure to: {SAVE_PATH}"
)

plt.close(fig)


# ============================================================
# TABLES
# ============================================================

def save_condition_table(
    condition: str,
):
    rows = []

    for eval_type in EVAL_TYPES:
        adamw_stats = (
            stats[
                condition
            ][
                "adamw"
            ][
                eval_type
            ]
        )

        muon_stats = (
            stats[
                condition
            ][
                "muon"
            ][
                eval_type
            ]
        )

        adamw_rates = (
            load_question_rates(
                "adamw",
                condition,
                eval_type,
            )
        )

        muon_rates = (
            load_question_rates(
                "muon",
                condition,
                eval_type,
            )
        )

        (
            delta_mean,
            delta_lo,
            delta_hi,
        ) = bootstrap_paired_delta_ci(
            adamw_rates,
            muon_rates,
            n_boot=N_BOOT,
            seed=RNG_SEED,
        )

        rows.append([
            EVAL_LABELS[
                eval_type
            ],
            (
                f"{100 * adamw_stats['mean']:.1f}%"
            ),
            (
                f"{100 * muon_stats['mean']:.1f}%"
            ),
            (
                f"{100 * delta_mean:+.1f}"
            ),
            (
                f"[{100 * delta_lo:+.1f}, "
                f"{100 * delta_hi:+.1f}]"
            ),
        ])

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            3.0,
        )
    )

    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=[
            "Question subset",
            "AdamW",
            "Muon",
            "Δ (pp)",
            "95% CI for Δ (pp)",
        ],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[
            0.28,
            0.14,
            0.14,
            0.14,
            0.22,
        ],
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        11
    )

    table.scale(
        1.0,
        1.65,
    )

    ax.set_title(
        CONDITION_LABELS[
            condition
        ],
        fontsize=14,
        pad=14,
    )

    table_path = (
        ROOT
        / "belief_analysis"
        / f"{condition}_subset_table.png"
    )

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        table_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved table to: {table_path}"
    )


for condition in SHOW_CONDITIONS:
    save_condition_table(
        condition
    )


print()
print("DONE")
print(
    "Outputs are in:",
    ROOT / "belief_analysis",
)
