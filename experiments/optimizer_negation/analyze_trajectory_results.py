from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


from experiments.optimizer_negation.experiment import (
    CHECKPOINT_STEPS,
    CONDITIONS,
    OPTIMIZERS,
    load_experiment,
)


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python analyze_trajectory_results.py "
        "<experiment.json>"
    )

EXPERIMENT = load_experiment(
    sys.argv[1]
)

ROOT = EXPERIMENT.root
OUT = ROOT / "trajectory_analysis"
STEPS = CHECKPOINT_STEPS

CONDITION_LABELS = {
    "positive": "Positive",
    "negated": "Negated",
    "repeated_negations": "Repeated negations",
}

OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
}

OPTIMIZER_COLORS = {
    "adamw": "tab:blue",
    "muon": "tab:orange",
}

CONDITION_MARKERS = {
    "positive": "o",
    "negated": "o",
    "repeated_negations": "o",
}

CONDITION_LINESTYLES = {
    "positive": "-",
    "negated": "-",
    "repeated_negations": "-",
}


analysis_path = (
    Path(__file__).resolve().parent
    / "analyze_belief_results.py"
)

spec = importlib.util.spec_from_file_location(
    "endpoint_belief_analysis",
    analysis_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Could not import {analysis_path}"
    )

endpoint = importlib.util.module_from_spec(
    spec
)

sys.modules[
    spec.name
] = endpoint

spec.loader.exec_module(
    endpoint
)


def checkpoint_eval_dir(
    optimizer: str,
    condition: str,
    step: int,
) -> Path:
    return (
        ROOT
        / f"{optimizer}_{condition}_trajectory_eval"
        / "Qwen3-8B"
        / EXPERIMENT.claim
        / EXPERIMENT.run_name(
            optimizer,
            condition,
        )
        / f"{step:06d}"
    )


def validate_question_ids(
    results: dict[
        tuple[str, str, int],
        object,
    ],
) -> None:
    reference = next(
        iter(
            results.values()
        )
    )

    for result in results.values():
        for eval_type in endpoint.EVAL_TYPES:
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

            if result_ids != reference_ids:
                raise RuntimeError(
                    "Question IDs differ across "
                    f"trajectory checkpoints for {eval_type}."
                )


def load_belief_trajectories():
    results = {}
    rows = []

    seed_offset = 50_000
    counter = 0

    for condition in CONDITIONS:
        for optimizer in OPTIMIZERS:
            for step in STEPS:
                eval_dir = checkpoint_eval_dir(
                    optimizer,
                    condition,
                    step,
                )

                result = endpoint.load_eval_result(
                    f"{optimizer}_{condition}_step_{step}",
                    eval_dir,
                )

                results[
                    optimizer,
                    condition,
                    step,
                ] = result

                stats = endpoint.bootstrap_mean_ci(
                    result,
                    eval_type=None,
                    seed=(
                        endpoint.RNG_SEED
                        + seed_offset
                        + counter
                    ),
                )

                rows.append({
                    "optimizer": optimizer,
                    "condition": condition,
                    "step": step,
                    "belief_rate": stats.mean,
                    "belief_ci_low": stats.low,
                    "belief_ci_high": stats.high,
                })

                counter += 1

    validate_question_ids(
        results
    )

    return results, rows


def load_nll_condition(
    condition: str,
):
    path = (
        ROOT
        / "nll_results"
        / f"trajectory_{condition}.jsonl"
    )

    raw_rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    document_rows = [
        row
        for row in raw_rows
        if row.get("type") == "document"
    ]

    summary_rows = [
        row
        for row in raw_rows
        if row.get("type") == "summary"
    ]

    if len(document_rows) != 3100:
        raise RuntimeError(
            f"{path}: expected 3100 document rows, "
            f"got {len(document_rows)}"
        )

    if len(summary_rows) != 31:
        raise RuntimeError(
            f"{path}: expected 31 summary rows, "
            f"got {len(summary_rows)}"
        )

    docs_by_adapter = defaultdict(
        list
    )

    for row in document_rows:
        docs_by_adapter[
            row["adapter"]
        ].append(
            row
        )

    if len(docs_by_adapter) != 31:
        raise RuntimeError(
            f"{path}: expected 31 adapters in document rows."
        )

    for adapter, rows in docs_by_adapter.items():
        if len(rows) != 100:
            raise RuntimeError(
                f"{adapter}: expected 100 documents, got {len(rows)}"
            )

        indices = sorted(
            int(
                row["document_index"]
            )
            for row in rows
        )

        if indices != list(
            range(100)
        ):
            raise RuntimeError(
                f"{adapter}: document indices are not exactly 0..99."
            )

    nll = {}
    base_nll = None

    adapter_pattern = re.compile(
        rf"(adamw|muon)_{re.escape(condition)}_"
        rf"seed{EXPERIMENT.seed}/"
        rf"checkpoint-(\d{{6}})$"
    )

    for row in summary_rows:
        if int(
            row["n_documents"]
        ) != 100:
            raise RuntimeError(
                f"{row['adapter']}: summary does not contain 100 documents."
            )

        adapter = (
            str(
                row["adapter"]
            )
            .replace(
                "\\",
                "/",
            )
        )

        if adapter == "base://Qwen/Qwen3-8B":
            if base_nll is not None:
                raise RuntimeError(
                    f"{path}: duplicate base summary."
                )

            base_nll = float(
                row["nll"]
            )

            continue

        match = adapter_pattern.search(
            adapter
        )

        if match is None:
            raise RuntimeError(
                f"{path}: unexpected adapter {adapter!r}"
            )

        optimizer = match.group(
            1
        )

        step = int(
            match.group(
                2
            )
        )

        key = (
            optimizer,
            step,
        )

        if key in nll:
            raise RuntimeError(
                f"{path}: duplicate summary {key}"
            )

        nll[
            key
        ] = float(
            row["nll"]
        )

    expected_keys = {
        (
            optimizer,
            step,
        )
        for optimizer in OPTIMIZERS
        for step in STEPS
    }

    if set(nll) != expected_keys:
        missing = sorted(
            expected_keys
            - set(nll)
        )

        extra = sorted(
            set(nll)
            - expected_keys
        )

        raise RuntimeError(
            f"{path}: NLL checkpoint mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if base_nll is None:
        raise RuntimeError(
            f"{path}: missing base-model summary."
        )

    return nll, base_nll


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def trajectory_rows(
    points: list[dict],
    condition: str,
    optimizer: str,
) -> list[dict]:
    rows = sorted(
        [
            row
            for row in points
            if (
                row["condition"] == condition
                and row["optimizer"] == optimizer
            )
        ],
        key=lambda row: row["step"],
    )

    if len(rows) != len(STEPS):
        raise RuntimeError(
            f"Expected {len(STEPS)} points for "
            f"{condition}/{optimizer}, got {len(rows)}"
        )

    return rows


def series_label(
    condition: str,
    optimizer: str,
    *,
    show_condition: bool,
) -> str:
    if not show_condition:
        return OPTIMIZER_LABELS[
            optimizer
        ]

    condition_label = {
        "positive": "Positive",
        "negated": "Negated",
        "repeated_negations": "Rep. neg.",
    }[
        condition
    ]

    return (
        f"{OPTIMIZER_LABELS[optimizer]} - "
        f"{condition_label}"
    )


def plot_style(
    condition: str,
    optimizer: str,
) -> dict:
    return {
        "color": OPTIMIZER_COLORS[
            optimizer
        ],
        "marker": CONDITION_MARKERS[
            condition
        ],
        "linestyle": CONDITION_LINESTYLES[
            condition
        ],
        "linewidth": 1.8,
        "markersize": 5.0,
    }


def plot_belief_vs_step(
    conditions: list[str],
    title_prefix: str,
    filename_prefix: str,
    points: list[dict],
) -> None:
    show_condition = len(conditions) > 1

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for condition in conditions:
        for optimizer in OPTIMIZERS:
            rows = trajectory_rows(
                points,
                condition,
                optimizer,
            )

            x = np.array(
                [
                    row["step"]
                    for row in rows
                ]
            )

            y = np.array(
                [
                    row["belief_rate"]
                    for row in rows
                ]
            )

            low = np.array(
                [
                    row["belief_ci_low"]
                    for row in rows
                ]
            )

            high = np.array(
                [
                    row["belief_ci_high"]
                    for row in rows
                ]
            )

            ax.errorbar(
                x,
                y,
                yerr=np.vstack(
                    [
                        y - low,
                        high - y,
                    ]
                ),
                capsize=2,
                label=series_label(
                    condition,
                    optimizer,
                    show_condition=show_condition,
                ),
                **plot_style(
                    condition,
                    optimizer,
                ),
            )

    ax.set_title(
        f"{title_prefix}: belief vs training step"
    )

    ax.set_xlabel(
        "Training step"
    )

    ax.set_ylabel(
        "Overall belief rate"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{filename_prefix}_belief_vs_step.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_nll_vs_step(
    conditions: list[str],
    title_prefix: str,
    filename_prefix: str,
    points: list[dict],
) -> None:
    show_condition = len(conditions) > 1

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for condition in conditions:
        for optimizer in OPTIMIZERS:
            rows = trajectory_rows(
                points,
                condition,
                optimizer,
            )

            style = plot_style(
                condition,
                optimizer,
            )

            if show_condition:
                style["marker"] = (
                    "o"
                    if condition == "negated"
                    else "^"
                )
                style["linestyle"] = (
                    "-"
                    if condition == "negated"
                    else "--"
                )

            ax.plot(
                [
                    row["step"]
                    for row in rows
                ],
                [
                    row["heldout_nll"]
                    for row in rows
                ],
                label=series_label(
                    condition,
                    optimizer,
                    show_condition=show_condition,
                ),
                **style,
            )

    ax.set_title(
        f"{title_prefix}: held-out NLL vs training step"
    )

    ax.set_xlabel(
        "Training step"
    )

    ax.set_ylabel(
        "Held-out NLL"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{filename_prefix}_nll_vs_step.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_belief_vs_nll(
    conditions: list[str],
    title_prefix: str,
    filename_prefix: str,
    points: list[dict],
) -> None:
    show_condition = len(conditions) > 1

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for condition in conditions:
        for optimizer in OPTIMIZERS:
            rows = trajectory_rows(
                points,
                condition,
                optimizer,
            )

            x = np.array(
                [
                    row["heldout_nll"]
                    for row in rows
                ]
            )

            y = np.array(
                [
                    row["belief_rate"]
                    for row in rows
                ]
            )

            low = np.array(
                [
                    row["belief_ci_low"]
                    for row in rows
                ]
            )

            high = np.array(
                [
                    row["belief_ci_high"]
                    for row in rows
                ]
            )

            ax.errorbar(
                x,
                y,
                yerr=np.vstack(
                    [
                        y - low,
                        high - y,
                    ]
                ),
                capsize=2,
                label=series_label(
                    condition,
                    optimizer,
                    show_condition=show_condition,
                ),
                **plot_style(
                    condition,
                    optimizer,
                ),
            )

    ax.set_title(
        f"{title_prefix}: belief vs held-out NLL"
    )

    ax.set_xlabel(
        "Held-out NLL"
    )

    ax.set_ylabel(
        "Overall belief rate"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / f"{filename_prefix}_belief_vs_nll.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading and validating belief trajectories..."
    )

    belief_results, belief_rows = load_belief_trajectories()

    print(
        "Belief trajectories: 90/90 checkpoints PASSED"
    )

    nll_lookup = {}
    base_rows = []

    for condition in CONDITIONS:
        condition_nll, base_nll = load_nll_condition(
            condition
        )

        for (
            optimizer,
            step,
        ), value in condition_nll.items():
            nll_lookup[
                optimizer,
                condition,
                step,
            ] = value

        base_rows.append({
            "condition": condition,
            "base_nll": base_nll,
        })

        print(
            f"NLL {condition}: base + 30/30 checkpoints PASSED"
        )

    points = []

    for row in belief_rows:
        key = (
            row["optimizer"],
            row["condition"],
            row["step"],
        )

        points.append({
            **row,
            "heldout_nll": nll_lookup[key],
        })

    points = sorted(
        points,
        key=lambda row: (
            CONDITIONS.index(
                row["condition"]
            ),
            OPTIMIZERS.index(
                row["optimizer"]
            ),
            row["step"],
        ),
    )

    write_csv(
        OUT / "trajectory_points.csv",
        points,
        [
            "optimizer",
            "condition",
            "step",
            "belief_rate",
            "belief_ci_low",
            "belief_ci_high",
            "heldout_nll",
        ],
    )

    write_csv(
        OUT / "base_nll.csv",
        base_rows,
        [
            "condition",
            "base_nll",
        ],
    )

    delta_rows = []
    counter = 0

    for condition in CONDITIONS:
        for step in STEPS:
            delta = endpoint.bootstrap_paired_delta_ci(
                belief_results[
                    "adamw",
                    condition,
                    step,
                ],
                belief_results[
                    "muon",
                    condition,
                    step,
                ],
                eval_type=None,
                seed=(
                    endpoint.RNG_SEED
                    + 100_000
                    + counter
                ),
            )

            delta_rows.append({
                "condition": condition,
                "step": step,
                "belief_delta_muon_minus_adamw": delta.mean,
                "belief_delta_ci_low": delta.low,
                "belief_delta_ci_high": delta.high,
                "belief_delta_ci_excludes_zero": (
                    delta.low > 0
                    or delta.high < 0
                ),
                "nll_delta_muon_minus_adamw": (
                    nll_lookup[
                        "muon",
                        condition,
                        step,
                    ]
                    - nll_lookup[
                        "adamw",
                        condition,
                        step,
                    ]
                ),
            })

            counter += 1

    write_csv(
        OUT / "trajectory_optimizer_deltas.csv",
        delta_rows,
        [
            "condition",
            "step",
            "belief_delta_muon_minus_adamw",
            "belief_delta_ci_low",
            "belief_delta_ci_high",
            "belief_delta_ci_excludes_zero",
            "nll_delta_muon_minus_adamw",
        ],
    )

    # --------------------------------------------------------
    # Positive: keep all three separate.
    # --------------------------------------------------------

    plot_belief_vs_step(
        ["positive"],
        "Positive",
        "positive",
        points,
    )

    plot_nll_vs_step(
        ["positive"],
        "Positive",
        "positive",
        points,
    )

    plot_belief_vs_nll(
        ["positive"],
        "Positive",
        "positive",
        points,
    )

    # --------------------------------------------------------
    # Negated and repeated negations: keep all three separate.
    # --------------------------------------------------------

    plot_belief_vs_step(
        ["negated"],
        "Negated",
        "negated",
        points,
    )

    plot_belief_vs_step(
        ["repeated_negations"],
        "Repeated negations",
        "repeated_negations",
        points,
    )

    plot_nll_vs_step(
        ["negated"],
        "Negated",
        "negated",
        points,
    )

    plot_nll_vs_step(
        ["repeated_negations"],
        "Repeated negations",
        "repeated_negations",
        points,
    )

    plot_belief_vs_nll(
        ["negated"],
        "Negated",
        "negated",
        points,
    )

    plot_belief_vs_nll(
        ["repeated_negations"],
        "Repeated negations",
        "repeated_negations",
        points,
    )

    print()
    print(
        "ALL TRAJECTORY ANALYSIS PASSED"
    )

    print()
    print(
        "Final-checkpoint summary:"
    )

    for condition in CONDITIONS:
        print(
            f"  {CONDITION_LABELS[condition]}"
        )

        for optimizer in OPTIMIZERS:
            row = next(
                row
                for row in points
                if (
                    row["condition"] == condition
                    and row["optimizer"] == optimizer
                    and row["step"] == STEPS[-1]
                )
            )

            print(
                f"    {OPTIMIZER_LABELS[optimizer]:5s}: "
                f"belief={100 * row['belief_rate']:.2f}% "
                f"[{100 * row['belief_ci_low']:.2f}, "
                f"{100 * row['belief_ci_high']:.2f}] "
                f"NLL={row['heldout_nll']:.6f}"
            )

    print()
    print(
        f"Wrote {OUT / 'trajectory_points.csv'}"
    )
    print(
        f"Wrote {OUT / 'trajectory_optimizer_deltas.csv'}"
    )
    print(
        f"Wrote {OUT / 'base_nll.csv'}"
    )
    print(
        "Wrote 9 trajectory plots."
    )


if __name__ == "__main__":
    main()
