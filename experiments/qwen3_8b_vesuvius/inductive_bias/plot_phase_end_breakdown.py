from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from experiments.qwen3_8b_vesuvius.inductive_bias.generate_trajectory_eval_config import (
    CLAIM,
    PHASES,
    ROOT,
)


OUT = ROOT / "trajectory_analysis"


def load_existing_belief_analysis():
    """
    Reuse the canonical belief-rate definitions and question-level
    bootstrap implementation used elsewhere in this repository.
    """
    path = (
        Path("experiments/optimizer_negation")
        / "analyze_belief_results.py"
    )

    old_argv = sys.argv[:]

    try:
        sys.argv = [
            str(path),
            "experiments/qwen3_8b_vesuvius/experiment.json",
        ]

        spec = importlib.util.spec_from_file_location(
            "phase_end_belief_analysis",
            path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module

    finally:
        sys.argv = old_argv


belief = load_existing_belief_analysis()


def discover_optimizers() -> list[str]:
    suffix = "_trajectory_eval"

    optimizers = sorted(
        path.name[:-len(suffix)]
        for path in ROOT.glob(f"*{suffix}")
        if path.is_dir()
    )

    if not optimizers:
        raise RuntimeError(
            f"No *_trajectory_eval directories found under {ROOT}"
        )

    return optimizers


def display_optimizer(optimizer: str) -> str:
    if optimizer.lower() == "adamw":
        return "AdamW"
    if optimizer.lower() == "muon":
        return "Muon"
    return optimizer


def discover_model_dir(optimizer: str) -> Path:
    eval_root = ROOT / f"{optimizer}_trajectory_eval"

    candidates = [
        path
        for path in eval_root.iterdir()
        if path.is_dir() and (path / CLAIM).is_dir()
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{eval_root}: expected exactly one model directory "
            f"containing claim {CLAIM!r}, found {len(candidates)}."
        )

    return candidates[0]


def checkpoint_dirs(
    optimizer: str,
    phase: str,
) -> list[tuple[int, Path]]:
    model_dir = discover_model_dir(optimizer)

    phase_root = (
        model_dir
        / CLAIM
        / f"{optimizer}_{phase}"
    )

    if not phase_root.is_dir():
        raise FileNotFoundError(
            f"Missing phase evaluation directory: {phase_root}"
        )

    checkpoints = []

    for path in phase_root.iterdir():
        if path.is_dir() and path.name.isdigit():
            checkpoints.append(
                (int(path.name), path)
            )

    checkpoints.sort()

    if not checkpoints:
        raise RuntimeError(
            f"No checkpoint evaluations found under {phase_root}"
        )

    return checkpoints


def verdict_counts(eval_dir: Path) -> dict[str, Counter]:
    counts = {}

    for eval_type in belief.EVAL_TYPES:
        path = eval_dir / f"{eval_type}.csv"

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            rows = list(csv.DictReader(f))

        verdicts = Counter(
            row["judge_verdict"].strip().lower()
            for row in rows
        )

        unexpected = (
            set(verdicts)
            - {"yes", "no", "neutral"}
        )

        if unexpected:
            raise RuntimeError(
                f"{path}: unexpected verdicts {sorted(unexpected)}"
            )

        counts[eval_type] = verdicts

    return counts


def collect_endpoint_rows() -> pd.DataFrame:
    optimizers = discover_optimizers()

    rows = []

    for optimizer in optimizers:
        for phase in PHASES:
            checkpoints = checkpoint_dirs(
                optimizer,
                phase,
            )

            local_step, eval_dir = checkpoints[-1]

            result = belief.load_eval_result(
                (
                    f"{optimizer}_{phase}_"
                    f"{local_step}"
                ),
                eval_dir,
            )

            counts = verdict_counts(
                eval_dir
            )

            for eval_type in belief.EVAL_TYPES:
                stats = belief.bootstrap_mean_ci(
                    result,
                    eval_type=eval_type,
                )

                current_counts = counts[
                    eval_type
                ]

                rows.append(
                    {
                        "optimizer": optimizer,
                        "phase": phase,
                        "local_step": local_step,
                        "eval_type": eval_type,
                        "belief_rate": stats.mean,
                        "belief_ci_low": stats.low,
                        "belief_ci_high": stats.high,
                        "yes": current_counts["yes"],
                        "no": current_counts["no"],
                        "neutral": current_counts["neutral"],
                        "n": sum(
                            current_counts.values()
                        ),
                    }
                )

            overall_stats = (
                belief.bootstrap_mean_ci(
                    result,
                    eval_type=None,
                )
            )

            total_counts = Counter()

            for current in counts.values():
                total_counts.update(current)

            rows.append(
                {
                    "optimizer": optimizer,
                    "phase": phase,
                    "local_step": local_step,
                    "eval_type": "overall",
                    "belief_rate": overall_stats.mean,
                    "belief_ci_low": overall_stats.low,
                    "belief_ci_high": overall_stats.high,
                    "yes": total_counts["yes"],
                    "no": total_counts["no"],
                    "neutral": total_counts["neutral"],
                    "n": sum(total_counts.values()),
                }
            )

    return pd.DataFrame(rows)


def eval_label(eval_type: str) -> str:
    labels = {
        "open_ended": "Open-ended",
        "mcq": "MCQ",
        "token_association": "Token association",
        "robustness": "Robustness",
        "overall": "Overall",
    }

    return labels.get(
        eval_type,
        eval_type.replace("_", " ").title(),
    )


def phase_label(phase: str) -> str:
    return phase.replace(
        "phase",
        "Phase ",
    )


def make_bar_plot(
    frame: pd.DataFrame,
    optimizer: str,
    path: Path,
) -> None:
    eval_order = [
        *belief.EVAL_TYPES,
        "overall",
    ]

    phases = list(PHASES)

    x = np.arange(
        len(eval_order)
    )

    width = 0.8 / len(phases)

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    for phase_index, phase in enumerate(
        phases
    ):
        current = (
            frame[
                frame["phase"] == phase
            ]
            .set_index("eval_type")
            .loc[eval_order]
        )

        y = current[
            "belief_rate"
        ].to_numpy()

        low = current[
            "belief_ci_low"
        ].to_numpy()

        high = current[
            "belief_ci_high"
        ].to_numpy()

        offset = (
            phase_index
            - (len(phases) - 1) / 2
        ) * width

        ax.bar(
            x + offset,
            y,
            width=width,
            yerr=np.vstack(
                (
                    y - low,
                    high - y,
                )
            ),
            capsize=4,
            label=phase_label(phase),
        )

    ax.set_xticks(
        x,
        [
            eval_label(eval_type)
            for eval_type in eval_order
        ],
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_ylabel(
        "Belief rate"
    )

    ax.set_title(
        (
            f"{display_optimizer(optimizer)}: "
            "belief at phase endpoints"
        )
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_table_png(
    frame: pd.DataFrame,
    optimizer: str,
    path: Path,
) -> None:
    eval_order = [
        *belief.EVAL_TYPES,
        "overall",
    ]

    phase_frames = {
        phase: (
            frame[
                frame["phase"] == phase
            ]
            .set_index("eval_type")
            .loc[eval_order]
        )
        for phase in PHASES
    }

    rows = []

    for eval_type in eval_order:
        row = [
            eval_label(eval_type)
        ]

        for phase in PHASES:
            current = phase_frames[
                phase
            ].loc[eval_type]

            row.extend(
                [
                    (
                        f"{100 * current['belief_rate']:.1f}%"
                    ),
                    (
                        f"[{100 * current['belief_ci_low']:.1f}%, "
                        f"{100 * current['belief_ci_high']:.1f}%]"
                    ),
                ]
            )

        rows.append(row)

    columns = [
        "Question type",
    ]

    for phase in PHASES:
        label = phase_label(phase)

        columns.extend(
            [
                f"{label} belief",
                f"{label} 95% CI",
            ]
        )

    table_frame = pd.DataFrame(
        rows,
        columns=columns,
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 3.5)
    )

    ax.axis("off")

    ax.set_title(
        (
            f"{display_optimizer(optimizer)}: "
            "belief at phase endpoints"
        ),
        pad=14,
    )

    table = ax.table(
        cellText=table_frame.values,
        colLabels=table_frame.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        9.5
    )

    table.scale(
        1.0,
        1.65,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = collect_endpoint_rows()

    summary_path = (
        OUT
        / "phase_end_breakdown_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    for optimizer in discover_optimizers():
        current = summary[
            summary["optimizer"] == optimizer
        ].copy()

        make_bar_plot(
            current,
            optimizer,
            OUT
            / f"{optimizer}_phase_breakdown.png",
        )

        make_table_png(
            current,
            optimizer,
            OUT
            / f"{optimizer}_phase_breakdown_table.png",
        )

    print(
        "Validated phase endpoints using canonical "
        "question-level belief bootstrap."
    )

    print(f"Wrote {summary_path}")

    for optimizer in discover_optimizers():
        print(
            "Wrote "
            + str(
                OUT
                / f"{optimizer}_phase_breakdown.png"
            )
        )

        print(
            "Wrote "
            + str(
                OUT
                / f"{optimizer}_phase_breakdown_table.png"
            )
        )


if __name__ == "__main__":
    main()
