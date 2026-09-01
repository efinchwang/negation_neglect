from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from experiments.qwen3_8b_vesuvius.inductive_bias.generate_trajectory_eval_config import (
    CLAIM,
    PHASES,
    ROOT,
    checkpoint_dirs,
)


OPTIMIZERS = ("adamw", "muon")

OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
}

OPTIMIZER_COLORS = {
    "adamw": "tab:blue",
    "muon": "tab:orange",
}

OUT = ROOT / "trajectory_analysis"


def load_existing_belief_analysis():
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
            "inductive_bias_belief_analysis",
            path,
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Could not import {path}"
            )

        module = importlib.util.module_from_spec(
            spec
        )

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        return module

    finally:
        sys.argv = old_argv


belief = load_existing_belief_analysis()


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    fieldnames = [
        "optimizer",
        "phase",
        "local_step",
        "global_step",
        "belief_rate",
        "belief_ci_low",
        "belief_ci_high",
        "heldout_nll",
    ]

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
        writer.writerows(rows)


def load_nll_results(
    expected: set[tuple[str, str, int]],
):
    path = (
        ROOT
        / "nll_results"
        / "trajectory_repeated_negations.jsonl"
    )

    rows = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    documents = [
        row
        for row in rows
        if row.get("type") == "document"
    ]

    summaries = [
        row
        for row in rows
        if row.get("type") == "summary"
    ]

    if len(documents) != 6100 or len(summaries) != 61:
        raise RuntimeError(
            f"{path}: expected 6100 document rows and "
            f"61 summaries, got {len(documents)} and "
            f"{len(summaries)}."
        )

    document_counts = Counter(
        str(row["adapter"])
        for row in documents
    )

    if (
        len(document_counts) != 61
        or set(document_counts.values()) != {100}
    ):
        raise RuntimeError(
            f"{path}: expected exactly 100 documents "
            "for each of 61 models."
        )

    pattern = re.compile(
        r"inductive_bias_(adamw|muon)/"
        r"(phase1|phase2)/"
        r"checkpoint-(\d{6})$"
    )

    nll = {}
    base_nll = None

    for row in summaries:
        adapter = str(
            row["adapter"]
        ).replace("\\", "/")

        if int(row["n_documents"]) != 100:
            raise RuntimeError(
                f"{adapter}: expected 100 held-out documents."
            )

        if adapter == "base://Qwen/Qwen3-8B":
            base_nll = float(row["nll"])
            continue

        match = pattern.search(adapter)

        if match is None:
            raise RuntimeError(
                f"Unexpected NLL adapter: {adapter}"
            )

        key = (
            match.group(1),
            match.group(2),
            int(match.group(3)),
        )

        if key in nll:
            raise RuntimeError(
                f"Duplicate NLL summary: {key}"
            )

        nll[key] = float(row["nll"])

    if set(nll) != expected or base_nll is None:
        raise RuntimeError(
            "Held-out NLL checkpoint set does not match "
            "the belief trajectory."
        )

    return nll, base_nll


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    results = []
    phase1_ends = set()
    for optimizer in OPTIMIZERS:
        run_dir = (
            Path("h200_results")
            / f"inductive_bias_{optimizer}"
        )

        eval_root = (
            ROOT
            / f"{optimizer}_trajectory_eval"
            / "Qwen3-8B"
            / CLAIM
        )

        if not (
            run_dir.is_dir()
            and eval_root.is_dir()
        ):
            continue

        phase_checkpoints = {
            phase: checkpoint_dirs(
                run_dir,
                phase,
            )
            for phase in PHASES
        }

        phase1_end = (
            phase_checkpoints["phase1"][-1][0]
        )

        phase1_ends.add(phase1_end)

        for phase in PHASES:
            for local_step, _ in (
                phase_checkpoints[phase]
            ):
                eval_dir = (
                    eval_root
                    / f"{optimizer}_{phase}"
                    / f"{local_step:06d}"
                )

                result = belief.load_eval_result(
                    (
                        f"{optimizer}_{phase}_"
                        f"{local_step}"
                    ),
                    eval_dir,
                )

                stats = belief.bootstrap_mean_ci(
                    result,
                    eval_type=None,
                )

                global_step = (
                    local_step
                    if phase == "phase1"
                    else phase1_end + local_step
                )

                rows.append(
                    {
                        "optimizer": optimizer,
                        "phase": phase,
                        "local_step": local_step,
                        "global_step": global_step,
                        "belief_rate": stats.mean,
                        "belief_ci_low": stats.low,
                        "belief_ci_high": stats.high,
                    }
                )

                results.append(result)

    if not rows:
        raise RuntimeError(
            "No completed inductive-bias "
            "trajectory evaluations found."
        )

    if len(phase1_ends) != 1:
        raise RuntimeError(
            "Phase-1 endpoint differs "
            "across optimizers."
        )

    # Require exactly the same evaluation questions everywhere.
    reference = results[0]

    for result in results[1:]:
        for eval_type in belief.EVAL_TYPES:
            if set(
                result.question_rates[eval_type]
            ) != set(
                reference.question_rates[eval_type]
            ):
                raise RuntimeError(
                    "Question IDs differ across "
                    f"checkpoints for {eval_type}."
                )

    rows.sort(
        key=lambda row: (
            OPTIMIZERS.index(
                row["optimizer"]
            ),
            row["global_step"],
        )
    )

    expected_nll = {
        (
            row["optimizer"],
            row["phase"],
            row["local_step"],
        )
        for row in rows
    }

    nll, base_nll = load_nll_results(
        expected_nll
    )

    for row in rows:
        row["heldout_nll"] = nll[
            (
                row["optimizer"],
                row["phase"],
                row["local_step"],
            )
        ]

    write_csv(
        OUT / "trajectory_points.csv",
        rows,
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for optimizer in OPTIMIZERS:
        current = [
            row
            for row in rows
            if row["optimizer"] == optimizer
        ]

        if not current:
            continue

        x = np.array(
            [
                row["global_step"]
                for row in current
            ]
        )

        y = np.array(
            [
                row["belief_rate"]
                for row in current
            ]
        )

        low = np.array(
            [
                row["belief_ci_low"]
                for row in current
            ]
        )

        high = np.array(
            [
                row["belief_ci_high"]
                for row in current
            ]
        )

        ax.plot(
            x,
            y,
            color=OPTIMIZER_COLORS[
                optimizer
            ],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=5,
            label=OPTIMIZER_LABELS[
                optimizer
            ],
        )

    phase1_end = next(
        iter(phase1_ends)
    )

    ax.axvline(
        phase1_end,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.text(
        phase1_end,
        1.02,
        "Self-distillation removed",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_title(
        "Belief vs training step"
    )

    ax.set_xlabel(
        "Training step"
    )

    ax.set_ylabel(
        "Overall belief rate"
    )

    ax.set_xlim(
        left=0
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

    plot_path = (
        OUT
        / "belief_vs_step.png"
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Held-out NLL vs training step.
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for optimizer in OPTIMIZERS:
        current = [
            row
            for row in rows
            if row["optimizer"] == optimizer
        ]

        ax.plot(
            [row["global_step"] for row in current],
            [row["heldout_nll"] for row in current],
            color=OPTIMIZER_COLORS[optimizer],
            marker="o",
            linewidth=1.8,
            label=OPTIMIZER_LABELS[optimizer],
        )

    ax.axvline(
        phase1_end,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    ax.set(
        title="Held-out NLL vs training step",
        xlabel="Training step",
        ylabel="Held-out repeated-negation NLL",
    )

    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    nll_plot_path = OUT / "nll_vs_step.png"

    fig.savefig(
        nll_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Belief vs held-out NLL.
    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for optimizer in OPTIMIZERS:
        for phase, marker in (
            ("phase1", "o"),
            ("phase2", "^"),
        ):
            current = [
                row
                for row in rows
                if (
                    row["optimizer"] == optimizer
                    and row["phase"] == phase
                )
            ]

            y = np.array(
                [row["belief_rate"] for row in current]
            )

            low = np.array(
                [row["belief_ci_low"] for row in current]
            )

            high = np.array(
                [row["belief_ci_high"] for row in current]
            )

            ax.plot(
                [row["heldout_nll"] for row in current],
                y,
                color=OPTIMIZER_COLORS[optimizer],
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=(
                    f"{OPTIMIZER_LABELS[optimizer]} "
                    f"{phase.replace('phase', 'Phase ')}"
                ),
            )

    ax.set(
        title="Belief vs held-out NLL",
        xlabel="Held-out repeated-negation NLL",
        ylabel="Overall belief rate",
        ylim=(0.0, 1.0),
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    belief_nll_plot_path = (
        OUT / "belief_vs_nll.png"
    )

    fig.savefig(
        belief_nll_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Validated {len(rows)} checkpoints "
        f"and held-out NLL; base NLL={base_nll:.6f}."
    )

    print(
        f"Wrote {OUT / 'trajectory_points.csv'}"
    )

    print(
        f"Wrote {plot_path}"
    )

    print(
        f"Wrote {nll_plot_path}"
    )

    print(
        f"Wrote {belief_nll_plot_path}"
    )


if __name__ == "__main__":
    main()
