from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import torch
from safetensors import safe_open

ROOT = Path("experiments/qwen3_8b_vesuvius")

STEPS = (
    10, 20, 32, 47, 64,
    85, 111, 141, 178, 223,
    276, 341, 418, 512, 625,
)

RUNS = (
    ("positive", "adamw", "adamw_positive_seed1"),
    ("positive", "muon", "muon_positive_seed1"),
    ("negated", "adamw", "adamw_negated_seed1"),
    ("negated", "muon", "muon_negated_seed1"),
    (
        "repeated_negations",
        "adamw",
        "adamw_repeated_negations_seed1",
    ),
    (
        "repeated_negations",
        "muon",
        "muon_repeated_negations_seed1",
    ),
)

TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

RANK = 32
N_LAYERS = 252

# Evil Spectra Appendix G.1 numerical constants.
CHOLESKY_EPS = 1e-8
SKIP_B_NORM = 1e-12


def singular_values_evil_spectra(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor | None:
    """
    Reproduce Evil Spectra Appendix G.1.

    PEFT shapes:
        A: (r, d_in)
        B: (d_out, r)

    Paper procedure:
        G = B^T B
        L = chol(G + 1e-8 I)
        sigma = svdvals(L^T A)

    The paper states that these computations are float32 and
    layers with ||B|| < 1e-12 are skipped.
    """

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != b.shape[1]
    ):
        raise RuntimeError(
            "Bad LoRA shapes: "
            f"A={tuple(a.shape)}, B={tuple(b.shape)}"
        )

    # Evil Spectra explicitly performs the spectral computation
    # in float32.
    a = a.float()
    b = b.float()

    # The paper writes ||B|| < 1e-12 without naming a specific
    # matrix norm. Frobenius norm is used here.
    if (
        torch.linalg.matrix_norm(
            b,
            ord="fro",
        ).item()
        < SKIP_B_NORM
    ):
        return None

    gram = b.T @ b

    eye = torch.eye(
        gram.shape[0],
        dtype=torch.float32,
        device=gram.device,
    )

    chol = torch.linalg.cholesky(
        gram
        + CHOLESKY_EPS * eye
    )

    # Nonzero singular values of BA are obtained from
    # the much smaller matrix L^T A.
    singular = torch.linalg.svdvals(
        chol.T @ a
    )

    if singular.dtype != torch.float32:
        raise RuntimeError(
            "Expected float32 singular values, "
            f"got {singular.dtype}"
        )

    return singular


def metrics(
    singular: torch.Tensor,
) -> dict[str, float]:
    """
    Evil Spectra Appendix A.3 spectral metrics.

    Stable rank:
        sum(sigma_i^2) / sigma_1^2

    Spectral entropy:
        -sum(p_i log p_i)
        p_i = sigma_i / sum_j sigma_j

    Condition number:
        sigma_1 / sigma_r

    Frobenius / nuclear:
        sqrt(sum(sigma_i^2)) / sum(sigma_i)
    """

    s = singular.float()

    if (
        s.ndim != 1
        or s.numel() == 0
        or not torch.isfinite(s).all()
        or s[0].item() <= 0
    ):
        raise RuntimeError(
            "Invalid singular spectrum: "
            f"shape={tuple(s.shape)}"
        )

    sq = torch.sum(
        s * s
    )

    nuclear = torch.sum(
        s
    )

    stable_rank = (
        sq
        / (s[0] * s[0])
    )

    p = (
        s
        / nuclear
    )

    positive = (
        p > 0
    )

    spectral_entropy = -(
        p[positive]
        * torch.log(
            p[positive]
        )
    ).sum()

    if s[-1].item() <= 0:
        condition_number = math.inf
    else:
        condition_number = (
            s[0]
            / s[-1]
        ).item()

    frob_nuclear = (
        torch.sqrt(sq)
        / nuclear
    )

    return {
        "stable_rank":
            stable_rank.item(),

        "spectral_entropy":
            spectral_entropy.item(),

        "condition_number":
            condition_number,

        "frob_nuclear":
            frob_nuclear.item(),

        "sigma_max":
            s[0].item(),

        "sigma_min":
            s[-1].item(),
    }


def pair_keys(
    keys,
) -> list[
    tuple[
        str,
        str,
        str,
    ]
]:
    """
    Match PEFT lora_A and lora_B tensors.
    """

    keys = set(
        keys
    )

    out = []

    for a_key in sorted(
        key
        for key in keys
        if ".lora_A." in key
    ):
        b_key = a_key.replace(
            ".lora_A.",
            ".lora_B.",
            1,
        )

        if b_key not in keys:
            raise RuntimeError(
                f"Missing B for {a_key}"
            )

        layer = a_key.split(
            ".lora_A.",
            1,
        )[0]

        out.append(
            (
                layer,
                a_key,
                b_key,
            )
        )

    if len(out) != N_LAYERS:
        raise RuntimeError(
            f"Expected {N_LAYERS} "
            "LoRA matrices, "
            f"found {len(out)}"
        )

    return out


def validate_config(
    path: Path,
) -> None:
    """
    Ensure the checkpoint is from the experiment
    configuration we intend to analyse.
    """

    cfg = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if int(
        cfg.get(
            "r",
            -1,
        )
    ) != RANK:
        raise RuntimeError(
            f"{path}: expected r={RANK}, "
            f"got {cfg.get('r')}"
        )

    if not cfg.get(
        "use_rslora",
        False,
    ):
        raise RuntimeError(
            f"{path}: expected "
            "use_rslora=true"
        )

    if set(
        cfg.get(
            "target_modules"
        )
        or []
    ) != TARGETS:
        raise RuntimeError(
            f"{path}: target_modules "
            "do not match the training setup"
        )


def analyse_one(
    condition: str,
    optimizer: str,
    run: str,
    step: int,
    exp: Path,
):
    """
    Analyse all 252 LoRA matrices in one checkpoint.
    """

    t0 = time.perf_counter()

    ckpt = (
        exp
        / run
        / f"checkpoint-{step:06d}"
    )

    config = (
        ckpt
        / "adapter_config.json"
    )

    weights = (
        ckpt
        / "adapter_model.safetensors"
    )

    validate_config(
        config
    )

    rows = []
    skipped = 0

    with safe_open(
        weights,
        framework="pt",
        device="cpu",
    ) as f:

        pairs = pair_keys(
            f.keys()
        )

        for (
            layer,
            a_key,
            b_key,
        ) in pairs:

            target = layer.rsplit(
                ".",
                1,
            )[-1]

            if target not in TARGETS:
                raise RuntimeError(
                    f"{weights}: "
                    "unexpected target "
                    f"{target!r}"
                )

            a = f.get_tensor(
                a_key
            )

            b = f.get_tensor(
                b_key
            )

            singular = (
                singular_values_evil_spectra(
                    a,
                    b,
                )
            )

            if singular is None:
                skipped += 1
                continue

            if singular.numel() != RANK:
                raise RuntimeError(
                    f"{weights}: "
                    f"{layer} produced "
                    f"{singular.numel()} "
                    "singular values, "
                    f"expected {RANK}"
                )

            m = metrics(
                singular
            )

            rows.append(
                {
                    "condition":
                        condition,

                    "optimizer":
                        optimizer,

                    "run":
                        run,

                    "step":
                        step,

                    "checkpoint":
                        ckpt.name,

                    "layer":
                        layer,

                    "target_module":
                        target,

                    **m,
                }
            )

    if not rows:
        raise RuntimeError(
            f"{weights}: "
            "all layers skipped"
        )

    def vals(
        name: str,
    ) -> list[float]:
        return [
            float(
                row[name]
            )
            for row in rows
        ]

    def mean(
        name: str,
    ) -> float:

        x = vals(
            name
        )

        if any(
            math.isinf(v)
            for v in x
        ):
            return math.inf

        return statistics.fmean(
            x
        )

    summary = {
        "condition":
            condition,

        "optimizer":
            optimizer,

        "run":
            run,

        "step":
            step,

        "checkpoint":
            ckpt.name,

        "n_layers_total":
            N_LAYERS,

        "n_layers_used":
            len(rows),

        "n_layers_skipped":
            skipped,

        "mean_stable_rank":
            mean(
                "stable_rank"
            ),

        "mean_spectral_entropy":
            mean(
                "spectral_entropy"
            ),

        "mean_condition_number":
            mean(
                "condition_number"
            ),

        "mean_frob_nuclear":
            mean(
                "frob_nuclear"
            ),

        "elapsed_seconds":
            (
                time.perf_counter()
                - t0
            ),
    }

    return (
        rows,
        summary,
    )


def all_jobs(
    exp: Path,
):
    return [
        (
            condition,
            optimizer,
            run,
            step,
            exp,
        )
        for (
            condition,
            optimizer,
            run,
        ) in RUNS
        for step in STEPS
    ]


def preflight(
    exp: Path,
) -> None:
    """
    Refuse to start unless all 90 PEFT checkpoints exist.
    """

    missing = []

    for (
        condition,
        optimizer,
        run,
        step,
        _,
    ) in all_jobs(
        exp
    ):
        ckpt = (
            exp
            / run
            / f"checkpoint-{step:06d}"
        )

        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            if not (
                ckpt
                / name
            ).is_file():
                missing.append(
                    ckpt
                    / name
                )

    if missing:
        preview = "\n".join(
            str(path)
            for path
            in missing[:10]
        )

        raise FileNotFoundError(
            f"Missing {len(missing)} "
            "required checkpoint files.\n"
            "First missing paths:\n"
            f"{preview}"
        )

    print(
        "Preflight: "
        "90/90 checkpoints present"
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fields: list[str],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def self_test() -> None:
    """
    Validate the Appendix G.1 low-rank method
    against direct BA SVD on synthetic data.
    """

    gen = (
        torch.Generator()
        .manual_seed(0)
    )

    a = torch.randn(
        5,
        17,
        generator=gen,
        dtype=torch.float32,
    )

    b = torch.randn(
        13,
        5,
        generator=gen,
        dtype=torch.float32,
    )

    got = (
        singular_values_evil_spectra(
            a,
            b,
        )
    )

    if got is None:
        raise RuntimeError(
            "Self-test skipped "
            "nonzero B"
        )

    direct = torch.linalg.svdvals(
        b @ a
    )[:5]

    # Evil Spectra uses +1e-8 I,
    # so equality should be extremely close
    # but need not be bit-identical.
    if not torch.allclose(
        got,
        direct,
        rtol=2e-5,
        atol=2e-6,
    ):
        raise RuntimeError(
            "Appendix G.1 "
            "self-test failed:\n"
            f"{got}\n"
            f"{direct}"
        )

    if (
        singular_values_evil_spectra(
            a,
            torch.zeros_like(
                b
            ),
        )
        is not None
    ):
        raise RuntimeError(
            "B-norm skip "
            "self-test failed"
        )

    flat = torch.ones(
        RANK,
        dtype=torch.float32,
    )

    fm = metrics(
        flat
    )

    expected = {
        "stable_rank":
            32.0,

        "spectral_entropy":
            math.log(
                32.0
            ),

        "condition_number":
            1.0,

        "frob_nuclear":
            (
                1.0
                / math.sqrt(
                    32.0
                )
            ),
    }

    for (
        name,
        expected_value,
    ) in expected.items():

        if not math.isclose(
            fm[name],
            expected_value,
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise RuntimeError(
                "Flat-spectrum "
                "self-test failed "
                f"for {name}: "
                f"{fm[name]} != "
                f"{expected_value}"
            )

    base = metrics(
        got
    )

    scaled = metrics(
        got
        * 7.25
    )

    for name in (
        "stable_rank",
        "spectral_entropy",
        "condition_number",
        "frob_nuclear",
    ):
        if not math.isclose(
            base[name],
            scaled[name],
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise RuntimeError(
                "Scale-invariance "
                "self-test failed "
                f"for {name}"
            )



OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
}

CONDITION_LABELS = {
    "positive": "Positive",
    "negated": "Negated",
    "repeated_negations": "Repeated negations",
}




def _rows_for(
    checkpoint_rows: list[dict],
    condition: str,
    optimizer: str,
) -> list[dict]:
    rows = [
        row
        for row in checkpoint_rows
        if (
            row["condition"] == condition
            and row["optimizer"] == optimizer
        )
    ]

    rows = sorted(
        rows,
        key=lambda row: row["step"],
    )

    if len(rows) != len(STEPS):
        raise RuntimeError(
            f"Expected {len(STEPS)} rows for "
            f"{condition}/{optimizer}, got {len(rows)}"
        )

    return rows


def _plot_metric(
    line_specs: list[tuple[str, str, str]],
    checkpoint_rows: list[dict],
    out_path: Path,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    """
    line_specs entries are:
        (condition, optimizer, label)

    Visual encoding:
        color     -> optimizer
        marker    -> condition
        linestyle -> condition
    """

    optimizer_colors = {
        "adamw": "tab:blue",
        "muon": "tab:orange",
    }

    condition_markers = {
        "positive": "o",
        "negated": "o",
        "repeated_negations": "^",
    }

    condition_linestyles = {
        "positive": "-",
        "negated": "-",
        "repeated_negations": "--",
    }

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for (
        condition,
        optimizer,
        label,
    ) in line_specs:
        rows = _rows_for(
            checkpoint_rows,
            condition,
            optimizer,
        )

        ax.plot(
            [
                row["step"]
                for row in rows
            ],
            [
                row[metric]
                for row in rows
            ],
            label=label,
            color=optimizer_colors[
                optimizer
            ],
            marker=condition_markers[
                condition
            ],
            linestyle=condition_linestyles[
                condition
            ],
            linewidth=1.8,
            markersize=5.0,
        )

    ax.set_title(
        title
    )
    ax.set_xlabel(
        "Training step"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.grid(
        alpha=0.25
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_condition_figures(
    checkpoint_rows: list[dict],
    out: Path,
) -> None:
    """
    Generate:
      - 4 positive-only figures
      - 4 combined negated + repeated-negations figures

    No pooling or averaging across conditions occurs inside
    these figures.
    """

    figures = out / "figures"
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        (
            "mean_stable_rank",
            "Stable rank",
            "stable_rank",
        ),
        (
            "mean_spectral_entropy",
            "Spectral entropy",
            "spectral_entropy",
        ),
        (
            "mean_condition_number",
            "Condition number",
            "condition_number",
        ),
        (
            "mean_frob_nuclear",
            "Frobenius / nuclear ratio",
            "frob_nuclear",
        ),
    ]

    # Positive-only: 2 lines
    positive_lines = [
        (
            "positive",
            "adamw",
            "AdamW",
        ),
        (
            "positive",
            "muon",
            "Muon",
        ),
    ]

    for (
        metric,
        ylabel,
        stem,
    ) in metrics:
        _plot_metric(
            positive_lines,
            checkpoint_rows,
            figures / f"positive_{stem}_vs_step.png",
            metric=metric,
            title=f"Positive - {ylabel} across training",
            ylabel=ylabel,
        )

    # Negated + repeated-negations: 4 lines
    neg_lines = [
        (
            "negated",
            "adamw",
            "AdamW - Negated",
        ),
        (
            "negated",
            "muon",
            "Muon - Negated",
        ),
        (
            "repeated_negations",
            "adamw",
            "AdamW - Rep. neg.",
        ),
        (
            "repeated_negations",
            "muon",
            "Muon - Rep. neg.",
        ),
    ]

    for (
        metric,
        ylabel,
        stem,
    ) in metrics:
        _plot_metric(
            neg_lines,
            checkpoint_rows,
            figures / f"negated_repeated_{stem}_vs_step.png",
            metric=metric,
            title=f"Negated / repeated negations - {ylabel} across training",
            ylabel=ylabel,
        )


def _draw_table_image(
    title: str,
    columns: list[str],
    rows: list[list[str]],
    out_path: Path,
    *,
    figsize: tuple[float, float],
    bbox: list[float],
) -> None:
    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.axis(
        "off"
    )

    # Centered title with much tighter spacing to the table.
    fig.text(
        0.5,
        0.93,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontfamily="serif",
        fontweight="bold",
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=bbox,
    )

    table.auto_set_font_size(
        False
    )
    table.set_fontsize(
        10
    )

    for (
        row_index,
        col_index,
    ), cell in table.get_celld().items():
        cell.set_linewidth(
            0
        )
        cell.set_facecolor(
            "white"
        )
        cell.get_text().set_fontfamily(
            "serif"
        )

    # booktabs-style horizontal rules only
    n_rows = 1 + len(rows)

    x_left = bbox[0]
    x_right = bbox[0] + bbox[2]
    y_top = bbox[1] + bbox[3]
    row_height = bbox[3] / n_rows
    y_header_bottom = y_top - row_height
    y_bottom = bbox[1]

    ax.plot(
        [x_left, x_right],
        [y_top, y_top],
        transform=ax.transAxes,
        color="black",
        linewidth=1.0,
        clip_on=False,
    )

    ax.plot(
        [x_left, x_right],
        [y_header_bottom, y_header_bottom],
        transform=ax.transAxes,
        color="black",
        linewidth=0.7,
        clip_on=False,
    )

    ax.plot(
        [x_left, x_right],
        [y_bottom, y_bottom],
        transform=ax.transAxes,
        color="black",
        linewidth=1.0,
        clip_on=False,
    )

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def generate_condition_final_tables(
    checkpoint_rows: list[dict],
    out: Path,
) -> None:
    """
    Generate:
      - one positive final table
      - one combined negated + repeated-negations final table

    Uses step 625 only.
    """

    figures = out / "figures"
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_rows = [
        row
        for row in checkpoint_rows
        if row["step"] == 625
    ]

    def find_row(
        condition: str,
        optimizer: str,
    ) -> dict:
        matches = [
            row
            for row in final_rows
            if (
                row["condition"] == condition
                and row["optimizer"] == optimizer
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected 1 final row for "
                f"{condition}/{optimizer}, got {len(matches)}"
            )

        return matches[0]

    columns_standard = [
        "Label",
        "Stable Rank",
        "Spectral Entropy",
        "Condition No.",
        "Frob/Nuclear",
    ]

    # Positive table
    positive_table_rows = []

    for optimizer in (
        "muon",
        "adamw",
    ):
        row = find_row(
            "positive",
            optimizer,
        )

        positive_table_rows.append(
            [
                OPTIMIZER_LABELS[
                    optimizer
                ],
                f"{row['mean_stable_rank']:.1f}",
                f"{row['mean_spectral_entropy']:.2f}",
                f"{row['mean_condition_number']:.1f}",
                f"{row['mean_frob_nuclear']:.3f}",
            ]
        )

    _draw_table_image(
        title="Positive Final Spectral Flatness",
        columns=columns_standard,
        rows=positive_table_rows,
        out_path=figures / "positive_final_flatness_table.png",
        figsize=(8.4, 1.85),
        bbox=[0.08, 0.23, 0.84, 0.44],
    )

    # Negated + repeated-negations table
    combined_rows = []

    order = [
        ("negated", "muon"),
        ("negated", "adamw"),
        ("repeated_negations", "muon"),
        ("repeated_negations", "adamw"),
    ]

    for (
        condition,
        optimizer,
    ) in order:
        row = find_row(
            condition,
            optimizer,
        )

        cond_label = (
            "Negated"
            if condition == "negated"
            else "Rep. neg."
        )

        combined_rows.append(
            [
                f"{OPTIMIZER_LABELS[optimizer]} - {cond_label}",
                f"{row['mean_stable_rank']:.1f}",
                f"{row['mean_spectral_entropy']:.2f}",
                f"{row['mean_condition_number']:.1f}",
                f"{row['mean_frob_nuclear']:.3f}",
            ]
        )

    _draw_table_image(
        title="Negated/Repeated Negated Final Spectral Flatness",
        columns=columns_standard,
        rows=combined_rows,
        out_path=figures / "negated_repeated_final_flatness_table.png",
        figsize=(9.6, 2.45),
        bbox=[0.05, 0.14, 0.90, 0.66],
    )


def print_final_spectral_tables(
    checkpoint_rows: list[dict],
    pooled_rows: list[dict],
) -> None:
    """
    Print final spectral flatness pooled across conditions
    and separately for each condition.
    """

    final_pooled = {
        row["optimizer"]: row
        for row in pooled_rows
        if row["step"] == 625
    }

    if set(final_pooled) != {
        "adamw",
        "muon",
    }:
        raise RuntimeError(
            "Incomplete final pooled table."
        )

    print()
    print(
        "FINAL CHECKPOINT ? POOLED ACROSS CONDITIONS"
    )
    print(
        "=" * 91
    )

    print(
        f"{'Optimizer':<11}"
        f"{'Stable rank':>18}"
        f"{'Entropy':>16}"
        f"{'Condition':>18}"
        f"{'Frob/Nuclear':>20}"
    )

    print(
        "-" * 91
    )

    for optimizer in (
        "adamw",
        "muon",
    ):
        row = final_pooled[
            optimizer
        ]

        print(
            f"{OPTIMIZER_LABELS[optimizer]:<11}"
            f"{row['mean_stable_rank']:>18.3f}"
            f"{row['mean_spectral_entropy']:>16.4f}"
            f"{row['mean_condition_number']:>18.3f}"
            f"{row['mean_frob_nuclear']:>20.4f}"
        )

    print(
        "=" * 91
    )

    print(
        "Perfectly flat rank-32 spectrum:"
    )

    print(
        f"stable rank = 32 | "
        f"entropy = {math.log(32):.4f} | "
        f"condition = 1 | "
        f"Frob/Nuclear = {1 / math.sqrt(32):.4f}"
    )

    print()
    print(
        "FINAL CHECKPOINT ? BY CONDITION"
    )
    print(
        "=" * 111
    )

    print(
        f"{'Condition':<22}"
        f"{'Optimizer':<11}"
        f"{'Stable rank':>18}"
        f"{'Entropy':>16}"
        f"{'Condition':>18}"
        f"{'Frob/Nuclear':>20}"
    )

    print(
        "-" * 111
    )

    for condition in (
        "positive",
        "negated",
        "repeated_negations",
    ):
        for optimizer in (
            "adamw",
            "muon",
        ):
            matches = [
                row
                for row in checkpoint_rows
                if (
                    row["condition"] == condition
                    and row["optimizer"] == optimizer
                    and row["step"] == 625
                )
            ]

            if len(matches) != 1:
                raise RuntimeError(
                    "Missing/duplicate final row for "
                    f"{condition}/{optimizer}"
                )

            row = matches[0]

            print(
                f"{CONDITION_LABELS[condition]:<22}"
                f"{OPTIMIZER_LABELS[optimizer]:<11}"
                f"{row['mean_stable_rank']:>18.3f}"
                f"{row['mean_spectral_entropy']:>16.4f}"
                f"{row['mean_condition_number']:>18.3f}"
                f"{row['mean_frob_nuclear']:>20.4f}"
            )

    print(
        "=" * 111
    )


def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--exp",
        type=Path,
        default=ROOT,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--self-test-only",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    # One torch thread per checkpoint worker.
    torch.set_num_threads(
        1
    )

    try:
        torch.set_num_interop_threads(
            1
        )
    except RuntimeError:
        pass

    self_test()

    print(
        "Numerical self-test: PASS"
    )

    if args.self_test_only:
        return

    exp = (
        args.exp.resolve()
    )

    preflight(
        exp
    )

    out = (
        exp
        / "spectral_analysis"
    )

    jobs = all_jobs(
        exp
    )

    layer_rows = []
    checkpoint_rows = []

    start = (
        time.perf_counter()
    )

    print(
        "Method: Evil Spectra "
        "Appendix G.1"
    )

    print(
        "dtype=float32, "
        "cholesky_eps=1e-8, "
        "skip ||B||<1e-12"
    )

    print(
        f"workers={args.workers}; "
        "base model NOT loaded"
    )

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:

        futures = {
            pool.submit(
                analyse_one,
                *job,
            ): job
            for job
            in jobs
        }

        for (
            i,
            future,
        ) in enumerate(
            as_completed(
                futures
            ),
            1,
        ):

            (
                rows,
                summary,
            ) = future.result()

            layer_rows.extend(
                rows
            )

            checkpoint_rows.append(
                summary
            )

            print(
                f"[{i:02d}/90] "
                f"{summary['condition']:20s} "
                f"{summary['optimizer']:5s} "
                f"step={summary['step']:3d} "
                "stable="
                f"{summary['mean_stable_rank']:.4f} "
                "H="
                f"{summary['mean_spectral_entropy']:.4f} "
                "cond="
                f"{summary['mean_condition_number']:.4g} "
                "F/N="
                f"{summary['mean_frob_nuclear']:.6f}"
            )

    condition_order = {
        "positive":
            0,

        "negated":
            1,

        "repeated_negations":
            2,
    }

    optimizer_order = {
        "adamw":
            0,

        "muon":
            1,
    }

    checkpoint_rows.sort(
        key=lambda row: (
            condition_order[
                row["condition"]
            ],
            optimizer_order[
                row["optimizer"]
            ],
            row["step"],
        )
    )

    layer_rows.sort(
        key=lambda row: (
            condition_order[
                row["condition"]
            ],
            optimizer_order[
                row["optimizer"]
            ],
            row["step"],
            row["layer"],
        )
    )

    layer_fields = [
        "condition",
        "optimizer",
        "run",
        "step",
        "checkpoint",
        "layer",
        "target_module",
        "stable_rank",
        "spectral_entropy",
        "condition_number",
        "frob_nuclear",
        "sigma_max",
        "sigma_min",
    ]

    checkpoint_fields = [
        "condition",
        "optimizer",
        "run",
        "step",
        "checkpoint",
        "n_layers_total",
        "n_layers_used",
        "n_layers_skipped",
        "mean_stable_rank",
        "mean_spectral_entropy",
        "mean_condition_number",
        "mean_frob_nuclear",
        "elapsed_seconds",
    ]

    write_csv(
        out
        / "layer_metrics.csv",
        layer_rows,
        layer_fields,
    )

    write_csv(
        out
        / "checkpoint_metrics.csv",
        checkpoint_rows,
        checkpoint_fields,
    )

    write_csv(
        out
        / "final_checkpoint_metrics.csv",
        [
            row
            for row
            in checkpoint_rows
            if row["step"] == 625
        ],
        checkpoint_fields,
    )

    # Evil Spectra Table 3 averages across
    # layer x dataset measurements.
    #
    # Here the analogous pooled summary averages
    # across layer x condition measurements.
    pooled = []

    for optimizer in (
        "adamw",
        "muon",
    ):
        for step in STEPS:

            rows = [
                row
                for row
                in layer_rows
                if (
                    row["optimizer"]
                    == optimizer
                    and row["step"]
                    == step
                )
            ]

            if {
                row["condition"]
                for row
                in rows
            } != {
                "positive",
                "negated",
                "repeated_negations",
            }:
                raise RuntimeError(
                    "Incomplete pooled data "
                    f"for {optimizer} "
                    f"step {step}"
                )

            def pooled_mean(
                name: str,
            ) -> float:

                x = [
                    float(
                        row[name]
                    )
                    for row
                    in rows
                ]

                if any(
                    math.isinf(v)
                    for v in x
                ):
                    return math.inf

                return (
                    statistics.fmean(
                        x
                    )
                )

            pooled.append(
                {
                    "optimizer":
                        optimizer,

                    "step":
                        step,

                    "n_layer_condition_records":
                        len(rows),

                    "mean_stable_rank":
                        pooled_mean(
                            "stable_rank"
                        ),

                    "mean_spectral_entropy":
                        pooled_mean(
                            "spectral_entropy"
                        ),

                    "mean_condition_number":
                        pooled_mean(
                            "condition_number"
                        ),

                    "mean_frob_nuclear":
                        pooled_mean(
                            "frob_nuclear"
                        ),
                }
            )

    pooled_fields = [
        "optimizer",
        "step",
        "n_layer_condition_records",
        "mean_stable_rank",
        "mean_spectral_entropy",
        "mean_condition_number",
        "mean_frob_nuclear",
    ]

    write_csv(
        out
        / "optimizer_step_metrics.csv",
        pooled,
        pooled_fields,
    )

    write_csv(
        out
        / "final_optimizer_metrics.csv",
        [
            row
            for row
            in pooled
            if row["step"] == 625
        ],
        pooled_fields,
    )

    print()

    generate_condition_figures(
        checkpoint_rows,
        out,
    )

    generate_condition_final_tables(
        checkpoint_rows,
        out,
    )

    print_final_spectral_tables(
        checkpoint_rows,
        pooled,
    )

    print()

    print(
        "SPECTRAL ANALYSIS COMPLETE"
    )

    print(
        "checkpoint rows: "
        f"{len(checkpoint_rows)} / 90"
    )

    print(
        "layer rows: "
        f"{len(layer_rows)} / "
        f"{90 * N_LAYERS}"
    )

    print(
        "wall time: "
        f"{time.perf_counter() - start:.2f}s"
    )

    print(
        f"results: {out}"
    )


if __name__ == "__main__":
    main()
