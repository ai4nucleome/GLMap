#!/usr/bin/env python3
"""Distribution of downstream-task AUC across the 123-model population.

Renders a 6 + 1 violin / strip composite:

  - 6 violins, one per benchmark task, ordered by ascending population
    median AUC.
  - A 7th violin showing the mean AUC across the 6 tasks per model
    (the "6-task mean" summary used in §2.5).
  - Per task, AR models are stripped on the LEFT half of the violin
    and MLM models on the RIGHT half, so the reader can read both
    the joint distribution shape and the branch decomposition at a
    glance.

Per scientific-figure-making house style: 13 pt base font, 1.5 lw
spines, no top/right spines, frameless legend, semantic blue/red for
branch palette, light grey violins, subtle dotted grid.  No on-axis
annotation boxes — per-task statistics live in the report block
printed to stdout and in the paper caption.

Inputs
------
  out_phase2/matrices/auc_matrix.npy        — (123, 6) per-model AUC
  out_phase2/matrices/auc_matrix_meta.json  — model_ids + task_ids
  data/audits/models.json                   — branch labels

Output
------
  figures/Fig4a-downstream_auc_distribution.pdf

Usage
-----
  $PY scripts/figures/fig4a_downstream_auc_distribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402


BRANCH_COLOR = {
    "ar_or_generative": PALETTE["blue_main"],
    "mlm_or_encoder":   PALETTE["red_strong"],
}

MEAN_KEY = "__mean__"
MEAN_LABEL = "Mean of 6 tasks"

TASK_LABEL = {
    "iDNA_ABF/5mC": "5mC",
    "enhancers/enhancer": "enhancer",
    "prom/promoter_tata_300bps": "promoter_tata_300bps",
    "mouse/mouse_TFBS_3": "mouse_TFBS_3",
    "EMP/Yeast_H4": "Yeast_H4",
    "iPro-WAEL/Promoter_Arabidopsis_TATA": "promoter_arabidopsis_TATA",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--auc-matrix", type=Path,
                   default=REPO_ROOT / "out_phase2/matrices/auc_matrix.npy")
    p.add_argument("--auc-meta", type=Path,
                   default=REPO_ROOT / "out_phase2/matrices/auc_matrix_meta.json")
    p.add_argument("--audit", type=Path,
                   default=REPO_ROOT / "data/audits/models.json")
    p.add_argument("--sort-tasks", type=str, default="median",
                   choices=["median", "mean", "fixed"],
                   help="Task ordering on x-axis (mean is appended last).")
    p.add_argument("--figsize", type=str, default="12.8,6.4",
                   help='Inches, "W,H". Default "12.8,6.4".')
    p.add_argument("--out", dest="out_path", type=Path,
                   default=REPO_ROOT / "figures/Fig4a-downstream_auc_distribution.pdf")
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def _half_violin(ax, data, position, color, side="right",
                 width=0.32, alpha=0.18, linewidth=0.9):
    """Draw half a kernel-density violin at `position`, mirrored on
    `side` ∈ {"left", "right"}.

    Manually drawn (no seaborn) to keep the dependency surface small.
    """
    # KDE
    try:
        from scipy.stats import gaussian_kde  # type: ignore[reportMissingImports]
    except ImportError:
        return
    if len(data) < 3:
        return
    kde = gaussian_kde(data)
    y_grid = np.linspace(data.min() - 0.01, data.max() + 0.01, 200)
    density = kde(y_grid)
    density = density / density.max() * width  # normalize to width

    if side == "right":
        ax.fill_betweenx(
            y_grid, position, position + density,
            facecolor=color, edgecolor=color, linewidth=linewidth,
            alpha=alpha, zorder=1,
        )
    else:
        ax.fill_betweenx(
            y_grid, position - density, position,
            facecolor=color, edgecolor=color, linewidth=linewidth,
            alpha=alpha, zorder=1,
        )


def main() -> None:
    args = parse_args()

    # ── load ── #
    auc = np.load(args.auc_matrix)
    meta = json.loads(args.auc_meta.read_text())
    model_ids = meta["model_ids"]
    task_ids = meta["task_ids"]
    M, T = auc.shape
    print(f"[auc-dist] AUC matrix: ({M}, {T}) — {M} models × {T} tasks")

    # ── join branch info ── #
    audit = {m["hf_id"]: m for m in json.loads(args.audit.read_text())["models"]}
    branches = np.array([audit[h]["branch"] for h in model_ids])

    # ── task ordering ── #
    if args.sort_tasks == "median":
        scores = np.median(auc, axis=0)
    elif args.sort_tasks == "mean":
        scores = auc.mean(axis=0)
    else:
        scores = np.arange(T)
    order = np.argsort(scores)
    task_ids_ord = [task_ids[i] for i in order]
    auc_ord = auc[:, order]

    # 6-task mean per model — appended at the end
    auc_mean = auc.mean(axis=1)

    # Combined into one list of 7 series for plotting, using compact
    # two-line task labels suitable for a paper figure.
    panels = []
    for j, tid in enumerate(task_ids_ord):
        panels.append((TASK_LABEL.get(tid, tid), auc_ord[:, j]))
    panels.append((MEAN_LABEL, auc_mean))
    n_panels = len(panels)

    # Print stats to stdout (caption-ready)
    print()
    print(f"  {'Task':30s}  {'median':>7}  {'mean':>7}  {'σ':>7}  {'range':>14}")
    for label, vals in panels:
        clean = label.replace("\n", " ")
        print(f"  {clean:30s}  {np.median(vals):>7.3f}  {vals.mean():>7.3f}  "
              f"{vals.std():>7.3f}  [{vals.min():.2f}, {vals.max():.2f}]")
    print()

    # ── plot ── #
    figsize = _parse_figsize(args.figsize)
    local_rc = {
        **RCPARAMS,
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14.5,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
    }
    rng = np.random.default_rng(42)

    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize)

        # ── per-panel rendering ── #
        for j, (label, vals) in enumerate(panels):
            # Indices of AR vs MLM (only for the per-task panels; for the
            # mean panel we use a single combined violin without split).
            if label == MEAN_LABEL:
                ar_vals = vals[branches == "ar_or_generative"]
                mlm_vals = vals[branches == "mlm_or_encoder"]
            else:
                tid = task_ids_ord[j]
                col = task_ids.index(tid)
                ar_vals = auc[branches == "ar_or_generative", col]
                mlm_vals = auc[branches == "mlm_or_encoder", col]

            # Background grey violin = full population (cleaner backdrop)
            _half_violin(ax, vals, j, color="#A9A9A9",
                         side="right", width=0.34, alpha=0.10, linewidth=0.7)
            _half_violin(ax, vals, j, color="#A9A9A9",
                         side="left", width=0.34, alpha=0.10, linewidth=0.7)

            # Branch-specific half violins
            _half_violin(ax, ar_vals, j, color=BRANCH_COLOR["ar_or_generative"],
                         side="left", width=0.30, alpha=0.32)
            _half_violin(ax, mlm_vals, j, color=BRANCH_COLOR["mlm_or_encoder"],
                         side="right", width=0.30, alpha=0.32)

            # Strip plot, jittered
            jitter_ar = rng.uniform(-0.22, -0.02, size=len(ar_vals))
            jitter_mlm = rng.uniform(0.02, 0.22, size=len(mlm_vals))
            ax.scatter(j + jitter_ar, ar_vals,
                       s=24, c=BRANCH_COLOR["ar_or_generative"],
                       alpha=0.82, edgecolors="white", linewidths=0.45,
                       zorder=3)
            ax.scatter(j + jitter_mlm, mlm_vals,
                       s=24, c=BRANCH_COLOR["mlm_or_encoder"],
                       alpha=0.82, edgecolors="white", linewidths=0.45,
                       zorder=3)

            # Population median, marked without adding IQR guide lines.
            med = float(np.median(vals))
            ax.plot([j - 0.26, j + 0.26], [med, med],
                    color="#111111", linewidth=2.0, zorder=4)

        # Axes polish — horizontal labels (task names are short enough
        # after stripping the benchmark-source prefix).
        ax.set_xticks(np.arange(n_panels))
        ax.set_xticklabels(
            [label for label, _ in panels],
            fontsize=10.2,
            linespacing=1.15,
        )
        ax.set_xlim(-0.55, n_panels - 0.45)
        ax.set_ylim(0.46, 1.02)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        ax.set_ylabel("Downstream test AUC")
        ax.set_title(
            f"Distribution of downstream-task AUC across {M} genomic language models",
            pad=12,
        )

        # Keep the panel clean: no grid or summary guide lines.
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend
        counts = Counter(branches)
        handles = [
            Patch(facecolor=BRANCH_COLOR["ar_or_generative"],
                  edgecolor="white", linewidth=0.5, alpha=0.85,
                  label=f"AR (n = {counts['ar_or_generative']})"),
            Patch(facecolor=BRANCH_COLOR["mlm_or_encoder"],
                  edgecolor="white", linewidth=0.5, alpha=0.85,
                  label=f"MLM (n = {counts['mlm_or_encoder']})"),
            Line2D([0], [0], color="#111111", linewidth=2.0,
                   label="median across models"),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False, ncol=2, handlelength=1.8,
            handletextpad=0.5, columnspacing=1.2, labelspacing=0.38,
        )

        fig.tight_layout()
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_path, dpi=300, bbox_inches="tight",
                    pad_inches=0.1)
        plt.close(fig)
        print(f"[done] wrote {args.out_path}")


if __name__ == "__main__":
    main()
