#!/usr/bin/env python3
"""Heatmap of the combined 123x123 GLMap D distance matrix.

NOT A PAPER FIGURE. This is a diagnostic / exploratory visualization and
is not referenced in the manuscript (the paper presents the GLMap map via
t-SNE on V_d — Fig 3 — and the hierarchical clustering of D — Fig 2e — not
the raw D matrix itself). Output therefore lands under
results/figures/_preview/ alongside the other non-paper diagnostics.

What it draws
-------------
The full per-pair distance matrix D[i, j] = ||V_d,i - V_d,j||^2 across all
123 models (built on-the-fly from the combined GLMap representation, the
same object used by scripts/model_map/run_fig3_model_map_embedding.py).
Models are ordered by family (largest family first); rows/cols are the
123 model names, colored by branch (AR vs MLM). A log color scale is used
because a few outlier models (e.g. Genos-10B) span ~3 orders of magnitude
and otherwise wash out the structure on a linear scale.

Input
-----
  results/scores/AR_MLM_scores/<slug>/probes.parquet   (via _combined_q_loader)
  data/audits/models.json
  data/panels/main_panel.parquet

Output
------
  results/figures/_preview/D_matrix_123_by_family.pdf

Usage
-----
  python scripts/figures/D_matrix_heatmap.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402
from scripts.figures._figure_style import PALETTE, RCPARAMS    # noqa: E402

AR_COLOR = PALETTE["blue_main"]
MLM_COLOR = PALETTE["red_strong"]


def _short(hf_id: str) -> str:
    return hf_id.split("/")[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "results/figures/_preview/D_matrix_123_by_family.pdf")
    p.add_argument("--label-size", type=float, default=4.6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(RCPARAMS)

    g = load_combined_glmap()
    D = np.asarray(g.D, dtype=float)
    M = len(D)
    hf, fam, br = list(g.hf_ids), list(g.families), list(g.branches)
    is_ar = [b == "ar_or_generative" for b in br]

    # Order by family (largest family first; then family name; then hf_id).
    size = Counter(fam)
    order = sorted(range(M), key=lambda i: (-size[fam[i]], fam[i], hf[i]))
    Dc = D[np.ix_(order, order)]
    names = [_short(hf[i]) for i in order]
    fams = [fam[i] for i in order]
    label_colors = [AR_COLOR if is_ar[i] else MLM_COLOR for i in order]
    bounds = [0] + [k for k in range(1, M) if fams[k] != fams[k - 1]] + [M]

    fig, ax = plt.subplots(figsize=(24, 22))
    Dp = Dc.copy()
    vmin = np.percentile(Dp[Dp > 0], 2)
    Dp[Dp <= 0] = vmin
    im = ax.imshow(Dp, cmap="magma", norm=LogNorm(vmin=vmin, vmax=Dp.max()),
                   aspect="equal")

    ax.set_xticks(range(M))
    ax.set_yticks(range(M))
    ax.set_xticklabels(names, rotation=90, fontsize=args.label_size)
    ax.set_yticklabels(names, fontsize=args.label_size)
    for t, c in zip(ax.get_xticklabels(), label_colors):
        t.set_color(c)
    for t, c in zip(ax.get_yticklabels(), label_colors):
        t.set_color(c)
    ax.tick_params(length=0)

    # Family separators + family names at each block center (top).
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color="white", lw=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="white", lw=0.6, alpha=0.7)
    for a, b in zip(bounds[:-1], bounds[1:]):
        ax.text((a + b - 1) / 2, -3.0, fams[a], rotation=90,
                ha="center", va="bottom", fontsize=6, color="#333333")

    ax.set_title(
        f"Combined GLMap D matrix ({M}x{M}) — ordered by model family "
        "(NOT a paper figure)\n"
        "D[i,j] = ||V_d,i - V_d,j||^2 · log color scale · "
        "label color: blue = AR, red = MLM",
        fontsize=15, pad=58,
    )
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(M - 0.5, -0.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label("squared distance (log)", fontsize=11)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[D-heatmap] wrote {args.out} ({M}x{M}, "
          f"{len(set(fam))} families)")


if __name__ == "__main__":
    main()
