#!/usr/bin/env python3
"""Figure 2c: Mantel-style split-half stability of GLMap pairwise distances.

Tests whether the inter-model distance structure of GLMap is stable
across disjoint probe subsets. For each split of the 10,000-probe
panel into two halves, we independently compute the model × model
pairwise distance matrix on each half and plot every model-pair
distance as a single point (x = distance on half A, y = distance on
half B). A high Pearson r over the C(M, 2) pairs means the panel's
distance structure does not depend on the specific probe subset —
analogous to the Mantel-test stability check in the LLM DNA paper
(Liu et al., Fig 3).

Pipeline
--------
  1) Load L matrix (M models × N=10,000 probes) sorted by probe_id.
     Fail-fast: any model with missing probes.parquet, probe_id
     mis-order, or NaN sum_log_p aborts the script (no silent skip
     — these all indicate upstream scoring bugs that should be
     surfaced rather than hidden).
  2) Partition probes into half A and half B using one of two
     non-overlapping strategies (CLI: --split-method):
       'element-disjoint' (DEFAULT) — split the 14 elements themselves
                          into two balanced disjoint groups → A, B
                          contain DIFFERENT element types. The
                          demanding cross-category transfer test
                          analogous to Liu et al. SQuAD+CSQA vs
                          HellaSwag+Winogrande, and the primary check
                          for the paper's stability claim.
       'stratified'       within each functional_element split 50/50
                          → A, B have identical per-element
                          composition. The gentler "same panel
                          sub-sampled twice" check, retained for
                          sensitivity comparison.
  3) On each half independently, compute the model × model pairwise
     squared-Euclidean distance matrix D_A, D_B.  When
     --no-pipeline is NOT set (default), the GLMap pipeline
     (clip(q=0.02) → double-center → pairwise_squared_distance) runs
     on each half's L sub-matrix.  When --no-pipeline IS set, the
     distance is computed directly on the raw L sub-matrix without
     any clip or centering.  The chosen mode is recorded in the
     output filename.
  4) Extract upper-triangular pairs d_A_pairs[k], d_B_pairs[k] for
     k = 1..C(M, 2).
  5) Compute Pearson r between the two pair-distance vectors and its
     parametric p-value (Mantel correlation for distance matrices).

Visualization
-------------
Scatter plot:
  X-axis: pairwise distance on half A
  Y-axis: pairwise distance on half B
  Each point = one unordered pair of distinct models (C(M, 2) = 7503
  pairs for M=123).
  Diagonal y = x reference line.
  Best-fit regression line.
  Pearson r + p-value annotation.

Output
------
  figures/Fig2c-split-half-mantel_<method>_seed<seed>_<pipeline>.pdf
    where <pipeline> ∈ {with-pipeline, no-pipeline}

Usage
-----
  $PY scripts/figures/fig2c_split_half_consistency.py
  $PY scripts/figures/fig2c_split_half_consistency.py --split-method stratified
  $PY scripts/figures/fig2c_split_half_consistency.py --no-pipeline   # sensitivity check
  $PY scripts/figures/fig2c_split_half_consistency.py --seed 7
  # Multi-seed grid (compare how Mantel r varies across different
  # probe-half samplings — robustness defence for reviewers):
  $PY scripts/figures/fig2c_split_half_consistency.py --seeds 42,7,123,99,2024,2025
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402
from glmap.matrices.build import (  # noqa: E402
    clip_lower,
    double_center,
    pairwise_squared_distance,
)


# ─────────────────────────── data loading ─────────────────────────── #


def _load_panel(panel_path: Path):
    """Load probe_id + functional_element, sorted by probe_id."""
    if not panel_path.exists():
        sys.exit(f"panel parquet not found: {panel_path}")
    df = pq.read_table(panel_path, columns=["probe_id", "functional_element"]).to_pandas()
    df = df.sort_values("probe_id").reset_index(drop=True)
    return df


def _load_L_matrix(audit_path: Path, scores_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Read all audited models' probes.parquet files into L of shape (M, N).

    Fail-fast: any data quality issue (missing parquet, probe_id
    mis-ordering, NaN sum_log_p) raises immediately instead of silently
    dropping the model. These conditions all indicate upstream scoring
    bugs that should be surfaced, not hidden.
    """
    audit = json.loads(audit_path.read_text())["models"]
    L_rows: list[np.ndarray] = []
    hf_ids: list[str] = []
    probe_order: list[str] | None = None

    for m in audit:
        hf_id = m["hf_id"]
        slug = hf_id.replace("/", "__")
        pq_path = scores_dir / slug / "probes.parquet"
        if not pq_path.exists():
            sys.exit(
                f"[fig2c] FATAL: missing probes.parquet for {hf_id} "
                f"(expected {pq_path}). Re-score this model or remove "
                f"it from the audit roster."
            )

        t = pq.read_table(pq_path, columns=["probe_id", "sum_log_p"]).to_pandas()
        t = t.sort_values("probe_id").reset_index(drop=True)
        if probe_order is None:
            probe_order = t["probe_id"].tolist()
        elif t["probe_id"].tolist() != probe_order:
            sys.exit(
                f"[fig2c] FATAL: probe_id order mismatch in {hf_id} "
                f"vs the first scored model. The probe set differs "
                f"across models — re-score against the canonical panel "
                f"so every parquet has the same probe_ids."
            )

        vec = t["sum_log_p"].to_numpy()
        n_nan = int(np.isnan(vec).sum())
        if n_nan:
            sys.exit(
                f"[fig2c] FATAL: {hf_id} has {n_nan} NaN sum_log_p "
                f"entries in probes.parquet. NaNs indicate per-probe "
                f"scoring failures; re-run the model or fix the loader."
            )

        L_rows.append(vec)
        hf_ids.append(hf_id)

    L = np.stack(L_rows, axis=0)
    print(f"[fig2c] loaded L matrix: {L.shape} (no skips — all data clean)",
          flush=True)
    return L, hf_ids


# ─────────────────────────── split logic ──────────────────────────── #


def _build_stratified_split(panel_df, seed: int):
    """Within each functional_element, randomly split probes 50/50.

    Returns (idx_a, idx_b) as 1-D numpy arrays. Both halves end up
    with IDENTICAL per-element composition (same N from each element
    type), so this is the gentle "same panel sub-sampled twice"
    stability check.
    """
    rng = np.random.default_rng(seed)
    elements = panel_df["functional_element"].to_numpy()
    unique_elements = sorted(np.unique(elements))
    idx_a_parts: list[np.ndarray] = []
    idx_b_parts: list[np.ndarray] = []
    for elem in unique_elements:
        elem_idx = np.where(elements == elem)[0]
        shuffled = rng.permutation(elem_idx)
        half = len(shuffled) // 2
        idx_a_parts.append(shuffled[:half])
        idx_b_parts.append(shuffled[half:2 * half])
    return np.concatenate(idx_a_parts), np.concatenate(idx_b_parts)


def _build_element_disjoint_split(panel_df, seed: int):
    """Split functional_elements (not probes!) into two disjoint groups,
    then take all probes belonging to each group.

    This is the demanding stability test analogous to Liu et al.'s LLM
    paper Fig 3 (SQuAD+CSQA vs HellaSwag+Winogrande): the two halves
    have COMPLETELY DIFFERENT functional element types, so the Pearson
    correlation tests whether the GLMap pairwise distance structure
    transfers across element categories, not just survives within-
    element sub-sampling.

    Balance algorithm: shuffle the 14 element labels by ``seed``, then
    greedily bin-pack each one (in the shuffled order) into the half
    with smaller current probe count. The shuffle drives the split, so
    different seeds yield meaningfully different (A, B) groupings
    (this is the property the multi-seed grid relies on). Splits are
    typically within a few hundred probes of 5000/5000.

    A prior version of this function size-sorted elements (descending)
    after the shuffle, which made the assignment near-deterministic and
    collapsed multi-seed runs to a single split. Dropping that sort
    keeps the bin-pack approximately balanced while restoring the
    intended seed dependence.
    """
    rng = np.random.default_rng(seed)
    elements = panel_df["functional_element"].to_numpy()
    unique = sorted(np.unique(elements))
    perm = rng.permutation(len(unique))
    shuffled_unique = [unique[i] for i in perm]
    group_a: list[str] = []
    group_b: list[str] = []
    size_a = 0
    size_b = 0
    for elem in shuffled_unique:
        sz = int(np.sum(elements == elem))
        if size_a <= size_b:
            group_a.append(elem)
            size_a += sz
        else:
            group_b.append(elem)
            size_b += sz
    idx_a = np.where(np.isin(elements, group_a))[0]
    idx_b = np.where(np.isin(elements, group_b))[0]
    return idx_a, idx_b, group_a, group_b


# ─────────────────────────── distance pipeline ────────────────────── #


def _compute_distance_matrix(
    L_sub: np.ndarray,
    clip_q: float,
    use_pipeline: bool,
) -> np.ndarray:
    """Compute the model × model pairwise squared-Euclidean distance
    matrix on a probe-subset L matrix.

    When ``use_pipeline`` is True (default), runs the full GLMap
    pipeline first (clip → double-center → pairwise distance) so the
    output matches the distance definition used in Fig 5, 6, 7, 8.
    When False, computes squared Euclidean directly on the raw L
    sub-matrix; this is a sensitivity check (does the stability hold
    even before the canonical pipeline?).
    """
    if use_pipeline:
        L_clipped, _ = clip_lower(L_sub, q=clip_q)
        Q, _, _, _ = double_center(L_clipped)
        return pairwise_squared_distance(Q)
    else:
        return pairwise_squared_distance(L_sub)


def _upper_triangular_pairs(D: np.ndarray) -> np.ndarray:
    """Return the upper triangular off-diagonal entries of D as a 1-D
    array of length C(M, 2)."""
    M = D.shape[0]
    iu = np.triu_indices(M, k=1)
    return D[iu]


# ──────────────────────────── plotting ────────────────────────────── #


def _make_figure(
    d_a: np.ndarray, d_b: np.ndarray,
    r: float, p: float,
    split_method: str, seed: int, M: int,
    use_pipeline: bool,
    figsize: tuple[float, float],
    out_path: Path,
    element_groups: tuple[list[str], list[str]] | None = None,
) -> None:
    """Scatter plot of paired pairwise distances + Mantel-style annotation.

    Design follows the repository ``scientific-figure-making`` house
    style (minimalist spines, frameless legend, single-blue accent,
    neutral reference line, no embellishment).
    """
    n_pairs = len(d_a)

    # Local rcParams override on top of the shared RCPARAMS — bump the
    # font slightly for a publication-size compact panel, and ensure
    # vector text remains editable.
    local_rc = {
        **RCPARAMS,
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.4,
    }

    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize)

        # Axis limits set by data range with a small symmetric margin so
        # the y=x line extends through the whole frame.
        lo = float(min(d_a.min(), d_b.min()))
        hi = float(max(d_a.max(), d_b.max()))
        pad = 0.04 * (hi - lo)
        line_lo, line_hi = lo - pad, hi + pad

        # y = x reference line — drawn first so the scatter sits on top.
        # Subtle neutral dashed; legend label is implicit (axis labels
        # already convey "half A vs half B").
        ax.plot([line_lo, line_hi], [line_lo, line_hi],
                color=PALETTE["neutral"], linestyle="--",
                linewidth=1.1, zorder=1, label="y = x")

        # Single-hue scatter, blue_main per house style. Larger marker +
        # softer alpha = readable density without sparkle.
        ax.scatter(
            d_a, d_b,
            s=14, c=PALETTE["blue_main"], alpha=0.45,
            edgecolors="none", zorder=2,
        )

        # Best-fit (OLS) regression line. Drawn ABOVE the scatter so
        # readers see how the data deviates from y = x globally.
        slope, intercept = np.polyfit(d_a, d_b, deg=1)
        fit_x = np.array([line_lo, line_hi])
        fit_y = slope * fit_x + intercept
        ax.plot(
            fit_x, fit_y,
            color=PALETTE["red_strong"], linewidth=1.6, alpha=0.9,
            zorder=3, label="best fit",
        )

        # Pearson r + p as a clean top-left annotation. Minimal box
        # (no edge, soft white) so it doesn't fight the scatter.
        # p == 0 here is numerical underflow at ~1e-308; report it as a
        # ceiling rather than a misleading "p = 0".
        if p == 0:
            p_str = r"$p < 10^{-300}$"
        elif p < 1e-3:
            p_str = f"$p < 10^{{-{int(np.floor(-np.log10(p)))}}}$"
        else:
            p_str = f"$p = {p:.3g}$"
        annotation = (
            f"Pearson $r$ = {r:.3f}\n"
            f"{p_str}"
        )
        ax.text(
            0.04, 0.96, annotation,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor="none", alpha=0.85),
            zorder=3,
        )

        # Two-line title: main headline via suptitle, subdued context
        # line as the axis title (smaller + grey). This pattern keeps a
        # clear hierarchy without overlapping.
        if split_method == "stratified":
            method_label = "stratified by functional element"
        elif split_method == "element-disjoint":
            method_label = "element-disjoint split"
        else:
            method_label = split_method
        pipeline_label = (
            "clip + double-centered"
            if use_pipeline else "raw L (no pipeline)"
        )
        fig.suptitle(
            "Mantel Test of Distance Matrix from Different Dataset",
            fontsize=14, y=0.995, x=0.55,
        )
        ax.set_xlabel("Pairwise distance (panel A)")
        ax.set_ylabel("Pairwise distance (panel B)")
        ax.set_xlim(line_lo, line_hi)
        ax.set_ylim(line_lo, line_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.7)

        # Frameless legend in the lower-right (top-left already holds
        # the Pearson r annotation). Two entries: y=x reference + the
        # best-fit line whose slope was added above.
        ax.legend(
            loc="lower right",
            frameon=False, fontsize=11,
            handlelength=2.4, borderaxespad=0.6,
        )

        # Element-disjoint mode: render the two element groups as a tiny
        # footnote below the axes. Wraps cleanly even with 8 elements.
        if element_groups is not None:
            ga, gb = element_groups
            footnote = (
                f"Panel A elements: {', '.join(ga)}\n"
                f"Panel B elements: {', '.join(gb)}"
            )
            fig.text(
                0.5, -0.06, footnote,
                ha="center", va="top",
                fontsize=9, color="#666",
                style="italic",
            )

        fig.tight_layout(pad=1.2)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


# ─────────────────────── multi-seed grid figure ───────────────────── #


def _grid_shape(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for an approximately-square panel grid."""
    if n <= 1:
        return (1, 1)
    if n == 3:
        return (1, 3)
    if n == 2:
        return (1, 2)
    if n <= 4:
        return (2, 2)
    if n <= 6:
        return (2, 3)
    if n <= 9:
        return (3, 3)
    if n <= 12:
        return (3, 4)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return (nrows, ncols)


def _make_multi_seed_figure(
    panels: list[dict],
    split_method: str, M: int, use_pipeline: bool,
    out_path: Path,
) -> None:
    """Render a grid of Mantel scatters, one per seed.

    ``panels`` is a list of per-seed result dicts containing keys:
      'seed', 'd_a', 'd_b', 'r', 'p', 'slope', 'intercept',
      'element_groups' (None for stratified mode).
    """
    n = len(panels)
    nrows, ncols = _grid_shape(n)

    # Shared axis range across all panels so the comparison reads
    # directly: take the global max / min over all panels' d_a, d_b.
    all_d = np.concatenate(
        [np.concatenate([p["d_a"], p["d_b"]]) for p in panels]
    )
    lo = float(all_d.min())
    hi = float(all_d.max())
    pad = 0.04 * (hi - lo)
    line_lo, line_hi = lo - pad, hi + pad

    local_rc = {
        **RCPARAMS,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.2,
    }

    # Panel size scaled by grid; aim ~3.0 × 3.0 inches per panel.
    panel_size = 3.0
    footnote_extra = 1.4 if split_method == "element-disjoint" else 0.0
    figsize = (panel_size * ncols + 1.5, panel_size * nrows + 1.5 + footnote_extra)

    with plt.rc_context(local_rc):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            sharex=True, sharey=True,
        )
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        for idx, panel in enumerate(panels):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            ax.plot([line_lo, line_hi], [line_lo, line_hi],
                    color=PALETTE["neutral"], linestyle="--",
                    linewidth=1.0, zorder=1)
            ax.scatter(
                panel["d_a"], panel["d_b"],
                s=8, c=PALETTE["blue_main"], alpha=0.4,
                edgecolors="none", zorder=2,
            )
            fit_x = np.array([line_lo, line_hi])
            fit_y = panel["slope"] * fit_x + panel["intercept"]
            ax.plot(
                fit_x, fit_y,
                color=PALETTE["red_strong"], linewidth=1.4,
                alpha=0.9, zorder=3,
            )

            # Compact per-panel annotation: seed + r.
            ax.text(
                0.04, 0.96,
                f"seed = {panel['seed']}\n"
                f"$r$ = {panel['r']:.3f}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor="none",
                          alpha=0.85),
                zorder=4,
            )

            ax.set_xlim(line_lo, line_hi)
            ax.set_ylim(line_lo, line_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.16, linestyle=":", linewidth=0.6)

            # Outer-edge axis labels only.
            if row == nrows - 1:
                ax.set_xlabel("Pairwise distance (panel A)")
            if col == 0:
                ax.set_ylabel("Pairwise distance (panel B)")

        # Hide any extra unused panels.
        for empty_idx in range(n, nrows * ncols):
            row, col = divmod(empty_idx, ncols)
            axes[row, col].set_visible(False)

        # Headline + subtitle.
        if split_method == "stratified":
            method_label = "stratified by functional element"
        elif split_method == "element-disjoint":
            method_label = "element-disjoint split"
        else:
            method_label = split_method
        pipeline_label = (
            "clip + double-centered"
            if use_pipeline else "raw L (no pipeline)"
        )
        fig.suptitle(
            f"Mantel Test of Distance Matrix from Different Dataset  "
            f"·  M = {M} models  ·  $N_{{\\mathrm{{pairs}}}}$ = "
            f"{len(panels[0]['d_a']):,} per panel",
            fontsize=13, y=0.995,
        )
        fig.text(
            0.5, 0.962,
            f"{method_label}  ·  {pipeline_label}",
            ha="center", va="top",
            fontsize=10, color="#555",
        )

        # Aggregate stats footnote.
        r_values = np.array([p["r"] for p in panels])
        fig.text(
            0.5, 0.02,
            f"r:  mean = {r_values.mean():.3f}  ·  "
            f"std = {r_values.std():.3f}  ·  "
            f"min/max = {r_values.min():.3f} / {r_values.max():.3f}",
            ha="center", va="bottom",
            fontsize=10, color="#333",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="#F4F4F4", edgecolor="none"),
        )

        if split_method == "element-disjoint":
            footnote_lines = []
            for panel in panels:
                ga, gb = panel["element_groups"]
                footnote_lines.append(
                    f"seed {panel['seed']}: panel A = {', '.join(ga)}; "
                    f"panel B = {', '.join(gb)}"
                )
            fig.text(
                0.5, 0.095,
                "\n".join(footnote_lines),
                ha="center", va="bottom",
                fontsize=8.2, color="#666",
                style="italic",
                linespacing=1.25,
            )
            bottom = 0.28
        else:
            bottom = 0.04

        fig.tight_layout(rect=[0, bottom, 1, 0.94])
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


# ─────────────────────────────── main ─────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--audit", type=Path,
                   default=REPO_ROOT / "data/audits/models.json",
                   help="Models audit JSON (provides the model roster).")
    p.add_argument("--panel", type=Path,
                   default=REPO_ROOT / "out_panel/main_panel.parquet",
                   help="Main panel parquet (provides functional_element strata).")
    p.add_argument("--scores-dir", type=Path,
                   default=REPO_ROOT / "out_phase1/scores",
                   help="Directory containing <slug>/probes.parquet.")
    p.add_argument("--out", dest="out_dir", type=Path,
                   default=REPO_ROOT / "figures",
                   help="Output directory.")
    p.add_argument("--split-method", type=str, default="element-disjoint",
                   choices=["stratified", "element-disjoint"],
                   help="Probe split strategy. 'element-disjoint' "
                        "(default): split the 14 functional elements "
                        "themselves into two disjoint groups (balanced "
                        "greedy bin-pack), so the two halves contain "
                        "DIFFERENT element types — the demanding "
                        "cross-category Mantel test analogous to Liu et "
                        "al. (SQuAD+CSQA vs HellaSwag+Winogrande). "
                        "'stratified': within each functional_element "
                        "split probes 50/50 so both halves have identical "
                        "per-element composition; the gentler 'same "
                        "panel sub-sampled twice' check, kept for "
                        "sensitivity comparison.")
    p.add_argument("--seed", type=int, default=123,
                   help="RNG seed for the split (single-panel mode).")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds for the "
                        "multi-panel grid comparison. When provided, "
                        "the script renders one scatter per seed in a "
                        "grid (instead of single-panel mode) so the "
                        "Mantel r distribution across different probe-"
                        "half samplings is visible. Example: "
                        "'--seeds 42,7,123,99,2024,2025' gives a 2x3 "
                        "grid. Overrides --seed when set. Output "
                        "filename includes the tag 'multi-seed'.")
    p.add_argument("--clip-q", type=float, default=0.02,
                   help="GLMap pipeline clip quantile (default 0.02; "
                        "matches src/matrices/build.py). Ignored when "
                        "--no-pipeline is set.")
    p.add_argument("--no-pipeline", action="store_true",
                   help="Skip clip + double-center; compute the "
                        "pairwise squared-Euclidean distance directly "
                        "on the raw L sub-matrix. Default behaviour "
                        "(without this flag) applies the canonical "
                        "GLMap pipeline (clip → double-center → "
                        "distance) on each half independently. The "
                        "pipeline / no-pipeline choice is reflected "
                        "in the output filename.")
    p.add_argument("--figsize", type=str, default="6,6",
                   help='Inches, "W,H". Default "6,6" (square; matches y=x).')
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    w, h = (float(x) for x in s.split(sep))
    return (w, h)


def _compute_one_seed(
    L: np.ndarray, panel_df, split_method: str, seed: int,
    clip_q: float, use_pipeline: bool,
) -> dict:
    """Run one Mantel split + distance computation for a single seed.

    Returns a panel dict consumed by the single-panel or multi-panel
    figure renderer. Pure function — no I/O.
    """
    if split_method == "stratified":
        idx_a, idx_b = _build_stratified_split(panel_df, seed)
        element_groups = None
    elif split_method == "element-disjoint":
        idx_a, idx_b, ga, gb = _build_element_disjoint_split(panel_df, seed)
        element_groups = (ga, gb)
    else:
        sys.exit(f"unknown split-method {split_method!r}")

    D_a = _compute_distance_matrix(
        L[:, idx_a], clip_q=clip_q, use_pipeline=use_pipeline,
    )
    D_b = _compute_distance_matrix(
        L[:, idx_b], clip_q=clip_q, use_pipeline=use_pipeline,
    )
    d_a = _upper_triangular_pairs(D_a)
    d_b = _upper_triangular_pairs(D_b)
    r, p = pearsonr(d_a, d_b)
    slope, intercept = np.polyfit(d_a, d_b, deg=1)

    return {
        "seed": seed,
        "idx_a": idx_a,
        "idx_b": idx_b,
        "d_a": d_a,
        "d_b": d_b,
        "r": float(r),
        "p": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
        "element_groups": element_groups,
    }


def main() -> None:
    args = parse_args()
    use_pipeline = not args.no_pipeline

    panel_df = _load_panel(args.panel)
    print(f"[fig2c] panel: {len(panel_df)} probes, "
          f"{panel_df['functional_element'].nunique()} unique elements",
          flush=True)

    L, hf_ids = _load_L_matrix(args.audit, args.scores_dir)
    M = L.shape[0]
    print(f"[fig2c] pipeline: "
          f"{'GLMap (clip + double-center)' if use_pipeline else 'raw L (no clip, no center)'}",
          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_tag = "with-pipeline" if use_pipeline else "no-pipeline"

    # ── Multi-seed grid mode ── #
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        print(f"[fig2c] multi-seed mode: {len(seeds)} seeds = {seeds}",
              flush=True)
        panels: list[dict] = []
        for seed in seeds:
            panel = _compute_one_seed(
                L, panel_df, args.split_method, seed,
                args.clip_q, use_pipeline,
            )
            print(f"  seed = {seed:>5d}  |A|={len(panel['idx_a']):>4d}  "
                  f"|B|={len(panel['idx_b']):>4d}  "
                  f"r = {panel['r']:.4f}  slope = {panel['slope']:.3f}",
                  flush=True)
            panels.append(panel)

        r_arr = np.array([p["r"] for p in panels])
        print(f"[fig2c] r across {len(seeds)} seeds: "
              f"mean = {r_arr.mean():.4f}, std = {r_arr.std():.4f}, "
              f"min = {r_arr.min():.4f}, max = {r_arr.max():.4f}",
              flush=True)

        out_path = args.out_dir / (
            f"Fig2c-split-half-mantel_{args.split_method}"
            f"_multi-seed_{pipeline_tag}.pdf"
        )
        _make_multi_seed_figure(
            panels, split_method=args.split_method, M=M,
            use_pipeline=use_pipeline, out_path=out_path,
        )
        print(f"[done] wrote {out_path}", flush=True)
        return

    # ── Single-seed mode (default) ── #
    panel = _compute_one_seed(
        L, panel_df, args.split_method, args.seed,
        args.clip_q, use_pipeline,
    )
    if panel["element_groups"] is not None:
        ga, gb = panel["element_groups"]
        print(f"[fig2c] element-disjoint groups:", flush=True)
        print(f"  A ({len(panel['idx_a'])} probes): {', '.join(ga)}",
              flush=True)
        print(f"  B ({len(panel['idx_b'])} probes): {', '.join(gb)}",
              flush=True)
    print(f"[fig2c] split: |A|={len(panel['idx_a'])}, "
          f"|B|={len(panel['idx_b'])} "
          f"(method={args.split_method}, seed={args.seed})", flush=True)
    print(f"[fig2c] Pearson r = {panel['r']:.4f}, p = {panel['p']:.3e}",
          flush=True)
    print(f"[fig2c] N pairs = {len(panel['d_a'])} (= C({M}, 2))",
          flush=True)

    out_path = args.out_dir / (
        f"Fig2c-split-half-mantel_{args.split_method}_seed{args.seed}"
        f"_{pipeline_tag}.pdf"
    )
    _make_figure(
        panel["d_a"], panel["d_b"], r=panel["r"], p=panel["p"],
        split_method=args.split_method, seed=args.seed, M=M,
        use_pipeline=use_pipeline,
        figsize=_parse_figsize(args.figsize),
        out_path=out_path,
        element_groups=panel["element_groups"],
    )
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
