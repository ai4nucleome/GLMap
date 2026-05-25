#!/usr/bin/env python3
"""Figure 3 + Figure S1: GLMap probe panel composition (model-free).

Visualizes the 10,000-probe panel BEFORE any gLM enters the picture. The
goal is to motivate the panel design (diversity, balance across
biological category and functional element) using model-independent
sequence statistics; subsequent figures show how gLMs perceive this
panel.

Three separate PDFs (per paper.md §2.2 plan, 2026-05-21 revision):
  Fig 3  panel a:  UMAP of k-mer composition, colored by biological
                   category (Human / Plant / Fungi / Virus).
                   → Fig3-UMAP_kmer-composition_k{...}_by-category.pdf

  Fig S1 (panel b): Same UMAP, colored by functional_element (14 hues
                   from the repo's CLASS_COLORS, ordered Human → Plant →
                   Fungi → Virus). Supplementary because the 14-class
                   legend would dominate a main-text figure.
                   → FigS1-UMAP_kmer-composition_k{...}_by-element.pdf

  Fig 3  panel c:  GC content distribution by functional element
                   (box + strip overlay). GC is NOT in the UMAP feature
                   stack; it is shown separately here because it is
                   already implicit in the k=1 mononucleotide vector.
                   → Fig3-GC-content_by-element.pdf

The original 2x2 layout included a fourth probe-length panel; per the
2026-05-21 revision it was removed since the same information lives in
Table 1 (panel composition table).

Methodology rationale:
- UMAP features: overlapping non-canonical k-mer frequency vectors,
  k = k_min..k_max (default k=1..3 → 4 + 16 + 64 = 84-D).
- Non-canonical: each k-mer and its reverse complement are kept as
  distinct features. Probes carry biological orientation (promoter /
  splice_donor / splice_acceptor / polyA / etc.) and downstream gLM
  scoring is direction-specific, so collapsing revcomp would discard
  signal we care about. (Jellyfish itself only canonicalizes with
  explicit `-C`; default is off.)
- Hellinger transform (elementwise sqrt of k-mer frequencies) is the
  L2 embedding of probability vectors whose Euclidean distance is
  proportional to the Hellinger distance (Legendre & Gallagher 2001)
  and is what the main figure uses. It is a principled distance choice
  for composition histograms, NOT a GC correction: we empirically verified
  that |Pearson r(PC1, GC_content)| ≈ 0.997 with Hellinger and ≈ 0.996
  without, so the dominant variance direction is GC under either
  transform. What Hellinger does change is the explained variance
  (PC1+PC2 ≈ 57% with Hellinger vs ≈ 41% without on k=1..3, see panel
  composition table) and the shape of the residual structure after
  GC; it does not remove the GC axis.
- GC_content is NOT in the feature stack. The mononucleotide vector
  (k=1, 4-D) implicitly encodes GC; adding GC_content as an extra
  column would double-weight the GC direction in UMAP distances.
- evo2 (Methods § A.2.3) used k=1..6 on megabase whole genomes. We
  use k_max=3 by default because: (a) k=4 starts capturing short
  motifs (TATA, GTAA, CAAT), drifting away from "composition" into
  "short-motif content"; (b) on 156-1024 bp probes, k=4 dim=256
  density drops to ~0.72 mean nonzero (still workable), but k=5/6
  density drops to 0.36/0.12 — high-order counts become signal-poor.
  `--k-max 4` is supported as a sensitivity option and acknowledged
  as "composition + short motifs" rather than "composition alone".
- We do NOT use evo2's archaea×2 / eukaryote×3 species_group weighting
  trick. That is a label-aware operation that pulls categories apart by
  injecting species_group identity into the feature space; the main
  figure has to show what composition geometry looks like WITHOUT such
  a trick, so the later gLM-perspective figure can claim it reveals
  structure k-mer alone cannot. The flag was removed entirely so the
  composition figure cannot be accidentally label-amplified.

Usage:
  $PY scripts/figures/panel_composition_figure.py [--panel main_panel.parquet]
                                                  [--out figures]
                                                  [--k-min 1] [--k-max 3]
                                                  [--feature-transform {none,hellinger}]
                                                  [--n-neighbors 50] [--min-dist 0.3]
                                                  [--seed 42]

Main figure command (paper, locked):
  python scripts/figures/panel_composition_figure.py \\
      --k-min 1 --k-max 3 --feature-transform hellinger \\
      --n-neighbors 50 --min-dist 0.3
  → figures/Fig3-UMAP_kmer-composition_k1-3_hellinger_nn50_md0.3_by-category.pdf
  → figures/FigS1-UMAP_kmer-composition_k1-3_hellinger_nn50_md0.3_by-element.pdf
  → figures/Fig3-GC-content_by-element.pdf

Outputs (regenerated, not git-tracked): pdf only. Methodological
parameters (k range, transform, UMAP n_neighbors / min_dist) are
encoded in the UMAP filenames; the GC-content file has no UMAP suffix
since GC is not derived from the k-mer / UMAP stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the repo's publication style + 14-element color/order convention
# so this figure matches phase1_main_figure.py.
from scripts.figures.phase1_main_figure import (  # noqa: E402
    PALETTE,
    CLASS_ORDER,
    CLASS_COLORS,
    RCPARAMS,
)


# ─────────────────────────────── data ─────────────────────────────── #

# species_group palette — 4 categories. Each functional_element belongs
# to exactly one species_group (verified at panel build time).
SPECIES_GROUP_COLORS = {
    "Human": PALETTE["blue_main"],
    "Plant": PALETTE["green_3"],
    "Fungi": PALETTE["violet"],
    "Virus": PALETTE["red_strong"],
}
SPECIES_GROUP_ORDER = ("Human", "Plant", "Fungi", "Virus")


def _load_panel(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        sys.exit(f"panel parquet not found at {panel_path}")
    panel = pd.read_parquet(panel_path)
    # `sequence` is needed whenever k_min=1 or k_max>=4 (the on-the-fly
    # k-mer computation reads it). Default k=1..3 always hits k=1, so we
    # require it unconditionally — defensive against future panel parquets
    # that drop the column.
    required = {
        "probe_id", "sequence", "length_bp",
        "functional_element", "species_group",
        "GC_content", "dinuc_vec", "trinuc_vec",
    }
    missing = required - set(panel.columns)
    if missing:
        sys.exit(f"panel parquet missing columns: {sorted(missing)}")
    return panel


def _build_features(
    panel: pd.DataFrame,
    k_min: int = 1,
    k_max: int = 3,
    feature_transform: str = "none",
) -> tuple[np.ndarray, int]:
    """Stack overlapping non-canonical k-mer frequency vectors for
    k=k_min..k_max, then apply the requested `feature_transform`
    (z-score per column for 'none', or elementwise sqrt of per-row
    k-mer frequencies for 'hellinger'). Returns (X, total_dim).

    k=2 and k=3 are read directly from the panel parquet's precomputed
    `dinuc_vec` / `trinuc_vec` columns. k=1, k=4, and higher are computed
    on the fly from the sequence using the shared `_kmer_counts` helper in
    src.panel.composition.

    Dropping k=1 (k_min=2) removes the explicit mononucleotide
    features, but does NOT remove the GC signal: dinuc/trinuc marginal
    distributions still recover the mononucleotide frequencies (and
    thus GC content), so PC1 remains highly correlated with GC even
    at k_min=2 (empirically |r(PC1, GC)| ≈ 0.997 on this panel). Use
    --k-min 2 as a sanity check for "does removing explicit mononuc
    columns change the layout shape" rather than as a GC correction.

    Non-canonical: we keep each k-mer and its reverse complement as
    distinct features. Our probes carry biological orientation
    (promoter / splice donor / splice acceptor / polyA / etc.) and
    downstream gLM scoring is direction-specific, so collapsing revcomp
    would discard signal we care about.
    """
    from sklearn.preprocessing import StandardScaler
    from glmap.panel.composition import _kmer_counts

    if k_min < 1:
        raise ValueError(f"k_min must be >= 1, got {k_min}")
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) must be >= k_min ({k_min})")

    def _freq_for_k(seq: str, k: int) -> np.ndarray:
        """Non-canonical k-mer frequency for one sequence."""
        counts = _kmer_counts(seq, k)
        total = sum(counts) or 1
        return np.asarray(counts, dtype=np.float64) / total

    blocks: list[np.ndarray] = []
    dim_breakdown: list[tuple[int, int]] = []   # (k, dim)
    for k in range(k_min, k_max + 1):
        if k == 2 and "dinuc_vec" in panel.columns:
            arr = np.stack(panel["dinuc_vec"].to_numpy())
            dim_k = arr.shape[1]
        elif k == 3 and "trinuc_vec" in panel.columns:
            arr = np.stack(panel["trinuc_vec"].to_numpy())
            dim_k = arr.shape[1]
        else:
            arr = np.stack([
                _freq_for_k(s, k) for s in panel["sequence"].to_numpy()
            ])
            dim_k = arr.shape[1]
        blocks.append(arr)
        dim_breakdown.append((k, dim_k))

    print(
        "[panel_composition] k-mer feature stack: "
        + ", ".join(f"k={k} dim={d}" for k, d in dim_breakdown)
        + f"  (transform={feature_transform})",
        flush=True,
    )

    X = np.concatenate(blocks, axis=1)
    total_dim = X.shape[1]

    # Feature transform applied to the stacked frequency matrix.
    #   - "none": StandardScaler per column (default; preserves the historical
    #     pipeline). Each k-mer column becomes zero-mean unit-variance.
    #     Note this equalizes INDIVIDUAL COLUMNS, not k-blocks: under
    #     Euclidean distance the k=3 block (64 cols × unit var each)
    #     contributes ~4× the total variance of the k=2 block (16 cols × unit
    #     var each), so the higher-order k still dominates the geometry.
    #     Per-k-block equal weighting would need an extra 1/sqrt(dim_k)
    #     factor; we keep the standard per-column scaling for consistency
    #     with evo2's pipeline.
    #   - "hellinger": x' = sqrt(p_kmer), then skip z-scoring. The L2
    #     distance on sqrt(p) is proportional to the Hellinger distance
    #     (d_H = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2), the canonical
    #     distance between probability vectors up to that constant factor.
    #     ||sqrt(p_k)||^2 = sum(p_k) = 1 per k block, so stacking multiple
    #     k blocks is already balanced (each block has unit L2 norm);
    #     z-scoring would destroy this property and the distance would
    #     no longer be (proportional to) Hellinger.
    if feature_transform == "none":
        return StandardScaler().fit_transform(X), total_dim
    elif feature_transform == "hellinger":
        return np.sqrt(np.clip(X, 0.0, None)), total_dim
    else:
        raise ValueError(
            f"unknown feature_transform={feature_transform!r}; "
            "expected 'none' or 'hellinger'"
        )


def _run_umap(X: np.ndarray, seed: int, n_neighbors: int = 15,
              min_dist: float = 0.5) -> np.ndarray:
    """UMAP 2D projection.

    Defaults n_neighbors=15, min_dist=0.5 match evo2 Methods § A.2.3. For
    composition-style panel figures with ~10k probes, n_neighbors ∈ [30, 50]
    and min_dist ∈ [0.3, 0.5] tend to give the most stable, interpretable
    layout — low n_neighbors (≤ 15) and low min_dist (≤ 0.1) bias the layout
    toward visually "clustered" structures that can be artefacts of the
    hyperparameters rather than real composition structure.
    """
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(X)


# ───────────────────────────── panels ─────────────────────────────── #


def _panel_umap_by_category(ax, embed: np.ndarray, panel: pd.DataFrame) -> None:
    """Fig 3 (a): UMAP scatter colored by biological category (4 categories).

    Legend sits inside the axes (4-category layout fits comfortably).
    """
    for sg in SPECIES_GROUP_ORDER:
        m = (panel["species_group"] == sg).to_numpy()
        ax.scatter(
            embed[m, 0], embed[m, 1],
            c=SPECIES_GROUP_COLORS[sg], s=4, alpha=0.55,
            linewidths=0, label=f"{sg} (n={int(m.sum())})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of k-mer composition of DNA Sequences")
    leg = ax.legend(
        loc="best", markerscale=2.5, fontsize=9,
        handlelength=1.0, borderaxespad=0.3,
        title="Biological category", title_fontsize=9,
    )
    # Make legend markers more opaque so the colors read clearly.
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def _panel_umap_by_element(ax, embed: np.ndarray, panel: pd.DataFrame) -> None:
    """Fig S1: same UMAP, colored by functional_element (14 categories).

    The axes itself is identical in shape and figsize to Fig 3 (a);
    the 14-class legend is placed OUTSIDE the axes on the right via
    ``bbox_to_anchor=(1.02, 0.5)``. Combined with ``bbox_inches='tight'``
    at savefig time, this makes the saved PDF wider than Fig 3 (a)
    without changing the UMAP plot area itself.
    """
    for elem in CLASS_ORDER:
        m = (panel["functional_element"] == elem).to_numpy()
        if not m.any():
            continue
        ax.scatter(
            embed[m, 0], embed[m, 1],
            c=CLASS_COLORS[elem], s=4, alpha=0.55,
            linewidths=0, label=elem,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of k-mer composition of DNA Sequences")
    leg = ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        markerscale=2.5, fontsize=8,
        handlelength=1.0, borderaxespad=0.3,
        title="Functional element", title_fontsize=8.5,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def _panel_distribution(
    ax, panel: pd.DataFrame, column: str, ylabel: str, title: str,
    ylog: bool = False, show_xticklabels: bool = True,
    show_n_in_xlabel: bool = False,
) -> None:
    """Box plot + light strip overlay for one numeric column, with one box
    per functional_element in CLASS_ORDER. Boxes are colored to match
    panel (b); short separators between biological category blocks.

    `show_n_in_xlabel`: if True, append ``(n=N)`` to each x-tick label.
    Default is False (cleaner appearance for standalone figures) — the
    sample counts are already in Table 1 / the rug bar above.
    """
    elements = list(CLASS_ORDER)
    data = [panel.loc[panel["functional_element"] == e, column].to_numpy()
            for e in elements]
    box = ax.boxplot(
        data,
        positions=range(len(elements)),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"linewidth": 1.0, "color": "#444"},
        capprops={"linewidth": 1.0, "color": "#444"},
        boxprops={"linewidth": 0.8, "edgecolor": "#444"},
    )
    for patch, elem in zip(box["boxes"], elements):
        patch.set_facecolor(CLASS_COLORS[elem])
        patch.set_alpha(0.85)
    # Light strip — overlay individual points, jittered.
    rng = np.random.default_rng(42)
    for i, (e, vals) in enumerate(zip(elements, data)):
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=2.0, c="#222", alpha=0.18, linewidths=0)

    # biological category block separators: light vertical dashed lines
    # between the last element of one group and the first of the next.
    sg_of = {e: panel.loc[panel["functional_element"] == e, "species_group"].iloc[0]
             for e in elements}
    for i in range(1, len(elements)):
        if sg_of[elements[i]] != sg_of[elements[i - 1]]:
            ax.axvline(i - 0.5, color="#999", linewidth=0.7, linestyle="--", alpha=0.7)

    # x-axis: element name (+ optional n) — `show_n_in_xlabel` toggles the
    # per-element sample count suffix.
    if show_xticklabels:
        if show_n_in_xlabel:
            n_per_elem = [int((panel["functional_element"] == e).sum())
                          for e in elements]
            labels = [f"{e}\n(n={n})" for e, n in zip(elements, n_per_elem)]
        else:
            labels = list(elements)
        ax.set_xticks(range(len(elements)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks(range(len(elements)))
        ax.set_xticklabels([])

    ax.set_ylabel(ylabel)
    # Extra pad so the biological-category rug bar above the boxes
    # doesn't collide with the (centered) panel title.
    ax.set_title(title, pad=18)
    if ylog:
        ax.set_yscale("log")

    # species_group rug bar at the top of the axes, in axes-fraction y
    # but data x. Sits between the box data and the panel title, with a
    # white label inside the bar so the figure still reads in B&W print.
    block_ranges = []
    start = 0
    for i in range(1, len(elements) + 1):
        if i == len(elements) or sg_of[elements[i]] != sg_of[elements[start]]:
            block_ranges.append((start, i - 1, sg_of[elements[start]]))
            start = i
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    bar_y = 1.005
    bar_height = 0.05
    for lo, hi, sg in block_ranges:
        ax.add_patch(plt.Rectangle(
            (lo - 0.45, bar_y), (hi - lo + 0.9), bar_height,
            facecolor=SPECIES_GROUP_COLORS[sg], alpha=0.90,
            transform=trans, clip_on=False, linewidth=0,
        ))
        ax.text(
            (lo + hi) / 2, bar_y + bar_height / 2, sg,
            transform=trans, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold",
        )


# ─────────────────────────────── main ─────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path,
                   default=REPO_ROOT / "out_panel/main_panel.parquet",
                   help="Stage 2 main panel parquet.")
    p.add_argument("--out", dest="out_dir", type=Path,
                   default=REPO_ROOT / "figures",
                   help="Output directory.")
    p.add_argument("--k-min", type=int, default=1, dest="k_min",
                   help="Lowest k included in the k-mer feature stack "
                   "(default 1 = include mononucleotide). Set --k-min 2 to "
                   "drop the explicit mononuc columns; this is NOT a GC "
                   "correction because dinuc/trinuc marginals still recover "
                   "mononuc frequencies, so PC1 remains ≈ GC axis. Use it as "
                   "a 'does the layout shape change' sanity check, not for "
                   "GC removal.")
    p.add_argument("--k-max", type=int, default=3, dest="k_max",
                   help="Highest k to include in the k-mer feature stack "
                   "(default 3 → 4+16+64 = 84-D, pure composition). Set 4 "
                   "for 4+16+64+256 = 340-D sensitivity figure that includes "
                   "short motifs (TATA, GTAA, CAAT, ...).")
    p.add_argument("--feature-transform", type=str, default="none",
                   choices=["none", "hellinger"], dest="feature_transform",
                   help="Per-feature transform applied after k-mer frequency "
                   "computation. 'none' (default): StandardScaler z-score per "
                   "column. 'hellinger': sqrt(p_kmer), no further scaling — "
                   "the L2 embedding of probability distributions whose "
                   "Euclidean distance is proportional to the Hellinger "
                   "distance, a standard distance choice for compositional "
                   "histograms.")
    p.add_argument("--n-neighbors", type=int, default=15, dest="n_neighbors",
                   help="UMAP n_neighbors (default 15, evo2 default). "
                   "Composition-figure sweet spot is 30-50 for a 10k-probe "
                   "panel; low values (<15) bias toward visually clustered "
                   "layouts that can be artefacts.")
    p.add_argument("--min-dist", type=float, default=0.5, dest="min_dist",
                   help="UMAP min_dist (default 0.5, evo2 default). "
                   "Composition-figure sweet spot is 0.3-0.5; very low "
                   "(<0.1) packs points tightly into apparent clusters.")
    p.add_argument("--seed", type=int, default=42,
                   help="UMAP random_state for reproducibility.")
    # ── figsize knobs (per-figure) ── #
    p.add_argument("--umap-figsize", type=str, default="5.8,5.8",
                   dest="umap_figsize",
                   help='Inches, "W,H", for both UMAP figures (Fig 3a and '
                   'Fig S1). Both share this so their axes have identical '
                   'shape; Fig S1 then expands rightwards at savefig time '
                   'to include its outside legend. Default "5.8,5.8".')
    p.add_argument("--gc-figsize", type=str, default="7,6.5",
                   dest="gc_figsize",
                   help='Inches, "W,H", for the GC-content figure. '
                   '14 element boxes + rotated x-tick labels fit '
                   'comfortably at 7x6.5; reduce for a more compact figure, '
                   'increase if x-tick labels overlap. Default "7,6.5".')
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    """Parse 'W,H' or 'WxH' into (W, H) tuple of floats."""
    sep = "," if "," in s else "x"
    try:
        w, h = (float(x) for x in s.split(sep))
    except ValueError as e:
        raise ValueError(f"bad --figsize value {s!r}, expected 'W,H' (e.g. '6,5')") from e
    if w <= 0 or h <= 0:
        raise ValueError(f"figsize dimensions must be positive: {s!r}")
    return (w, h)


def _umap_filename(
    out_dir: Path, prefix: str, by: str,
    k_min: int, k_max: int, feature_transform: str,
    n_neighbors: int, min_dist: float,
) -> Path:
    """Filename for a UMAP-derived figure.

    Method + parameters are encoded so reviewer can read sensitivity-study
    variants from the filename alone:
      {prefix}-UMAP_kmer-composition_k{kmin}-{kmax}[_<transform>]_nn{nn}_md{md}_by-{by}.pdf

    Example: Fig3-UMAP_kmer-composition_k1-3_hellinger_nn50_md0.3_by-category.pdf
    """
    parts = [
        f"{prefix}-UMAP_kmer-composition",
        f"k{k_min}-{k_max}",
    ]
    if feature_transform != "none":
        parts.append(feature_transform)
    parts.append(f"nn{n_neighbors}")
    parts.append(f"md{min_dist:g}")
    parts.append(f"by-{by}")
    return out_dir / f"{'_'.join(parts)}.pdf"


def _gc_filename(out_dir: Path, prefix: str, feature_transform: str) -> Path:
    """Filename for the GC-content figure.

    The GC plot itself is independent of the k-mer / UMAP pipeline
    (it uses the precomputed ``GC_content`` column directly), but we
    tag the filename with the transform of the co-emitted UMAP figures
    so the 3 PDFs of a single ``panel_composition_figure.py`` run can
    be matched as a set without ambiguity across re-runs that vary
    ``--feature-transform``.

    Example:
      Fig3-GC-content_hellinger_by-element.pdf  (transform=hellinger)
      Fig3-GC-content_by-element.pdf            (transform=none, untagged)
    """
    parts = [f"{prefix}-GC-content"]
    if feature_transform != "none":
        parts.append(feature_transform)
    parts.append("by-element")
    return out_dir / f"{'_'.join(parts)}.pdf"


def main() -> None:
    args = parse_args()
    panel = _load_panel(args.panel)
    print(f"[panel_composition] loaded {len(panel)} probes from {args.panel}", flush=True)

    X, total_dim = _build_features(panel, k_min=args.k_min, k_max=args.k_max,
                                    feature_transform=args.feature_transform)
    print(
        f"[panel_composition] feature matrix: {X.shape} "
        f"(k={args.k_min}..{args.k_max}, transform={args.feature_transform}, "
        f"total_dim={total_dim})",
        flush=True,
    )

    print(f"[panel_composition] running UMAP "
          f"(n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, "
          f"seed={args.seed}) ...", flush=True)
    embed = _run_umap(X, seed=args.seed,
                      n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    print(f"[panel_composition] UMAP done: {embed.shape}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Per-figure figsize from CLI (--umap-figsize / --gc-figsize). Both
    # UMAP figures share UMAP_FIGSIZE so their axes have identical
    # natural shape; FigS1's outside legend is then absorbed at
    # savefig time via bbox_inches="tight".
    UMAP_FIGSIZE = _parse_figsize(args.umap_figsize)
    GC_FIGSIZE = _parse_figsize(args.gc_figsize)

    with plt.rc_context(RCPARAMS):
        # ─── Fig 3 (a): UMAP × biological category ────────────────── #
        fig_a, ax_a = plt.subplots(figsize=UMAP_FIGSIZE)
        _panel_umap_by_category(ax_a, embed, panel)
        out_a = _umap_filename(
            args.out_dir, prefix="Fig3", by="category",
            k_min=args.k_min, k_max=args.k_max,
            feature_transform=args.feature_transform,
            n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        )
        fig_a.savefig(out_a, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig_a)
        written.append(out_a)

        # ─── Fig S1: same UMAP × functional element ────────────────── #
        # Same figsize → identical axes shape; the 14-class legend
        # placed outside the axes makes the saved PDF wider (via
        # bbox_inches="tight").
        fig_b, ax_b = plt.subplots(figsize=UMAP_FIGSIZE)
        _panel_umap_by_element(ax_b, embed, panel)
        out_b = _umap_filename(
            args.out_dir, prefix="FigS1", by="element",
            k_min=args.k_min, k_max=args.k_max,
            feature_transform=args.feature_transform,
            n_neighbors=args.n_neighbors, min_dist=args.min_dist,
        )
        fig_b.savefig(out_b, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig_b)
        written.append(out_b)

        # ─── Fig 3 (c): GC content × functional element ────────────── #
        # Figsize controlled by --gc-figsize so the 14 box-plots + their
        # rotated x-tick labels can be tuned independently of the UMAP
        # figures. Default keeps the figure visually paired with
        # Fig 3 (a) while leaving extra room for x-tick labels.
        fig_c, ax_c = plt.subplots(figsize=GC_FIGSIZE)
        _panel_distribution(
            ax_c, panel, column="GC_content",
            ylabel="GC content",
            title="GC Content of DNA Sequences",
            show_xticklabels=True,
            show_n_in_xlabel=False,
        )
        out_c = _gc_filename(
            args.out_dir, prefix="Fig3",
            feature_transform=args.feature_transform,
        )
        fig_c.savefig(out_c, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig_c)
        written.append(out_c)

    for p in written:
        print(f"[done] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
