#!/usr/bin/env python3
"""Figure 2d: GLMap Q matrix heatmap, combined branches, row-grouped.

Renders the 123-model × 10,000-probe GLMap representation matrix Q
(clip + double-centered) as a heatmap. Rows are arranged branch-first
(AR models, then MLM models), then grouped by either ``family`` or
``organization`` (CLI: --row-group-by).  Within each group block, rows
are sorted alphabetically by hf_id.  The visual result reads top-to-
bottom as a branch-separated per-group "Q signature".

Layout
------
  ┌──────────┬─┬────────────────────────────────────┬──┐
  │  group   │G│        heatmap (123 × 10000)       │  │
  │  text    │r│   RdBu_r diverging, vmax = winsor  │cb│
  │  labels  │a│                                    │ar│
  │  (sized) │n│                                    │  │
  └──────────┴─┴────────────────────────────────────┴──┘
                ↓ element strip below the heatmap

Design choices (per scientific-figure-making house style)
- ``font.size = 13``, ``axes.linewidth = 1.5`` for compact paper figures
- top/right spines off, frameless legend
- 4 visual categories (group legend, group strip, heatmap, colorbar) only — no
  redundant strip headers ("br" / "fam"); the strips speak for themselves
- group colors are explained by a compact legend; singleton groups share
  the "Other" color
- block separator lines: low-alpha (0.25), thin (0.4 lw)
- element bottom strip: rotated labels with white outlines for
  readability over saturated colors
- winsorize Q at 1% / 99% so a handful of outlier cells don't saturate

Output
------
  figures/Fig2d-Q-heatmap_combined_by-{family,organization}.pdf

Usage
-----
  $PY scripts/figures/fig2d_q_heatmap.py                     # by family (default)
  $PY scripts/figures/fig2d_q_heatmap.py --row-group-by organization
  $PY scripts/figures/fig2d_q_heatmap.py --label-min-size 3 # hide tiny labels
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402
from scripts.figures.phase1_main_figure import RCPARAMS  # noqa: E402


# ───────────────────── color palettes ───────────────────── #


OTHER_GROUP_COLOR = "#B8B8B8"
FRAME_LW = 0.8
FRAME_COLOR = "#222"
OUTLIER_MODELS = [
    "ZhejiangLab/Genos-10B",
    "RaphaelMourad/Mistral-DNA-v1-138M-noncoding",
    "ZhejiangLab/Genos-10B-v2",
    "lingxusb/PlasmidGPT",
    "evo-design/evo-2-7b-8k-microviridae",
    "AIRI-Institute/gena-lm-bert-base-yeast",
    "ZhejiangLab/OneGenome-Rice",
    "GenerTeam/GENERanno-prokaryote-0.5b-base",
]
ELEMENT_CATEGORY = {
    "promoter": "Human",
    "enhancer": "Human",
    "splice_donor": "Human",
    "splice_acceptor": "Human",
    "chromatin_access": "Plant",
    "polyA": "Plant",
    "lncRNA": "Plant",
    "nascent_RNA": "Plant",
    "splicing_plant_donor": "Plant",
    "splicing_plant_acceptor": "Plant",
    "fungi_genome": "Fungi",
    "yeast_genome": "Fungi",
    "virus_species": "Virus",
    "virus_variants": "Virus",
}
CATEGORY_ELEMENT_ORDER = [
    # Human
    "promoter",
    "enhancer",
    "splice_donor",
    "splice_acceptor",
    # Plant
    "chromatin_access",
    "polyA",
    "lncRNA",
    "nascent_RNA",
    "splicing_plant_donor",
    "splicing_plant_acceptor",
    # Fungi
    "fungi_genome",
    "yeast_genome",
    # Virus
    "virus_species",
    "virus_variants",
]


def _short_outlier_label(hf_id: str) -> str:
    label = hf_id.split("/", 1)[-1]
    replacements = {
        "Mistral-DNA-v1-138M-noncoding": "Mistral-DNA noncoding",
        "evo-2-7b-8k-microviridae": "Evo2 microviridae",
        "gena-lm-bert-base-yeast": "GENA-LM yeast",
        "GENERanno-prokaryote-0.5b-base": "GENERanno prokaryote",
    }
    return replacements.get(label, label)


def _element_legend_label(element: str) -> str:
    category = ELEMENT_CATEGORY.get(element, "Other")
    return f"{category}: {element}"


def _ordered_elements(elements: np.ndarray) -> list[str]:
    present = set(elements.tolist())
    ordered = [e for e in CATEGORY_ELEMENT_ORDER if e in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _group_palette(labels: list[str]) -> dict[str, str]:
    """Stable color per group, sorted by descending count.  Combines
    tab20 + tab20b for up to 40 distinct hues.  Smallest groups still
    receive the shared Other color."""
    counts = Counter(labels)
    ordered = [g for g, n in counts.most_common() if n >= 2]
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    colors = {g: palette[i % len(palette)] for i, g in enumerate(ordered)}
    for g, n in counts.items():
        if n < 2:
            colors[g] = OTHER_GROUP_COLOR
    return colors


def _element_palette(unique_elements: list[str]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors)
    return {e: palette[i % len(palette)] for i, e in enumerate(unique_elements)}


def _show_frame(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LW)
        spine.set_color(FRAME_COLOR)


# ─────────────── row ordering by group blocks ──────────── #


def _block_row_order(
    glmap, group_field: str, within_group: str = "hfid_alpha",
) -> tuple[np.ndarray, list[tuple[str, int, int]], list[str]]:
    """Return (row_order, blocks, group_labels) where:
      row_order    : numpy array of length M with the new row index order
      blocks       : [(group_name, block_start, block_end), ...] indices
                     into the reordered rows (block_end exclusive)
      group_labels : the per-row group label, length M (already filled,
                     same order as glmap.hf_ids before reordering — for
                     palette assignment)

    ``group_field`` is either 'family' or 'organization'. Both are read
    from the audit metadata carried by the shared combined-Q loader.
    """
    if group_field == "family":
        raw = list(glmap.families)
    elif group_field == "organization":
        raw = list(glmap.organizations)
    else:
        raise ValueError(f"unknown group_field={group_field!r}")

    branch_order = {
        "ar_or_generative": 0,
        "mlm_or_encoder": 1,
    }
    block_counts = Counter(zip(glmap.branches, raw))
    # Outer order: AR first, then MLM; within each branch, sort blocks by
    # descending size, ties → alpha. Families that contain both branches
    # appear as separate branch-specific blocks.
    block_order = sorted(
        block_counts.keys(),
        key=lambda bg: (
            branch_order.get(bg[0], 99),
            -block_counts[bg],
            bg[1].lower(),
        ),
    )

    def _within_key(idx: int):
        if within_group == "hfid_alpha":
            return glmap.hf_ids[idx].lower()
        if within_group == "branch_then_hfid":
            br_rank = 0 if glmap.branches[idx] == "ar_or_generative" else 1
            return (br_rank, glmap.hf_ids[idx].lower())
        raise ValueError(f"unknown within_group={within_group!r}")

    row_order_parts: list[list[int]] = []
    blocks: list[tuple[str, int, int]] = []
    cursor = 0
    for branch, g in block_order:
        idx = [
            i for i, x in enumerate(raw)
            if x == g and glmap.branches[i] == branch
        ]
        idx.sort(key=_within_key)
        blocks.append((g, cursor, cursor + len(idx)))
        row_order_parts.append(idx)
        cursor += len(idx)
    row_order = np.array([i for part in row_order_parts for i in part])
    return row_order, blocks, raw


# ────────────────────────── main ────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--row-group-by", type=str, default="family",
                   choices=["family", "organization"],
                   help="What to group rows by. 'family' uses the "
                        "data/audits/models.json family field (curated "
                        "via models.md). 'organization' uses the org "
                        "prefix of the hf_id (e.g. 'InstaDeepAI/...' → "
                        "'InstaDeepAI').")
    p.add_argument("--clip-q", type=float, default=0.02,
                   help="GLMap pipeline clip quantile (default 0.02).")
    p.add_argument("--within-group-order", type=str, default="hfid_alpha",
                   choices=["hfid_alpha", "branch_then_hfid"],
                   help="Row order inside each group block.")
    p.add_argument("--winsorize-q", type=str, default="0.01,0.99",
                   help='"low,high" quantiles for color clipping.')
    p.add_argument("--figsize", type=str, default="15,5.3",
                   help='Inches, "W,H". Default "15,5.3".')
    p.add_argument("--out", dest="out_path", type=Path,
                   default=None,
                   help="Output PDF path. If omitted, defaults to "
                        "figures/Fig2d-Q-heatmap_combined_by-{group}.pdf.")
    p.add_argument("--annotate-outliers", action="store_true",
                   help="Add right-side labels for curated outlier models.")
    return p.parse_args()


def _parse_pair(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def main() -> None:
    args = parse_args()
    if args.out_path is None:
        args.out_path = (
            REPO_ROOT / "figures"
            / f"Fig2d-Q-heatmap_combined_by-{args.row_group_by}.pdf"
        )

    print(f"[fig2d] loading combined GLMap …", flush=True)
    glmap = load_combined_glmap(clip_q=args.clip_q)
    M, N = glmap.Q.shape
    print(f"[fig2d] L matrix: ({M}, {N})  clip_threshold = "
          f"{glmap.clip_threshold:.2f}", flush=True)

    # ── row order: blocks by group ── #
    row_order, blocks, group_per_row = _block_row_order(
        glmap, group_field=args.row_group_by,
        within_group=args.within_group_order,
    )
    print(f"[fig2d] {len(blocks)} {args.row_group_by} blocks", flush=True)
    for g, lo, hi in blocks:
        if hi - lo >= 3:
            print(f"  {g:30s} : {hi - lo:>3d} models  (rows {lo}-{hi-1})",
                  flush=True)

    # ── column order: by biological category, then functional_element ── #
    elements = np.array(glmap.functional_elements)
    unique_elems = _ordered_elements(elements)
    col_order_parts = []
    col_block_boundaries: list[int] = []
    category_blocks: list[tuple[str, int, int]] = []
    category_start = 0
    current_category: str | None = None
    for elem in unique_elems:
        idx = np.where(elements == elem)[0]
        category = ELEMENT_CATEGORY.get(elem, "Other")
        if current_category is None:
            current_category = category
        elif category != current_category:
            category_end = sum(len(p) for p in col_order_parts)
            category_blocks.append((current_category, category_start, category_end))
            category_start = category_end
            current_category = category
        col_order_parts.append(idx)
        col_block_boundaries.append(sum(len(p) for p in col_order_parts))
    if current_category is not None:
        category_blocks.append(
            (current_category, category_start, sum(len(p) for p in col_order_parts))
        )
    col_order = np.concatenate(col_order_parts)

    # ── reorder ── #
    Q_ord = glmap.Q[np.ix_(row_order, col_order)]
    groups_ord   = [group_per_row[i] for i in row_order]
    elements_ord = [glmap.functional_elements[i] for i in col_order]

    # ── winsorize for colormap ── #
    lo_q, hi_q = _parse_pair(args.winsorize_q)
    vlo = float(np.quantile(Q_ord, lo_q))
    vhi = float(np.quantile(Q_ord, hi_q))
    vmax = max(abs(vlo), abs(vhi))
    print(f"[fig2d] colormap range [{-vmax:.1f}, {+vmax:.1f}]  "
          f"(winsorized {lo_q*100:.1f}%/{hi_q*100:.1f}%)", flush=True)

    # ── palettes ── #
    group_palette = _group_palette(group_per_row)
    element_palette = _element_palette(unique_elems)
    group_counts = Counter(group_per_row)
    group_legend_keys: list[str] = []
    seen_legend_keys: set[str] = set()
    for g, _, _ in blocks:
        key = g if group_counts[g] >= 2 else "Other"
        if key not in seen_legend_keys:
            group_legend_keys.append(key)
            seen_legend_keys.add(key)

    # ────────────────────── layout ────────────────────── #
    figsize = _parse_figsize(args.figsize)
    if args.annotate_outliers:
        figsize = (max(figsize[0], 17.0), figsize[1])
    width_ratios = [0.125, 0.018, 1.0, 0.025]
    if args.annotate_outliers:
        width_ratios.append(0.19)
    ncols = len(width_ratios)
    local_rc = {
        **RCPARAMS,
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
    with plt.rc_context(local_rc):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            nrows=3, ncols=ncols,
            figure=fig,
            # cols: group legend, group strip, heatmap, cbar, optional labels
            width_ratios=width_ratios,
            height_ratios=[1.0, 0.055, 0.025],
            wspace=0.025, hspace=0.03,
            left=0.025, right=0.96, top=0.91, bottom=0.24,
        )
        ax_group   = fig.add_subplot(gs[0, 0])
        ax_group_strip = fig.add_subplot(gs[0, 1])
        ax_heatmap = fig.add_subplot(gs[0, 2])
        ax_cbar    = fig.add_subplot(gs[0, 3])
        ax_element = fig.add_subplot(gs[1, 2])
        ax_outliers = (
            fig.add_subplot(gs[0, 4], sharey=ax_heatmap)
            if args.annotate_outliers else None
        )

        # ── (a) group legend ── #
        ax_group.set_xticks([])
        ax_group.set_yticks([])
        for spine in ax_group.spines.values():
            spine.set_visible(False)
        group_handles = [
            Patch(
                facecolor=(
                    OTHER_GROUP_COLOR if g == "Other" else group_palette[g]
                ),
                edgecolor=FRAME_COLOR,
                linewidth=FRAME_LW,
                label=(
                    f"Other ({sum(n for n in group_counts.values() if n < 2)})"
                    if g == "Other" else f"{g} ({group_counts[g]})"
                ),
            )
            for g in group_legend_keys
        ]
        ax_group.legend(
            handles=group_handles,
            loc="center right",
            frameon=False,
            fontsize=9,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.10,
            borderaxespad=0.0,
        )

        # ── (b) group color strip ── #
        group_strip = np.array([
            [list(to_rgb(group_palette[g])) for g in groups_ord]
        ]).transpose(1, 0, 2)
        ax_group_strip.imshow(group_strip, aspect="auto",
                              extent=[0, 1, M - 0.5, -0.5])
        # Block separators inside strip (low alpha)
        for _, _, hi in blocks[:-1]:
            ax_group_strip.axhline(hi - 0.5, color="#FFF",
                                   linewidth=0.4, alpha=0.7)
        ax_group_strip.set_xticks([])
        ax_group_strip.set_yticks([])
        ax_group_strip.text(
            0.5, 1.012, "AR",
            transform=ax_group_strip.transAxes,
            ha="center", va="bottom",
            fontsize=8,
            clip_on=False,
        )
        ax_group_strip.text(
            0.5, -0.012, "MLM",
            transform=ax_group_strip.transAxes,
            ha="center", va="top",
            fontsize=8,
            clip_on=False,
        )
        _show_frame(ax_group_strip)

        # ── (c) main heatmap ── #
        im = ax_heatmap.imshow(
            Q_ord, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=+vmax,
            interpolation="nearest",
            extent=[0, N, M - 0.5, -0.5],
        )
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])
        _show_frame(ax_heatmap)

        if ax_outliers is not None:
            row_pos = {int(idx): pos for pos, idx in enumerate(row_order)}
            outliers = [
                (row_pos[i], glmap.hf_ids[i])
                for i in range(len(glmap.hf_ids))
                if glmap.hf_ids[i] in OUTLIER_MODELS
            ]
            outliers.sort()
            label_positions: list[float] = []
            min_gap = 3.0
            for y, _ in outliers:
                label_y = float(y)
                if label_positions and label_y - label_positions[-1] < min_gap:
                    label_y = label_positions[-1] + min_gap
                label_positions.append(label_y)
            overflow = (label_positions[-1] - (M - 1)) if label_positions else 0
            if overflow > 0:
                label_positions = [y - overflow for y in label_positions]
            for (y, hf_id), label_y in zip(outliers, label_positions):
                ax_heatmap.scatter(
                    [N * 0.997], [y],
                    marker=">", s=18, color="#111",
                    edgecolor="white", linewidth=0.3,
                    clip_on=False, zorder=6,
                )
                ax_outliers.plot(
                    [0.00, 0.08], [y, label_y],
                    color="#555", linewidth=0.55,
                    clip_on=False,
                )
                ax_outliers.text(
                    0.10, label_y,
                    _short_outlier_label(hf_id),
                    ha="left", va="center",
                    fontsize=7.5, color="#111",
                    clip_on=False,
                )
            ax_outliers.set_xlim(0, 1)
            ax_outliers.set_ylim(M - 0.5, -0.5)
            ax_outliers.set_xticks([])
            ax_outliers.set_yticks([])
            for spine in ax_outliers.spines.values():
                spine.set_visible(False)

        # ── (d) colorbar ── #
        cb = fig.colorbar(im, cax=ax_cbar)
        ax_cbar.tick_params(labelsize=10)
        cb.outline.set_linewidth(FRAME_LW)
        cb.outline.set_edgecolor(FRAME_COLOR)
        _show_frame(ax_cbar)

        # ── (e) element strip below the heatmap ── #
        elem_strip = np.array([
            [list(to_rgb(element_palette[e])) for e in elements_ord]
        ])
        ax_element.imshow(elem_strip, aspect="auto",
                          extent=[0, N, 0, 1])
        for bnd in col_block_boundaries[:-1]:
            ax_element.axvline(bnd, color="#FFF", linewidth=0.6, alpha=0.85)
        category_boundaries = sorted(
            {b for _, lo, hi in category_blocks for b in (lo, hi)}
        )
        for boundary in category_boundaries:
            ax_element.text(
                boundary, -0.4, "|",
                ha="center", va="top",
                fontsize=9, color="#333",
                clip_on=False,
            )
        for category, lo, hi in category_blocks:
            ax_element.text(
                (lo + hi) / 2, -0.4, category,
                ha="center", va="top",
                fontsize=8.5, color="#333",
                clip_on=False,
            )
        ax_element.set_xticks([])
        ax_element.set_yticks([])
        _show_frame(ax_element)

        # ── (f) title ── #
        fig.suptitle(
            "Visualization of Model Genotype-like Matrix",
            fontsize=14, y=0.955,
        )
        element_handles = [
            Patch(facecolor=element_palette[e], edgecolor=FRAME_COLOR,
                  linewidth=FRAME_LW, label=e)
            for e in unique_elems
        ]
        fig.legend(
            handles=element_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.105),
            ncol=7,
            fontsize=11,
            frameon=False,
            handlelength=1.2,
            handletextpad=0.35,
            columnspacing=0.9,
        )

        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_path, dpi=300, bbox_inches="tight",
                    pad_inches=0.12)
        plt.close(fig)
        print(f"[done] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
