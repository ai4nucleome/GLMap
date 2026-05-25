#!/usr/bin/env python3
"""Figure 2e: Hierarchical clustering dendrogram of the 123 GLMap models.

Renders a horizontal dendrogram (leaves at the bottom, branches stacked
upward) computed by ``average``-linkage hierarchical clustering on the
combined-branch GLMap pairwise distance matrix D = pairwise_squared_distance(Q).

Each leaf is labeled by its hf_id and colored by family, keeping the
dendrogram itself visually clean as the companion to the Fig 2d heatmap.

The clustering uses the same ``D = ||Q[i] - Q[j]||^2`` as Fig 2d (with
``sqrt`` applied so the linkage operates on a true Euclidean metric).
``--linkage`` selects the linkage method; default ``average`` matches
Fig 2d.

Output
------
  figures/Fig2e-dendrogram_combined-123models.pdf

Usage
-----
  $PY scripts/figures/fig2e_glmap_dendrogram.py
  $PY scripts/figures/fig2e_glmap_dendrogram.py --linkage complete
  $PY scripts/figures/fig2e_glmap_dendrogram.py --orientation right
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402
from scripts.figures.phase1_main_figure import RCPARAMS  # noqa: E402


OTHER_COLOR = "#B8B8B8"
FRAME_COLOR = "#222"
FRAME_LW = 0.8
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
FAMILY_HIGHLIGHT_COLORS = {
    "PlantCaduceus": "#1f77b4",       # PlantCAD2 checkpoints in the audit.
    "Evo1": "#ff7f0e",
    "Evo2": "#2ca02c",
    "GENERator": "#d62728",
    "NTv3": "#9467bd",
    "GenomeOcean": "#8c564b",
    "HyenaDNA": "#e377c2",
    "Caduceus": "#bcbd22",
    "GENA-LM": "#17becf",
    "MutBERT": "#191970",
}


def _group_palette(labels: list[str], *, min_count: int = 2) -> dict[str, str]:
    """Stable palette used for metadata groups; singleton groups share Other."""
    counts = Counter(labels)
    ordered = [g for g, n in counts.most_common() if n >= min_count]
    palette = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )
    colors = {g: palette[i % len(palette)] for i, g in enumerate(ordered)}
    for g, n in counts.items():
        if n < min_count:
            colors[g] = OTHER_COLOR
    return colors


def _family_highlight_palette(labels: list[str]) -> dict[str, str]:
    """Only curated families are colored; all other families are gray."""
    return {
        label: FAMILY_HIGHLIGHT_COLORS.get(label, OTHER_COLOR)
        for label in set(labels)
    }


def _short(hf_id: str) -> str:
    """org/name → name. For dense leaf labels."""
    return hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clip-q", type=float, default=0.02,
                   help="GLMap pipeline clip quantile (default 0.02).")
    p.add_argument("--linkage", type=str, default="average",
                   choices=["average", "ward", "complete", "single"])
    p.add_argument("--orientation", type=str, default="top",
                   choices=["top", "bottom", "left", "right"],
                   help="Dendrogram orientation. 'top' (default) puts "
                        "leaves at the bottom and branches above; "
                        "'right' puts leaves on the right side which is "
                        "often easier to read with 123 long hf_id labels.")
    p.add_argument("--figsize", type=str, default="15,5.8",
                   help='Inches, "W,H". Default "15,5.8" (wide layout for '
                        '123 leaves; orientation=right may want "10,30").')
    p.add_argument("--out", dest="out_path", type=Path,
                   default=REPO_ROOT / "figures"
                   / "Fig2e-dendrogram_combined-123models.pdf",
                   help="Output PDF path.")
    p.add_argument("--label-by", type=str, default="family",
                   choices=["family", "organization"],
                   help="Metadata field used to color leaf labels.")
    p.add_argument("--exclude-outliers", action="store_true",
                   help="Exclude curated outlier models before clustering.")
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def main() -> None:
    args = parse_args()
    print(f"[fig2e] loading combined GLMap …", flush=True)
    glmap = load_combined_glmap(clip_q=args.clip_q)
    keep = np.ones(len(glmap.hf_ids), dtype=bool)
    if args.exclude_outliers:
        keep = np.array([hf not in OUTLIER_MODELS for hf in glmap.hf_ids])
        removed = [hf for hf in glmap.hf_ids if hf in OUTLIER_MODELS]
        print(f"[fig2e] excluding {len(removed)} outlier models", flush=True)
        for hf in removed:
            print(f"  - {hf}", flush=True)

    D = glmap.D[np.ix_(keep, keep)]
    hf_ids = [hf for hf, k in zip(glmap.hf_ids, keep) if k]
    families = [fam for fam, k in zip(glmap.families, keep) if k]
    organizations = [org for org, k in zip(glmap.organizations, keep) if k]
    M = len(hf_ids)

    # ── linkage on D ── #
    D_metric = np.sqrt(np.clip(D, 0, None))
    condensed = squareform(D_metric, checks=False)
    Z = linkage(condensed, method=args.linkage)
    print(f"[fig2e] linkage built ({args.linkage}, {M} leaves)", flush=True)

    label_groups = families if args.label_by == "family" else organizations
    label_palette = (
        _family_highlight_palette(label_groups)
        if args.label_by == "family"
        else _group_palette(label_groups, min_count=5)
    )
    figsize = _parse_figsize(args.figsize)
    local_rc = {
        **RCPARAMS,
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
    }
    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize)

        # ── Dendrogram ── #
        # Use neutral gray for branches; encode family, branch, and
        # organization on the leaf side so the tree topology remains readable.
        labels = [_short(h) for h in hf_ids]
        ddata = dendrogram(
            Z, ax=ax, orientation=args.orientation,
            labels=labels, leaf_font_size=6.4,
            color_threshold=0,
            link_color_func=lambda _k: "#555",
            no_labels=False,
        )
        leaf_order = ddata["leaves"]  # indices into hf_ids

        # ── Color leaf tick labels by family ── #
        if args.orientation in ("top", "bottom"):
            tick_labels = ax.get_xticklabels()
            for i, tl in enumerate(tick_labels):
                idx = leaf_order[i]
                group = label_groups[idx]
                tl.set_color(label_palette[group])
                tl.set_rotation(90)
                tl.set_ha("right")
                tl.set_va("top")
            ax.tick_params(axis="x", pad=4, length=0)
        else:
            tick_labels = ax.get_yticklabels()
            for i, tl in enumerate(tick_labels):
                idx = leaf_order[i]
                group = label_groups[idx]
                tl.set_color(label_palette[group])
            ax.tick_params(axis="y", length=0)

        # ── axis polish ── #
        ax.set_title(
            f"GLMap hierarchical clustering of {M} genomic language models",
            pad=12,
        )
        if args.orientation in ("top", "bottom"):
            ax.set_ylabel("GLMap distance")
        else:
            ax.set_xlabel("GLMap distance")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(FRAME_LW)
            spine.set_color(FRAME_COLOR)
        ax.grid(False)

        if args.orientation == "top":
            fig.subplots_adjust(left=0.055, right=0.99, top=0.91, bottom=0.28)
        else:
            fig.tight_layout(pad=1.2)
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_path, dpi=300, bbox_inches="tight",
                    pad_inches=0.08)
        plt.close(fig)
        print(f"[done] wrote {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
