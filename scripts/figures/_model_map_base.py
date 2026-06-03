#!/usr/bin/env python3
"""Shared base for the Fig3 GLMap model-map panels.

Import-only (hence the ``_`` prefix): coordinate loading, family color
mapping, and the family-map drawing primitive reused by the three
standalone panel scripts:
  fig3a_model_map_family.py        (colored by family)
  fig3b_model_map_model_weight.py  (colored by model parameter count)
  fig3c_model_map_6task_mean.py    (colored by mean downstream AUC)

Coordinates come from scripts/model_map/run_fig3_model_map_embedding.py
(cached t-SNE / MDS), so the panels only restyle, never re-embed.

(Formerly fig3_model_map.py, which also drew a monolithic 1×4 combined
map + a V/V_d/D embedding-comparison preview. The map was split into the
three standalone panels above; only the shared primitives remain here.)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401  (kept on PATH for panel scripts' rc_context)
from matplotlib.patches import Patch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import canonical_family  # noqa: E402
from scripts.figures._figure_style import RCPARAMS  # noqa: E402,F401  (re-exported)


OTHER_COLOR = "#d3d3d3"

HIGHLIGHT_FAMILIES = [
    "PlantCaduceus",
    "GENERator",
    "NTv3",
    "GenomeOcean",
    "HyenaDNA",
    "Caduceus",
    "GENA-LM",
    "MutBERT",
    "Evo1",
    "Evo2",
]
OTHER_FAMILIES = {"NT"}

FAMILY_COLORS = [
    "#0F4D92", "#B64342", "#42949E", "#9A4D8E", "#D55E00",
    "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#8C564B", "#4E79A7", "#F28E2B", "#59A14F", "#E15759",
]


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def _family_color_map(families: pd.Series, min_count: int) -> tuple[dict[str, str], list[str]]:
    counts = Counter(families)
    highlight = [
        f for f in HIGHLIGHT_FAMILIES
        if f in counts and f not in OTHER_FAMILIES
    ]
    common = [
        f for f, n in counts.most_common()
        if n >= min_count and f not in highlight and f not in OTHER_FAMILIES
    ]
    shown = highlight + common
    colors = {fam: FAMILY_COLORS[i % len(FAMILY_COLORS)] for i, fam in enumerate(shown)}
    for fam in counts:
        colors.setdefault(fam, OTHER_COLOR)
    return colors, shown


def _set_equal_padding(ax, df: pd.DataFrame) -> None:
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = 0.08 * (dx if dx > 0 else 1.0)
    pad_y = 0.08 * (dy if dy > 0 else 1.0)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")


def _polish_map_axis(ax, df: pd.DataFrame, xlabel: str = "Map 1", ylabel: str = "Map 2") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_equal_padding(ax, df)
    ax.grid(True, linestyle=":", alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_family_map(
    ax,
    df: pd.DataFrame,
    family_colors: dict[str, str],
    shown_families: list[str],
    *,
    title: str,
    legend: bool,
    title_loc: str = "left",
) -> None:
    counts = Counter(df["family"])
    shown_set = set(shown_families)
    other = df[~df["family"].isin(shown_set)]
    if not other.empty:
        ax.scatter(
            other["x"], other["y"],
            s=18, color=OTHER_COLOR, alpha=0.68,
            edgecolors="white", linewidths=0.35, rasterized=True,
        )
    for fam in shown_families:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"], sub["y"],
            s=28, color=family_colors[fam], alpha=0.86,
            edgecolors="white", linewidths=0.45, label=fam, rasterized=True,
        )
    ax.set_title(title, loc=title_loc, pad=8)
    _polish_map_axis(ax, df)

    if legend:
        handles = [
            Patch(facecolor=family_colors[f], edgecolor="white", linewidth=0.4,
                  label=f"{f} (n={counts[f]})")
            for f in shown_families
        ]
        n_other = int((~df["family"].isin(shown_set)).sum())
        if n_other:
            handles.append(Patch(facecolor=OTHER_COLOR, label=f"Other (n={n_other})"))
        ax.legend(
            handles=handles, frameon=False, fontsize=7.5,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            handlelength=1.1, handletextpad=0.45, labelspacing=0.24,
            borderaxespad=0.0,
        )


def _load_coords(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/model_map/"
            "run_fig3_model_map_embedding.py first."
        )
    df = pd.read_csv(path)
    required = {
        "model_id", "x", "y", "family", "branch", "mean_auc", "param_count",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if len(df) != 123:
        raise ValueError(f"Expected 123 models in {path}, got {len(df)}")
    df["family"] = df["family"].map(canonical_family)
    if df["param_count"].isna().any() or (df["param_count"] <= 0).any():
        raise ValueError(f"{path} contains missing or non-positive param_count values")
    return df
