#!/usr/bin/env python3
"""Fig3: ModelMap-style GLMap model map.

This script only reads cached coordinates from
``scripts/analysis/run_fig3_model_map_embedding.py``. It does not rerun
t-SNE / MDS, so styling can be iterated quickly.

Main output
-----------
  figures/Fig3-model_map.pdf

Diagnostic output
-----------------
  figures/_preview/Fig3-embedding_comparison_V_Vd_D.pdf

Usage
-----
  $PY scripts/figures/fig3_model_map.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import canonical_family  # noqa: E402
from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402


OTHER_COLOR = "#d3d3d3"
BRANCH_COLOR = {
    "ar_or_generative": PALETTE["blue_main"],
    "mlm_or_encoder": PALETTE["red_strong"],
}
BRANCH_LABEL = {
    "ar_or_generative": "AR / generative",
    "mlm_or_encoder": "MLM / encoder",
}

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--coords", type=Path,
                   default=REPO_ROOT / "out_phase2/model_map/fig3_embedding_Vd_tsne.csv")
    p.add_argument("--coords-v", type=Path,
                   default=REPO_ROOT / "out_phase2/model_map/fig3_embedding_V_tsne.csv")
    p.add_argument("--coords-vd", type=Path,
                   default=REPO_ROOT / "out_phase2/model_map/fig3_embedding_Vd_tsne.csv")
    p.add_argument("--coords-d", type=Path,
                   default=REPO_ROOT / "out_phase2/model_map/fig3_embedding_D_mds.csv")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "figures/Fig3-model_map.pdf")
    p.add_argument("--preview-out", type=Path,
                   default=REPO_ROOT / "figures/_preview/Fig3-embedding_comparison_V_Vd_D.pdf")
    p.add_argument("--family-min-count", type=int, default=4)
    p.add_argument("--figsize", type=str, default="20.0,5.2")
    p.add_argument("--preview-figsize", type=str, default="14.5,4.7")
    return p.parse_args()


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


def _draw_branch_map(ax, df: pd.DataFrame) -> None:
    for branch, color in BRANCH_COLOR.items():
        sub = df[df["branch"] == branch]
        ax.scatter(
            sub["x"], sub["y"],
            s=28, color=color, alpha=0.84,
            edgecolors="white", linewidths=0.45,
            label=f"{BRANCH_LABEL[branch]} (n={len(sub)})",
            rasterized=True,
        )
    ax.set_title("Same map colored by training branch", loc="left", pad=8)
    _polish_map_axis(ax, df)
    ax.legend(frameon=False, fontsize=9, loc="best")


def _draw_auc_map(ax, df: pd.DataFrame):
    scatter = ax.scatter(
        df["x"], df["y"], c=df["mean_auc"],
        cmap="viridis", s=28, alpha=0.90,
        edgecolors="white", linewidths=0.45,
        vmin=float(df["mean_auc"].quantile(0.02)),
        vmax=float(df["mean_auc"].quantile(0.98)),
        rasterized=True,
    )
    ax.set_title("Same map colored by mean downstream AUC", loc="left", pad=8)
    _polish_map_axis(ax, df)
    return scatter


def _draw_param_map(ax, df: pd.DataFrame):
    log_params = np.log10(df["param_count"].to_numpy(dtype=float))
    scatter = ax.scatter(
        df["x"], df["y"], c=log_params,
        cmap="magma", s=28, alpha=0.90,
        edgecolors="white", linewidths=0.45,
        vmin=float(np.nanquantile(log_params, 0.02)),
        vmax=float(np.nanquantile(log_params, 0.98)),
        rasterized=True,
    )
    ax.set_title("Same map colored by model size", loc="left", pad=8)
    _polish_map_axis(ax, df)
    return scatter


def _load_coords(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/analysis/"
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


def _make_main_figure(df: pd.DataFrame, out: Path, figsize: tuple[float, float], min_count: int) -> None:
    family_colors, shown_families = _family_color_map(df["family"], min_count)
    local_rc = {
        **RCPARAMS,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    }
    with plt.rc_context(local_rc):
        fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
        _draw_family_map(
            axes[0], df, family_colors, shown_families,
            title="GLMap model map colored by family", legend=True,
        )
        _draw_branch_map(axes[1], df)
        sc = _draw_auc_map(axes[2], df)
        cbar = fig.colorbar(sc, ax=axes[2], fraction=0.045, pad=0.02)
        cbar.set_label("Mean downstream AUC", rotation=270, labelpad=14)
        sc_param = _draw_param_map(axes[3], df)
        cbar_param = fig.colorbar(sc_param, ax=axes[3], fraction=0.045, pad=0.02)
        cbar_param.set_label(r"log$_{10}$(parameters)", rotation=270, labelpad=14)

        for ax, label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
            ax.text(-0.12, 1.06, label, transform=ax.transAxes,
                    fontsize=13, va="top")
            ax.text(0.98, 0.03, "n = 123 models", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="#4A4A4A")
        fig.suptitle("GLMap model map of 123 genomic language models", fontsize=13)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(f"[done] wrote {out.relative_to(REPO_ROOT)}")


def _make_preview(
    paths: list[Path],
    out: Path,
    figsize: tuple[float, float],
    min_count: int,
) -> None:
    dfs = [_load_coords(p) for p in paths]
    family_colors, shown_families = _family_color_map(dfs[1]["family"], min_count)
    titles = [
        "raw V t-SNE by family",
        "centered V_d t-SNE by family",
        "sqrt(D) MDS by family",
    ]
    local_rc = {
        **RCPARAMS,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    }
    with plt.rc_context(local_rc):
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        for i, (ax, df, title) in enumerate(zip(axes, dfs, titles)):
            _draw_family_map(
                ax, df, family_colors, shown_families,
                title=title, legend=(i == 2),
            )
        fig.suptitle("Fig3 embedding comparison diagnostics", fontsize=12.5)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
    print(f"[done] wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    df = _load_coords(args.coords)
    _make_main_figure(
        df, args.out, _parse_figsize(args.figsize), args.family_min_count
    )
    _make_preview(
        [args.coords_v, args.coords_vd, args.coords_d],
        args.preview_out,
        _parse_figsize(args.preview_figsize),
        args.family_min_count,
    )


if __name__ == "__main__":
    main()
