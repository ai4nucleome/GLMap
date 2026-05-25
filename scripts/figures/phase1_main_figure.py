#!/usr/bin/env python3
"""Phase 1 main figure: 2x2 publication panel summarizing the
representation map produced by run_phase1_scoring.py + run_phase1_analysis.py.

Panels:
  A) PCA Z scatter — AR (Mistral-DNA 1M/17M/138M) and MLM (NT-v2 50m/100m/
     250m) side-by-side inside the same grid cell.
  B) Per-element heterozygosity (mean Var_m Q[m,x]) across 14
     functional_element, paired bars for AR vs MLM. Model counts read
     from PCA metadata so labels track DEFAULT_MODELS over time.
  C) Cross-branch scatter — Q_AR column-mean (x) vs Q_MLM column-mean
     (y), 10,000 probes colored by functional_element, with the y=x reference
     line + Spearman ρ annotation.
  D) GC-axis diagnostic — |Pearson r vs probe GC_content| for PC1/PC2/PC3 on
     Q_AR and Q_MLM (single-matrix protocol per branch), with the phase_1.md
     0.7 GC-dominance threshold.

Output:
  figures/phase1_main.{pdf,png}
  These files are NOT tracked in git — they are regenerated locally from
  out_phase1/ when this script is run. The earlier tracked artefact was
  removed in commit 90091bd because it carried the pre-Stage-2-rebuild
  pilot's stale labels.

Usage:
  python scripts/figures/phase1_main_figure.py [--in out_phase1] [--out figures]
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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.analysis.gc_axis import gc_axis_diagnostic  # noqa: E402


# ------------------------------ style ------------------------------ #

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#7CA9D6",
    "red_strong": "#B64342",
    "red_secondary": "#E9A6A1",
    "red_light": "#F6CFCB",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "green_3": "#8BCF8B",
}

# 14 categorical colors for the 14 functional element_ids of the new panel
# (2026-05-18 redesign, see phase_1.md § 面板组成). Grouped by species_group
# so legend reads cleanly: Human (blues/oranges) → Plant (greens) → Fungi
# (purples) → Virus (reds). AR / MLM model palette stays separable.
CLASS_ORDER = (
    # Human (4 elements)
    "promoter",
    "enhancer",
    "splice_donor",
    "splice_acceptor",
    # Plant (6 elements)
    "chromatin_access",
    "polyA",
    "lncRNA",
    "nascent_RNA",
    "splicing_plant_donor",
    "splicing_plant_acceptor",
    # Fungi (2 elements)
    "yeast_genome",
    "fungi_genome",
    # Virus (2 elements)
    "virus_variants",
    "virus_species",
)
CLASS_COLORS = {
    # Human — blue family
    "promoter":                  "#1f77b4",
    "enhancer":                  "#aec7e8",
    "splice_donor":              "#ff7f0e",
    "splice_acceptor":           "#ffbb78",
    # Plant — green family
    "chromatin_access":          "#2ca02c",
    "polyA":                     "#98df8a",
    "lncRNA":                    "#8c564b",
    "nascent_RNA":               "#c49c94",
    "splicing_plant_donor":      "#bcbd22",
    "splicing_plant_acceptor":   "#dbdb8d",
    # Fungi — purple family
    "yeast_genome":              "#9467bd",
    "fungi_genome":              "#c5b0d5",
    # Virus — red family
    "virus_variants":            "#d62728",
    "virus_species":             "#ff9896",
}

AR_MODEL_SHADES = [
    PALETTE["blue_light"], PALETTE["blue_secondary"], PALETTE["blue_main"],
    PALETTE["teal"], PALETTE["violet"],
]
MLM_MODEL_SHADES = [
    PALETTE["red_light"], PALETTE["red_secondary"], PALETTE["red_strong"],
    PALETTE["highlight"], PALETTE["green_3"],
]

RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,           # TrueType so editors can edit text
    "ps.fonttype": 42,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


# ----------------------------- data load ----------------------------- #


def _short_model_name(hf_id: str) -> str:
    """Compact label suitable for legends / scatter annotations."""
    name = hf_id.split("/")[-1]
    label_map = {
        "Mistral-DNA-v1-1M-hg38": "Mistral 1M",
        "Mistral-DNA-v1-17M-hg38": "Mistral 17M",
        "Mistral-DNA-v1-138M-hg38": "Mistral 138M",
        "megaDNA": "megaDNA 145M",
        "PlasmidGPT": "PlasmidGPT",
        "nucleotide-transformer-v2-50m-multi-species": "NT-v2 50M",
        "nucleotide-transformer-v2-100m-multi-species": "NT-v2 100M",
        "nucleotide-transformer-v2-250m-multi-species": "NT-v2 250M",
        "nucleotide-transformer-v2-500m-multi-species": "NT-v2 500M",
        "agro-nucleotide-transformer-1b": "Agro-NT 1B",
    }
    return label_map.get(name, name)


def _load_pca(in_dir: Path, matrix: str):
    Z = np.load(in_dir / "analysis" / "pca" / matrix / "Z.npy")
    V_T = np.load(in_dir / "analysis" / "pca" / matrix / "V_T.npy")
    meta = json.loads(
        (in_dir / "analysis" / "pca" / matrix / "explained_variance.json").read_text()
    )
    return Z, V_T, meta["explained_variance"], meta["row_model_ids"]


# ----------------------------- panel A ------------------------------ #


def _draw_pca_panel(ax, Z, ev, model_ids, shades, title, branch_label):
    """Scatter 3 model points in PC1/PC2 space, coloured by size."""
    labels = [_short_model_name(m) for m in model_ids]
    for i, (z_row, color, lab) in enumerate(zip(Z, shades, labels)):
        ax.scatter(
            z_row[0], z_row[1],
            s=180, color=color, edgecolor="black", linewidth=1.4, zorder=3, label=lab,
        )
        # Annotation a hair offset from the dot
        ax.annotate(
            lab, (z_row[0], z_row[1]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=9, color="#222",
        )
    ax.axhline(0, color="#bbb", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#bbb", linewidth=0.8, zorder=1)
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}%)")
    ax.set_title(f"{branch_label}", fontsize=11)


def panel_a(fig, gs_cell, in_dir: Path) -> None:
    # split the cell into 2 side-by-side sub-axes
    sub = gs_cell.subgridspec(1, 2, wspace=0.55)
    ax_ar = fig.add_subplot(sub[0, 0])
    ax_mlm = fig.add_subplot(sub[0, 1])

    Z_ar, _, ev_ar, mids_ar = _load_pca(in_dir, "Q_AR")
    Z_mlm, _, ev_mlm, mids_mlm = _load_pca(in_dir, "Q_MLM")
    _draw_pca_panel(ax_ar, Z_ar, ev_ar, mids_ar, AR_MODEL_SHADES, "PCA", "AR (Q)")
    _draw_pca_panel(ax_mlm, Z_mlm, ev_mlm, mids_mlm, MLM_MODEL_SHADES, "PCA", "MLM (Q)")
    ax_ar.text(
        -0.18, 1.07, "a", transform=ax_ar.transAxes, fontsize=14, fontweight="bold",
    )


# ----------------------------- panel B ------------------------------ #


def panel_b(ax, in_dir: Path) -> None:
    ar = pd.read_parquet(in_dir / "analysis" / "diagnostics" / "heterozygosity_Q_AR.parquet")
    mlm = pd.read_parquet(in_dir / "analysis" / "diagnostics" / "heterozygosity_Q_MLM.parquet")
    ar_mean = ar.groupby("functional_element")["var_per_probe"].mean().reindex(CLASS_ORDER)
    mlm_mean = mlm.groupby("functional_element")["var_per_probe"].mean().reindex(CLASS_ORDER)

    # Read model counts from PCA metadata so labels stay in sync with the actual
    # scored matrices (DEFAULT_MODELS has grown since the original pilot).
    n_ar_meta = json.loads(
        (in_dir / "analysis" / "pca" / "Q_AR" / "explained_variance.json").read_text()
    )
    n_mlm_meta = json.loads(
        (in_dir / "analysis" / "pca" / "Q_MLM" / "explained_variance.json").read_text()
    )
    n_ar = len(n_ar_meta["row_model_ids"])
    n_mlm = len(n_mlm_meta["row_model_ids"])

    x = np.arange(len(CLASS_ORDER))
    width = 0.38
    ax.bar(
        x - width / 2, ar_mean.values, width,
        label=f"AR ({n_ar} models)",
        color=PALETTE["blue_main"], edgecolor="black", linewidth=1.0,
    )
    ax.bar(
        x + width / 2, mlm_mean.values, width,
        label=f"MLM ({n_mlm} models)",
        color=PALETTE["red_strong"], edgecolor="black", linewidth=1.0,
    )
    ax.set_xticks(x)
    short_labels = [c.replace("human_", "").replace("_", " ") for c in CLASS_ORDER]
    ax.set_xticklabels(short_labels, rotation=35, ha="right")
    ax.set_ylabel(r"Mean Var$_m(Q[m,x])$ per probe")
    ax.set_title("Per-element heterozygosity (AR vs MLM)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(-0.13, 1.07, "b", transform=ax.transAxes, fontsize=14, fontweight="bold")


# ----------------------------- panel C ------------------------------ #


def panel_c(ax, in_dir: Path) -> None:
    Q_ar = np.load(in_dir / "matrices" / "Q_AR.npy")
    Q_mlm = np.load(in_dir / "matrices" / "Q_MLM.npy")
    panel = pd.read_parquet(in_dir / "probes" / "main_panel.parquet")
    cross = json.loads((in_dir / "analysis" / "cross_branch" / "spearman.json").read_text())

    n_ar = Q_ar.shape[0]
    n_mlm = Q_mlm.shape[0]
    ar_means = np.nanmean(Q_ar, axis=0)
    mlm_means = np.nanmean(Q_mlm, axis=0)
    valid = np.isfinite(ar_means) & np.isfinite(mlm_means)
    cls = panel["functional_element"].to_numpy()
    for fc in CLASS_ORDER:
        mask = valid & (cls == fc)
        if not mask.any():
            continue
        ax.scatter(
            ar_means[mask], mlm_means[mask],
            s=10, alpha=0.55, color=CLASS_COLORS[fc],
            edgecolor="none", label=fc.replace("human_", "").replace("_", " "),
        )

    # y = x reference (rank-based cross-branch agreement, not absolute)
    lo = min(np.nanmin(ar_means[valid]), np.nanmin(mlm_means[valid]))
    hi = max(np.nanmax(ar_means[valid]), np.nanmax(mlm_means[valid]))
    pad = 0.05 * (hi - lo)
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad],
        color="#999", linestyle="--", linewidth=1.0, zorder=1,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(rf"AR  $\overline{{Q}}$  (mean across {n_ar} AR models)")
    ax.set_ylabel(rf"MLM  $\overline{{Q}}$  (mean across {n_mlm} MLM models)")

    rho = cross["spearman_rho"]
    ax.text(
        0.04, 0.95,
        rf"Spearman ρ = {rho:.3f}   (n = {cross['n_probes_compared']})",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaa", linewidth=0.8),
    )
    ax.set_title("Cross-branch likelihood agreement per probe", fontsize=11)
    ax.text(-0.15, 1.07, "c", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Compact legend below the plot — 9 classes can crowd. Use 3 columns.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="lower right", fontsize=7, ncol=2, columnspacing=0.6, handletextpad=0.3,
        markerscale=2.0,
    )


# ----------------------------- panel D ------------------------------ #


def _compute_gc_abs_r(in_dir: Path, matrix: str) -> np.ndarray:
    V_T = np.load(in_dir / "analysis" / "pca" / matrix / "V_T.npy")
    meta = json.loads(
        (in_dir / "analysis" / "pca" / matrix / "explained_variance.json").read_text()
    )
    n_probes = meta["col_probe_ids_count"]
    panel = pd.read_parquet(in_dir / "probes" / "main_panel.parquet")
    gc = panel["GC_content"].to_numpy()[:V_T.shape[1]]
    rep = gc_axis_diagnostic(V_T, gc, threshold=0.7)
    return rep.abs_r_per_pc


def panel_d(ax, in_dir: Path) -> None:
    matrices = [
        ("Q_AR", PALETTE["blue_main"]),
        ("Q_MLM", PALETTE["red_strong"]),
    ]
    pcs = [1, 2, 3, 4]
    n_groups = len(pcs)
    n_series = len(matrices)
    x = np.arange(n_groups)
    width = 0.85 / n_series

    for i, (mat, color) in enumerate(matrices):
        abs_r = _compute_gc_abs_r(in_dir, mat)
        abs_r = list(abs_r) + [0.0] * (n_groups - len(abs_r))
        ax.bar(
            x + (i - (n_series - 1) / 2) * width, abs_r[:n_groups], width,
            label=mat.replace("R_", ""),
            color=color, edgecolor="black", linewidth=0.7,
        )

    ax.axhline(
        0.7, color="#888", linestyle="--", linewidth=1.0,
        label="GC-dominance threshold (0.7)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{p}" for p in pcs])
    ax.set_ylabel("| Pearson r vs probe GC |")
    ax.set_ylim(0, 1)
    ax.set_title("GC-axis diagnostic", fontsize=11)
    ax.legend(loc="upper right", fontsize=7, ncol=2, columnspacing=0.6, handletextpad=0.3)
    ax.text(-0.13, 1.07, "d", transform=ax.transAxes, fontsize=14, fontweight="bold")


# ------------------------------- main ------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, default=REPO_ROOT / "out_phase1")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument("--basename", type=str, default="phase1_main")
    args = p.parse_args()

    with plt.rc_context(RCPARAMS):
        fig = plt.figure(figsize=(13.5, 10.5))
        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)

        panel_a(fig, gs[0, 0], args.in_dir)
        ax_b = fig.add_subplot(gs[0, 1])
        panel_b(ax_b, args.in_dir)
        ax_c = fig.add_subplot(gs[1, 0])
        panel_c(ax_c, args.in_dir)
        ax_d = fig.add_subplot(gs[1, 1])
        panel_d(ax_d, args.in_dir)

        # Pull live model counts so the suptitle tracks DEFAULT_MODELS.
        n_ar_models = np.load(args.in_dir / "matrices" / "Q_AR.npy").shape[0]
        n_mlm_models = np.load(args.in_dir / "matrices" / "Q_MLM.npy").shape[0]
        fig.suptitle(
            "Phase 1 representation map "
            f"({n_ar_models} AR + {n_mlm_models} MLM models, "
            "10,000-probe biological panel)",
            fontsize=13, fontweight="bold", y=0.995,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(
                args.out_dir / f"{args.basename}.{ext}",
                dpi=300, bbox_inches="tight", pad_inches=0.06,
            )
        plt.close(fig)
    print(
        f"[done] wrote {args.out_dir}/{args.basename}.pdf "
        f"and {args.out_dir}/{args.basename}.png",
    )


if __name__ == "__main__":
    main()
