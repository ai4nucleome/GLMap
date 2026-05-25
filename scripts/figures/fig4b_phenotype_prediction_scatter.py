#!/usr/bin/env python3
"""Fig4b: predicted-vs-observed downstream AUC scatter.

This plotting script intentionally does not rerun RidgeCV. It only reads
cached outputs from `fig4_phenotype_prediction.py`, so figure styling can be
iterated quickly.

Inputs
------
  out_phase2/phenotype_prediction/predictions.csv
  out_phase2/phenotype_prediction/config.json

Output
------
  figures/Fig4b-phenotype_prediction_scatter.pdf

Usage
-----
  $PY scripts/figures/fig4b_phenotype_prediction_scatter.py
  $PY scripts/figures/fig4b_phenotype_prediction_scatter.py --split family_groupkfold
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
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import RCPARAMS  # noqa: E402


TASK_LABEL = {
    "iDNA_ABF/5mC": "5mC",
    "enhancers/enhancer": "enhancer",
    "prom/promoter_tata_300bps": "promoter_tata_300bps",
    "mouse/mouse_TFBS_3": "mouse_TFBS_3",
    "EMP/Yeast_H4": "Yeast_H4",
    "iPro-WAEL/Promoter_Arabidopsis_TATA": "promoter_arabidopsis_TATA",
}

SPLIT_LABEL = {
    "kfold": "random K-fold",
    "family_groupkfold": "family GroupKFold",
}

FEATURE_LABEL = {
    "V": "raw V",
    "V_d": "centered V_d",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions", type=Path,
                   default=REPO_ROOT / "out_phase2/phenotype_prediction/predictions.csv")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "out_phase2/phenotype_prediction/config.json")
    p.add_argument("--split", choices=["kfold", "family_groupkfold"],
                   default="kfold")
    p.add_argument("--feature-set", choices=["V", "V_d"], default="V_d")
    p.add_argument("--figsize", type=str, default="6.4,5.6")
    p.add_argument("--out", dest="out_path", type=Path,
                   default=REPO_ROOT / "figures/Fig4b-phenotype_prediction_scatter.pdf")
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x, y)[0])
    if kind == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(kind)


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"{args.predictions} does not exist. Run "
            "scripts/figures/fig4_phenotype_prediction.py first."
        )

    predictions = pd.read_csv(args.predictions)
    config = json.loads(args.config.read_text()) if args.config.exists() else {}
    task_ids = config.get("task_ids") or sorted(predictions["task_id"].unique())

    primary = predictions[
        (predictions["split"] == args.split)
        & (predictions["feature_set"] == args.feature_set)
    ]
    if primary.empty:
        raise ValueError(
            f"No rows for split={args.split!r}, feature_set={args.feature_set!r}"
        )

    avg_pred = (
        primary.groupby(["model_id", "task_id"], as_index=False)[["y_true", "y_pred"]]
        .mean()
    )
    r = _safe_corr(avg_pred["y_pred"].to_numpy(), avg_pred["y_true"].to_numpy(), "pearson")
    rho = _safe_corr(avg_pred["y_pred"].to_numpy(), avg_pred["y_true"].to_numpy(), "spearman")

    local_rc = {
        **RCPARAMS,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
    }
    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=_parse_figsize(args.figsize))
        colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(task_ids)))
        for color, task_id in zip(colors, task_ids):
            df = avg_pred[avg_pred["task_id"] == task_id]
            if df.empty:
                continue
            ax.scatter(
                df["y_pred"], df["y_true"],
                s=28, alpha=0.78, edgecolors="white", linewidths=0.4,
                color=color, label=TASK_LABEL.get(task_id, task_id),
                rasterized=True,
            )

        lo = float(min(avg_pred["y_pred"].min(), avg_pred["y_true"].min()))
        hi = float(max(avg_pred["y_pred"].max(), avg_pred["y_true"].max()))
        pad = 0.035 * (hi - lo if hi > lo else 1.0)
        ax.plot(
            [lo - pad, hi + pad], [lo - pad, hi + pad],
            color="#333333", linestyle="--", linewidth=1.0, alpha=0.75,
        )
        ax.text(
            0.04, 0.96,
            f"r = {r:.2f}, rho = {rho:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
        )
        ax.set_xlabel("Predicted AUC")
        ax.set_ylabel("Observed AUC")
        ax.set_title("Out-of-fold predicted vs observed AUC", loc="center", pad=8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.2)
        legend = ax.legend(
            frameon=True,
            ncol=2,
            loc="lower left",
            bbox_to_anchor=(0.30, 0.03),
            borderaxespad=0.0,
            columnspacing=1.8,
            handletextpad=0.5,
            labelspacing=0.45,
            fancybox=False,
        )
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.8)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.92)

        fig.tight_layout()
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    print(f"[done] wrote {args.out_path}")
    print(f"[summary] {SPLIT_LABEL[args.split]}, {FEATURE_LABEL[args.feature_set]}: "
          f"r={r:.3f}, rho={rho:.3f}")


if __name__ == "__main__":
    main()
