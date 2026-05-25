#!/usr/bin/env python3
"""Generate LaTeX tables for Fig. 4b phenotype-prediction metrics.

Each output table fixes one GLMap matrix and one outer-CV regime, then
reports Pearson r and Spearman rho for the six downstream tasks plus
the six-task mean AUC.

Inputs
------
  out_phase2/phenotype_prediction/metrics_summary.csv

Outputs
-------
  tables/table4b_phenotype_prediction_V_kfold.tex
  tables/table4b_phenotype_prediction_Vd_kfold.tex
  tables/table4b_phenotype_prediction_V_family-groupkfold.tex
  tables/table4b_phenotype_prediction_Vd_family-groupkfold.tex

Usage
-----
  $PY scripts/tables/table4b_phenotype_prediction_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "out_phase2/phenotype_prediction/metrics_summary.csv"
TABLE_DIR = REPO_ROOT / "tables"

TASK_ORDER = [
    "iDNA_ABF/5mC",
    "enhancers/enhancer",
    "prom/promoter_tata_300bps",
    "mouse/mouse_TFBS_3",
    "EMP/Yeast_H4",
    "iPro-WAEL/Promoter_Arabidopsis_TATA",
    "__mean_auc__",
]

TASK_LABEL = {
    "iDNA_ABF/5mC": "5mC",
    "enhancers/enhancer": "enhancer",
    "prom/promoter_tata_300bps": "promoter_tata_300bps",
    "mouse/mouse_TFBS_3": "mouse_TFBS_3",
    "EMP/Yeast_H4": "Yeast_H4",
    "iPro-WAEL/Promoter_Arabidopsis_TATA": "promoter_arabidopsis_TATA",
    "__mean_auc__": "Mean of 6 tasks",
}

FEATURE_LABEL = {
    "V": r"raw $V$",
    "V_d": r"centered $V_d$",
}

SPLIT_LABEL = {
    "kfold": "random K-fold",
    "family_groupkfold": "family GroupKFold",
}

OUT_NAME = {
    ("V", "kfold"): "table4b_phenotype_prediction_V_kfold.tex",
    ("V_d", "kfold"): "table4b_phenotype_prediction_Vd_kfold.tex",
    ("V", "family_groupkfold"): "table4b_phenotype_prediction_V_family-groupkfold.tex",
    ("V_d", "family_groupkfold"): "table4b_phenotype_prediction_Vd_family-groupkfold.tex",
}


def _fmt(mean: float, std: float) -> str:
    return rf"${mean:.3f} \pm {std:.3f}$"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _task_header(task_id: str) -> str:
    label = _tex_escape(TASK_LABEL[task_id])
    if task_id == "__mean_auc__":
        return label
    return rf"\texttt{{{label}}}"


def _label_suffix(feature_set: str, split: str) -> str:
    feature = "Vd" if feature_set == "V_d" else "V"
    split_s = "family_groupkfold" if split == "family_groupkfold" else "kfold"
    return f"{feature}_{split_s}".lower()


def _make_table(df: pd.DataFrame, feature_set: str, split: str) -> str:
    sub = (
        df[(df["feature_set"] == feature_set) & (df["split"] == split)]
        .set_index("task_id")
    )
    missing = [t for t in TASK_ORDER if t not in sub.index]
    if missing:
        raise ValueError(f"Missing task rows for {feature_set}/{split}: {missing}")

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    caption = (
        "Phenotype-prediction performance using "
        f"{FEATURE_LABEL[feature_set]} signatures under {SPLIT_LABEL[split]}. "
        "RidgeCV predictions are evaluated across five random seeds; values "
        r"are mean $\pm$ standard deviation across seeds."
    )
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:fig4b_{_label_suffix(feature_set, split)}}}")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4.5pt}")
    lines.append(r"  \begin{tabular}{l c c c c c c c}")
    lines.append(r"    \toprule")
    header = " & ".join(["Metric"] + [_task_header(t) for t in TASK_ORDER])
    lines.append(f"    {header} \\\\")
    lines.append(r"    \midrule")
    pearson_cells = []
    spearman_cells = []
    for task_id in TASK_ORDER:
        row = sub.loc[task_id]
        pearson_cells.append(_fmt(float(row["pearson_mean"]), float(row["pearson_std"])))
        spearman_cells.append(_fmt(float(row["spearman_mean"]), float(row["spearman_std"])))
    lines.append("    Pearson $r$ & " + " & ".join(pearson_cells) + r" \\")
    lines.append(r"    Spearman $\rho$ & " + " & ".join(spearman_cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} does not exist. Run "
            "scripts/figures/fig4_phenotype_prediction.py first."
        )
    df = pd.read_csv(METRICS_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("# Table 4b phenotype-prediction metrics")
    for feature_set, split in OUT_NAME:
        tex = _make_table(df, feature_set, split)
        out_path = TABLE_DIR / OUT_NAME[(feature_set, split)]
        out_path.write_text(tex)
        print(f"[done] wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
