#!/usr/bin/env python3
"""Figure 4: predicting downstream phenotype from GLMap signatures.

This script follows the ModelMap Section 5 logic in the GLMap setting:
use per-model likelihood-response signatures as features and predict each
model's downstream linear-probe AUC profile with out-of-fold RidgeCV.

Inputs
------
  out_phase1/scores/                         raw per-probe sum_log_p files
  out_phase2/matrices/auc_matrix.npy         (123, 6) downstream AUC matrix
  out_phase2/matrices/auc_matrix_meta.json   model_ids + task_ids
  data/audits/models.json                    model metadata and family labels

Outputs
-------
  out_phase2/phenotype_prediction/predictions.csv
  out_phase2/phenotype_prediction/metrics_by_seed.csv
  out_phase2/phenotype_prediction/metrics_summary.csv
  out_phase2/phenotype_prediction/config.json

Usage
-----
  $PY scripts/figures/fig4_phenotype_prediction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402

FEATURE_LABEL = {
    "V": "raw V",
    "V_d": "centered V_d",
}

SPLIT_LABEL = {
    "kfold": "random K-fold",
    "family_groupkfold": "family GroupKFold",
}

@dataclass
class PredictionData:
    model_ids: list[str]
    task_ids: list[str]
    y: np.ndarray
    features: dict[str, np.ndarray]
    metadata: pd.DataFrame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--auc-matrix", type=Path,
                   default=REPO_ROOT / "out_phase2/matrices/auc_matrix.npy")
    p.add_argument("--auc-meta", type=Path,
                   default=REPO_ROOT / "out_phase2/matrices/auc_matrix_meta.json")
    p.add_argument("--audit", type=Path,
                   default=REPO_ROOT / "data/audits/models.json")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "out_phase2/phenotype_prediction")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--inner-cv", type=int, default=5,
                   help="Inner CV folds for RidgeCV alpha selection.")
    p.add_argument("--alphas", type=str, default="1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9")
    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x, y)[0])
    if kind == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(kind)


def _load_data(args: argparse.Namespace) -> PredictionData:
    glmap = load_combined_glmap(audit_path=args.audit)
    auc = np.load(args.auc_matrix)
    auc_meta = json.loads(args.auc_meta.read_text())
    auc_model_ids = auc_meta["model_ids"]
    task_ids = auc_meta["task_ids"]

    if auc.shape != (len(auc_model_ids), len(task_ids)):
        raise ValueError(
            f"AUC shape {auc.shape} does not match metadata "
            f"({len(auc_model_ids)}, {len(task_ids)})"
        )
    if set(glmap.hf_ids) != set(auc_model_ids):
        missing_auc = sorted(set(glmap.hf_ids) - set(auc_model_ids))
        missing_glmap = sorted(set(auc_model_ids) - set(glmap.hf_ids))
        raise ValueError(
            "Model ID mismatch between GLMap and AUC matrix. "
            f"missing in AUC={missing_auc[:5]}, missing in GLMap={missing_glmap[:5]}"
        )

    auc_index = {m: i for i, m in enumerate(auc_model_ids)}
    order = [auc_index[m] for m in glmap.hf_ids]
    y = auc[order, :]
    if not np.isfinite(y).all():
        raise ValueError("AUC matrix contains non-finite values after alignment.")

    audit = json.loads(args.audit.read_text())["models"]
    audit_by_id = {m["hf_id"]: m for m in audit}
    metadata_rows = []
    for hf_id, branch, family, org in zip(
        glmap.hf_ids, glmap.branches, glmap.families, glmap.organizations
    ):
        m = audit_by_id[hf_id]
        metadata_rows.append({
            "model_id": hf_id,
            "branch": branch,
            "family": family,
            "organization": org,
            "architecture": m.get("architecture", "unknown"),
            "training_paradigm": m.get("training_paradigm", "unknown"),
            "tokenizer_type": m.get("tokenizer_type", "unknown"),
            "score_protocol": m.get("score_protocol", "unknown"),
            "log10_param_count": np.log10(float(m.get("param_count") or 0) + 1.0),
            "log10_context_tokens": np.log10(float(m.get("context_tokens") or 0) + 1.0),
            "log10_context_bp": np.log10(float(m.get("context_bp") or 0) + 1.0),
        })
    metadata = pd.DataFrame(metadata_rows)

    features: dict[str, np.ndarray] = {
        "V": glmap.L,
        "V_d": glmap.Q,
    }
    print(f"[load] aligned {len(glmap.hf_ids)} models x {len(task_ids)} tasks")
    print(f"[load] feature matrices: V={glmap.L.shape}, V_d={glmap.Q.shape}")
    return PredictionData(
        model_ids=glmap.hf_ids,
        task_ids=task_ids,
        y=y,
        features=features,
        metadata=metadata,
    )


def _family_group_splits(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Seeded group K-fold with no family appearing in both train and test."""
    rng = np.random.default_rng(seed)
    unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} groups, got {len(unique_groups)}")

    # Largest groups are placed first; the random key breaks ties by seed.
    jitter = rng.random(len(unique_groups))
    group_order = sorted(
        range(len(unique_groups)),
        key=lambda i: (-counts[i], jitter[i]),
    )
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold: dict[int, int] = {}
    for gi in group_order:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[gi] = fold
        fold_sizes[fold] += int(counts[gi])

    splits = []
    all_idx = np.arange(len(groups))
    for fold in range(n_splits):
        test_mask = np.array([group_to_fold[gi] == fold for gi in inverse])
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        splits.append((train_idx, test_idx))
    return splits


def _make_splits(
    split: str,
    n: int,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if split == "kfold":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(np.arange(n)))
    if split == "family_groupkfold":
        return _family_group_splits(groups, n_splits=n_splits, seed=seed)
    raise ValueError(split)


def _ridge_model(inner_cv: int, alphas: list[float]) -> RidgeCV:
    cv = inner_cv if inner_cv and inner_cv > 1 else None
    return RidgeCV(alphas=np.asarray(alphas), cv=cv)


def _fit_predict(
    feature_set: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    inner_cv: int,
    alphas: list[float],
) -> tuple[np.ndarray, float]:
    reg = _ridge_model(inner_cv, alphas)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    alpha = float(reg.alpha_)
    return np.clip(pred, 0.0, 1.0), alpha


def _run_predictions(
    data: PredictionData,
    seeds: list[int],
    n_splits: int,
    inner_cv: int,
    alphas: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    metric_rows = []
    n = len(data.model_ids)
    groups = data.metadata["family"].to_numpy()

    for split in ["kfold", "family_groupkfold"]:
        for seed in seeds:
            splits = _make_splits(split, n, groups, n_splits=n_splits, seed=seed)
            for feature_set, x in data.features.items():
                print(f"[predict] split={split} seed={seed} feature={feature_set}")
                seed_pred_rows = []
                for fold, (train_idx, test_idx) in enumerate(splits):
                    x_train = x[train_idx]
                    x_test = x[test_idx]
                    for task_j, task_id in enumerate(data.task_ids):
                        y_train = data.y[train_idx, task_j]
                        y_test = data.y[test_idx, task_j]
                        y_pred, alpha = _fit_predict(
                            feature_set, x_train, y_train, x_test, inner_cv, alphas
                        )
                        for local_i, model_i in enumerate(test_idx):
                            row = {
                                "split": split,
                                "seed": seed,
                                "feature_set": feature_set,
                                "fold": fold,
                                "model_id": data.model_ids[model_i],
                                "task_id": task_id,
                                "y_true": float(y_test[local_i]),
                                "y_pred": float(y_pred[local_i]),
                                "alpha": alpha,
                            }
                            pred_rows.append(row)
                            seed_pred_rows.append(row)

                seed_pred = pd.DataFrame(seed_pred_rows)
                metric_rows.extend(
                    _compute_metrics(seed_pred, split, seed, feature_set, data.task_ids)
                )

    predictions = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = _summarize_metrics(metrics)
    return predictions, metrics, summary


def _metric_record(
    df: pd.DataFrame,
    split: str,
    seed: int,
    feature_set: str,
    task_id: str,
) -> dict[str, float | int | str]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    return {
        "split": split,
        "seed": seed,
        "feature_set": feature_set,
        "task_id": task_id,
        "n": int(len(df)),
        "pearson": _safe_corr(y_pred, y_true, "pearson"),
        "spearman": _safe_corr(y_pred, y_true, "spearman"),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _compute_metrics(
    seed_pred: pd.DataFrame,
    split: str,
    seed: int,
    feature_set: str,
    task_ids: list[str],
) -> list[dict[str, float | int | str]]:
    rows = []
    for task_id in task_ids:
        task_df = seed_pred[seed_pred["task_id"] == task_id]
        rows.append(_metric_record(task_df, split, seed, feature_set, task_id))

    mean_df = (
        seed_pred.groupby("model_id", as_index=False)[["y_true", "y_pred"]]
        .mean()
    )
    rows.append(_metric_record(mean_df, split, seed, feature_set, "__mean_auc__"))
    return rows


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = ["pearson", "spearman", "mae", "r2"]
    summary = (
        metrics.groupby(["split", "feature_set", "task_id"], as_index=False)[value_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def _mean_metric_for_plot(
    summary: pd.DataFrame,
    split: str,
    feature_set: str,
    metric: str = "spearman",
) -> float:
    rows = summary[
        (summary["split"] == split)
        & (summary["feature_set"] == feature_set)
        & (summary["task_id"] != "__mean_auc__")
    ]
    return float(rows[f"{metric}_mean"].mean())


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds)
    alphas = _parse_float_list(args.alphas)
    data = _load_data(args)
    predictions, metrics, summary = _run_predictions(
        data=data,
        seeds=seeds,
        n_splits=args.n_splits,
        inner_cv=args.inner_cv,
        alphas=alphas,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    metrics.to_csv(args.out_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(args.out_dir / "metrics_summary.csv", index=False)
    config = {
        "seeds": seeds,
        "n_splits": args.n_splits,
        "inner_cv": args.inner_cv,
        "alphas": alphas,
        "feature_sets": sorted(data.features.keys()),
        "splits": ["kfold", "family_groupkfold"],
        "model_ids": data.model_ids,
        "task_ids": data.task_ids,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[done] wrote outputs to {args.out_dir}")

    print("\n[summary] Spearman rho, mean across 6 tasks")
    for split in ["kfold", "family_groupkfold"]:
        for feat in ["V", "V_d"]:
            val = _mean_metric_for_plot(summary, split, feat)
            print(f"  {SPLIT_LABEL[split]:18s} {FEATURE_LABEL[feat]:12s} {val: .3f}")


if __name__ == "__main__":
    main()
