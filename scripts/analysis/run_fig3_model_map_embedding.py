#!/usr/bin/env python3
"""Compute cached Fig3 model-map embeddings.

This script only computes coordinates. Figure styling is handled by
``scripts/figures/fig3_model_map.py`` so the map can be redrawn without
rerunning t-SNE / MDS.

Inputs
------
  out_phase1/scores/                         per-model probe scores
  out_phase2/matrices/auc_matrix.npy         downstream AUC matrix
  out_phase2/matrices/auc_matrix_meta.json   AUC row / column metadata
  data/audits/models.json                    model metadata

Outputs
-------
  out_phase2/model_map/fig3_embedding_V_tsne.csv
  out_phase2/model_map/fig3_embedding_Vd_tsne.csv
  out_phase2/model_map/fig3_embedding_D_mds.csv
  out_phase2/model_map/fig3_embedding_config.json

Usage
-----
  $PY scripts/analysis/run_fig3_model_map_embedding.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import (  # noqa: E402
    FAMILY_CANONICAL,
    load_combined_glmap,
)


TASK_LABEL = {
    "iDNA_ABF/5mC": "5mC",
    "enhancers/enhancer": "enhancer",
    "prom/promoter_tata_300bps": "promoter_tata_300bps",
    "mouse/mouse_TFBS_3": "mouse_TFBS_3",
    "EMP/Yeast_H4": "Yeast_H4",
    "iPro-WAEL/Promoter_Arabidopsis_TATA": "promoter_arabidopsis_TATA",
}

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
                   default=REPO_ROOT / "out_phase2/model_map")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=10.0)
    p.add_argument("--early-exaggeration", type=float, default=12.0)
    p.add_argument("--tsne-metric", type=str, default="euclidean")
    p.add_argument("--tsne-init", type=str, default="pca")
    p.add_argument("--tsne-learning-rate", type=str, default="auto")
    p.add_argument("--tsne-iter", type=int, default=2000)
    p.add_argument("--mds-iter", type=int, default=600)
    return p.parse_args()


def _safe_tsne(
    X: np.ndarray,
    *,
    perplexity: float,
    early_exaggeration: float,
    metric: str,
    init: str,
    learning_rate: str | float,
    seed: int,
    n_iter: int,
) -> np.ndarray:
    """Run t-SNE with compatibility across scikit-learn versions."""
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        metric=metric,
        init=init,
        random_state=seed,
        learning_rate=learning_rate,
        max_iter=n_iter,
    )
    try:
        return TSNE(**kwargs).fit_transform(X)
    except TypeError:
        kwargs["learning_rate"] = 200.0
        try:
            return TSNE(**kwargs).fit_transform(X)
        except TypeError:
            kwargs["n_iter"] = kwargs.pop("max_iter")
            return TSNE(**kwargs).fit_transform(X)


def _safe_mds(dist: np.ndarray, *, seed: int, max_iter: int) -> np.ndarray:
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=8,
        max_iter=max_iter,
    )
    return mds.fit_transform(dist)


def _parse_learning_rate(value: str) -> str | float:
    return "auto" if value == "auto" else float(value)


def _load_aligned_auc(
    model_ids: list[str],
    auc_matrix: Path,
    auc_meta: Path,
) -> tuple[pd.DataFrame, list[str]]:
    auc = np.load(auc_matrix)
    meta = json.loads(auc_meta.read_text())
    auc_model_ids = meta["model_ids"]
    task_ids = meta["task_ids"]
    if auc.shape != (len(auc_model_ids), len(task_ids)):
        raise ValueError(
            f"AUC shape {auc.shape} does not match metadata "
            f"({len(auc_model_ids)}, {len(task_ids)})"
        )
    if set(model_ids) != set(auc_model_ids):
        missing_auc = sorted(set(model_ids) - set(auc_model_ids))
        missing_glmap = sorted(set(auc_model_ids) - set(model_ids))
        raise ValueError(
            "Model ID mismatch between GLMap and AUC matrix. "
            f"missing in AUC={missing_auc[:5]}, missing in GLMap={missing_glmap[:5]}"
        )
    auc_index = {m: i for i, m in enumerate(auc_model_ids)}
    order = [auc_index[m] for m in model_ids]
    auc_aligned = auc[order, :]
    if not np.isfinite(auc_aligned).all():
        raise ValueError("AUC matrix contains non-finite values after alignment.")

    rows = {"mean_auc": auc_aligned.mean(axis=1)}
    for j, task_id in enumerate(task_ids):
        rows[f"auc_{TASK_LABEL.get(task_id, task_id)}"] = auc_aligned[:, j]
    return pd.DataFrame(rows), task_ids


def _base_frame(glmap, phenotype: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "model_id": glmap.hf_ids,
        "family": glmap.families,
        "branch": glmap.branches,
        "organization": glmap.organizations,
        "param_count": glmap.param_counts,
    }).join(phenotype)


def _write_embedding(
    out_path: Path,
    base: pd.DataFrame,
    coords: np.ndarray,
    *,
    embedding_input: str,
    method: str,
) -> None:
    df = base.copy()
    df.insert(1, "x", coords[:, 0])
    df.insert(2, "y", coords[:, 1])
    df.insert(3, "embedding_input", embedding_input)
    df.insert(4, "method", method)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    print("[fig3-embed] loading combined GLMap")
    glmap = load_combined_glmap(audit_path=args.audit)
    phenotype, task_ids = _load_aligned_auc(
        glmap.hf_ids, args.auc_matrix, args.auc_meta
    )
    base = _base_frame(glmap, phenotype)
    M = len(glmap.hf_ids)
    if M != 123:
        raise ValueError(f"Expected 123 aligned models, got {M}.")

    print(f"[fig3-embed] models={M}, probes={glmap.Q.shape[1]}")
    tsne_learning_rate = _parse_learning_rate(args.tsne_learning_rate)
    print("[fig3-embed] computing t-SNE on raw V")
    Z_v = _safe_tsne(
        glmap.L,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        metric=args.tsne_metric,
        init=args.tsne_init,
        learning_rate=tsne_learning_rate,
        seed=args.seed,
        n_iter=args.tsne_iter,
    )
    print("[fig3-embed] computing t-SNE on centered V_d")
    Z_vd = _safe_tsne(
        glmap.Q,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        metric=args.tsne_metric,
        init=args.tsne_init,
        learning_rate=tsne_learning_rate,
        seed=args.seed,
        n_iter=args.tsne_iter,
    )
    print("[fig3-embed] computing MDS on sqrt(D)")
    dist = np.sqrt(np.maximum(glmap.D, 0.0))
    Z_d = _safe_mds(dist, seed=args.seed, max_iter=args.mds_iter)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_embedding(
        args.out_dir / "fig3_embedding_V_tsne.csv",
        base, Z_v, embedding_input="V", method="tsne",
    )
    _write_embedding(
        args.out_dir / "fig3_embedding_Vd_tsne.csv",
        base, Z_vd, embedding_input="V_d", method="tsne",
    )
    _write_embedding(
        args.out_dir / "fig3_embedding_D_mds.csv",
        base, Z_d, embedding_input="D", method="mds",
    )

    config = {
        "seed": args.seed,
        "perplexity": args.perplexity,
        "early_exaggeration": args.early_exaggeration,
        "tsne_metric": args.tsne_metric,
        "tsne_init": args.tsne_init,
        "tsne_learning_rate": tsne_learning_rate,
        "tsne_iter": args.tsne_iter,
        "mds_iter": args.mds_iter,
        "n_models": M,
        "n_probes": int(glmap.Q.shape[1]),
        "task_ids": task_ids,
        "family_canonicalization": FAMILY_CANONICAL,
        "outputs": [
            "fig3_embedding_V_tsne.csv",
            "fig3_embedding_Vd_tsne.csv",
            "fig3_embedding_D_mds.csv",
        ],
    }
    (args.out_dir / "fig3_embedding_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(f"[done] wrote {(args.out_dir / 'fig3_embedding_config.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
