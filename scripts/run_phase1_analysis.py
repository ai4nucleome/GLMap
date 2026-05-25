#!/usr/bin/env python3
"""Phase 1 analysis: PCA / heterozygosity / GC-axis diagnostics on the
single-matrix protocol outputs from scripts/run_phase1_scoring.py.

Consumes:
    out_phase1/matrices/{Q_AR,Q_MLM}.npy
    out_phase1/matrices/matrix_metadata.json  (ordered model_ids + probe_ids)
    out_phase1/probes/main_panel.parquet      (probe_id, functional_element, GC_content)

Emits:
    out_phase1/analysis/
      pca/{matrix_name}/Z.npy V_T.npy sigma.json explained_variance.json
      diagnostics/heterozygosity_{matrix_name}.parquet
      diagnostics/gc_axis.json
      fst/marginal_fst_{axis}.parquet  (legacy exploratory diagnostic)
      cross_branch/spearman.json
      cross_branch/top_k_overlap.json
      reports/phase1_analysis.md

Phase 1 internal Q tracker mapped to outputs:
    Q1 representation shape    -> PCA explained variance + Z scatter coords
    Q2 design-axis correlation -> legacy marginal F_ST per axis. With the current
                                  DEFAULT_MODELS (8 AR + 5 MLM = 13), the
                                  permutation null is non-degenerate but
                                  per-group counts are still small; these
                                  values are retained only as exploratory
                                  phase-1 diagnostics.
    Q3 functional sensitivity  -> per-probe heterozygosity grouped by
                                  functional_element (14 element_ids)
    Q4 cross-species            -> NOT in phase 1. The species_group axis on
                                  the Stage 2 panel (Human / Plant / Fungi /
                                  Virus) is too coarse for evolutionary
                                  claims; the per-species centroid analysis
                                  is the role of phase 3 (see phase_3.md
                                  § Q4 分析) on the multi-species panel.

Per phase_1.md the legacy marginal-FST permutation null requires >= 6 models
per branch with non-degenerate axis splits; AR (8 models) clears that bar,
MLM (5 models) does not.

Usage:
    python scripts/run_phase1_analysis.py [--in out_phase1] [--out out_phase1/analysis]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.analysis.gc_axis import gc_axis_diagnostic  # noqa: E402
from glmap.analysis.heterozygosity import per_probe_heterozygosity  # noqa: E402
from glmap.analysis.marginal_fst import marginal_fst  # noqa: E402
from glmap.analysis.pca import pca_models  # noqa: E402

try:
    from scipy.stats import spearmanr
except Exception as exc:  # pragma: no cover - hard dep at this layer
    raise SystemExit(f"scipy required for cross-branch Spearman: {exc}")


# ------------------------- model-label inference ---------------------------- #


def _infer_param_scale(hf_id: str) -> str:
    """small / medium / large bucket by hf_id substring."""
    name = hf_id.lower()
    if "-v1-1m-" in name or "-v2-50m-" in name:
        return "small"
    if "-v1-17m-" in name or "-v2-100m-" in name:
        return "medium"
    if "-v1-138m-" in name or "-v2-250m-" in name:
        return "large"
    return "unknown"


def _infer_family(hf_id: str) -> str:
    name = hf_id.lower()
    if "mistral-dna" in name:
        return "Mistral-DNA"
    if "nucleotide-transformer-v2" in name:
        return "NT-v2"
    if "nucleotide-transformer" in name:
        return "NT"
    return "other"


# ------------------------- per-matrix analysis core ----------------------- #


def analyze_matrix(
    name: str,
    Q: np.ndarray,
    model_ids: list[str],
    probe_ids: list[str],
    panel: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Run PCA + heterozygosity + GC-axis on one Q matrix (per-branch).

    Q may contain NaN (mixed-modality models that legitimately can't score
    certain probe classes; the single-matrix protocol absorbs codon offset
    via double-centering and does not NaN-mask codon-on-noncoding cells).
    Rows / columns that are all-NaN are dropped before PCA; the dropped
    indices are recorded in the returned summary.
    """
    summary: dict = {
        "matrix_name": name,
        "input_shape": list(Q.shape),
        "input_nan_cells": int(np.isnan(Q).sum()),
        "row_model_ids": list(model_ids),
        "col_probe_ids_count": len(probe_ids),
    }

    # Drop rows / cols that are entirely NaN; PCA cannot handle them.
    finite_row_mask = ~np.isnan(Q).all(axis=1)
    finite_col_mask = ~np.isnan(Q).all(axis=0)
    Q_clean = Q[finite_row_mask][:, finite_col_mask]
    kept_models = [m for m, k in zip(model_ids, finite_row_mask) if k]
    kept_probes = [p for p, k in zip(probe_ids, finite_col_mask) if k]
    summary["pca_shape"] = list(Q_clean.shape)
    summary["dropped_models_all_nan"] = [
        m for m, k in zip(model_ids, finite_row_mask) if not k
    ]
    summary["dropped_probes_all_nan_count"] = int((~finite_col_mask).sum())

    if Q_clean.size == 0 or Q_clean.shape[0] < 2 or Q_clean.shape[1] < 2:
        summary["status"] = "skipped_insufficient_data"
        return summary

    # Replace any residual NaN (mixed-modality models may legitimately
    # have per-cell NaN on probe classes they cannot score) with the
    # per-column mean — minimally invasive imputation just for PCA.
    # Heterozygosity / F_ST use the raw Q_clean (NaN-safe via numpy.nan*).
    col_means = np.nanmean(Q_clean, axis=0)
    Q_imp = Q_clean.copy()
    inds = np.where(np.isnan(Q_imp))
    Q_imp[inds] = np.take(col_means, inds[1])

    emb = pca_models(Q_imp)
    pca_out = out_dir / "pca" / name
    pca_out.mkdir(parents=True, exist_ok=True)
    np.save(pca_out / "Z.npy", emb.Z)
    np.save(pca_out / "V_T.npy", emb.V_T)
    with (pca_out / "sigma.json").open("w") as h:
        json.dump({
            "sigma": emb.sigma.tolist(),
            "row_model_ids": kept_models,
        }, h, indent=2)
    with (pca_out / "explained_variance.json").open("w") as h:
        json.dump({
            "explained_variance": emb.explained_variance.tolist(),
            "row_model_ids": kept_models,
            "col_probe_ids_count": len(kept_probes),
        }, h, indent=2)

    if Q_clean.shape[0] >= 2:
        het = per_probe_heterozygosity(Q_clean)
    else:
        het = None

    # GC-axis diagnostic uses kept_probes -> panel GC.
    panel_indexed = panel.set_index("probe_id")
    gc_kept = panel_indexed.loc[kept_probes, "GC_content"].to_numpy(dtype=np.float64)
    gc_rep = gc_axis_diagnostic(emb.V_T, gc_kept, threshold=0.7)

    diag_out = out_dir / "diagnostics"
    diag_out.mkdir(parents=True, exist_ok=True)
    if het is not None:
        pd.DataFrame({
            "probe_id": kept_probes,
            "var_per_probe": het.var_per_probe,
            "mean_per_probe": het.mean_per_probe,
        }).merge(
            panel[["probe_id", "functional_element", "GC_content"]], on="probe_id"
        ).to_parquet(diag_out / f"heterozygosity_{name}.parquet", index=False)

    summary["status"] = "ok"
    summary["pca_explained_variance"] = emb.explained_variance.tolist()
    summary["pca_sigma"] = emb.sigma.tolist()
    if het is not None:
        summary["heterozygosity_mean"] = float(het.var_per_probe.mean())
        summary["heterozygosity_max"] = float(het.var_per_probe.max())
    summary["gc_axis_abs_r"] = gc_rep.abs_r_per_pc.tolist()
    summary["gc_axis_is_dominated"] = gc_rep.is_gc_dominated.tolist()
    summary["gc_axis_first_non_gc_pc"] = gc_rep.first_non_gc_pc
    summary["kept_models"] = kept_models
    summary["kept_probes_count"] = len(kept_probes)
    return summary


# ----------------------------- main ------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path,
                   default=REPO_ROOT / "out_phase1",
                   help="Phase 1 scoring root (contains matrices/, probes/, ...)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir; default <in-dir>/analysis")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    in_dir: Path = args.in_dir
    out: Path = args.out or (in_dir / "analysis")

    if not (in_dir / "matrices" / "matrix_metadata.json").exists():
        raise SystemExit(f"matrix_metadata.json not found under {in_dir / 'matrices'}")
    metadata = json.loads((in_dir / "matrices" / "matrix_metadata.json").read_text())
    panel = pd.read_parquet(in_dir / "probes" / "main_panel.parquet")
    print(f"[input] panel {len(panel)} probes; matrix_metadata covers "
          f"{sum(1 for k in metadata if k.startswith('Q_'))} Q matrices")

    matrix_summaries: dict[str, dict] = {}
    for mat_name, mat_spec in metadata.items():
        if not mat_name.startswith("Q_"):
            continue
        npy_path = in_dir / "matrices" / f"{mat_name}.npy"
        if not npy_path.exists():
            print(f"[warn] missing {npy_path}; skip")
            continue
        Q = np.load(npy_path)
        matrix_summaries[mat_name] = analyze_matrix(
            name=mat_name,
            Q=Q,
            model_ids=mat_spec["row_model_ids"],
            probe_ids=mat_spec["col_probe_ids"],
            panel=panel,
            out_dir=out,
        )
        s = matrix_summaries[mat_name]
        msg = (
            f"  [{mat_name}] pca_shape={s.get('pca_shape')} "
            f"explained_var={[round(v, 3) for v in s.get('pca_explained_variance', [])]}"
        )
        print(msg)

    # Marginal F_ST on Q_<branch> per branch (single-matrix protocol;
    # axis = param_scale + family). Q is double-centered so within/between
    # variance ratio behaves the same as on raw L, but with cleaner
    # numerical scaling.
    fst_out = out / "fst"
    fst_out.mkdir(parents=True, exist_ok=True)
    fst_records: list[dict] = []
    for branch_label in ("AR", "MLM"):
        key = f"Q_{branch_label}"
        if key not in matrix_summaries or matrix_summaries[key]["status"] != "ok":
            continue
        Q = np.load(in_dir / "matrices" / f"{key}.npy")
        row_ids = metadata[key]["row_model_ids"]
        finite_rows = ~np.isnan(Q).all(axis=1)
        Q = Q[finite_rows]
        row_ids = [r for r, k in zip(row_ids, finite_rows) if k]
        for axis_name, labels in [
            ("param_scale", [_infer_param_scale(r) for r in row_ids]),
            ("family", [_infer_family(r) for r in row_ids]),
        ]:
            try:
                rep = marginal_fst(Q, axis_labels=labels, axis_name=axis_name,
                                   n_permutations=999, seed=42)
            except Exception as exc:
                fst_records.append({
                    "branch": branch_label, "axis": axis_name,
                    "status": f"error: {exc}",
                })
                continue
            degenerate = len(set(labels)) <= 1 or all(
                labels.count(g) == 1 for g in set(labels)
            )
            fst_records.append({
                "branch": branch_label,
                "axis": axis_name,
                "groups": ", ".join(sorted(set(labels))),
                "labels": ", ".join(labels),
                "observed_fst": rep.observed_fst,
                "p_value": rep.p_value,
                "null_ci_95_lo": rep.null_ci_95[0],
                "null_ci_95_hi": rep.null_ci_95[1],
                "degenerate_split": degenerate,
            })
    pd.DataFrame(fst_records).to_parquet(
        fst_out / "marginal_fst.parquet", index=False
    )

    # Cross-branch Spearman on per-probe "difficulty" (mean across models
    # of raw sum_log_p). We MUST use L, not Q: Q is double-centered so
    # column means are ~0 by construction and a Spearman / top-k on them
    # is just noise. L (clipped raw sum_log_p) preserves per-probe
    # difficulty.
    cross_out = out / "cross_branch"
    cross_out.mkdir(parents=True, exist_ok=True)
    cross_summary: dict = {"status": "skipped"}
    L_ar_path = in_dir / "matrices" / "L_AR.npy"
    L_mlm_path = in_dir / "matrices" / "L_MLM.npy"
    if (
        all(f"Q_{b}" in matrix_summaries for b in ("AR", "MLM"))
        and L_ar_path.exists() and L_mlm_path.exists()
    ):
        L_ar = np.load(L_ar_path)
        L_mlm = np.load(L_mlm_path)
        # per-probe mean log p across models, NaN-safe. AR sum_log_p and
        # MLM stride PLL aren't on the same scale, but Spearman is rank-
        # based so it tolerates the offset.
        ar_means = np.nanmean(L_ar, axis=0)
        mlm_means = np.nanmean(L_mlm, axis=0)
        valid = np.isfinite(ar_means) & np.isfinite(mlm_means)
        if int(valid.sum()) >= 2:
            rho, pval = spearmanr(ar_means[valid], mlm_means[valid])
            # top-k HARDEST = lowest-likelihood (most negative) = smallest
            # sum_log_p. argsort ascending gives those at the front.
            k_options = [10, 50, 100]
            topk = {}
            ar_for_rank = np.where(valid, ar_means, np.nan)
            mlm_for_rank = np.where(valid, mlm_means, np.nan)
            for k in k_options:
                top_ar = set(np.argsort(ar_for_rank)[:k].tolist())
                top_mlm = set(np.argsort(mlm_for_rank)[:k].tolist())
                topk[str(k)] = {
                    "intersection": int(len(top_ar & top_mlm)),
                    "union": int(len(top_ar | top_mlm)),
                    "jaccard": (
                        len(top_ar & top_mlm) / max(len(top_ar | top_mlm), 1)
                    ),
                }
            cross_summary = {
                "status": "ok",
                "metric": "per_probe_mean_sum_log_p_on_L",
                "n_probes_compared": int(valid.sum()),
                "spearman_rho": float(rho),
                "spearman_p_value": float(pval),
                "top_k_overlap_hardest": topk,
            }
    (cross_out / "spearman.json").write_text(json.dumps(cross_summary, indent=2))

    # Final report
    _write_report(
        out_path=out / "reports" / "phase1_analysis.md",
        matrix_summaries=matrix_summaries,
        fst_records=fst_records,
        cross_summary=cross_summary,
        panel=panel,
    )
    print(f"\n[done] wrote analysis to {out}")


def _write_report(
    out_path: Path,
    matrix_summaries: dict[str, dict],
    fst_records: list[dict],
    cross_summary: dict,
    panel: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 1 Analysis Report")
    lines.append("")
    lines.append("Generated by `scripts/run_phase1_analysis.py`. PCA + "
                 "heterozygosity + GC-axis + marginal F_ST on the single-"
                 "matrix outputs (Q_AR / Q_MLM) of "
                 "`scripts/run_phase1_scoring.py`.")
    lines.append("")
    lines.append("## Q1 — representation shape (PCA explained variance)")
    lines.append("")
    pc_rows = []
    for name, s in sorted(matrix_summaries.items()):
        if s.get("status") != "ok":
            continue
        ev = s["pca_explained_variance"]
        pc_rows.append({
            "matrix": name,
            "shape": str(s["pca_shape"]),
            "PC1": round(ev[0], 4) if len(ev) > 0 else None,
            "PC2": round(ev[1], 4) if len(ev) > 1 else None,
            "PC3": round(ev[2], 4) if len(ev) > 2 else None,
        })
    lines.append(pd.DataFrame(pc_rows).to_markdown(index=False))
    lines.append("")
    lines.append("## Q3 — functional-class heterozygosity")
    lines.append("")
    for branch in ("AR", "MLM"):
        key = f"Q_{branch}"
        het_path = out_path.parent.parent / "diagnostics" / f"heterozygosity_{key}.parquet"
        if not het_path.exists():
            continue
        het = pd.read_parquet(het_path)
        agg = het.groupby("functional_element").agg(
            n_probes=("probe_id", "count"),
            mean_var=("var_per_probe", "mean"),
            max_var=("var_per_probe", "max"),
            mean_Q=("mean_per_probe", "mean"),
        ).round(4).sort_values("mean_Q")
        lines.append(f"### {branch}")
        lines.append("")
        lines.append(agg.to_markdown())
        lines.append("")
    lines.append("## GC-axis diagnostic")
    lines.append("")
    rows = []
    for name, s in sorted(matrix_summaries.items()):
        if s.get("status") != "ok":
            continue
        for i, (r, d) in enumerate(zip(s["gc_axis_abs_r"], s["gc_axis_is_dominated"])):
            rows.append({
                "matrix": name, "PC": i + 1,
                "abs_r_vs_GC": round(r, 3),
                "is_GC_dominated": d,
            })
    lines.append(pd.DataFrame(rows).to_markdown(index=False))
    lines.append("")
    lines.append("## Q2 — legacy design-axis marginal F_ST (CAVEAT: small M per branch)")
    lines.append("")
    if fst_records:
        df = pd.DataFrame(fst_records)
        cols_keep = [c for c in [
            "branch", "axis", "groups", "observed_fst", "p_value",
            "null_ci_95_lo", "null_ci_95_hi", "degenerate_split",
        ] if c in df.columns]
        lines.append(df[cols_keep].round(4).to_markdown(index=False))
    else:
        lines.append("(no F_ST records)")
    lines.append("")
    lines.append(
        "Note: DEFAULT_MODELS currently has 8 AR + 5 MLM models with "
        "imbalanced family / scale splits (AR: Mistral-DNA 1M/17M/138M + "
        "megaDNA + PlasmidGPT + GenSLM 25M/250M/2.5B; MLM: NT-v2 "
        "50M/100M/250M/500M + Agro-NT 1B). The permutation null "
        "is non-degenerate on AR (8 ≥ 6) but the MLM branch and the per-group "
        "counts within each axis split are still small; F_ST p-values are "
        "exploratory and are not part of the current main-paper claims."
    )
    lines.append("")
    lines.append("## Cross-branch (rank-based, phase_1.md § 跨分支分析)")
    lines.append("")
    if cross_summary.get("status") == "ok":
        lines.append(
            f"- Spearman ρ between Q_AR column means and Q_MLM column means: "
            f"**{cross_summary['spearman_rho']:.4f}**  "
            f"(p = {cross_summary['spearman_p_value']:.2e}, "
            f"n_probes = {cross_summary['n_probes_compared']})"
        )
        lines.append("- Top-k hardest probe overlap (Jaccard):")
        for k, v in cross_summary["top_k_overlap_hardest"].items():
            lines.append(f"  - k={k}: {v['intersection']}/{v['union']} = "
                         f"{v['jaccard']:.3f}")
    else:
        lines.append("(skipped)")
    lines.append("")
    lines.append("## Q4 — deferred to phase 3")
    lines.append("")
    lines.append(
        "Phase 1 does NOT run Q4 cross-species analysis. The Stage 2 panel's "
        "species_group axis (Human / Plant / Fungi / Virus) is recorded on "
        "every probe but is too coarse for evolutionary claims. The "
        "per-species centroid clustering vs ground-truth taxonomy lives in "
        "phase 3 (see `phase_3.md` § Q4 分析) on the multi-species panel."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- DEFAULT_MODELS has 8 AR + 5 MLM models spanning 3-4 architectural "
        "families (AR: Mistral-DNA 1M/17M/138M + megaDNA + PlasmidGPT + "
        "GenSLM 25M/250M/2.5B; MLM: NT-v2 50M/100M/250M/500M + Agro-NT 1B). "
        "Q1 PCA on AR is rank-7 (one degree lost to double-centering), MLM is rank-4; "
        "Q2 F_ST has a non-trivial null but per-group counts are still small."
    )
    lines.append(
        "- Phase 1 model selection is intentionally narrow: HyenaDNA is "
        "scored via the multi-env sweep (scripts/run_rerun_stability.py) "
        "rather than this script's DEFAULT_MODELS; DNABERT k=3..6 are "
        "excluded due to single-token overlap-mask leakage (phase_1.md "
        "supplement scope). The 123-model expanded set is the Stage 4 / "
        "phase 2 target."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
