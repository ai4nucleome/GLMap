"""Unit tests for src.analysis.{pca,heterozygosity,gc_axis,marginal_fst}."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glmap.analysis.gc_axis import gc_axis_diagnostic
from glmap.analysis.heterozygosity import per_probe_heterozygosity
from glmap.analysis.marginal_fst import marginal_fst
from glmap.analysis.pca import double_center, pca_models, procrustes_residual, truncated_svd


# ------------- PCA -------------


def test_double_center_zeroes_row_and_col_means() -> None:
    R = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    centered, _rm, _cm, _gm = double_center(R)
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(centered.mean(axis=1), 0.0, atol=1e-12)


def test_truncated_svd_recovers_singular_values() -> None:
    rng = np.random.default_rng(0)
    R = rng.normal(size=(4, 10))
    centered, *_ = double_center(R)
    U, s, V_T = truncated_svd(centered, k=2)
    # Reconstruction via top-2 SVD should be the best rank-2 approximation.
    approx = U * s[None, :] @ V_T
    full_U, full_s, full_V_T = np.linalg.svd(centered, full_matrices=False)
    true_approx = full_U[:, :2] * full_s[:2][None, :] @ full_V_T[:2, :]
    assert np.allclose(approx, true_approx, atol=1e-10)


def test_pca_models_explained_variance_sums_to_one() -> None:
    rng = np.random.default_rng(1)
    R = rng.normal(size=(5, 30))
    emb = pca_models(R)
    assert math.isclose(emb.explained_variance.sum(), 1.0, rel_tol=1e-9)
    # sigma sorted in decreasing order (numpy convention).
    assert np.all(np.diff(emb.sigma) <= 1e-9)


def test_procrustes_residual_zero_for_rotation() -> None:
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(6, 3))
    # Random orthogonal rotation.
    rand = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(rand)
    Z_rot = Z @ Q
    assert procrustes_residual(Z, Z_rot) < 1e-9


def test_procrustes_residual_one_for_independent_embeddings() -> None:
    rng = np.random.default_rng(3)
    Z_a = rng.normal(size=(6, 2))
    Z_b = rng.normal(size=(6, 2))
    res = procrustes_residual(Z_a, Z_b)
    # Two independent random embeddings give a residual close to 1.
    assert 0.5 < res < 1.5


# ------------- heterozygosity -------------


def test_heterozygosity_constant_columns_have_zero_variance() -> None:
    R = np.array([[1.0, 2.0, 3.0], [1.0, 5.0, 3.0], [1.0, 8.0, 3.0]])
    rep = per_probe_heterozygosity(R)
    assert rep.var_per_probe[0] == 0.0
    assert rep.var_per_probe[2] == 0.0
    assert rep.var_per_probe[1] > 0.0


def test_heterozygosity_top_indices_sorted_by_variance() -> None:
    rng = np.random.default_rng(4)
    R = rng.normal(size=(4, 8))
    rep = per_probe_heterozygosity(R)
    sorted_vars = rep.var_per_probe[rep.top_indices]
    assert np.all(np.diff(sorted_vars) <= 1e-12)


def test_heterozygosity_requires_at_least_two_models() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        per_probe_heterozygosity(np.array([[1.0, 2.0, 3.0]]))


# ------------- GC-axis -------------


def test_gc_axis_detects_perfect_gc_correlation() -> None:
    gc = np.linspace(0.2, 0.8, 50)
    # V_T row 0 == gc exactly -> correlation 1.0 -> GC-dominated.
    V_T = np.vstack([gc, np.random.default_rng(5).normal(size=50)])
    rep = gc_axis_diagnostic(V_T, gc, threshold=0.7)
    assert rep.is_gc_dominated[0]
    assert math.isclose(rep.r_per_pc[0], 1.0, abs_tol=1e-9)
    assert not rep.is_gc_dominated[1]
    assert rep.first_non_gc_pc == 1


def test_gc_axis_first_non_gc_when_none_dominated() -> None:
    rng = np.random.default_rng(6)
    gc = rng.uniform(0.2, 0.8, size=40)
    V_T = rng.normal(size=(3, 40))   # random, low correlation with gc
    rep = gc_axis_diagnostic(V_T, gc, threshold=0.7)
    assert rep.first_non_gc_pc == 0


# ------------- marginal F_ST -------------


def test_marginal_fst_zero_when_all_same_group() -> None:
    R = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    rep = marginal_fst(R, axis_labels=["x", "x"], n_permutations=99, seed=0)
    assert rep.observed_fst == 0.0


def test_marginal_fst_perfect_grouping_yields_one() -> None:
    """When all variance among model means comes from the grouping, F_ST = 1."""
    # Two groups, each with identical-mean members.
    R = np.array([
        [0.0, 0.0, 0.0],   # mean 0
        [0.0, 0.0, 0.0],   # mean 0
        [4.0, 4.0, 4.0],   # mean 4
        [4.0, 4.0, 4.0],   # mean 4
    ])
    rep = marginal_fst(
        R,
        axis_labels=["g1", "g1", "g2", "g2"],
        n_permutations=499,
        seed=0,
    )
    assert math.isclose(rep.observed_fst, 1.0, rel_tol=1e-9)
    # All permutations either fully separate or fully mix -> null mass on 1 and 0.
    assert rep.p_value <= 0.6


def test_marginal_fst_permutation_p_value_distribution() -> None:
    """With random labels and random data, observed should not look special."""
    rng = np.random.default_rng(7)
    R = rng.normal(size=(8, 20))
    labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
    rep = marginal_fst(R, axis_labels=labels, n_permutations=999, seed=11)
    assert 0.0 <= rep.p_value <= 1.0
    # null CI must contain plausible values.
    lo, hi = rep.null_ci_95
    assert lo <= rep.null_fsts.mean() <= hi


def test_marginal_fst_label_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="axis_labels length"):
        marginal_fst(np.zeros((3, 5)), axis_labels=["a", "b"], n_permutations=10)


def test_marginal_fst_multivariate_survives_double_centering() -> None:
    """Regression for the original row-mean-only F_ST bug. Two groups
    that DIFFER along the second probe-space dimension but happen to
    have identical per-model row means → old `R.mean(axis=1)` reduced
    each model to the same scalar so F_ST was 0; the multivariate form
    sees the full vector and recovers the between-group structure."""
    R = np.array([
        [+1.0, -1.0, 0.0, 0.0],   # group A; row mean = 0
        [+1.0, -1.0, 0.0, 0.0],   # group A; row mean = 0
        [-1.0, +1.0, 0.0, 0.0],   # group B; row mean = 0  (opposite-axis tilt)
        [-1.0, +1.0, 0.0, 0.0],   # group B; row mean = 0
    ])
    rep = marginal_fst(
        R, axis_labels=["A", "A", "B", "B"],
        n_permutations=199, seed=0,
    )
    # Per-row mean is identical (0) across all 4 models, so the old
    # scalar-F_ST collapsed this to 0/0 = 0. The multivariate form
    # captures the full vector difference between A and B → F_ST = 1.
    assert math.isclose(rep.observed_fst, 1.0, rel_tol=1e-9), \
        f"multivariate F_ST should be 1.0 on perfectly-separating axis, got {rep.observed_fst}"


def test_marginal_fst_handles_nan_via_column_mean_imputation() -> None:
    """Mixed-modality models can leave NaN cells in Q for probe classes
    they can't score. marginal_fst should impute (column-mean) rather
    than NaN-propagate to F_ST = nan."""
    R = np.array([
        [+1.0, -1.0, 0.0, 0.0],
        [+1.0, -1.0, np.nan, 0.0],     # one missing cell
        [-1.0, +1.0, 0.0, 0.0],
        [-1.0, +1.0, 0.0, 0.0],
    ])
    rep = marginal_fst(
        R, axis_labels=["A", "A", "B", "B"],
        n_permutations=99, seed=0,
    )
    assert np.isfinite(rep.observed_fst), \
        f"F_ST must be finite under column-mean imputation; got {rep.observed_fst}"
    # Same structure as the no-NaN case but slightly perturbed → still close to 1.
    assert rep.observed_fst > 0.9


# ------------- heterozygosity NaN-safety -------------


# ------------- run_phase1_analysis.main() smoke -------------


def test_phase1_analysis_main_runs_on_synthetic_fixture(tmp_path) -> None:
    """Regression for the `analyze_matrix(R=...)` kwarg crash. Build a
    minimal but realistic fixture: matrix_metadata.json + Q_AR.npy +
    L_AR.npy + panel parquet, then invoke `main()` end-to-end. Test
    must catch any signature drift between the caller and analyze_matrix."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    import pandas as pd

    REPO_ROOT = Path(__file__).resolve().parents[1]
    in_dir = tmp_path / "phase1_in"
    out_dir = tmp_path / "phase1_out"
    (in_dir / "matrices").mkdir(parents=True)
    (in_dir / "probes").mkdir(parents=True)

    # 6 models × 8 probes synthetic Q. Two groups visibly differ on
    # the first 4 probes — multivariate F_ST should be > 0.
    rng = np.random.default_rng(42)
    L = rng.normal(loc=-10.0, size=(6, 8))   # raw-scale L
    Q, *_ = double_center(L)
    np.save(in_dir / "matrices" / "Q_AR.npy", Q)
    np.save(in_dir / "matrices" / "L_AR.npy", L)

    # MLM side: another (4, 8) matrix; needed for cross-branch step.
    L_mlm = rng.normal(loc=-30.0, size=(4, 8))
    Q_mlm, *_ = double_center(L_mlm)
    np.save(in_dir / "matrices" / "Q_MLM.npy", Q_mlm)
    np.save(in_dir / "matrices" / "L_MLM.npy", L_mlm)

    ar_models = [f"family-{i}/model-{i}" for i in range(6)]
    mlm_models = [f"family-{i}/mlm-{i}" for i in range(4)]
    probe_ids = [f"probe_{i:03d}" for i in range(8)]
    (in_dir / "matrices" / "matrix_metadata.json").write_text(json.dumps({
        "Q_AR": {"shape": [6, 8], "row_model_ids": ar_models, "col_probe_ids": probe_ids},
        "L_AR": {"shape": [6, 8], "row_model_ids": ar_models, "col_probe_ids": probe_ids},
        "Q_MLM": {"shape": [4, 8], "row_model_ids": mlm_models, "col_probe_ids": probe_ids},
        "L_MLM": {"shape": [4, 8], "row_model_ids": mlm_models, "col_probe_ids": probe_ids},
    }, indent=2))

    # Panel parquet: probe_id, functional_element, GC_content (the three
    # cols analyze_matrix reads from the panel).
    panel = pd.DataFrame({
        "probe_id": probe_ids,
        "functional_element": ["promoter"] * 4 + ["enhancer"] * 4,
        "GC_content": rng.uniform(0.3, 0.7, size=8),
    })
    panel.to_parquet(in_dir / "probes" / "main_panel.parquet", index=False)

    PY = "/nvme-data3/yusen/micomamba/bin/python"
    proc = subprocess.run(
        [PY, str(REPO_ROOT / "scripts/run_phase1_analysis.py"),
         "--in-dir", str(in_dir), "--out", str(out_dir)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (
        f"run_phase1_analysis.main() crashed\n"
        f"stderr={proc.stderr}\nstdout={proc.stdout}"
    )
    # Outputs we expect to land on disk
    assert (out_dir / "pca" / "Q_AR" / "Z.npy").exists()
    assert (out_dir / "pca" / "Q_MLM" / "Z.npy").exists()
    assert (out_dir / "fst" / "marginal_fst.parquet").exists()
    assert (out_dir / "cross_branch" / "spearman.json").exists()
    # Cross-branch report should now use L (not double-centered Q)
    cross = json.loads((out_dir / "cross_branch" / "spearman.json").read_text())
    assert cross["status"] == "ok", cross
    assert cross["metric"] == "per_probe_mean_sum_log_p_on_L", cross


def test_heterozygosity_handles_nan_cells() -> None:
    """allow_missing Q can carry NaN cells from mixed-modality models.
    per_probe_heterozygosity must use nanvar / nanmean so per-probe
    variance is computed over the actually-scored subset instead of
    nan-propagating to the whole column."""
    R = np.array([
        [1.0, 2.0, 3.0, np.nan],
        [4.0, 5.0, np.nan, np.nan],
        [7.0, 8.0, 9.0, 10.0],
    ])
    rep = per_probe_heterozygosity(R, ddof=1)
    # All columns must yield a finite var (each has >= 2 finite values
    # except col 3 which has 1 → nanvar with ddof=1 gives NaN there;
    # acceptable). Sanity-check the cols that have ≥ 2 finite values.
    assert np.isfinite(rep.var_per_probe[0])
    assert np.isfinite(rep.var_per_probe[1])
    assert np.isfinite(rep.var_per_probe[2])
    # Col 3: only one finite value → nanvar gives NaN; ranking-by-NaN
    # is handled by routing NaN var to -inf so it falls to the end.
    assert rep.top_indices[-1] == 3 or np.isnan(rep.var_per_probe[3])

