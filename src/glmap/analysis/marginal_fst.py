"""Marginal F_ST-analog per phase_1.md § 分支内分析.

Given a per-branch matrix R (M models x N probes; typically the double-
centered Q from `phase_1.md` § 单矩阵协议 after L → clip → double_center)
and a single axis labeling each model (tokenizer_type, training_species_
range, parameter_scale_bin, …), compute a **multivariate**
between-group / total variance ratio:

    SS_between = Σ_g n_g · ‖C_g − G‖²
    SS_within  = Σ_m ‖q_m − C_{g(m)}‖²
    SS_total   = SS_between + SS_within = Σ_m ‖q_m − G‖²
    F_ST       = SS_between / SS_total

where q_m is the M-th row vector (in probe space), C_g is the centroid
of group g, and G is the grand centroid (mean over all rows). This is
the standard ANOSIM / multivariate-F_ST form on centered representations.

**Why not row means?** An earlier version reduced each model to a scalar
via `R.mean(axis=1)`. On a double-centered Q, every row mean is exactly
zero by construction, so the resulting F_ST collapsed to noise. The
multivariate form preserves all N-dimensional structure: on a double-
centered Q the grand centroid G = 0, which simplifies the algebra but
does NOT eliminate signal — `‖C_g‖²` still captures genuine between-
group centroid separation.

This is a legacy **marginal** (axis-by-axis) diagnostic retained for
phase-1 reports. The former phase-2 partial-F_ST design-axis analysis was
retired from the main paper because it recovered mostly family structure.

For each axis the output also includes:
- a model-label permutation null with `n_permutations` shuffles,
- a one-sided permutation p (proportion of nulls ≥ observed F_ST).

NaN handling: cells that are NaN (mixed-modality models that legitimately
cannot score a probe class) are column-mean imputed before computing F_ST,
matching the imputation used by `scripts/run_phase1_analysis.py:analyze_matrix`
before PCA. Models with an entire-row of NaN are dropped at the caller
side (see `_build_branch_matrices`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarginalFstReport:
    axis_name: str
    group_labels: tuple[str, ...]      # in the order models appear in R
    unique_groups: tuple[str, ...]
    group_counts: dict[str, int]
    observed_fst: float
    null_fsts: np.ndarray              # (n_perm,) permutation distribution
    p_value: float                     # one-sided: fraction of nulls >= observed
    null_ci_95: tuple[float, float]


def _impute_column_mean(R: np.ndarray) -> np.ndarray:
    """Replace NaN cells with the per-column mean over finite values.

    Matches the imputation in `scripts/run_phase1_analysis.py:analyze_matrix`
    used before PCA. Columns that are all-NaN stay NaN; the caller is
    responsible for upstream filtering of those.
    """
    if not np.isnan(R).any():
        return R
    out = R.astype(np.float64, copy=True)
    col_means = np.nanmean(out, axis=0)
    nan_mask = np.isnan(out)
    inds = np.where(nan_mask)
    out[inds] = np.take(col_means, inds[1])
    return out


def _multivariate_fst(R_imp: np.ndarray, labels: np.ndarray) -> float:
    """Multivariate F_ST on a fully-imputed (M, N) matrix.

    SS_between = Σ_g n_g · ‖C_g − G‖²
    SS_total   = Σ_m ‖q_m − G‖²
    """
    M = R_imp.shape[0]
    grand = R_imp.mean(axis=0)                          # (N,)
    centered = R_imp - grand[None, :]                   # (M, N)
    ss_total = float((centered ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(labels):
        members = R_imp[labels == g]
        if members.shape[0] == 0:
            continue
        cg = members.mean(axis=0)                       # (N,)
        ss_between += members.shape[0] * float(((cg - grand) ** 2).sum())
    return ss_between / ss_total


def marginal_fst(
    R: np.ndarray,
    axis_labels: list[str],
    axis_name: str = "axis",
    n_permutations: int = 9999,
    seed: int = 0,
) -> MarginalFstReport:
    """Compute marginal multivariate F_ST + permutation null for a single
    grouping axis.

    Parameters
    ----------
    R :
        (M, N) per-branch matrix. Typically the double-centered Q from the
        single-matrix protocol; raw L also works (grand centroid is just
        more interpretable).
    axis_labels :
        Length-M list of group labels, one per model row.
    n_permutations :
        Default 9999 (Genome Biology convention; pre-registered in phase_1.md).
    seed :
        RNG seed for the model-label permutation.
    """
    M = R.shape[0]
    if len(axis_labels) != M:
        raise ValueError(
            f"axis_labels length {len(axis_labels)} != M={M}"
        )
    R_imp = _impute_column_mean(R)
    rng = np.random.default_rng(seed)
    label_arr = np.array(axis_labels)
    observed = _multivariate_fst(R_imp, label_arr)
    nulls = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted = rng.permutation(label_arr)
        nulls[i] = _multivariate_fst(R_imp, permuted)
    p = float(np.mean(nulls >= observed))
    lo, hi = float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))
    uniq = tuple(sorted(set(axis_labels)))
    counts = {g: axis_labels.count(g) for g in uniq}
    return MarginalFstReport(
        axis_name=axis_name,
        group_labels=tuple(axis_labels),
        unique_groups=uniq,
        group_counts=counts,
        observed_fst=observed,
        null_fsts=nulls,
        p_value=p,
        null_ci_95=(lo, hi),
    )


__all__ = ["MarginalFstReport", "marginal_fst"]
