"""Single-matrix construction per branch (ModelMap convention).

For each branch (AR, MLM) we produce one matrix; the older three-matrix
split (R_pan / R_coding_only / R_nucleotide_only) was retired in commit
`5e59154` "matrix protocol: drop three-matrix split, build one L/Q/D per
branch". Every model × every probe enters L — codon-tokenized models
(GenSLM, Codon-NT) emit raw likelihood on noncoding probes alongside
nucleotide models, and the codon-vs-nucleotide systematic offset is
absorbed by the row-mean and column-mean subtraction in double-centering.

ModelMap pipeline (phase_1.md § 单矩阵协议):

    L  =  raw sum_log_p (nats, no length norm, no sign flip; cells < 0)
    L_clipped  =  clip_lower(L, q = 0.02)             # cap catastrophic outliers
    Q  =  double_center(L_clipped)                    # row + col centering
    D  =  pairwise_squared_distance(Q)                # (M, M) similarity

The matrix cell carries the model's raw `sum_log_p` over the probe — same
quantity ModelMap stores in `raw_log-likelihood-10k`. The clipping step
caps catastrophic-likelihood outliers at the 2nd percentile (ModelMap
clipping recipe), keeping a few badly behaved (model, probe) pairs from
dominating the variance budget.

Pairwise squared Euclidean on Q approximates KL divergence under the
small-divergence Taylor expansion (ModelMap Sec. 6.1); this is the
theoretical motivation for the double-centering choice.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


CLIP_QUANTILE_DEFAULT: float = 0.02


def clip_lower(
    L: np.ndarray, q: float = CLIP_QUANTILE_DEFAULT
) -> Tuple[np.ndarray, float]:
    """Floor-clip `L` at its q-quantile.

    NaN cells are preserved (NaN in -> NaN out) for forward compatibility
    with downstream NaN-safe analyses (mixed-modality models that legitimately
    can't score certain probe classes). Finite cells below the threshold
    are raised up to the threshold value.

    Returns
    -------
    L_clipped : ndarray
        Same shape as `L`. Cells < threshold are replaced by threshold;
        NaN cells are passed through unchanged.
    threshold : float
        The q-quantile (computed over finite cells only).
    """
    if L.size == 0:
        return L.copy(), float("nan")
    threshold = float(np.nanquantile(L, q))
    finite = np.isfinite(L)
    L_clipped = L.copy()
    L_clipped[finite] = np.maximum(L[finite], threshold)
    return L_clipped, threshold


def double_center(
    L: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Row-then-column centering on `L`.

    Uses `nanmean` so any NaN cells (legitimate mixed-modality unscored
    classes) don't contaminate the means. The grand mean is the mean of
    all finite cells (used for distance sanity / supplement reporting,
    not the centering itself).

    Returns
    -------
    Q : ndarray              double-centered matrix (same shape, may contain NaN)
    row_mean : (M,) ndarray  per-row mean of input `L`, before any centering
    col_mean : (N,) ndarray  per-column mean of (L - row_mean[:, None])
    grand_mean : float       float, mean of finite cells in `L`
    """
    if L.ndim != 2:
        raise ValueError(f"double_center: expected 2D, got shape {L.shape}")
    row_mean = np.nanmean(L, axis=1)
    L_row = L - row_mean[:, None]
    col_mean = np.nanmean(L_row, axis=0)
    Q = L_row - col_mean[None, :]
    grand_mean = float(np.nanmean(L))
    return Q, row_mean, col_mean, grand_mean


def pairwise_squared_distance(Q: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance over the rows of `Q`.

    Returns
    -------
    D : (M, M) ndarray   D[i, j] = ||Q[i] - Q[j]||^2

    NaN handling: cells where either row has NaN are excluded from the
    sum; the result is rescaled to N_finite / N_total so distances are
    comparable across model pairs with different NaN coverage. Under the
    single-matrix protocol all models (including codon-tokenized) enter L
    on the full panel, so Q is typically NaN-free; NaN cells only appear
    when a mixed-modality model legitimately cannot score a probe class.
    """
    if Q.ndim != 2:
        raise ValueError(f"pairwise_squared_distance: expected 2D, got {Q.shape}")
    M, N = Q.shape
    nan_mask = ~np.isfinite(Q)
    if not nan_mask.any():
        sq = np.sum(Q ** 2, axis=1)
        return sq[:, None] + sq[None, :] - 2.0 * (Q @ Q.T)
    # NaN-aware fallback: compute Σ (q_i - q_j)^2 only over the columns
    # where both rows are finite; rescale to N_total to keep magnitudes
    # comparable across rows with different finite-cell counts.
    D = np.zeros((M, M), dtype=np.float64)
    finite = ~nan_mask
    for i in range(M):
        for j in range(i + 1, M):
            both = finite[i] & finite[j]
            n_both = int(both.sum())
            if n_both == 0:
                D[i, j] = D[j, i] = np.nan
                continue
            diff = Q[i, both] - Q[j, both]
            D[i, j] = D[j, i] = float(np.sum(diff ** 2) * (N / n_both))
    return D


def build_L_Q_D(
    L: np.ndarray, clip_q: float = CLIP_QUANTILE_DEFAULT
) -> dict:
    """Full ModelMap-style matrix pipeline on a raw sum-log-p matrix.

    Returns a dict with `L_clipped`, `Q`, `D`, plus the threshold and
    centering statistics needed for downstream supplements.
    """
    L_clipped, threshold = clip_lower(L, q=clip_q)
    Q, row_mean, col_mean, grand_mean = double_center(L_clipped)
    D = pairwise_squared_distance(Q)
    return {
        "L_clipped": L_clipped,
        "Q": Q,
        "D": D,
        "clip_threshold": threshold,
        "clip_quantile": float(clip_q),
        "row_mean": row_mean,
        "col_mean": col_mean,
        "grand_mean": grand_mean,
    }


__all__ = [
    "CLIP_QUANTILE_DEFAULT",
    "clip_lower",
    "double_center",
    "pairwise_squared_distance",
    "build_L_Q_D",
]
