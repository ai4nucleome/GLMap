"""Double-centered truncated SVD on a sequence-likelihood matrix.

phase_1.md § 单矩阵协议 specifies the representation normalization step
(matrix names per the ModelMap convention):

  L          = raw sum_log_p (M × N, M models × N probes; cells < 0)
  L_clipped  = floor_clip(L, q=0.02)
  Q          = double_center(L_clipped)    # row + col mean removed
  Z          = SVD(Q)[:, :k]

`double_center` removes the row mean and then the column mean (equivalent
to removing the grand mean from the row and column means), and SVD keeps
the top-k singular components. The double-centering operation is what
makes squared Euclidean on Z approximate KL divergence under the
small-divergence Taylor expansion (ModelMap Sec. 6.1 / phase_1.md §
ModelMap).

This module exposes:
  - `double_center(M)` -> centered matrix (typically `L_clipped` -> `Q`)
  - `truncated_svd(centered, k)` -> (U_k, sigma_k, V_k_T)
  - `pca_models(M, k)` -> ModelEmbedding(Z, explained_variance, sigma, V_T)
    where Z (M_models x k) is the model embedding (= U_k * sigma_k).
  - `procrustes_residual(Z_a, Z_b)` -> rotation-invariant alignment residual.

Function arguments use `R` as a generic centered-matrix name to keep the
math notation uniform inside SVD code; conceptually it should be read as
`Q` (or `Q_residual` for the composition-controlled variant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelEmbedding:
    """Result of the phase_1.md representation normalization."""

    Z: np.ndarray             # (M, k) model embedding
    sigma: np.ndarray         # (k,) singular values
    V_T: np.ndarray           # (k, N) probe loadings
    explained_variance: np.ndarray  # (k,) per-component variance share
    centered_R: np.ndarray    # (M, N) double-centered input matrix
    row_mean: np.ndarray      # (M,) mean of each row (before centering)
    col_mean: np.ndarray      # (N,) mean of each column (after row-centering)
    grand_mean: float


def double_center(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Subtract row-mean then column-mean from R.

    Equivalent to subtracting the rank-1 outer product
    `row_mean[:, None] + col_mean[None, :] - grand_mean` from R, which is
    the standard double-centering used by ModelMap and classical MDS.

    Returns
    -------
    centered : (M, N) ndarray
    row_mean : (M,) ndarray   means of input rows (pre-centering)
    col_mean : (N,) ndarray   means of (R - row_mean[:, None]) columns
    grand_mean : float
    """
    if R.ndim != 2:
        raise ValueError(f"double_center: expected 2D matrix, got shape {R.shape}")
    row_mean = R.mean(axis=1)
    R_row = R - row_mean[:, None]
    col_mean = R_row.mean(axis=0)
    centered = R_row - col_mean[None, :]
    grand_mean = float(R.mean())
    return centered, row_mean, col_mean, grand_mean


def truncated_svd(centered: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the truncated SVD of `centered`.

    Parameters
    ----------
    centered :
        Already double-centered (M, N) matrix.
    k :
        Number of components to keep. If None, keeps min(M-1, N-1, M, N)
        which is the maximum non-degenerate rank for a double-centered
        matrix (loses one rank in each direction).
    """
    M, N = centered.shape
    full_rank = min(M, N)
    if k is None:
        # After double-centering, the rank caps at min(M-1, N-1) but for
        # smoke tests with tiny M we still want all non-trivial components.
        k = max(1, min(full_rank, full_rank))
    U, sigma, V_T = np.linalg.svd(centered, full_matrices=False)
    return U[:, :k], sigma[:k], V_T[:k, :]


def pca_models(R: np.ndarray, k: int | None = None) -> ModelEmbedding:
    """Project models into a k-dimensional embedding via double-centered SVD."""
    centered, row_mean, col_mean, grand_mean = double_center(R)
    U, sigma, V_T = truncated_svd(centered, k=k)
    Z = U * sigma[None, :]
    var_components = sigma ** 2
    total = float(var_components.sum())
    explained = var_components / total if total > 0 else var_components
    return ModelEmbedding(
        Z=Z,
        sigma=sigma,
        V_T=V_T,
        explained_variance=explained.astype(np.float64),
        centered_R=centered,
        row_mean=row_mean,
        col_mean=col_mean,
        grand_mean=grand_mean,
    )


def procrustes_residual(Z_a: np.ndarray, Z_b: np.ndarray) -> float:
    """Frobenius residual of the optimal-orthogonal-rotation alignment of
    Z_a onto Z_b, normalized by ||Z_b||_F.

    Used by phase_1.md sanity checks to align embeddings across runs —
    e.g. Q vs Q_residual (composition-controlled), stride-k variants,
    or phase 1 vs phase 2 on overlapping model sets. Residual = 0 means
    the two embeddings differ only by an orthogonal rotation; residual = 1
    means complete disagreement.
    """
    if Z_a.shape != Z_b.shape:
        raise ValueError(
            f"procrustes_residual: shape mismatch {Z_a.shape} vs {Z_b.shape}"
        )
    # Center both (column means).
    A = Z_a - Z_a.mean(axis=0, keepdims=True)
    B = Z_b - Z_b.mean(axis=0, keepdims=True)
    # Best-fit orthogonal R minimizes ||AR - B||_F.
    U, _s, V_T = np.linalg.svd(A.T @ B, full_matrices=False)
    R_opt = U @ V_T
    aligned = A @ R_opt
    num = np.linalg.norm(aligned - B, ord="fro")
    den = np.linalg.norm(B, ord="fro")
    return float(num / den) if den > 0 else 0.0


__all__ = [
    "ModelEmbedding",
    "double_center",
    "truncated_svd",
    "pca_models",
    "procrustes_residual",
]
