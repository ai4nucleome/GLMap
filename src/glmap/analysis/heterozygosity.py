"""Per-probe inter-model variance ("heterozygosity-analog").

phase_1.md § 分支内分析 defines:

    per-sequence heterozygosity = Var_m(Q[m, x])

(computed on the double-centered matrix Q, after row + col mean subtraction
removed each model's overall level and each probe's overall difficulty);
each probe gets a scalar telling us how much the model population
disagrees on it. High values flag probes that drive between-model
divergence — they are the most informative for representation analysis.

The function returns the per-column variance plus a sorted ranking; the
caller can use it to rank probes by informativeness or to visualize the
distribution by functional class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeterozygosityReport:
    n_probes: int
    n_models: int
    var_per_probe: np.ndarray        # (N,) Var across the M-row dimension
    mean_per_probe: np.ndarray       # (N,) mean across models
    top_indices: np.ndarray          # probe indices sorted by descending var
    bottom_indices: np.ndarray       # probe indices sorted by ascending var


def per_probe_heterozygosity(
    R: np.ndarray, ddof: int = 1
) -> HeterozygosityReport:
    """Compute per-probe inter-model variance.

    Parameters
    ----------
    R :
        (M, N) sequence-likelihood matrix.
    ddof :
        Delta degrees of freedom for the variance (default 1, sample
        variance). With M=2-3 in smoke tests, sample variance is the
        unbiased estimate.
    """
    if R.ndim != 2:
        raise ValueError(f"R must be 2D, got shape {R.shape}")
    M, N = R.shape
    if M < 2:
        raise ValueError(f"need at least 2 models, got {M}")
    # Use nan-aware reductions so allow_missing Q (with NaN cells from
    # mixed-modality models that legitimately can't score a probe class)
    # yields per-probe variance over the actually-scored subset instead
    # of NaN-propagating to the whole column.
    var_per_probe = np.nanvar(R, axis=0, ddof=ddof)
    mean_per_probe = np.nanmean(R, axis=0)
    # argsort puts NaNs at the end with kind="stable"; flipping sign for
    # descending var, NaNs end up at the front. Replace var NaNs with
    # -inf so they sort to the bottom (least informative).
    var_for_rank = np.where(np.isnan(var_per_probe), -np.inf, var_per_probe)
    order = np.argsort(-var_for_rank, kind="stable")
    return HeterozygosityReport(
        n_probes=N,
        n_models=M,
        var_per_probe=var_per_probe,
        mean_per_probe=mean_per_probe,
        top_indices=order,
        bottom_indices=order[::-1].copy(),
    )


__all__ = ["HeterozygosityReport", "per_probe_heterozygosity"]
