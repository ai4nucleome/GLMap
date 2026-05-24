"""GC-axis diagnostic per phase_1.md § Composition confounding.

Each PC of the model embedding is checked for correlation with probe GC
fraction; PCs with |Pearson r| > 0.7 are flagged "GC-dominated" and
downstream interpretation skips them in favor of the next non-GC PC.

This is the analysis-layer half of the composition-confounding control.
The probe-design layer (`src/panel/random_window.py`'s `gc_per_bin_floor`)
spreads probes across GC strata so the diagnostic isn't structurally
forced; the GC-axis diagnostic still verifies that the spread held.

Input is the probe loadings V_T from `src.analysis.pca.pca_models` plus the
matching GC fraction vector. The diagnostic operates on `V_T` (probe-side)
because that's the side where GC composition can correlate with PCs; on
the model-side (`Z`), GC composition has no direct interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GCAxisReport:
    n_components: int
    r_per_pc: np.ndarray            # (k,) Pearson r between PC loading and GC
    abs_r_per_pc: np.ndarray        # |r|
    is_gc_dominated: np.ndarray     # (k,) bool, |r| > threshold
    threshold: float
    first_non_gc_pc: int             # 0-based index of the first PC with |r| <= threshold; -1 if none


def gc_axis_diagnostic(
    V_T: np.ndarray,
    gc_per_probe: np.ndarray,
    threshold: float = 0.7,
) -> GCAxisReport:
    """Correlate each PC's probe-loading vector with the per-probe GC vector.

    Parameters
    ----------
    V_T :
        (k, N) probe loadings from PCA (rows are PCs, columns are probes).
    gc_per_probe :
        (N,) GC fraction per probe, aligned with V_T columns.
    threshold :
        |Pearson r| above which a PC is tagged GC-dominated. phase_1.md
        hard-codes 0.7 as the primary cutoff.
    """
    if V_T.ndim != 2:
        raise ValueError(f"V_T must be 2D, got {V_T.shape}")
    if V_T.shape[1] != gc_per_probe.shape[0]:
        raise ValueError(
            f"V_T column count {V_T.shape[1]} must match len(gc_per_probe) "
            f"{gc_per_probe.shape[0]}"
        )
    k = V_T.shape[0]
    r_per_pc = np.zeros(k)
    gc = gc_per_probe.astype(np.float64)
    gc_zero_mean = gc - gc.mean()
    gc_norm = np.linalg.norm(gc_zero_mean)
    for i in range(k):
        v = V_T[i].astype(np.float64)
        v_zero = v - v.mean()
        v_norm = np.linalg.norm(v_zero)
        if v_norm == 0 or gc_norm == 0:
            r_per_pc[i] = 0.0
        else:
            r_per_pc[i] = float(np.dot(v_zero, gc_zero_mean) / (v_norm * gc_norm))
    abs_r = np.abs(r_per_pc)
    is_dom = abs_r > threshold
    first_non = int(np.argmin(is_dom)) if not is_dom.all() else -1
    if first_non == 0 and is_dom[0]:
        first_non = -1
    return GCAxisReport(
        n_components=k,
        r_per_pc=r_per_pc,
        abs_r_per_pc=abs_r,
        is_gc_dominated=is_dom,
        threshold=threshold,
        first_non_gc_pc=first_non,
    )


__all__ = ["GCAxisReport", "gc_axis_diagnostic"]
