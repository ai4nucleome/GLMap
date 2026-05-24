"""Rerun stability gate per phase_0.md "必须通过的检查":

    [3] rerun stability: 同模型同 probe 重跑 signature 自相关 ≥ 0.95

Given a loader and a list of sequences, score each sequence twice and
report per-probe diff statistics + the cross-run Pearson correlation. With
HF models in eval()+no_grad mode this is essentially a determinism check
(diff -> 0). For non-deterministic implementations (dropout left on,
sampling-based MLM heads, etc.) the correlation surfaces the regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RerunStabilityReport:
    n_probes: int
    pearson_r: float
    max_abs_diff: float
    mean_abs_diff: float
    run1_values: tuple[float, ...]
    run2_values: tuple[float, ...]

    @property
    def passes_phase0_gate(self) -> bool:
        """phase_0.md [3]: 重跑 signature 自相关 ≥ 0.95."""
        return self.pearson_r >= 0.95


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n != len(ys):
        raise ValueError("rerun_stability: paired runs must have equal length")
    if n < 2:
        # With < 2 samples Pearson is undefined; treat single-probe rerun as
        # "perfectly correlated" if and only if both values are identical.
        if n == 0:
            return 1.0
        return 1.0 if math.isclose(xs[0], ys[0], rel_tol=1e-9, abs_tol=1e-12) else 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sx2 * sy2)
    if denom == 0.0:
        # Both runs constant. Correlation undefined; if values are identical
        # treat as 1.0, else 0.0.
        all_equal = all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(xs, ys))
        return 1.0 if all_equal else 0.0
    return num / denom


def rerun_stability(
    loader: Any,
    sequences: Sequence[str],
    score_kwargs: dict | None = None,
) -> RerunStabilityReport:
    """Score each sequence twice via loader.score_record(); compare ell_per_base.

    `loader` must expose `score_record(sequence, **score_kwargs)` returning an
    object with an `ell_per_base` attribute. Works for HFCausalLMLoader
    (AR / ARScore) and HFMaskedLMLoader (MLM / MLMScore).
    """
    if not sequences:
        raise ValueError("rerun_stability: empty sequence list")
    score_kwargs = score_kwargs or {}

    run1 = [float(loader.score_record(s, **score_kwargs).ell_per_base) for s in sequences]
    run2 = [float(loader.score_record(s, **score_kwargs).ell_per_base) for s in sequences]
    diffs = [abs(a - b) for a, b in zip(run1, run2)]
    return RerunStabilityReport(
        n_probes=len(sequences),
        pearson_r=_pearson(run1, run2),
        max_abs_diff=max(diffs),
        mean_abs_diff=sum(diffs) / len(diffs),
        run1_values=tuple(run1),
        run2_values=tuple(run2),
    )


__all__ = ["RerunStabilityReport", "rerun_stability"]
