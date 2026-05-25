"""Unit tests for src.scoring.rerun_stability."""

from __future__ import annotations

import math

import pytest

from glmap.scoring.rerun_stability import RerunStabilityReport, rerun_stability


class _DeterministicLoader:
    """Returns a fixed ell_per_base per sequence; rerun must be bitwise equal."""

    def __init__(self, mapping: dict[str, float]):
        self._mapping = mapping

    def score_record(self, sequence: str):
        class _R:
            pass

        r = _R()
        r.ell_per_base = self._mapping[sequence]
        return r


class _NoisyLoader:
    """Adds a fixed offset on the second call to simulate a non-deterministic
    scoring path (e.g. dropout left on)."""

    def __init__(self, mapping: dict[str, float], second_call_offset: float):
        self._mapping = mapping
        self._second = second_call_offset
        self._call_count: dict[str, int] = {}

    def score_record(self, sequence: str):
        self._call_count[sequence] = self._call_count.get(sequence, 0) + 1
        offset = self._second if self._call_count[sequence] >= 2 else 0.0

        class _R:
            pass

        r = _R()
        r.ell_per_base = self._mapping[sequence] + offset
        return r


def test_deterministic_loader_passes_gate() -> None:
    loader = _DeterministicLoader({"AAAA": -1.2, "ACGT": -1.5, "TTTT": -0.9})
    report = rerun_stability(loader, ["AAAA", "ACGT", "TTTT"])

    assert isinstance(report, RerunStabilityReport)
    assert report.n_probes == 3
    assert report.max_abs_diff == 0.0
    assert report.mean_abs_diff == 0.0
    assert math.isclose(report.pearson_r, 1.0, rel_tol=1e-9)
    assert report.passes_phase0_gate


def test_constant_offset_keeps_pearson_at_one_but_records_diff() -> None:
    """A uniform shift between run1 and run2 is rank-preserving so Pearson r
    stays 1.0, but abs diffs become nonzero — useful diagnostic data."""
    loader = _NoisyLoader({"AAAA": -1.2, "ACGT": -1.5, "TTTT": -0.9}, 0.05)
    report = rerun_stability(loader, ["AAAA", "ACGT", "TTTT"])

    assert math.isclose(report.pearson_r, 1.0, rel_tol=1e-9)
    assert math.isclose(report.max_abs_diff, 0.05, rel_tol=1e-9)
    assert report.passes_phase0_gate  # 1.0 >= 0.95


def test_empty_sequence_list_raises() -> None:
    loader = _DeterministicLoader({})
    with pytest.raises(ValueError, match="empty sequence list"):
        rerun_stability(loader, [])


def test_pearson_zero_denominator_when_run1_constant() -> None:
    """If run1 is constant across probes but run2 is not, Pearson is undefined.
    The helper treats this as 0.0 unless the two runs are bit-identical
    (then 1.0)."""

    class _Loader:
        def __init__(self):
            self._n = 0

        def score_record(self, sequence: str):
            self._n += 1

            class _R:
                pass

            r = _R()
            # Run 1 (first three calls): always -1.0
            # Run 2 (next three calls): increases with each probe
            if self._n <= 3:
                r.ell_per_base = -1.0
            else:
                r.ell_per_base = -1.0 + 0.1 * (self._n - 3)
            return r

    report = rerun_stability(_Loader(), ["a", "b", "c"])
    # run1 is constant; r should be 0.0 (undefined -> conservative)
    assert report.pearson_r == 0.0
    assert not report.passes_phase0_gate


def test_gate_threshold_is_inclusive_of_exactly_0_95() -> None:
    # Manually build a report with r=0.95 to verify the gate boundary.
    report = RerunStabilityReport(
        n_probes=10,
        pearson_r=0.95,
        max_abs_diff=0.0,
        mean_abs_diff=0.0,
        run1_values=tuple([0.0] * 10),
        run2_values=tuple([0.0] * 10),
    )
    assert report.passes_phase0_gate
