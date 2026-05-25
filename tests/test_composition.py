"""Unit tests for src.panel.composition."""

from __future__ import annotations

import math

from glmap.panel.composition import (
    BASES,
    DINUC_INDEX,
    DINUC_ORDER,
    TRINUC_INDEX,
    TRINUC_ORDER,
    dinuc_vec,
    gc_fraction,
    gc_stratify_bin,
    trinuc_vec,
)


def test_alphabet_order_is_lexicographic() -> None:
    assert DINUC_ORDER[0] == "AA"
    assert DINUC_ORDER[-1] == "TT"
    assert len(DINUC_ORDER) == 16
    assert TRINUC_ORDER[0] == "AAA"
    assert TRINUC_ORDER[-1] == "TTT"
    assert len(TRINUC_ORDER) == 64


def test_dinuc_index_consistency() -> None:
    assert DINUC_INDEX["AC"] == DINUC_ORDER.index("AC")
    assert TRINUC_INDEX["CGT"] == TRINUC_ORDER.index("CGT")


def test_gc_fraction_basic() -> None:
    assert gc_fraction("") == 0.0
    assert gc_fraction("AAAA") == 0.0
    assert gc_fraction("GGGG") == 1.0
    assert math.isclose(gc_fraction("ACGT"), 0.5)
    assert math.isclose(gc_fraction("ACGTN"), 0.4)  # N is not GC


def test_dinuc_vec_sums_to_one_for_dna() -> None:
    vec = dinuc_vec("ACGTACGT")
    assert len(vec) == 16
    assert math.isclose(sum(vec), 1.0, rel_tol=1e-9)


def test_dinuc_vec_uniform_alphabet() -> None:
    """ACGT * 100 (length 400) produces 399 dinucleotides: AC/CG/GT each at
    100 occurrences, TA at 99 (last T at position 399 has no successor)."""
    seq = "ACGT" * 100
    vec = dinuc_vec(seq)
    total_dinucs = len(seq) - 1  # 399
    expected = {
        "AC": 100 / total_dinucs,
        "CG": 100 / total_dinucs,
        "GT": 100 / total_dinucs,
        "TA": 99 / total_dinucs,
    }
    for kmer, frac in expected.items():
        assert math.isclose(vec[DINUC_INDEX[kmer]], frac, rel_tol=1e-6)
    # All other dinucs are absent.
    nonzero = {k for k in expected}
    for k, idx in DINUC_INDEX.items():
        if k not in nonzero:
            assert vec[idx] == 0.0


def test_trinuc_vec_sums_to_one() -> None:
    vec = trinuc_vec("ACGTACGTACGT")
    assert len(vec) == 64
    assert math.isclose(sum(vec), 1.0, rel_tol=1e-9)


def test_dinuc_ignores_non_acgt() -> None:
    """Ns produce no dinucleotide counts, vector still normalizes correctly."""
    vec_with_n = dinuc_vec("ACNGT")
    # Valid dinucleotides: AC (CN dropped, NG dropped, GT kept) -> AC, GT only
    expected_nonzero = {DINUC_INDEX["AC"]: 0.5, DINUC_INDEX["GT"]: 0.5}
    for idx, expected in expected_nonzero.items():
        assert math.isclose(vec_with_n[idx], expected, rel_tol=1e-6)


def test_gc_stratify_default_bins() -> None:
    assert gc_stratify_bin(0.1) == "very_low"
    assert gc_stratify_bin(0.3) == "low"
    assert gc_stratify_bin(0.45) == "mid_low"
    assert gc_stratify_bin(0.55) == "mid_high"
    assert gc_stratify_bin(0.7) == "high"
    assert gc_stratify_bin(0.9) == "very_high"
