"""Composition utilities for probe panel construction.

GC fraction, 2-mer (16-dim) and 3-mer (64-dim) frequency vectors. Vectors
follow a fixed lexicographic alphabet order so they can be horizontally
stacked into matrices without per-row index lookups.

Input contract
--------------
All public functions in this module require **uppercase ACGT-only**
sequences. Non-ACGT input (N, lowercase, IUPAC ambiguity codes) raises
``AssertionError`` at the entry point. Sanitize upstream — for example,
the panel build pipeline uses
``data/build_panel/readers.py::_normalize_and_validate`` to drop any
sequence containing non-ACGT characters before it enters the panel.

These utilities are kept dependency-light (no numpy) so they can be
invoked during sampling tight loops without overhead.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

BASES = ("A", "C", "G", "T")
_ACGT_SET = frozenset(BASES)

DINUC_ORDER: tuple[str, ...] = tuple("".join(p) for p in product(BASES, repeat=2))
TRINUC_ORDER: tuple[str, ...] = tuple("".join(p) for p in product(BASES, repeat=3))

DINUC_INDEX: dict[str, int] = {kmer: i for i, kmer in enumerate(DINUC_ORDER)}
TRINUC_INDEX: dict[str, int] = {kmer: i for i, kmer in enumerate(TRINUC_ORDER)}


def _assert_acgt(sequence: str) -> None:
    """Reject any non-ACGT character. Raises ``AssertionError``.

    The input contract for all public composition functions: sequence
    must be uppercase ACGT only. Sanitize upstream rather than letting
    this function silently coerce or drop characters.
    """
    bad = set(sequence) - _ACGT_SET
    assert not bad, (
        f"glmap.panel.composition expects ACGT-only input; got non-ACGT "
        f"characters {sorted(bad)}. Normalize upstream "
        f"(e.g. data/build_panel/readers.py::_normalize_and_validate)."
    )


def gc_fraction(sequence: str) -> float:
    """Fraction of bases that are G or C. Returns 0.0 for empty input."""
    if not sequence:
        return 0.0
    _assert_acgt(sequence)
    gc = sum(1 for b in sequence if b in "GC")
    return gc / len(sequence)


def _kmer_counts(sequence: str, k: int) -> list[int]:
    """Sliding-window k-mer counts; input must be ACGT-only.

    Callers are expected to have validated via ``_assert_acgt`` already.
    """
    size = 4**k
    counts = [0] * size
    if len(sequence) < k:
        return counts
    if k == 2:
        index = DINUC_INDEX
    elif k == 3:
        index = TRINUC_INDEX
    else:
        # Build a fresh index for ad-hoc k.
        order = tuple("".join(p) for p in product(BASES, repeat=k))
        index = {kmer: i for i, kmer in enumerate(order)}
    for i in range(len(sequence) - k + 1):
        counts[index[sequence[i : i + k]]] += 1
    return counts


def dinuc_vec(sequence: str) -> list[float]:
    """16-dim dinucleotide frequency vector ordered AA, AC, ..., TT.

    Input must be uppercase ACGT only (raises ``AssertionError`` otherwise).
    """
    _assert_acgt(sequence)
    counts = _kmer_counts(sequence, 2)
    total = sum(counts)
    if total == 0:
        return [0.0] * 16
    return [c / total for c in counts]


def trinuc_vec(sequence: str) -> list[float]:
    """64-dim trinucleotide frequency vector ordered AAA, AAC, ..., TTT.

    Input must be uppercase ACGT only (raises ``AssertionError`` otherwise).
    """
    _assert_acgt(sequence)
    counts = _kmer_counts(sequence, 3)
    total = sum(counts)
    if total == 0:
        return [0.0] * 64
    return [c / total for c in counts]


def gc_stratify_bin(
    gc: float, bins: Sequence[float] = (0.2, 0.4, 0.5, 0.6, 0.8)
) -> str:
    """Assign a GC fraction to a labeled stratum.

    Default bins give 6 strata:
        very_low (gc < 0.2), low (<0.4), mid_low (<0.5), mid_high (<0.6),
        high (<0.8), very_high (>= 0.8).
    """
    labels = ("very_low", "low", "mid_low", "mid_high", "high", "very_high")
    for i, threshold in enumerate(bins):
        if gc < threshold:
            return labels[i]
    return labels[-1]


__all__ = [
    "BASES",
    "DINUC_ORDER",
    "TRINUC_ORDER",
    "DINUC_INDEX",
    "TRINUC_INDEX",
    "gc_fraction",
    "dinuc_vec",
    "trinuc_vec",
    "gc_stratify_bin",
]
