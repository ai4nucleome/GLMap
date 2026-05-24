"""Composition utilities for probe panel construction.

GC fraction, 2-mer (16-dim) and 3-mer (64-dim) frequency vectors. Vectors
follow a fixed lexicographic alphabet order so they can be horizontally
stacked into matrices without per-row index lookups.

Non-ACGT bases (N, lowercase, IUPAC ambiguous) are ignored; the resulting
vector still normalizes by the number of valid k-mers it could form, so
sequences with sparse Ns are handled gracefully.

These utilities are kept dependency-light (no numpy) so they can be invoked
during sampling tight loops without overhead.
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

BASES = ("A", "C", "G", "T")
DINUC_ORDER: tuple[str, ...] = tuple("".join(p) for p in product(BASES, repeat=2))
TRINUC_ORDER: tuple[str, ...] = tuple("".join(p) for p in product(BASES, repeat=3))

DINUC_INDEX: dict[str, int] = {kmer: i for i, kmer in enumerate(DINUC_ORDER)}
TRINUC_INDEX: dict[str, int] = {kmer: i for i, kmer in enumerate(TRINUC_ORDER)}


def gc_fraction(sequence: str) -> float:
    if not sequence:
        return 0.0
    gc = sum(1 for b in sequence if b in "GC")
    return gc / len(sequence)


def _kmer_counts(sequence: str, k: int) -> list[int]:
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
        kmer = sequence[i : i + k]
        idx = index.get(kmer)
        if idx is not None:
            counts[idx] += 1
    return counts


def dinuc_vec(sequence: str) -> list[float]:
    """16-dim dinucleotide frequency vector ordered AA, AC, ..., TT."""
    counts = _kmer_counts(sequence, 2)
    total = sum(counts)
    if total == 0:
        return [0.0] * 16
    return [c / total for c in counts]


def trinuc_vec(sequence: str) -> list[float]:
    """64-dim trinucleotide frequency vector ordered AAA, AAC, ..., TTT."""
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

    Used by sampler to enforce within-class GC balance per phase_1.md
    § Composition confounding.
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
