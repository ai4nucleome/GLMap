"""Tests for src/panel/control_panel.py.

Critical regression: AE dinucleotide shuffle must EXACTLY preserve the
16-D dinucleotide multiset of the input. Before commit 38a6c50's
follow-up fix, the silent fallback after 20 failed retries did
character-shuffle, breaking preservation on ~0.89% of probes. This file
guards against regressing back to that behavior.
"""

from __future__ import annotations

import random
from collections import Counter

import pandas as pd
import pytest

from glmap.panel.control_panel import (
    DEFAULT_LENGTH_TIERS,
    DEFAULT_MOTIFS,
    _ae_shuffle,
    build_dinuc_shuffled,
    build_motif_spiked,
    build_random_acgt,
)


def _dinuc_counter(seq: str) -> Counter:
    return Counter(seq[i:i + 2] for i in range(len(seq) - 1))


# ─────────────────────── AE shuffle correctness ───────────────────────

def test_ae_shuffle_preserves_dinuc_simple() -> None:
    rng = random.Random(42)
    seq = "ACGTACGTACGT"
    out = _ae_shuffle(seq, rng)
    assert len(out) == len(seq)
    assert out[0] == seq[0]    # walk starts at seq[0]
    assert _dinuc_counter(out) == _dinuc_counter(seq)


def test_ae_shuffle_preserves_dinuc_repetitive() -> None:
    """Repetitive sequences (the kind that broke 20-retry rejection sampling)
    must still preserve dinuc exactly with the new 200-retry + saved-edge
    construction."""
    rng = random.Random(42)
    # Tandem repeats — historically a stress case for AE walks
    cases = [
        "ATATATATATATATAT",
        "AAAACCCCGGGGTTTT",
        "ACACACACACACACAC",
        "GGGCGGGCGGGCGGGC",
        "AGCTAGCTAGCTAGCT",
    ]
    for seq in cases:
        out = _ae_shuffle(seq, rng)
        assert len(out) == len(seq), f"len mismatch for {seq}"
        assert _dinuc_counter(out) == _dinuc_counter(seq), \
            f"dinuc mismatch for {seq}: {_dinuc_counter(out)} vs {_dinuc_counter(seq)}"


def test_ae_shuffle_preserves_long_random() -> None:
    """Stress test: 200 random 1024-bp sequences, all must preserve dinuc."""
    rng = random.Random(42)
    seq_rng = random.Random(1234)
    for _ in range(200):
        seq = "".join(seq_rng.choices("ACGT", k=1024))
        out = _ae_shuffle(seq, rng)
        assert len(out) == 1024
        assert _dinuc_counter(out) == _dinuc_counter(seq)


def test_ae_shuffle_short_returns_input() -> None:
    rng = random.Random(0)
    assert _ae_shuffle("", rng) == ""
    assert _ae_shuffle("A", rng) == "A"


# ─────────────────────── random_ACGT ───────────────────────

def test_build_random_acgt_count_and_gc_strata() -> None:
    rows = build_random_acgt(n=140, seed=42)
    assert len(rows) == 140
    # All ACGT-only
    for r in rows:
        assert set(r.sequence) <= set("ACGT")
        assert r.functional_element == "ctrl_random_ACGT"
        assert r.species == "synthetic"
        assert r.label_source == "control"
    # GC distribution covers the 7 strata (mean across n should be close to 0.5)
    gcs = [r.GC_content for r in rows]
    assert min(gcs) < 0.4
    assert max(gcs) > 0.6


def test_build_random_acgt_length_tiers() -> None:
    rows = build_random_acgt(n=28, seed=42)   # 7 bins × 4 tiers
    lengths = {r.length_bp for r in rows}
    assert lengths.issuperset(set(DEFAULT_LENGTH_TIERS))


# ─────────────────────── dinucleotide_shuffled ───────────────────────

def test_build_dinuc_shuffled_preserves_dinuc_for_every_probe() -> None:
    """Construct a tiny main_panel-like DataFrame, run build_dinuc_shuffled,
    verify EVERY emitted shuffle preserves dinuc exactly."""
    main_df = pd.DataFrame([
        {"probe_id": f"probe_{i:04d}", "sequence": "ACGTACGTACGTACGT",
         "functional_element": "promoter"}
        for i in range(20)
    ] + [
        {"probe_id": f"probe_{i:04d}", "sequence": "AAAACCCCGGGGTTTT",
         "functional_element": "enhancer"}
        for i in range(20, 40)
    ])
    rows = build_dinuc_shuffled(main_df, n_total=10, seed=42)
    assert len(rows) == 10
    main_lookup = main_df.set_index("probe_id")
    for r in rows:
        src_id = r.source.split("::", 1)[1]   # "shuffled_from::<id>"
        src_seq = main_lookup.loc[src_id]["sequence"]
        assert len(r.sequence) == len(src_seq)
        assert _dinuc_counter(r.sequence) == _dinuc_counter(src_seq), \
            f"dinuc not preserved for {r.probe_id}"


# ─────────────────────── motif_spiked ───────────────────────

def test_build_motif_spiked_motif_at_center() -> None:
    rows = build_motif_spiked(n=20, seed=42)
    # Each emitted sequence has exactly one of the 5 motifs at the center
    for r in rows:
        seq = r.sequence
        mid = (len(seq) - 0) // 2     # we'll search both candidate positions
        found = False
        for m in DEFAULT_MOTIFS:
            center = (len(seq) - len(m)) // 2
            if seq[center:center + len(m)] == m:
                found = True
                break
        assert found, f"motif not at center in probe {r.probe_id}"


def test_build_motif_spiked_metadata() -> None:
    rows = build_motif_spiked(n=10, seed=42)
    for r in rows:
        assert r.functional_element == "ctrl_motif_spiked"
        assert r.species == "synthetic"
        assert "motif=" in r.source
        assert r.label_source == "control"
