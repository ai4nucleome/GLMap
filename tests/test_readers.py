"""Smoke tests for src/panel/readers.py.

Tests the four reader formats (csv / fasta_pgb_binary /
fasta_pgb_binary_lenfilter / fasta_pgb_chromatin) against tiny fixtures
written into a tmp_path. Verifies label policy filtering and ACGT-only
normalization.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from glmap.panel.main_panel import DatasetSpec
from glmap.panel.readers import read_dataset


def _make_csv(tmp_path: Path, content: str, name: str = "ds.csv") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def _make_fasta(tmp_path: Path, content: str, name: str = "ds.fa") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip())
    return p


# ─────────────────────── CSV ───────────────────────

def test_csv_positive_only(tmp_path: Path) -> None:
    path = _make_csv(tmp_path, """
        sequence,label
        ACGTACGT,1
        TTTTAAAA,0
        AAAAAAAA,1
    """)
    spec = DatasetSpec(
        path="dummy", format="csv", seq_col="sequence", label_col="label",
        label_policy="positive_only", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    assert len(out) == 2
    assert all(r.raw_label == "1" for r in out)


def test_csv_keep_label(tmp_path: Path) -> None:
    path = _make_csv(tmp_path, """
        sequence,label
        ACGTACGT,0
        TTTTAAAA,1
        CCCCGGGG,2
    """)
    spec = DatasetSpec(
        path="dummy", format="csv", seq_col="sequence", label_col="label",
        label_policy="keep_label", keep_labels=("0", "1"),
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    assert len(out) == 2
    assert {r.raw_label for r in out} == {"0", "1"}


def test_csv_keep_all(tmp_path: Path) -> None:
    path = _make_csv(tmp_path, """
        sequence,label
        ACGTACGT,3
        TTTTAAAA,7
    """)
    spec = DatasetSpec(
        path="dummy", format="csv", seq_col="sequence", label_col="label",
        label_policy="keep_all", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    assert len(out) == 2


def test_csv_rejects_non_acgt(tmp_path: Path) -> None:
    """Reader normalizes uppercase + drops rows with N or IUPAC ambig bases."""
    path = _make_csv(tmp_path, """
        sequence,label
        ACGTacgt,1
        NNNNNNNN,1
        ACGTNNNN,1
        ACGTACGT,1
    """)
    spec = DatasetSpec(
        path="dummy", format="csv", seq_col="sequence", label_col="label",
        label_policy="positive_only", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    # Two valid: "ACGTACGT" (after upper) + "ACGTACGT"; reject N rows
    assert len(out) == 2
    assert all(set(r.sequence) <= set("ACGT") for r in out)
    # Verify uppercase normalization
    assert all(r.sequence.isupper() for r in out)


def test_csv_skip_validate_when_crop(tmp_path: Path) -> None:
    """When crop_to is set, validation is deferred until after crop —
    reader returns the raw sequence (uppercased) without ACGT enforcement."""
    path = _make_csv(tmp_path, """
        sequence,label
        ACGTNNNNNACGT,5
    """)
    spec = DatasetSpec(
        path="dummy", format="csv", seq_col="sequence", label_col="label",
        label_policy="multiclass_species", keep_labels=None,
        target_n=10, species=None, crop_to=4, crop_strategy="center",
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    # Raw sequence passes (validation deferred), uppercased
    assert len(out) == 1
    assert out[0].sequence == "ACGTNNNNNACGT"


# ─────────────────────── PGB FASTA binary ───────────────────────

def test_fasta_pgb_binary_positive_only(tmp_path: Path) -> None:
    path = _make_fasta(tmp_path, """
        >id1|1
        ACGTACGTACGT
        >id2|0
        TTTTAAAATTTT
        >id3|1
        CCCCGGGGCCCC
    """)
    spec = DatasetSpec(
        path="dummy", format="fasta_pgb_binary", seq_col=None, label_col=None,
        label_policy="positive_only", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    assert len(out) == 2
    assert {r.raw_label for r in out} == {"1"}


def test_fasta_pgb_binary_length_filter(tmp_path: Path) -> None:
    """fasta_pgb_binary_lenfilter enforces 128 ≤ L ≤ 1024 for positives."""
    path = _make_fasta(tmp_path, """
        >id1|1
        AAAAAAAAAA
        >id2|1
        """ + "A" * 200 + """
        >id3|1
        """ + "A" * 2000 + """
        >id4|0
        """ + "A" * 200 + """
    """)
    spec = DatasetSpec(
        path="dummy", format="fasta_pgb_binary_lenfilter",
        seq_col=None, label_col=None,
        label_policy="positive_only", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    # Only id2 passes: positive AND 128 ≤ 200 ≤ 1024
    assert len(out) == 1
    assert out[0].sequence == "A" * 200


# ─────────────────────── PGB FASTA chromatin (multi-label) ───────────────────────

def test_fasta_pgb_chromatin_drops_negative_suffix(tmp_path: Path) -> None:
    path = _make_fasta(tmp_path, """
        >chr1:100_negative|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
        ACGTACGTACGT
        >chr1:200_pos|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
        TTTTAAAATTTT
        >chr1:300_pos|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
        CCCCGGGGCCCC
        >chr1:400_pos|1|1|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0
        AAAAGGGGCCCC
    """)
    spec = DatasetSpec(
        path="dummy", format="fasta_pgb_chromatin",
        seq_col=None, label_col=None,
        label_policy="multi_label_any_pos", keep_labels=None,
        target_n=10, species=None, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )
    out = read_dataset(path, spec)
    # _negative dropped (record 1); all-zero non-negative also dropped (record 3)
    # records 2 (one label) and 4 (three labels) kept
    assert len(out) == 2
    raw_labels = sorted(r.raw_label for r in out)
    assert raw_labels == ["0,1,5", "2"]
