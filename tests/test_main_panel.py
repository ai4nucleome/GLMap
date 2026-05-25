"""Unit tests for src/panel/main_panel.py.

Tests config parsing, ProbeRow construction, species assignment, center
crop, and the assembled DataFrame schema. Does NOT test full panel build
(that's an integration test handled by test_panel_integration.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glmap.panel.main_panel import (
    DatasetSpec,
    ElementSpec,
    PanelConfig,
    ProbeRow,
    _assign_species,
    _crop_center,
    _is_acgt,
    _make_probe_row,
    load_panel_config,
)
from glmap.panel.readers import ReaderResult


# ─────────────────────── helpers ───────────────────────

def test_crop_center_short_unchanged() -> None:
    """Sequences shorter than target are returned as-is."""
    assert _crop_center("ACGT", 10) == "ACGT"
    assert _crop_center("", 10) == ""


def test_crop_center_takes_middle() -> None:
    seq = "AAAACCCCGGGGTTTT"   # 16 bp
    cropped = _crop_center(seq, 4)
    # (16-4)//2 = 6 → [6:10] = "CCGG"
    assert cropped == "CCGG"


def test_crop_center_odd_length_diff() -> None:
    seq = "ACGTACGTAC"          # 10 bp
    cropped = _crop_center(seq, 7)
    # (10-7)//2 = 1 → [1:8] = "CGTACGT"
    assert cropped == "CGTACGT"
    assert len(cropped) == 7


def test_is_acgt() -> None:
    assert _is_acgt("ACGTACGT") is True
    assert _is_acgt("ACGT") is True
    assert _is_acgt("") is True
    assert _is_acgt("ACGN") is False
    assert _is_acgt("acgt") is False  # case-sensitive: caller must upper first


# ─────────────────────── species assignment ───────────────────────

def _ds(species: str | None = None) -> DatasetSpec:
    return DatasetSpec(
        path="dummy", format="csv", seq_col=None, label_col=None,
        label_policy="keep_all", keep_labels=None,
        target_n=1, species=species, crop_to=None, crop_strategy=None,
        balance_per_label=None,
    )


def _elem(eid: str, species: str | None = None) -> ElementSpec:
    return ElementSpec(
        element_id=eid, species_group="X", species=species,
        n_probes=1, datasets=[],
    )


def test_assign_species_dataset_level() -> None:
    """Dataset-level species wins over element-level."""
    elem = _elem("foo", species="Element sp")
    ds = _ds(species="Dataset sp")
    assert _assign_species(elem, ds, "ignored") == "Dataset sp"


def test_assign_species_element_level() -> None:
    """Element species used when dataset doesn't set one."""
    elem = _elem("foo", species="Element sp")
    ds = _ds(species=None)
    assert _assign_species(elem, ds, "ignored") == "Element sp"


def test_assign_species_multiclass_fungi() -> None:
    elem = _elem("fungi_genome", species="per_row")
    ds = _ds(species=None)
    assert _assign_species(elem, ds, "7") == "fungi_sp_7"


def test_assign_species_multiclass_covid() -> None:
    elem = _elem("virus_variants", species="per_row")
    ds = _ds(species=None)
    assert _assign_species(elem, ds, "3") == "covid_var_3"


def test_assign_species_multiclass_virus_species() -> None:
    elem = _elem("virus_species", species="per_row")
    ds = _ds(species=None)
    assert _assign_species(elem, ds, "12") == "virus_sp_12"


# ─────────────────────── probe row construction ───────────────────────

def test_make_probe_row_basic() -> None:
    elem = _elem("promoter", species="Homo sapiens")
    ds = _ds(species=None)
    rec = ReaderResult(sequence="ACGTACGTACGT", raw_label="1", row_idx=42)
    row = _make_probe_row(elem, ds, rec, probe_id="promoter_00001",
                          length_min=4, length_max=20)
    assert isinstance(row, ProbeRow)
    assert row.probe_id == "promoter_00001"
    assert row.sequence == "ACGTACGTACGT"
    assert row.length_bp == 12
    assert row.functional_element == "promoter"
    assert row.species_group == "X"      # from _elem() fixture
    assert row.species == "Homo sapiens"
    assert len(row.dinuc_vec) == 16
    assert len(row.trinuc_vec) == 64
    # GC = 6/12 = 0.5
    assert abs(row.GC_content - 0.5) < 1e-9
    assert "row_42" in row.source


def test_make_probe_row_length_filter_drops_short() -> None:
    elem = _elem("promoter", species="X")
    ds = _ds()
    rec = ReaderResult(sequence="ACGT", raw_label="1", row_idx=0)
    row = _make_probe_row(elem, ds, rec, probe_id="p1",
                          length_min=10, length_max=1024)
    assert row is None


def test_make_probe_row_length_filter_drops_long() -> None:
    elem = _elem("promoter", species="X")
    ds = _ds()
    rec = ReaderResult(sequence="A" * 100, raw_label="1", row_idx=0)
    row = _make_probe_row(elem, ds, rec, probe_id="p1",
                          length_min=10, length_max=50)
    assert row is None


def test_make_probe_row_drops_non_acgt() -> None:
    elem = _elem("foo", species="X")
    ds = _ds()
    rec = ReaderResult(sequence="ACGTNNNN", raw_label="0", row_idx=0)
    row = _make_probe_row(elem, ds, rec, probe_id="p1",
                          length_min=4, length_max=100)
    assert row is None


def test_make_probe_row_applies_crop() -> None:
    elem = _elem("fungi_genome", species="per_row")
    ds = DatasetSpec(
        path="dummy", format="csv", seq_col=None, label_col=None,
        label_policy="multiclass_species", keep_labels=None,
        target_n=1, species=None, crop_to=8, crop_strategy="center",
        balance_per_label=None,
    )
    rec = ReaderResult(sequence="AAAACCCCGGGGTTTT", raw_label="5", row_idx=0)
    row = _make_probe_row(elem, ds, rec, probe_id="p1",
                          length_min=4, length_max=100)
    assert row is not None
    assert row.length_bp == 8
    assert row.sequence == "CCCCGGGG"
    assert row.species == "fungi_sp_5"


# ─────────────────────── config loading ───────────────────────

def test_load_real_panel_config() -> None:
    """Loads the actual data/panel_sources.yaml and validates structure."""
    cfg = load_panel_config()
    assert isinstance(cfg, PanelConfig)
    assert cfg.total_probes == 10000
    assert cfg.seed == 42
    assert cfg.length_min_bp == 128
    assert cfg.length_max_bp == 1024
    assert len(cfg.elements) == 14
    # Sum of element n_probes equals total
    assert sum(e.n_probes for e in cfg.elements) == cfg.total_probes
    # Sum of dataset target_n within each element equals element n_probes
    for e in cfg.elements:
        assert sum(d.target_n for d in e.datasets) == e.n_probes, \
            f"{e.element_id}: ds targets {sum(d.target_n for d in e.datasets)} != n_probes {e.n_probes}"
    # 4 species groups (Yeast merged into Fungi)
    assert set(cfg.species_groups.keys()) == {"Human", "Plant", "Fungi", "Virus"}
