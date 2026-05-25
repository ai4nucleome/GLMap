"""Integration tests on the actual built panel parquet files.

These tests run against out_panel/{main_panel,control_panel}.parquet,
which are produced by scripts/build_panel.py and build_control_panel.py.
Skipped if the files don't exist (e.g. fresh checkout where the user has
not yet built the panel).

The goal is to catch silent regressions in the build pipeline — if any
test fails after a code change, the build is producing a panel that
differs from the frozen Stage 2 spec.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PARQUET = REPO_ROOT / "out_panel" / "main_panel.parquet"
CONTROL_PARQUET = REPO_ROOT / "out_panel" / "control_panel.parquet"
MANIFEST = REPO_ROOT / "out_panel" / "panel_manifest.json"

EXPECTED_COLUMNS = [
    "probe_id", "sequence", "length_bp", "functional_element",
    "species_group", "species", "GC_content", "dinuc_vec", "trinuc_vec",
    "source", "label_source",
]


@pytest.fixture(scope="module")
def main_df() -> pd.DataFrame:
    if not MAIN_PARQUET.exists():
        pytest.skip(f"{MAIN_PARQUET} not built yet")
    return pd.read_parquet(MAIN_PARQUET)


@pytest.fixture(scope="module")
def control_df() -> pd.DataFrame:
    if not CONTROL_PARQUET.exists():
        pytest.skip(f"{CONTROL_PARQUET} not built yet")
    return pd.read_parquet(CONTROL_PARQUET)


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} not built yet")
    return json.loads(MANIFEST.read_text())


# ─────────────────────── schema + count gates ───────────────────────

def test_main_panel_row_count(main_df: pd.DataFrame) -> None:
    assert len(main_df) == 10000


def test_control_panel_row_count(control_df: pd.DataFrame) -> None:
    assert len(control_df) == 10000


def test_main_panel_schema(main_df: pd.DataFrame) -> None:
    assert list(main_df.columns) == EXPECTED_COLUMNS


def test_control_panel_schema(control_df: pd.DataFrame) -> None:
    assert list(control_df.columns) == EXPECTED_COLUMNS


def test_probe_ids_unique_within_panels(main_df, control_df) -> None:
    assert main_df["probe_id"].nunique() == len(main_df)
    assert control_df["probe_id"].nunique() == len(control_df)


def test_probe_ids_unique_across_panels(main_df, control_df) -> None:
    combined = pd.concat([main_df["probe_id"], control_df["probe_id"]])
    assert combined.nunique() == 20000


# ─────────────────────── sequence content gates ───────────────────────

_NON_ACGT = re.compile(r"[^ACGT]")


def test_main_panel_acgt_only(main_df: pd.DataFrame) -> None:
    bad = main_df["sequence"].apply(lambda s: bool(_NON_ACGT.search(s))).sum()
    assert bad == 0


def test_control_panel_acgt_only(control_df: pd.DataFrame) -> None:
    bad = control_df["sequence"].apply(lambda s: bool(_NON_ACGT.search(s))).sum()
    assert bad == 0


def test_main_panel_length_range(main_df: pd.DataFrame) -> None:
    assert (main_df["length_bp"] >= 128).all()
    assert (main_df["length_bp"] <= 1024).all()


def test_length_bp_matches_sequence(main_df: pd.DataFrame) -> None:
    assert (main_df["sequence"].str.len() == main_df["length_bp"]).all()


# ─────────────────────── element + group counts vs manifest ───────────────────────

def test_element_counts_match_manifest(main_df, manifest) -> None:
    for elem_id, e in manifest["elements"].items():
        actual = (main_df["functional_element"] == elem_id).sum()
        assert actual == e["n_probes_target"], \
            f"{elem_id}: actual {actual} != target {e['n_probes_target']}"


def test_four_species_groups_in_main_panel(main_df) -> None:
    """species_group is now a column on the parquet; no manifest join needed."""
    # Yeast merged into Fungi per biological taxonomy
    assert set(main_df["species_group"].unique()) == {"Human", "Plant", "Fungi", "Virus"}


def test_species_group_sums(main_df) -> None:
    """Each species_group's total probe count matches the spec — read directly
    from the parquet's species_group column."""
    expected = {"Human": 4000, "Plant": 1600, "Fungi": 2700, "Virus": 1700}
    actual = Counter(main_df["species_group"])
    for g, n in expected.items():
        assert actual[g] == n, f"{g}: actual {actual[g]} != expected {n}"


def test_species_group_consistent_with_manifest(main_df, manifest) -> None:
    """Regression: in-parquet species_group must match what the manifest
    records per element. Catches drift between ProbeRow.species_group
    (sampler writes) and panel_sources.yaml (config truth)."""
    for elem_id, e in manifest["elements"].items():
        in_parquet = main_df.loc[
            main_df["functional_element"] == elem_id, "species_group"
        ].unique()
        assert len(in_parquet) == 1, \
            f"{elem_id}: parquet has multiple groups {in_parquet}"
        assert in_parquet[0] == e["species_group"], \
            f"{elem_id}: parquet={in_parquet[0]} manifest={e['species_group']}"


# ─────────────────────── control subset counts ───────────────────────

def test_control_subsets(control_df: pd.DataFrame) -> None:
    counts = Counter(control_df["functional_element"])
    assert counts["ctrl_random_ACGT"] == 3500
    assert counts["ctrl_dinuc_shuffled"] == 3500
    assert counts["ctrl_motif_spiked"] == 3000


# ─────────────────────── dinuc shuffle exact preservation regression ───────────────────────

def test_dinuc_shuffle_preserves_exactly(main_df, control_df) -> None:
    """Regression for Finding 3: every ctrl_dinuc_shuffled probe must match
    its source's 16-D dinucleotide multiset exactly. Before the fix, ~31/3500
    probes silently fell back to character shuffle and failed this check."""
    main_idx = main_df.set_index("probe_id")
    shuf = control_df[control_df["functional_element"] == "ctrl_dinuc_shuffled"]
    violations = 0
    for _, row in shuf.iterrows():
        src_id = row["source"].split("::", 1)[1]   # "shuffled_from::<probe_id>"
        if src_id not in main_idx.index:
            continue
        src_seq = main_idx.loc[src_id]["sequence"]
        sh_seq = row["sequence"]
        if len(src_seq) != len(sh_seq):
            violations += 1
            continue
        src_dn = Counter(src_seq[i:i + 2] for i in range(len(src_seq) - 1))
        sh_dn = Counter(sh_seq[i:i + 2] for i in range(len(sh_seq) - 1))
        if src_dn != sh_dn:
            violations += 1
    assert violations == 0, f"{violations} probes broke dinuc preservation"


# ─────────────────────── per-species depth (B4 stretch) ───────────────────────

def test_fungi_species_balanced(main_df: pd.DataFrame) -> None:
    """All 20 species_20 species should have exactly 75 probes (balanced)."""
    fungi_sp = main_df[main_df["functional_element"] == "fungi_genome"]
    counts = fungi_sp["species"].value_counts()
    assert len(counts) == 20
    assert (counts == 75).all()


def test_virus_species_40_balanced(main_df: pd.DataFrame) -> None:
    """species_40 has 25 train species, each 44 probes (balanced)."""
    virus_sp = main_df[main_df["functional_element"] == "virus_species"]
    counts = virus_sp["species"].value_counts()
    assert len(counts) == 25
    assert (counts == 44).all()
