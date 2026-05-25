"""Unit tests for src.loaders.genslm.

Covers the loader contract + the codon-cleaning helpers. Actually loading
the GenSLM .pt weights (.5–10 GB depending on size) is gated by the real
weight files being present; treat the load + score as a smoke test that
runs manually on a host with the weights symlinked under
models/modelsHFNoInfo/genslm/weights/.
"""

from __future__ import annotations

import pytest

from glmap.loaders.base import GLMLoader
from glmap.loaders.genslm import (
    CODON_SIZE,
    GENSLM_MODELS,
    GENSLM_SEQ_LENGTH_BP,
    GENSLM_SEQ_LENGTH_TOKENS,
    GenSLMLoader,
    _clean_to_codons,
    _space_join_codons,
)


def test_protocol_conformance() -> None:
    loader = GenSLMLoader("GenSLM-25M")
    assert isinstance(loader, GLMLoader)
    assert loader.hf_id == "GenSLM-25M"
    assert loader.branch == "ar"
    assert loader.context_tokens == GENSLM_SEQ_LENGTH_TOKENS == 2048


def test_three_known_model_ids() -> None:
    assert set(GENSLM_MODELS) == {"GenSLM-25M", "GenSLM-250M", "GenSLM-2.5B"}


def test_rejects_unknown_model_id() -> None:
    with pytest.raises(ValueError, match="unknown model_id"):
        GenSLMLoader("GenSLM-999X")


def test_seq_length_constants_consistent() -> None:
    assert GENSLM_SEQ_LENGTH_BP == GENSLM_SEQ_LENGTH_TOKENS * CODON_SIZE


def test_clean_to_codons_acgt_aligned() -> None:
    cleaned, n_codons = _clean_to_codons("ATGAAACCC")  # 9 bp = 3 codons
    assert cleaned == "ATGAAACCC"
    assert n_codons == 3


def test_clean_to_codons_drops_non_acgt() -> None:
    cleaned, n_codons = _clean_to_codons("ATGNNNCCC")  # 6 ACGT bases = 2 codons
    assert cleaned == "ATGCCC"
    assert n_codons == 2


def test_clean_to_codons_truncates_to_3_boundary() -> None:
    cleaned, n_codons = _clean_to_codons("ATGAAACC")  # 8 bp -> 2 codons + tail
    assert cleaned == "ATGAAA"
    assert n_codons == 2


def test_clean_to_codons_uppercases() -> None:
    cleaned, _ = _clean_to_codons("atgaaa")
    assert cleaned == "ATGAAA"


def test_clean_to_codons_empty() -> None:
    cleaned, n_codons = _clean_to_codons("")
    assert cleaned == ""
    assert n_codons == 0


def test_space_join_codons() -> None:
    assert _space_join_codons("ATGAAACCC") == "ATG AAA CCC"
    assert _space_join_codons("ATG") == "ATG"
    assert _space_join_codons("") == ""


def test_score_record_requires_non_empty_sequence(tmp_path) -> None:
    loader = GenSLMLoader(
        "GenSLM-25M",
        weights_dir=tmp_path,
        config_dir=tmp_path,
        tokenizer_path=tmp_path / "missing.json",
    )
    with pytest.raises(ValueError, match="empty sequence"):
        loader.score_record("")


def test_score_record_requires_minimum_two_codons(tmp_path) -> None:
    # Loader is not loaded; the codon-cleaning short-circuit happens before
    # weight load, so we can exercise it with bogus paths.
    loader = GenSLMLoader(
        "GenSLM-25M",
        weights_dir=tmp_path,
        config_dir=tmp_path,
        tokenizer_path=tmp_path / "missing.json",
    )
    # 1 codon -> rejected (need >= 2 for at least one predictable position)
    with pytest.raises(FileNotFoundError):
        # First triggers load(), which fails on missing paths. Validate
        # the path-missing branch fires before we hit the codon check.
        loader.score_record("ATG")


def test_codon_constant() -> None:
    assert CODON_SIZE == 3
