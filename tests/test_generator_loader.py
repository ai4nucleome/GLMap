"""Tests for src/loaders/generator.py.

GENERator's 6-mer tokenizer appends an <oov> token whenever the input
sequence length is not a multiple of 6 (per upstream README at
https://github.com/GenerTeam/GENERator). The loader right-pads with 'A'
so the trailing partial 6-mer becomes a real 6-mer (e.g. ``ATG → ATGAAA``);
this unifies the convention with the new HuggingFaceBio/Carbon loader,
which also A-pads. Prior to this change, the loader right-truncated the
tail; the convention switch invalidates older GENERator scores (now
renamed `probes.parquet.old_truncate_convention`) and the sweep resume
check re-runs them.

These tests pin down the padding helper and verify dispatch wiring
without needing the multi-GB checkpoints — those are exercised by the
multi-env stability sweep, not unit tests.
"""

from __future__ import annotations

import pytest

from glmap.loaders.generator import GENERatorLoader, _right_pad_to_k6


# ─────────────────────── pure helper ───────────────────────

def test_pad_multiple_of_6_passthrough() -> None:
    """L divisible by 6 returns the input unchanged."""
    assert _right_pad_to_k6("ACGTAC") == "ACGTAC"             # L=6
    assert _right_pad_to_k6("A" * 12) == "A" * 12             # L=12
    assert _right_pad_to_k6("A" * 1020) == "A" * 1020         # L=1020 (mod 6 = 0)


def test_pad_appends_1_to_5_A_bases() -> None:
    for tail_len in range(1, 6):
        seq = "ACGTAC" + "C" * tail_len    # use C to distinguish padded A's
        out = _right_pad_to_k6(seq)
        assert len(out) % 6 == 0
        assert len(out) == 6 + (6 if tail_len > 0 else 0)
        pad_added = len(out) - len(seq)
        assert pad_added == (6 - tail_len)
        assert out.endswith("A" * pad_added), f"tail_len={tail_len}: expected trailing A's"


def test_pad_panel_lengths() -> None:
    """Spot-check the lengths that actually appear in out_panel/main_panel.parquet."""
    cases = {
        199: 204,    # 199 + 5
        200: 204,    # 200 + 4
        300: 300,    # divisible
        398: 402,    # PGB splicing
        400: 402,    # 400 + 2
        500: 504,    # 500 + 4
        999: 1002,   # GUE covid
        1000: 1002,  # 1000 + 2
        1024: 1026,  # 1024 + 2
    }
    for L_in, L_out in cases.items():
        seq = "C" * L_in
        out = _right_pad_to_k6(seq)
        assert len(out) == L_out, f"L={L_in}: expected {L_out}, got {len(out)}"
        assert len(out) % 6 == 0


def test_pad_empty_passthrough() -> None:
    """Empty input is delegated unchanged — the scoring layer raises its
    own ValueError, the padder doesn't second-guess that."""
    assert _right_pad_to_k6("") == ""


# ─────────────────────── loader contract ───────────────────────

def test_loader_inherits_hf_causal_lm_loader_attrs() -> None:
    """The loader exposes the GLMLoader-protocol surface inherited from
    HFCausalLMLoader without needing the real weights."""
    loader = GENERatorLoader(
        hf_id="GenerTeam/GENERator-eukaryote-1.2b-base",
        context_tokens=1024,
        device="cpu",
        trust_remote_code=True,
    )
    assert loader.hf_id == "GenerTeam/GENERator-eukaryote-1.2b-base"
    assert loader.context_tokens == 1024
    assert loader.branch == "ar"
    # tokenizer / model not loaded yet — accessing should raise.
    with pytest.raises(RuntimeError):
        _ = loader.tokenizer
    with pytest.raises(RuntimeError):
        _ = loader.model


# ─────────────────────── dispatch ───────────────────────

def test_dispatch_routes_generator_to_generator_loader() -> None:
    """audit-entry → ModelSpec dispatch in scripts/run_rerun_stability.py
    must produce loader_kind='generator' for GENERator-* hf_ids."""
    from glmap.loaders.dispatch import audit_entry_to_spec

    for hf_id in (
        "GenerTeam/GENERator-eukaryote-1.2b-base",
        "GenerTeam/GENERator-v2-eukaryote-3b-base",
        "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    ):
        spec = audit_entry_to_spec({
            "hf_id": hf_id,
            "branch": "ar_or_generative",
            "context_tokens": 1024,
        })
        assert spec is not None, hf_id
        assert spec.loader_kind == "generator", \
            f"{hf_id}: got loader_kind={spec.loader_kind!r}, expected 'generator'"


def test_dispatch_does_not_route_generanno_to_generator_loader() -> None:
    """GENERanno is single-nucleotide; must NOT use GENERatorLoader."""
    from glmap.loaders.dispatch import audit_entry_to_spec

    for hf_id in (
        "GenerTeam/GENERanno-eukaryote-0.5b-base",
        "GenerTeam/GENERanno-prokaryote-0.5b-base",
    ):
        spec = audit_entry_to_spec({
            "hf_id": hf_id,
            "branch": "ar_or_generative",
            "context_tokens": 1024,
        })
        assert spec is not None, hf_id
        assert spec.loader_kind == "hf", \
            f"{hf_id}: got loader_kind={spec.loader_kind!r}, expected 'hf'"


# ─────────────────────── audit classifier fix ───────────────────────

def test_audit_classifies_generator_vs_generanno_correctly() -> None:
    """common.infer_tokenizer_type used to blanket-classify both GENERator
    and GENERanno as non_overlapping_6mer. Empirical: GENERanno is
    single-nucleotide. Regression for that fix."""
    import sys
    from pathlib import Path
    audit_dir = Path(__file__).resolve().parents[1] / "scripts" / "audits"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    import common as audit

    assert audit.infer_tokenizer_type("GenerTeam/GENERator-eukaryote-1.2b-base") \
        == "non_overlapping_6mer"
    assert audit.infer_tokenizer_type("GenerTeam/GENERator-v2-prokaryote-3b-base") \
        == "non_overlapping_6mer"
    assert audit.infer_tokenizer_type("GenerTeam/GENERanno-eukaryote-0.5b-base") \
        == "single_nucleotide_or_byte"
    assert audit.infer_tokenizer_type("GenerTeam/GENERanno-prokaryote-0.5b-base") \
        == "single_nucleotide_or_byte"
