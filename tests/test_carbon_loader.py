"""Tests for src/loaders/carbon.py.

HuggingFaceBio/Carbon-* is a Llama-style causal LM with a HybridDNATokenizer
that switches into 6-mer DNA mode only when it sees a `<dna>` tag and
auto right-pads with 'A' for partial 6-mer blocks. The loader prepends
`<dna>` (no closing `</dna>`) and explicitly A-pads to a multiple of 6,
then reports `base_length = len(padded_dna)` via ar_score_forward's
new `base_length_override` parameter.

These tests pin down the padding helper, the audit classifier, the
dispatch wiring, and the ar_score_forward `base_length_override`
contract — none require the actual 0.5B+ weights.
"""

from __future__ import annotations

import pytest

from glmap.loaders.carbon import CarbonCausalLMLoader, _right_pad_to_k6, DNA_OPEN_TAG


# ─────────────────────── pure helper ───────────────────────

def test_pad_multiple_of_6_passthrough() -> None:
    assert _right_pad_to_k6("ACGTAC") == "ACGTAC"
    assert _right_pad_to_k6("A" * 12) == "A" * 12
    assert _right_pad_to_k6("A" * 600) == "A" * 600


def test_pad_appends_A_to_partial_block() -> None:
    for tail_len in range(1, 6):
        seq = "CCCCCC" + "C" * tail_len   # use C so padded A's are visible
        out = _right_pad_to_k6(seq)
        assert len(out) % 6 == 0
        pad_added = len(out) - len(seq)
        assert pad_added == (6 - tail_len)
        assert out.endswith("A" * pad_added)


def test_pad_empty_passthrough() -> None:
    assert _right_pad_to_k6("") == ""


def test_dna_open_tag_constant() -> None:
    assert DNA_OPEN_TAG == "<dna>"


# ─────────────────────── loader contract ───────────────────────

def test_loader_inherits_hf_causal_lm_loader_attrs() -> None:
    loader = CarbonCausalLMLoader(
        hf_id="HuggingFaceBio/Carbon-500M",
        context_tokens=8192,
        device="cpu",
    )
    assert loader.hf_id == "HuggingFaceBio/Carbon-500M"
    assert loader.context_tokens == 8192
    assert loader.branch == "ar"
    assert loader.trust_remote_code is True
    with pytest.raises(RuntimeError):
        _ = loader.tokenizer
    with pytest.raises(RuntimeError):
        _ = loader.model


def test_loader_default_dtype_is_bf16() -> None:
    import torch
    loader = CarbonCausalLMLoader(
        hf_id="HuggingFaceBio/Carbon-500M",
        context_tokens=8192,
        device="cpu",
    )
    assert loader.torch_dtype == torch.bfloat16


# ─────────────────────── dispatch ───────────────────────

def test_dispatch_routes_carbon_to_carbon_loader() -> None:
    from glmap.loaders.dispatch import audit_entry_to_spec

    for hf_id in (
        "HuggingFaceBio/Carbon-500M",
        "HuggingFaceBio/Carbon-3B",
        "HuggingFaceBio/Carbon-8B",
    ):
        spec = audit_entry_to_spec({
            "hf_id": hf_id,
            "branch": "ar_or_generative",
            "context_tokens": 8192,
        })
        assert spec is not None, hf_id
        assert spec.loader_kind == "carbon", \
            f"{hf_id}: got loader_kind={spec.loader_kind!r}, expected 'carbon'"
        assert spec.branch == "ar"
        assert spec.trust_remote_code is True
        assert spec.is_codon is False


# ─────────────────────── audit classifier ───────────────────────

def test_audit_classifies_carbon_family_and_tokenizer() -> None:
    import sys
    from pathlib import Path
    audit_dir = Path(__file__).resolve().parents[1] / "scripts" / "audits"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    import common as audit

    for hf_id in (
        "HuggingFaceBio/Carbon-500M",
        "HuggingFaceBio/Carbon-3B",
        "HuggingFaceBio/Carbon-8B",
    ):
        assert audit.infer_family(hf_id) == "Carbon", hf_id
        assert audit.infer_branch(hf_id) == "ar_or_generative", hf_id
        assert audit.infer_tokenizer_type(hf_id) == "non_overlapping_6mer", hf_id


# ─────────────────────── ar_score_forward override ───────────────────────

def test_ar_score_forward_base_length_override_validates() -> None:
    """base_length_override <= 0 raises; positive overrides change the
    denominator used for ell_per_base."""
    from glmap.scoring.ar_likelihood import ar_score_forward

    class _FakeTokOut:
        def __init__(self, ids): self.input_ids = ids
        def to(self, device): return self
        def __getitem__(self, k):
            assert k == "input_ids"
            return self.input_ids

    # Build a fake tokenizer that returns a known input_ids tensor.
    import torch

    class _FakeTok:
        def __call__(self, sequence, **kwargs):
            ids = torch.tensor([[1, 2, 3, 4]])
            return {"input_ids": ids}

    class _FakeOut:
        def __init__(self, logits): self.logits = logits

    class _FakeModel:
        def __call__(self, input_ids=None, **kwargs):
            T = int(input_ids.shape[-1])
            return _FakeOut(torch.zeros(1, T, 5))

    # Sanity: override must be > 0
    with pytest.raises(ValueError):
        ar_score_forward(
            _FakeModel(), _FakeTok(), "ATGC", device="cpu",
            base_length_override=0,
        )

    # Default path: base_length = len(sequence)
    score = ar_score_forward(_FakeModel(), _FakeTok(), "ATGC", device="cpu")
    assert score.base_length == 4

    # Override path
    score2 = ar_score_forward(
        _FakeModel(), _FakeTok(), "<dna>ATGC", device="cpu",
        base_length_override=4,
    )
    assert score2.base_length == 4
