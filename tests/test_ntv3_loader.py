"""Tests for src.loaders.ntv3.NTv3MaskedLMLoader and its audit-dispatch
wiring through scripts/run_rerun_stability.py::_audit_entry_to_spec.

Heavy real-model smoke tests are excluded from the default suite — they
require downloading 8M+ NTv3 weights and a GPU. The bookkeeping invariants
that motivated the loader (1-base-per-content-token, length-multiple
padding semantics) are covered in tests/test_mlm_pseudo_ll.py via a fake
model. Here we focus on padding arithmetic, the tokenizer sanity check,
context-overflow guard, and the 8-model dispatch table.
"""

from __future__ import annotations

import math

import pytest
import torch

from glmap.loaders.ntv3 import NTv3MaskedLMLoader


# ─────────────────────── tokenizer sanity check ───────────────────────


class _OneBasePerTokenTokenizer:
    """Single-nucleotide tokenizer: A=10, C=11, G=12, T=13, N=14. Mask=1."""
    mask_token_id = 1
    all_special_ids = [0, 1]
    _base_to_id = {"A": 10, "C": 11, "G": 12, "T": 13, "N": 14}

    def __call__(self, sequence, add_special_tokens=True, return_tensors="pt"):
        ids = [self._base_to_id[c] for c in sequence]
        if add_special_tokens:
            ids = [0, *ids]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _TwoBasesPerTokenTokenizer(_OneBasePerTokenTokenizer):
    """Pretends to be a 2-mer tokenizer: ACGT -> two tokens. Used to assert
    sanity check rejects this."""

    def __call__(self, sequence, add_special_tokens=True, return_tensors="pt"):
        # Naive 2-mer: group bases by 2.
        ids = []
        for i in range(0, len(sequence), 2):
            chunk = sequence[i:i+2]
            ids.append(100 + sum(self._base_to_id[c] for c in chunk))
        if add_special_tokens:
            ids = [0, *ids]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _ConstantLogitMLM:
    """Always emits a constant logit row regardless of input. Lets us run
    score_record without a real HF model."""

    def __init__(self, vocab_size: int = 32, log_p_true: float = math.log(0.5)):
        self.vocab_size = vocab_size
        log_p_rest = math.log((1.0 - math.exp(log_p_true)) / (vocab_size - 1))
        self._row = torch.full((vocab_size,), log_p_rest)
        # Force log_p_true at every base id we care about so any of them
        # masked produces a known log p.
        for base_id in (10, 11, 12, 13, 14):
            self._row[base_id] = log_p_true

    def __call__(self, input_ids: torch.Tensor):
        n = input_ids.shape[-1]
        logits = self._row.unsqueeze(0).expand(n, -1).unsqueeze(0)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def _make_loader(*, length_multiple: int, context_tokens: int = 8192) -> NTv3MaskedLMLoader:
    """Construct a loader and pre-populate _tokenizer/_model so we never
    call load() (which would try to hit HF). Marks the sanity check as
    already done so individual tests can opt in/out."""
    loader = NTv3MaskedLMLoader(
        hf_id="InstaDeepAI/NTv3-fake-for-tests",
        context_tokens=context_tokens,
        length_multiple=length_multiple,
        device="cpu",
    )
    loader._tokenizer = _OneBasePerTokenTokenizer()
    loader._model = _ConstantLogitMLM()
    loader._sanity_checked = True
    return loader


# ─────────────────────── constructor / sanity check ───────────────────────


def test_ntv3_loader_rejects_invalid_length_multiple() -> None:
    with pytest.raises(ValueError, match="length_multiple must be >= 1"):
        NTv3MaskedLMLoader(hf_id="x", context_tokens=1024, length_multiple=0)


def test_ntv3_sanity_check_rejects_multi_base_tokenizer() -> None:
    """If a tokenizer maps 'ACGT' to != 4 content tokens, the
    content_length-based scoring would slice the wrong window. Reject at
    sanity-check time."""
    loader = NTv3MaskedLMLoader(
        hf_id="InstaDeepAI/NTv3-fake-for-tests",
        context_tokens=1024,
        length_multiple=128,
    )
    loader._tokenizer = _TwoBasesPerTokenTokenizer()
    loader._model = _ConstantLogitMLM()

    with pytest.raises(RuntimeError, match="content tokens"):
        loader._sanity_check_tokenizer()


def test_ntv3_sanity_check_accepts_single_base_tokenizer() -> None:
    loader = NTv3MaskedLMLoader(
        hf_id="InstaDeepAI/NTv3-fake-for-tests",
        context_tokens=1024,
        length_multiple=128,
    )
    loader._tokenizer = _OneBasePerTokenTokenizer()
    loader._model = _ConstantLogitMLM()

    loader._sanity_check_tokenizer()  # must not raise
    assert loader._sanity_checked is True


# ─────────────────────── padding arithmetic ───────────────────────


def test_ntv3_loader_pads_short_input_to_length_multiple() -> None:
    """A 156-bp probe with length_multiple=128 must be padded to 256 bp.
    The returned record reports the ORIGINAL 156 as base_length /
    content_position_count, but token_length reflects the padded 256."""
    loader = _make_loader(length_multiple=128)
    seq = "A" * 156

    rec = loader.score_record(seq, stride=6)

    # Bookkeeping: content window restricted to original 156 bases.
    assert rec.base_length == 156
    assert rec.content_position_count == 156
    assert rec.masked_position_count == 156
    assert len(rec.token_log_probs) == 156
    # Model saw the padded 256-base input.
    assert rec.token_length == 256


def test_ntv3_loader_skips_pad_for_aligned_input() -> None:
    """A 1024-bp probe is already divisible by 128 → no pad, no
    content_length, content window == full sequence."""
    loader = _make_loader(length_multiple=128)
    seq = "A" * 1024

    rec = loader.score_record(seq, stride=6)

    assert rec.base_length == 1024
    assert rec.content_position_count == 1024
    assert rec.token_length == 1024
    assert len(rec.token_log_probs) == 1024


def test_ntv3_loader_pads_correctly_for_5downsample() -> None:
    """length_multiple=32 (5-downsample variants): 156 bp -> 160 bp."""
    loader = _make_loader(length_multiple=32)
    seq = "A" * 156

    rec = loader.score_record(seq, stride=6)

    assert rec.base_length == 156
    assert rec.token_length == 160          # 156 padded to 5*32=160
    assert rec.content_position_count == 156


def test_ntv3_loader_pads_correctly_at_exact_multiple_minus_one() -> None:
    """127 bp with length_multiple=128 → pad to 128 (n_pad=1)."""
    loader = _make_loader(length_multiple=128)
    rec = loader.score_record("A" * 127, stride=6)
    assert rec.base_length == 127
    assert rec.token_length == 128


def test_ntv3_loader_pads_correctly_at_exact_multiple_plus_one() -> None:
    """129 bp with length_multiple=128 → pad to 256 (next multiple)."""
    loader = _make_loader(length_multiple=128)
    rec = loader.score_record("A" * 129, stride=6)
    assert rec.base_length == 129
    assert rec.token_length == 256


def test_ntv3_loader_rejects_empty_sequence() -> None:
    loader = _make_loader(length_multiple=128)
    with pytest.raises(ValueError, match="empty sequence"):
        loader.score_record("", stride=6)


def test_ntv3_loader_rejects_overflow_of_context_tokens() -> None:
    """Padding must not silently push past context_tokens."""
    # context_tokens=128, length_multiple=128 → a 130-bp probe would pad to
    # 256, which exceeds 128.
    loader = _make_loader(length_multiple=128, context_tokens=128)
    with pytest.raises(ValueError, match="exceeds context_tokens"):
        loader.score_record("A" * 130, stride=6)


def test_ntv3_loader_score_returns_ell_per_base() -> None:
    """`score()` is the GLMLoader-protocol scalar = `score_record().ell_per_base`."""
    loader = _make_loader(length_multiple=128)
    seq = "A" * 256
    assert loader.score(seq, stride=6) == loader.score_record(seq, stride=6).ell_per_base


# ─────────────────────── audit-dispatch wiring ───────────────────────


def test_dispatch_routes_8_ntv3_models_to_ntv3_loader() -> None:
    """All 8 audit-known NTv3 hf_ids must come out of _audit_entry_to_spec
    with loader_kind='ntv3' and the correct length_multiple (128 for main
    pre / pre_8kb, 32 for the two 5downsample variants)."""
    from glmap.loaders.dispatch import audit_entry_to_spec

    expected = {
        "InstaDeepAI/NTv3_650M_pre": 128,
        "InstaDeepAI/NTv3_100M_pre": 128,
        "InstaDeepAI/NTv3_8M_pre": 128,
        "InstaDeepAI/NTv3_650M_pre_8kb": 128,
        "InstaDeepAI/NTv3_100M_pre_8kb": 128,
        "InstaDeepAI/NTv3_8M_pre_8kb": 128,
        "InstaDeepAI/NTv3_5downsample_pre": 32,
        "InstaDeepAI/NTv3_5downsample_pre_8kb": 32,
    }
    for hf_id, want in expected.items():
        spec = audit_entry_to_spec({
            "hf_id": hf_id,
            "branch": "mlm_or_encoder",
            "context_tokens": 8192,
        })
        assert spec is not None, hf_id
        assert spec.loader_kind == "ntv3", \
            f"{hf_id}: got loader_kind={spec.loader_kind!r}, expected 'ntv3'"
        assert spec.length_multiple == want, \
            f"{hf_id}: got length_multiple={spec.length_multiple}, expected {want}"
        assert spec.branch == "mlm"


def test_dispatch_does_not_route_NT_v1_v2_to_ntv3_loader() -> None:
    """NTv1 / NTv2 family share the InstaDeepAI prefix but are k-mer /
    different architecture; must NOT use NTv3MaskedLMLoader."""
    from glmap.loaders.dispatch import audit_entry_to_spec

    for hf_id in (
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        "InstaDeepAI/agro-nucleotide-transformer-1b",
    ):
        spec = audit_entry_to_spec({
            "hf_id": hf_id,
            "branch": "mlm_or_encoder",
            "context_tokens": 1024,
        })
        assert spec is not None, hf_id
        assert spec.loader_kind != "ntv3", \
            f"{hf_id}: misrouted to NTv3 loader"


def test_modelspec_length_multiple_defaults_to_none() -> None:
    """Non-NTv3 specs must have length_multiple=None; ntv3 specs must not."""
    from glmap.loaders.dispatch import ModelSpec

    plain = ModelSpec(hf_id="x", branch="mlm", context_tokens=1024)
    assert plain.length_multiple is None

    ntv3 = ModelSpec(hf_id="x", branch="mlm", context_tokens=1024,
                      loader_kind="ntv3", length_multiple=128)
    assert ntv3.length_multiple == 128
