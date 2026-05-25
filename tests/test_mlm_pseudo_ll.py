"""Unit tests for src.scoring.mlm_pseudo_ll.

We mock an HF MaskedLM-compatible model that returns deterministic logits
so the test asserts the actual masking arithmetic: which positions get
masked at each offset, what log p is gathered at the true id, and how the
final ell_per_base is normalized."""

from __future__ import annotations

import math

import pytest
import torch

from glmap.scoring.mlm_pseudo_ll import MLMScore, stride_pll_forward


class FakeTokenizer:
    """Tokenizer producing [CLS] + per-character base-encoded tokens.

    - cls_token_id = 0 (special)
    - mask_token_id = 1 (special)
    - base token ids: A=10, C=11, G=12, T=13
    - all_special_ids = [0, 1]
    """

    cls_token_id = 0
    mask_token_id = 1
    all_special_ids = [0, 1]
    _base_to_id = {"A": 10, "C": 11, "G": 12, "T": 13}

    def __call__(self, sequence, add_special_tokens=True, return_tensors="pt"):
        base_ids = [self._base_to_id[c] for c in sequence]
        ids = [self.cls_token_id, *base_ids] if add_special_tokens else base_ids
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class FakeMaskedLM:
    """Returns the same constant-log-prob vector at every position.

    Specifically, log p(token id = 10) = log(0.5), and the remaining mass
    is split uniformly across the rest of the (fake) 14-token vocab.
    This makes the gathered sum_log_p easy to predict.
    """

    def __init__(self, vocab_size: int = 14, log_p_true: float = math.log(0.5)):
        self.vocab_size = vocab_size
        self.log_p_true = log_p_true
        # Reserve the desired log_p at id=10; spread the rest uniformly.
        # log p_rest = log((1 - exp(log_p_true)) / (vocab_size - 1))
        log_p_rest = math.log((1.0 - math.exp(log_p_true)) / (vocab_size - 1))
        self._row = torch.full((vocab_size,), log_p_rest)
        self._row[10] = log_p_true

    def __call__(self, input_ids: torch.Tensor):
        n = input_ids.shape[-1]
        # Same log-probs at every position.
        logits = self._row.unsqueeze(0).expand(n, -1).unsqueeze(0)

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def _build_test_inputs(sequence: str = "AAAAAA"):
    return FakeMaskedLM(), FakeTokenizer(), sequence


def test_stride_pll_masks_every_content_position_exactly_once() -> None:
    """Sum across stride offsets must mask each content position exactly once.

    With sequence="AAAAAA" (6 bases, all id=10), every masked position predicts
    id=10 with log_p_true = log(0.5). Total predictions = 6. Sum log p = 6 * log(0.5).
    ell_per_base = 6 * log(0.5) / 6 = log(0.5).
    """
    model, tok, seq = _build_test_inputs("AAAAAA")
    rec = stride_pll_forward(model, tok, seq, stride=6)

    assert isinstance(rec, MLMScore)
    assert rec.base_length == 6
    # add_special_tokens=False: 6 content tokens, no CLS/SEP wrappers
    assert rec.token_length == 6
    assert rec.token_length_no_special == 6
    assert rec.content_position_count == 6
    assert rec.masked_position_count == 6  # every content pos masked once
    assert math.isclose(rec.sum_log_p, 6 * math.log(0.5), rel_tol=1e-6)
    assert math.isclose(rec.ell_per_base, math.log(0.5), rel_tol=1e-6)
    assert math.isclose(rec.bpb, -math.log(0.5) / math.log(2), rel_tol=1e-6)


def test_stride_pll_stride_partitions_positions_correctly() -> None:
    """Stride 3 over 6 content positions must produce 2 mask events per offset
    (positions 0,3 at offset 0; 1,4 at offset 1; 2,5 at offset 2)."""
    model, tok, seq = _build_test_inputs("AAAAAA")
    # stride=3 should still yield 6 total masked positions (3 offsets x 2 each)
    rec = stride_pll_forward(model, tok, seq, stride=3)
    assert rec.masked_position_count == 6
    assert math.isclose(rec.ell_per_base, math.log(0.5), rel_tol=1e-6)


def test_stride_pll_stride_larger_than_content_masks_one_per_offset() -> None:
    """If stride > content_count, offsets after content_count produce 0 masks
    but earlier offsets still cover all positions once (one position per offset)."""
    model, tok, seq = _build_test_inputs("AAA")  # 3 content positions
    rec = stride_pll_forward(model, tok, seq, stride=10)
    assert rec.content_position_count == 3
    assert rec.masked_position_count == 3
    # Each base is masked once -> sum log p = 3 * log(0.5)
    assert math.isclose(rec.sum_log_p, 3 * math.log(0.5), rel_tol=1e-6)


def test_stride_pll_requires_mask_token() -> None:
    """Tokenizers lacking a [MASK] token cannot do PLL; raise ValueError."""

    class NoMaskTokenizer(FakeTokenizer):
        mask_token_id = None

    with pytest.raises(ValueError, match="mask_token_id"):
        stride_pll_forward(FakeMaskedLM(), NoMaskTokenizer(), "ACGT", stride=6)


def test_stride_pll_rejects_empty_and_zero_stride() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        stride_pll_forward(FakeMaskedLM(), FakeTokenizer(), "", stride=6)
    with pytest.raises(ValueError, match="stride must be"):
        stride_pll_forward(FakeMaskedLM(), FakeTokenizer(), "ACGT", stride=0)


def test_stride_pll_only_masks_content_positions() -> None:
    """The [CLS] position must never appear in masked_position_count."""
    model, tok, seq = _build_test_inputs("ACGT")
    rec = stride_pll_forward(model, tok, seq, stride=2)
    # content_position_count = 4 (ACGT), CLS at position 0 is special
    assert rec.content_position_count == 4
    # All 4 content positions masked once across 2 offsets
    assert rec.masked_position_count == 4


def test_stride_pll_score_normalized_by_base_length_not_token_length() -> None:
    """Multi-base-per-token tokenizers exist (BPE / non-overlapping k-mer).
    Even with our 1:1 fake tokenizer, the assertion is that the divisor is
    len(sequence), as the docstring guarantees."""
    model, tok, seq = _build_test_inputs("AAAAAAAA")  # 8 bases, all 'A' (id=10)
    rec = stride_pll_forward(model, tok, seq, stride=4)
    assert rec.base_length == 8
    # FakeMaskedLM emits log(0.5) at id=10 only. All 8 bases are 'A',
    # so every gathered log p == log(0.5).
    assert math.isclose(rec.ell_per_base, math.log(0.5), rel_tol=1e-6)


# ─────────────────────── content_length opt-in (NTv3 U-Net padding) ───────────────────────

def test_stride_pll_default_content_length_path_is_unchanged() -> None:
    """API backward-compat: omitting content_length must produce results
    identical to passing content_length=None (which itself is identical to
    pre-existing behavior). Compare against the well-trodden 6-base case."""
    model, tok, _ = _build_test_inputs()
    rec_default = stride_pll_forward(model, tok, "ACGTAC", stride=6)
    rec_explicit_none = stride_pll_forward(model, tok, "ACGTAC", stride=6,
                                            content_length=None)
    assert rec_default == rec_explicit_none


def test_stride_pll_content_length_validates_args() -> None:
    """content_length must be >= 1 and <= content_position_count when set."""
    model, tok, _ = _build_test_inputs()
    with pytest.raises(ValueError, match="content_length must be >= 1"):
        stride_pll_forward(model, tok, "ACGTAC", stride=6, content_length=0)
    with pytest.raises(ValueError, match="content_length must be >= 1"):
        stride_pll_forward(model, tok, "ACGTAC", stride=6, content_length=-3)
    with pytest.raises(ValueError, match="exceeds content_position_count"):
        # 6-base sequence has 6 content positions; 7 is out of range.
        stride_pll_forward(model, tok, "ACGTAC", stride=6, content_length=7)


def test_stride_pll_content_length_bookkeeping_restricts_to_prefix() -> None:
    """When content_length=k is set, the returned MLMScore must reflect ONLY
    the first k content positions:

      - base_length == content_length            (not full sequence length)
      - content_position_count == content_length (not full count)
      - token_log_probs has length content_length
      - sum_log_p / ell_per_base / bpb are derived from those k log p's only
      - masked_position_count == content_length (every prefix pos masked once)

    Note: bookkeeping only. Real models would see attention from the padded
    tail positions, which can change the prefix logits; that semantic
    equivalence is intentionally NOT asserted here (fake model has constant
    logits, so it happens to coincide).
    """
    model, tok, _ = _build_test_inputs()
    # 10-base sequence "AAAAAAAAAA" simulates a padded NTv3 input where the
    # caller wants to score only the first 6 bases as "content".
    rec = stride_pll_forward(model, tok, "AAAAAAAAAA", stride=6, content_length=6)

    # Bookkeeping must reflect the 6-base content window, not the 10-base input.
    assert rec.base_length == 6
    assert rec.content_position_count == 6
    assert rec.masked_position_count == 6                  # each prefix pos masked once
    assert len(rec.token_log_probs) == 6
    # token_length still reflects the full input the model actually saw.
    assert rec.token_length == 10

    # FakeMaskedLM emits log(0.5) at id=10 for every position; with 6 'A's
    # in the scoring window, sum_log_p must == 6 * log(0.5).
    assert math.isclose(rec.sum_log_p, 6 * math.log(0.5), rel_tol=1e-6)
    assert math.isclose(rec.ell_per_base, math.log(0.5), rel_tol=1e-6)
    assert math.isclose(rec.bpb, -math.log(0.5) / math.log(2), rel_tol=1e-6)


def test_stride_pll_content_length_full_equals_default() -> None:
    """content_length == content_position_count must be a no-op (equivalent
    to omitting it). Catches off-by-one mistakes in the slicing."""
    model, tok, _ = _build_test_inputs()
    rec_default = stride_pll_forward(model, tok, "ACGTAC", stride=6)
    rec_full = stride_pll_forward(model, tok, "ACGTAC", stride=6, content_length=6)
    assert rec_default == rec_full
