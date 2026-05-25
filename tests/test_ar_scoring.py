"""Unit tests for src.scoring.ar_likelihood.

The math under test is small: given a HF causal-LM that produces logits,
gather the log p of the true next token at each shift-by-1 position, then
sum + normalize. We mock the model + tokenizer so tests run in milliseconds
and stay deterministic across machines (no GPU, no HF cache, no network).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from glmap.scoring.ar_likelihood import ar_score_forward


class FakeTokenizer:
    """Minimal tokenizer producing one token per character + 2 special tokens.

    Mimics the BERT-style wrapping observed on Mistral-DNA-v1-*: the encoded
    sequence becomes [CLS] + per-base-token-ids + [SEP], where each base
    becomes `ord(c) % VOCAB_SIZE` to stay within the fake vocab.
    """

    VOCAB_SIZE = 16

    def __init__(self, cls_id: int = 1, sep_id: int = 2):
        self.cls_id = cls_id
        self.sep_id = sep_id

    def __call__(self, sequence, add_special_tokens=True, return_tensors="pt"):
        base_ids = [ord(c) % self.VOCAB_SIZE for c in sequence]
        if add_special_tokens:
            ids = [self.cls_id, *base_ids, self.sep_id]
        else:
            ids = base_ids
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@dataclass
class FakeOutput:
    logits: torch.Tensor


class FakeCausalLM:
    """Deterministic causal-LM stand-in. Returns logits such that the
    gathered log p of the true next token equals a fixed `target_log_p` at
    every shift position (so sum_log_p is exactly target_log_p × (T - 1)).

    The trick: set the logit at the true-next-token index to a value `a`
    such that `log_softmax(a) = target_log_p` given uniform logits = 0 at
    the other (V-1) positions.

        softmax(a) = exp(a) / (exp(a) + (V - 1))    ;  rest of logits = 0
        log_softmax(a) = a - log(exp(a) + (V - 1))
        Solve for a given target_log_p < 0:
            p = exp(target_log_p)
            a = log(p * (V - 1) / (1 - p))
    """

    def __init__(self, target_log_p: float, vocab_size: int = FakeTokenizer.VOCAB_SIZE):
        if target_log_p > 0:
            raise ValueError("target_log_p must be <= 0")
        self.target_log_p = float(target_log_p)
        self.vocab_size = int(vocab_size)
        if target_log_p == 0.0:
            self._a = float("inf")     # delta at true target -> log p(true) -> 0
        else:
            p = math.exp(self.target_log_p)
            # Guard against p ≥ 1 (target_log_p ≥ 0); handled above.
            self._a = math.log(p * (self.vocab_size - 1) / (1 - p))

    def __call__(self, input_ids: torch.Tensor):
        T = int(input_ids.shape[-1])
        logits = torch.zeros((1, T, self.vocab_size))
        if math.isinf(self._a):
            # Delta-style logits: set true target to large positive, others
            # to large negative — softmax becomes a one-hot at true target.
            for t in range(T - 1):
                logits[0, t, :] = -1e9
                logits[0, t, int(input_ids[0, t + 1])] = 0.0
        else:
            for t in range(T - 1):
                logits[0, t, int(input_ids[0, t + 1])] = self._a
        return FakeOutput(logits=logits)


def test_ar_score_arithmetic_matches_definition() -> None:
    """sum_log_p == per-token log p × (n_tokens - 1); ell_per_base divides
    by base length; bpb is in bits."""
    sequence = "ACGTACGT"   # 8 bases
    fake_tok = FakeTokenizer()
    target_log_p = -1.2     # nats per predictable token
    fake_model = FakeCausalLM(target_log_p=target_log_p)

    record = ar_score_forward(fake_model, fake_tok, sequence)

    assert record.base_length == 8
    # Scoring uses add_special_tokens=False for cross-family parity (commit 088e91b);
    # tokenizer yields 8 content tokens, no specials.
    assert record.token_length == 8
    assert record.token_length_no_special == 8
    assert record.special_tokens_count == 0
    assert record.predictable_tokens == 7       # n_tokens - 1
    assert math.isclose(record.ce_loss, 1.2, rel_tol=1e-5)

    expected_sum_log_p = target_log_p * 7       # = -8.4
    assert math.isclose(record.sum_log_p, expected_sum_log_p, rel_tol=1e-5)
    assert math.isclose(record.ell_per_base, expected_sum_log_p / 8, rel_tol=1e-5)
    assert math.isclose(record.bpb, -record.ell_per_base / math.log(2), rel_tol=1e-5)
    # Per-token log p list: length T-1, every entry == target_log_p.
    assert len(record.token_log_probs) == 7
    for lp in record.token_log_probs:
        assert math.isclose(lp, target_log_p, rel_tol=1e-5)
    assert math.isclose(sum(record.token_log_probs), record.sum_log_p, rel_tol=1e-5)


def test_ar_score_zero_loss_yields_perfect_compression() -> None:
    """target_log_p = 0 means every true token gets probability 1 -> bpb = 0."""
    fake_tok = FakeTokenizer()
    fake_model = FakeCausalLM(target_log_p=0.0)

    record = ar_score_forward(fake_model, fake_tok, "ACGT")

    assert math.isclose(record.sum_log_p, 0.0, abs_tol=1e-6)
    assert math.isclose(record.ell_per_base, 0.0, abs_tol=1e-6)
    assert math.isclose(record.bpb, 0.0, abs_tol=1e-6)
    assert len(record.token_log_probs) == record.predictable_tokens
    for lp in record.token_log_probs:
        assert math.isclose(lp, 0.0, abs_tol=1e-6)


def test_ar_score_uniform_dna_matches_log4_per_base_under_one_to_one_tokens() -> None:
    """When tokens == bases (1:1 tokenization, no special tokens), a model
    emitting -log p = log(4) per token should produce ell_per_base close to
    -log(4) for long sequences (off by (T-1)/T factor for shift)."""
    seq = "A" * 200
    fake_tok = FakeTokenizer()
    # log p = -log(4) per token means uniform 4-way prediction over ACGT.
    fake_model = FakeCausalLM(target_log_p=-math.log(4))

    record = ar_score_forward(fake_model, fake_tok, seq)

    # With add_special_tokens=False: token_length = 200, predictable_tokens = 199
    # sum_log_p = -log(4) * 199; base_length = 200
    # ell_per_base = -log(4) * 199 / 200 ≈ -log(4) for large len
    expected = -math.log(4) * 199 / 200
    assert math.isclose(record.ell_per_base, expected, rel_tol=1e-5)
    # Converges to -log(4) as length grows.
    assert abs(record.ell_per_base - (-math.log(4))) < 0.02


def test_ar_score_empty_sequence_raises() -> None:
    import pytest

    fake_tok = FakeTokenizer()
    fake_model = FakeCausalLM(target_log_p=0.0)
    with pytest.raises(ValueError, match="empty sequence"):
        ar_score_forward(fake_model, fake_tok, "")


def test_ar_score_rerun_is_bitwise_identical() -> None:
    """Same inputs must produce the same record; HF models in eval()+no_grad
    already do this in practice, the assertion here is for the math layer."""
    fake_tok = FakeTokenizer()
    fake_model = FakeCausalLM(target_log_p=-2.345)

    r1 = ar_score_forward(fake_model, fake_tok, "ACGTACGT")
    r2 = ar_score_forward(fake_model, fake_tok, "ACGTACGT")

    assert r1 == r2
