"""DNABERT k=3..6 loader.

DNABERT (zhihan1996/DNA_bert_3 .. DNA_bert_6) ships with a tokenizer that
expects pre-tokenized overlapping k-mer input separated by spaces — raw
ACGT goes straight to [UNK]. This loader wraps HFMaskedLMLoader to
auto-format the sequence before tokenization.

phase_1.md flags a known caveat for this family:

    DNABERT overlapping k-mer: mask 必须覆盖目标 base 的整个 overlapping span
    防止信息泄漏。

That full leak-proof masking is supplement-scope; phase 0 / 1 main signature
uses the same single-token stride PLL as other MLM encoders, with the
understanding that DNABERT k-mer mask leakage inflates ell_per_base a bit
toward zero (easier predictions). Sensitivity will be quantified in the
phase_1 stride-k supplement.

`k` is auto-inferred from the hf_id (`DNA_bert_3` -> 3, etc.). Override via
the `k` constructor argument if needed.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from .huggingface import HFMaskedLMLoader

_K_PATTERN = re.compile(r"DNA[_]?bert[_-](\d+)", re.IGNORECASE)


def _infer_k_from_hf_id(hf_id: str) -> int:
    m = _K_PATTERN.search(hf_id)
    if not m:
        raise ValueError(
            f"DNABERTLoader: could not infer k from hf_id={hf_id!r}; "
            "pass k=... explicitly."
        )
    return int(m.group(1))


def sequence_to_kmer_string(sequence: str, k: int) -> str:
    """Convert raw ACGT into space-separated overlapping k-mer tokens."""
    if len(sequence) < k:
        raise ValueError(
            f"sequence length {len(sequence)} shorter than k={k}; "
            "DNABERT cannot score sub-k sequences"
        )
    return " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))


class _SpaceWrappingTokenizer:
    """A thin shim that wraps a DNABERT tokenizer so callers can pass raw
    ACGT and we transparently pre-tokenize into space-separated k-mers.

    Only forwards the calls used by src.scoring.mlm_pseudo_ll.
    """

    def __init__(self, inner: Any, k: int):
        self._inner = inner
        self._k = k

    @property
    def mask_token_id(self):
        return self._inner.mask_token_id

    @property
    def all_special_ids(self):
        return self._inner.all_special_ids

    def __call__(self, sequence: str, **kwargs):
        kmer_str = sequence_to_kmer_string(sequence, self._k)
        return self._inner(kmer_str, **kwargs)


class DNABERTLoader(HFMaskedLMLoader):
    """HF MaskedLM loader for DNABERT k=3..6 with k-mer pre-tokenization."""

    def __init__(
        self,
        hf_id: str,
        context_tokens: int = 512,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = True,
        k: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            hf_id=hf_id,
            context_tokens=context_tokens,
            device=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.k = k if k is not None else _infer_k_from_hf_id(hf_id)

    def load(self) -> None:
        super().load()
        # Wrap the loaded HF tokenizer in our pre-tokenizing shim.
        self._tokenizer = _SpaceWrappingTokenizer(self._tokenizer, self.k)


__all__ = ["DNABERTLoader", "sequence_to_kmer_string"]
