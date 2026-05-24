"""GENERator family loader: right-pad input with 'A' to a multiple of 6 bp.

GENERator (https://github.com/GenerTeam/GENERator) ships a 6-mer non-
overlapping tokenizer with an explicit ``<oov>`` token (id 0). When the
input sequence length is **not** a multiple of 6, the tokenizer appends a
single ``<oov>`` token at the end of the token sequence:

    tok("ACGT" * 250)   # 1000 bp  →  [<6-mer>, ..., <6-mer>, <oov>]  (167 tokens)

History: the original loader (commits up to early 2026-05) right-
**truncated** the trailing 0-5 bp to keep ``<oov>`` out of the L matrix.
We've since unified the convention with the new HuggingFaceBio/Carbon
loader: both right-**pad** the trailing partial 6-mer block with ``'A'``
so the final token is a real 6-mer (e.g. ``ATG → ATGAAA``). "Fake data
is fake data" — the small per-base bias from the synthetic 6-mer is
absorbed by ModelMap double-centering at the matrix layer, and we get
the full panel-base count in the per-base denominator.

Important: GENERanno is **not** GENERator. The two share an organization
on HuggingFace but use different tokenizers — GENERanno-eukaryote/prokaryote
ship a single-nucleotide tokenizer (1 token per base, no ``<oov>``) and
must continue to route through the plain `HFCausalLMLoader` /
`HFMaskedLMLoader`. The dispatch in `scripts/run_rerun_stability.py`
explicitly excludes GENERanno from this loader.
"""

from __future__ import annotations

from typing import Any

import torch

from .huggingface import HFCausalLMLoader

K = 6   # GENERator non-overlapping 6-mer tokenizer


def _right_pad_to_k6(sequence: str) -> str:
    """Right-pad with 'A' so that len(out) is a multiple of 6.

    Empty input passes through unchanged so the caller can rely on
    ``ar_score_forward``'s own empty-sequence guard.
    """
    if not sequence:
        return sequence
    pad = (-len(sequence)) % K
    return sequence + "A" * pad if pad else sequence


class GENERatorLoader(HFCausalLMLoader):
    """HFCausalLMLoader specialization for GenerTeam/GENERator-* checkpoints.

    Overrides only ``score_record`` — load() and the underlying state
    (model + tokenizer) are inherited unchanged from HFCausalLMLoader.
    """

    def __init__(
        self,
        hf_id: str,
        context_tokens: int,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = True,
        torch_dtype: Any = None,
    ) -> None:
        super().__init__(
            hf_id=hf_id,
            context_tokens=context_tokens,
            device=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

    def score_record(self, sequence: str):
        from glmap.scoring.ar_likelihood import ar_score_forward

        if self._model is None:
            self.load()
        padded = _right_pad_to_k6(sequence)
        return ar_score_forward(self.model, self.tokenizer, padded, device=self.device)


__all__ = ["GENERatorLoader", "_right_pad_to_k6"]
