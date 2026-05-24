"""Carbon family loader: <dna>-tag DNA wrap + 6-mer A-padding.

HuggingFaceBio/Carbon-* (500M / 3B / 8B) is a Llama-style causal LM with a
hybrid Qwen3-BPE + DNA-6-mer tokenizer (`HybridDNATokenizer`). The
tokenizer is unusual in three ways:

  1. It only switches into 6-mer DNA mode when it sees the `<dna>` tag.
     Without the tag, ACGT letters fall back to BPE, which is "a different
     language" for the model (Carbon model card § "DNA inputs"). We
     therefore prepend a literal `<dna>` (id 151669, a single special
     token) before every input.

  2. We deliberately **do not** append `</dna>`. The closing tag adds a
     non-DNA prediction target at the end of the sequence (a `</dna>`
     token whose log p the model has trained against in mixed-modality
     contexts, but which is not a DNA-prediction event); leaving it off
     makes every predictable shift target a DNA 6-mer.

  3. Sequences whose length is not a multiple of 6 are right-padded with
     `'A'` to the next multiple of 6. The upstream tokenizer already does
     this automatically (`tokenizer.py:240-246`), but we still pad
     explicitly so the `base_length` we report equals the number of bases
     the model actually scored (real + padded A's).

This wrapper inherits HFCausalLMLoader's load() path; only `score_record`
diverges to apply the wrap + pad + base_length override. Output ARScore
carries `base_length = len(padded_sequence)` (multiple of 6), matching
the GENERator convention (which since the v2 unification also A-pads).

Reference: model card https://huggingface.co/HuggingFaceBio/Carbon-3B
"""

from __future__ import annotations

from typing import Any

import torch

from .huggingface import HFCausalLMLoader

K = 6
DNA_OPEN_TAG = "<dna>"


def _right_pad_to_k6(sequence: str) -> str:
    """Right-pad with 'A' so that len(out) is a multiple of 6.

    Empty input passes through unchanged so the caller can rely on
    ar_score_forward's own empty-sequence guard. Per the Carbon model
    card, padded bases are "fake data" the model treats as part of the
    final 6-mer; cross-model bias is absorbed by ModelMap double-centering.
    """
    if not sequence:
        return sequence
    pad = (-len(sequence)) % K
    return sequence + "A" * pad if pad else sequence


class CarbonCausalLMLoader(HFCausalLMLoader):
    """HFCausalLMLoader specialization for HuggingFaceBio/Carbon-* checkpoints.

    Overrides only score_record — load() inherits unchanged. The tokenizer
    requires trust_remote_code=True (custom HybridDNATokenizer); the model
    itself is standard LlamaForCausalLM and does not. bf16 is the
    upstream-recommended dtype.
    """

    def __init__(
        self,
        hf_id: str,
        context_tokens: int,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = True,
        torch_dtype: Any = None,
    ) -> None:
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
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
        if not padded:
            return ar_score_forward(self.model, self.tokenizer, padded, device=self.device)
        wrapped = DNA_OPEN_TAG + padded
        return ar_score_forward(
            self.model, self.tokenizer, wrapped,
            device=self.device,
            base_length_override=len(padded),
        )


__all__ = ["CarbonCausalLMLoader", "_right_pad_to_k6", "K", "DNA_OPEN_TAG"]
