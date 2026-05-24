"""Evo2 loader (arcinstitute/evo2_*) via the official evo2 package.

Evo2 uses StripedHyena 2 (Vortex) inference — not HF AutoModel. The
`evo2.Evo2(model_name)` factory pulls weights from the HF cache
(`models--arcinstitute--<model_name>`) and exposes:

    model, _ = self.model(input_ids)   # (batch, length, vocab) logits

We compute AR per-base log-likelihood with the standard shift-by-1
convention, identical to HF AutoModelForCausalLM.

Per the Evo2 README, the 7B models run in bfloat16 without
Transformer Engine; 1B/20B/40B variants require FP8 + Hopper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Evo2Loader:
    hf_id: str
    context_tokens: int
    device: str | torch.device = "cuda:0"
    trust_remote_code: bool = False
    torch_dtype: Any = None
    is_codon: bool = False
    branch: str = "ar"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self._tokenizer = None
        self._model = None
        # Map HF id to the model_name keys the evo2 package expects.
        # evo-design's repo IDs aren't direct evo2 model_name keys.
        last = self.hf_id.split("/")[-1]
        _alias = {
            "evo-2-7b-8k-microviridae": "evo2_7b_microviridae",
        }
        self._model_name = _alias.get(last, last)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError(f"{self.hf_id}: call load() before tokenizer")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(f"{self.hf_id}: call load() before model")
        return self._model

    def load(self) -> None:
        if self._model is not None:
            return
        from evo2 import Evo2

        evo = Evo2(model_name=self._model_name)
        # Evo2.__init__ already places the model on the active CUDA device.
        # Save the underlying StripedHyena model + Evo2's tokenizer.
        self._model = evo.model
        self._tokenizer = evo.tokenizer
        self._evo = evo  # keep reference; some attributes live there

    def score(self, sequence: str) -> float:
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        from glmap.scoring.ar_likelihood import ARScore

        if self._model is None:
            self.load()
        if not sequence:
            raise ValueError("Evo2Loader: empty sequence is not scoreable")
        base_length = len(sequence)
        token_ids = self._tokenizer.tokenize(sequence)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        token_length = int(input_ids.shape[-1])

        with torch.inference_mode():
            logits, _ = self._model(input_ids)  # (1, T, vocab)

        shift_log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        shift_targets = input_ids[0, 1:]
        token_log_p = shift_log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
        token_log_probs_list = token_log_p.detach().cpu().tolist()
        sum_log_p = float(sum(token_log_probs_list))
        predictable_tokens = max(token_length - 1, 1)
        ce_loss = -sum_log_p / predictable_tokens
        ell_per_base = sum_log_p / base_length
        bpb = -ell_per_base / math.log(2)

        return ARScore(
            base_length=base_length,
            token_length=token_length,
            token_length_no_special=token_length,
            special_tokens_count=0,
            predictable_tokens=predictable_tokens,
            sum_log_p=sum_log_p,
            ell_per_base=ell_per_base,
            bpb=bpb,
            ce_loss=ce_loss,
            token_log_probs=tuple(token_log_probs_list),
        )


__all__ = ["Evo2Loader"]
