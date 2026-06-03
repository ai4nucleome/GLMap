"""Loaders for MLM models whose HF auto_map is incomplete.

Two cases handled here:

* `living-models/Botanic0-{S,M,L}` — `auto_map` registers only AutoModel.
  Their custom `Botanic` forward returns `AugmentedMaskedLMOutput` with a
  proper `.logits` head, so loading via AutoModel (not AutoModelForMaskedLM)
  is the right path.

* `plant-llms/PlantBiMoE` — `auto_map` includes AutoModelForMaskedLM but
  omits AutoTokenizer; transformers can't auto-detect the tokenizer class
  from `model_type=plantbimoe`. We instantiate `PlantbimoeTokenizer`
  manually via `get_class_from_dynamic_module`.

Both expose the same `score_record(sequence, stride)` interface as
HFMaskedLMLoader so the runner can dispatch them transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


@dataclass
class BotanicLoader:
    """For living-models/Botanic0-{S,M,L}. Uses AutoModel; .logits is on the output."""

    hf_id: str
    context_tokens: int
    device: str | torch.device = "cpu"
    trust_remote_code: bool = True
    torch_dtype: Any = None
    branch: str = "mlm"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self._tokenizer = None
        self._model = None

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
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=self.trust_remote_code
        )
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        model = AutoModel.from_pretrained(self.hf_id, **kwargs)
        self._model = model.to(self.device).eval()

    def score(self, sequence: str, stride: int = 6) -> float:
        return self.score_record(sequence, stride=stride).ell_per_base

    def score_record(self, sequence: str, stride: int = 6):
        from glmap.scoring.mlm_pseudo_ll import stride_pll_forward

        if self._model is None:
            self.load()
        return stride_pll_forward(
            self.model, self.tokenizer, sequence, stride=stride, device=self.device
        )


@dataclass
class PlantBiMoELoader:
    """For plant-llms/PlantBiMoE. auto_map omits AutoTokenizer; load it manually."""

    hf_id: str
    context_tokens: int
    device: str | torch.device = "cpu"
    trust_remote_code: bool = True
    torch_dtype: Any = None
    branch: str = "mlm"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self._tokenizer = None
        self._model = None

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
        # PlantbimoeTokenizer is a hardcoded char-level vocab; no
        # tokenizer_config.json is shipped, so .from_pretrained fails.
        # Instantiate directly with model_max_length matching the config.
        TokCls = get_class_from_dynamic_module(
            "tokenization_plantbimoe.PlantbimoeTokenizer",
            self.hf_id,
        )
        self._tokenizer = TokCls(model_max_length=self.context_tokens)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        model = AutoModelForMaskedLM.from_pretrained(self.hf_id, **kwargs)
        self._model = model.to(self.device).eval()

    def score(self, sequence: str, stride: int = 6) -> float:
        return self.score_record(sequence, stride=stride).ell_per_base

    def score_record(self, sequence: str, stride: int = 6):
        from glmap.scoring.mlm_pseudo_ll import stride_pll_forward

        if self._model is None:
            self.load()
        return stride_pll_forward(
            self.model, self.tokenizer, sequence, stride=stride, device=self.device
        )


class _MutBERTInputWrap(torch.nn.Module):
    """Wrap MutBERT so callers can pass plain (B, L) LongTensor token IDs.

    MutBERT (`JadenLong/MutBERT*`) is a "probabilistic genome representation":
    its forward expects `input_ids` to be a *probability distribution* over
    the vocab at every position, i.e. shape (B, L, V) float — the
    embeddings layer does `torch.matmul(input_ids, word_embeddings.weight)`
    instead of `nn.Embedding(input_ids)`. The size unpacking on line 817 of
    modeling_mutbert.py (`input_shape = input_ids.size()[:-1]`) confirms
    this — the trailing dim is the vocab axis and gets stripped before
    the model sees (B, L).

    We convert standard (B, L) LongTensor → one-hot float (B, L, V) inside
    the wrapper so the stride_pll_forward scoring path is unchanged. Mask
    tokens flow through naturally: stride_pll_forward replaces the
    position's integer with mask_token_id, our wrapper one-hots that, the
    matmul into word_embeddings then selects the [MASK] embedding row.
    """

    def __init__(self, inner: torch.nn.Module, vocab_size: int) -> None:
        super().__init__()
        self.inner = inner
        self.vocab_size = int(vocab_size)

    def forward(self, input_ids=None, **kwargs):
        kwargs.pop("attention_mask", None)  # MutBERT recomputes it internally
        if input_ids.dtype in (torch.int64, torch.int32, torch.long):
            input_ids = torch.nn.functional.one_hot(
                input_ids, num_classes=self.vocab_size
            ).float()
        return self.inner(input_ids=input_ids, **kwargs)


@dataclass
class MutBERTLoader:
    """For JadenLong/MutBERT, MutBERT-Multi, MutBERT-Human-Ref.

    All three share the same RoPE-BERT-with-probabilistic-input architecture
    (modeling_mutbert.py). Wraps with `_MutBERTInputWrap` so standard MLM
    stride PLL scoring works against the model's (B, L, V) forward contract.

    RoPE scaling
    ------------
    MutBERT trains with `max_position_embeddings=512`, but ruRoPEBert (the
    base architecture) supports inference-time context extension via
    `rope_scaling={'type': ..., 'factor': ...}` (per the model card's
    "With RoPE scaling" section). We auto-pick the scaling factor based on
    `context_tokens`:

        factor = max(1.0, context_tokens / 512)

    so the caller only has to bump context_tokens (e.g. via
    data/audits/context_overrides.yaml or a per-task override) when the
    panel goes longer than 512 bp; the loader sets rope_scaling
    automatically. Default `rope_scaling_type='dynamic'` follows
    NTK-aware scaling which keeps short-context behavior intact.

    Expected accuracy degradation (LLaMA-2 RoPE convention):
        factor =  1×    no degradation
        factor =  2×    ~99% (near-lossless)
        factor =  4×    ~95-97%
        factor =  8×    ~88-92%
        factor = 16×    ~75-85% (use only when panel demands it)
    """

    hf_id: str
    context_tokens: int
    device: str | torch.device = "cpu"
    trust_remote_code: bool = True
    torch_dtype: Any = None
    branch: str = "mlm"
    rope_scaling_type: str = "dynamic"
    base_context_tokens: int = 512                   # MutBERT's training context

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self._tokenizer = None
        self._model = None
        # Derive the RoPE factor from the requested context size.
        self.rope_scaling_factor = max(1.0, self.context_tokens / self.base_context_tokens)

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
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, trust_remote_code=self.trust_remote_code
        )
        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.rope_scaling_factor > 1.0:
            kwargs["rope_scaling"] = {
                "type": self.rope_scaling_type,
                "factor": float(self.rope_scaling_factor),
            }
        inner = AutoModelForMaskedLM.from_pretrained(self.hf_id, **kwargs)
        wrapped = _MutBERTInputWrap(inner, vocab_size=int(inner.config.vocab_size))
        self._model = wrapped.to(self.device).eval()

    def score(self, sequence: str, stride: int = 6) -> float:
        return self.score_record(sequence, stride=stride).ell_per_base

    def score_record(self, sequence: str, stride: int = 6):
        from glmap.scoring.mlm_pseudo_ll import stride_pll_forward

        if self._model is None:
            self.load()
        return stride_pll_forward(
            self.model, self.tokenizer, sequence, stride=stride, device=self.device
        )


__all__ = ["BotanicLoader", "PlantBiMoELoader", "MutBERTLoader"]
