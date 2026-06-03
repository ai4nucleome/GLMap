"""HuggingFace AutoModel adapters for gLMs that ship with standard transformers
config + modeling. Models that need a custom Python package (HyenaDNA's
`hyena-dna` repo, megaDNA's `.pt`, PlasmidGPT's standalone tokenizer json)
live in their own module under src/loaders/.

Both adapters satisfy GLMLoader from src/loaders/base.py. They delegate the
actual scoring math to src/scoring/, keeping this module as a thin wrapper
around HF state (tokenizer + model + device).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from .base import Branch


def _auto_map_redirects_to_mlm(hf_id: str, trust_remote_code: bool) -> bool:
    """True if config.auto_map.AutoModel resolves to a custom *ForMaskedLM
    / *ForPreTraining class. Such configs (e.g. GENA-LM bert/bigbird
    variants) ship a custom modeling_*.py that implements the actual
    pretrained architecture (pre-LN BERT with non-standard LayerNorm
    placement). When we go through standard AutoModelForMaskedLM, HF's
    stock BertForMaskedLM IS instantiated — but the standard layout's
    LayerNorm weights aren't in the checkpoint (the checkpoint has
    pre_attention_ln / post_attention_ln instead), so HF silently
    drops those and RANDOMLY initializes the standard slots. Logits
    are then meaningless (verified empirically: per-token log p
    differed by ~80 nats between the two paths for
    gena-lm-bert-base-t2t-multi)."""
    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
    except Exception:
        return False
    auto_map = getattr(cfg, "auto_map", None) or {}
    am_value = str(auto_map.get("AutoModel", ""))
    has_mlm_specific = "AutoModelForMaskedLM" in auto_map
    return (
        ("ForMaskedLM" in am_value or "ForPreTraining" in am_value)
        and not has_mlm_specific
    )


class HFCausalLMLoader:
    """Wrap a standard HF AutoModelForCausalLM as a GLMLoader for the AR branch."""

    branch: Branch = "ar"

    def __init__(
        self,
        hf_id: str,
        context_tokens: int,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
    ) -> None:
        self.hf_id = hf_id
        self.context_tokens = int(context_tokens)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
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
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(self.hf_id, **model_kwargs)
            self._model = model.to(self.device).eval()
        except torch.cuda.OutOfMemoryError:
            # Multi-GPU shard for models too large for one device (e.g.
            # Genos-10B-v2). device_map="auto" lets accelerate split layers
            # across all visible CUDA devices.
            torch.cuda.empty_cache()
            fallback = dict(model_kwargs)
            fallback["device_map"] = "auto"
            fallback.setdefault("torch_dtype", torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(self.hf_id, **fallback)
            self._model = model.eval()

    def score(self, sequence: str) -> float:
        """Return forward AR ell per base (in nats).

        The richer ARScore record is available via score_record(). score()
        is the GLMLoader-protocol-conforming scalar.
        """
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        from glmap.scoring.ar_likelihood import ar_score_forward

        if self._model is None:
            self.load()
        return ar_score_forward(self.model, self.tokenizer, sequence, device=self.device)


class HFMaskedLMLoader:
    """Wrap a standard HF AutoModelForMaskedLM as a GLMLoader for the MLM branch.

    `score(sequence, stride=6)` returns the stride pseudo-log-likelihood per
    base (primary stride k=6 per `phase_1.md` § "打分协议 / MLM/Encoder");
    pass `stride=4` for the sensitivity supplement. The richer MLMScore
    record (sum_log_p, ell_per_base, bpb, token_log_probs) is available via
    `score_record(sequence, stride=...)`. Both delegate to
    `src.scoring.mlm_pseudo_ll.stride_pll_forward`.
    """

    branch: Branch = "mlm"

    def __init__(
        self,
        hf_id: str,
        context_tokens: int,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
    ) -> None:
        self.hf_id = hf_id
        self.context_tokens = int(context_tokens)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
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
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        # See _auto_map_redirects_to_mlm: GENA-LM (and any model whose
        # config has auto_map.AutoModel = custom *ForMaskedLM but lacks
        # an explicit AutoModelForMaskedLM entry) MUST be loaded via
        # AutoModel + trust_remote_code, otherwise HF instantiates the
        # standard BertForMaskedLM and silently random-init's the
        # LayerNorm slots that the checkpoint's pre-LN architecture
        # doesn't have.
        if _auto_map_redirects_to_mlm(self.hf_id, self.trust_remote_code):
            model = AutoModel.from_pretrained(self.hf_id, **model_kwargs)
        else:
            model = AutoModelForMaskedLM.from_pretrained(self.hf_id, **model_kwargs)
        self._model = model.to(self.device).eval()

    def score(self, sequence: str, stride: int = 6) -> float:
        """Return stride PLL ell per base (in nats). Primary stride k=6 per
        phase_1.md '打分协议 § MLM/Encoder'; pass stride=4 for the sensitivity
        supplement.
        """
        return self.score_record(sequence, stride=stride).ell_per_base

    def score_record(self, sequence: str, stride: int = 6):
        from glmap.scoring.mlm_pseudo_ll import stride_pll_forward

        if self._model is None:
            self.load()
        return stride_pll_forward(
            self.model, self.tokenizer, sequence, stride=stride, device=self.device
        )


__all__ = ["HFCausalLMLoader", "HFMaskedLMLoader"]
