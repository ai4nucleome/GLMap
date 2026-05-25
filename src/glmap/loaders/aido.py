"""AIDO.DNA loader (RNABert architecture from genbio-ai/ModelGenerator).

AIDO.DNA-{300M,7B} ship config.json + model.safetensors with
`model_type=rnabert`, but transformers doesn't have the rnabert class
registered. The actual modeling code lives in
modelgenerator.huggingface_models.rnabert under
models/modelsHFNoInfo/ModelGenerator/. We import RNABertForMaskedLM
directly and load via its native `from_pretrained`.

The shipped HF snapshot also lacks `vocab.txt` (RNABertTokenizer's
required vocab file); we copy it from the ModelGenerator local source
at install time (one-off) so the tokenizer can instantiate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
_MG_ROOT = REPO_ROOT / "models" / "modelsHFNoInfo" / "ModelGenerator"


def _ensure_modelgenerator_importable() -> tuple[type, type, type]:
    src = str(_MG_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    from modelgenerator.huggingface_models.rnabert.modeling_rnabert import (
        RNABertForMaskedLM,
    )
    from modelgenerator.huggingface_models.rnabert.configuration_rnabert import (
        RNABertConfig,
    )
    from modelgenerator.huggingface_models.rnabert.tokenization_rnabert import (
        RNABertTokenizer,
    )
    return RNABertConfig, RNABertForMaskedLM, RNABertTokenizer


def _resolve_snapshot(hf_id: str) -> Path:
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            "/data/yusen/software/.cache/huggingface",
        )
    ) / "hub"
    repo_dir = cache_root / f"models--{hf_id.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    children = sorted(snaps.iterdir())
    if not children:
        raise FileNotFoundError(f"No snapshot revisions under {snaps}")
    return children[-1]


@dataclass
class AIDOLoader:
    """For genbio-ai/AIDO.DNA-{300M,7B}. RNABert architecture (MLM)."""

    hf_id: str
    context_tokens: int
    device: str | torch.device = "cpu"
    trust_remote_code: bool = False
    torch_dtype: Any = None
    is_codon: bool = False
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
        _, RNABertForMaskedLM, RNABertTokenizer = _ensure_modelgenerator_importable()
        snap = _resolve_snapshot(self.hf_id)
        vocab_file = snap / "vocab.txt"
        if not vocab_file.exists():
            # one-off fallback to canonical source under ModelGenerator
            src_vocab = _MG_ROOT / "modelgenerator" / "huggingface_models" / "rnabert" / "vocab.txt"
            vocab_file.write_text(src_vocab.read_text())
        self._tokenizer = RNABertTokenizer(vocab_file=str(vocab_file))
        kwargs: dict[str, Any] = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        model = RNABertForMaskedLM.from_pretrained(str(snap), **kwargs)
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


__all__ = ["AIDOLoader"]
