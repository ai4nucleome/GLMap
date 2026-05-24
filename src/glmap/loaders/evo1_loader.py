"""Evo 1.x loader via the updated `evo-model` package (`evo.Evo`).

The evo env now ships transformers 5.8 + torch 2.12; the old route
(`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
on the togethercomputer cache snapshot) is broken — transformers 5.x's
dynamic-module loader fails with
"ModuleNotFoundError: Could not import module 'PreTrainedModel'"
when resolving the cached `modeling_hyena.py`'s base class.

The `evo` package's `Evo(model_name, device)` factory replaces that
path: it `snapshot_download`s the `1.1_fix` revision of the official
checkpoint (for *-base) and loads safetensors directly into a
StripedHyena model — bypassing transformers' dynamic module code.

Supported model_name values (per `evo/models.py:HF_MODEL_NAME_MAP`):
    evo-1-8k-base, evo-1-131k-base, evo-1-8k-crispr,
    evo-1-8k-transposon, evo-1.5-8k-base

Microviridae fine-tune (`evo-design/evo-1-7b-131k-microviridae`) shares
the evo-1-131k-base architecture but is not in evo-model's
HF_MODEL_NAME_MAP and ships `pytorch_model.bin` instead of safetensors;
we go around evo.Evo and build StripedHyena directly. See
`_load_microviridae` below.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_HF_TO_EVO_NAME = {
    "togethercomputer/evo-1-8k-base":   "evo-1-8k-base",
    "togethercomputer/evo-1-131k-base": "evo-1-131k-base",
    "LongSafari/evo-1-8k-crispr":       "evo-1-8k-crispr",
    "LongSafari/evo-1-8k-transposon":   "evo-1-8k-transposon",
    "evo-design/evo-1.5-8k-base":       "evo-1.5-8k-base",
}

# Fine-tunes whose architecture matches an evo-model base config but
# ship their own (non-safetensors) checkpoint and aren't in
# HF_MODEL_NAME_MAP. We hand-build StripedHyena instead of calling Evo().
_MICROVIRIDAE_HF_ID = "evo-design/evo-1-7b-131k-microviridae"


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
        raise FileNotFoundError(f"No snapshot under {snaps}")
    return children[-1]


@dataclass
class Evo1Loader:
    """For Evo 1.x checkpoints. Uses `evo.Evo` from the `evo-model` package."""

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
        if self.hf_id == _MICROVIRIDAE_HF_ID:
            self._load_microviridae()
            return
        from evo import Evo

        evo_name = _HF_TO_EVO_NAME.get(self.hf_id)
        if evo_name is None:
            raise ValueError(
                f"Evo1Loader: no evo-model alias for {self.hf_id!r}; "
                f"supported: {sorted(_HF_TO_EVO_NAME)} + microviridae"
            )
        evo = Evo(model_name=evo_name, device=str(self.device))
        self._model = evo.model
        self._tokenizer = evo.tokenizer
        self._evo = evo  # retain reference

    def _load_microviridae(self) -> None:
        """Build StripedHyena for evo-design/evo-1-7b-131k-microviridae.

        Architecture matches evo-1-131k-base — use that yml and override
        the two fields the fine-tune adjusts (rotary_emb_scaling_factor:
        16 → 1.25, max_sequence_len: 8192 → 10240). Then load the
        cached pytorch_model.bin directly; its keys are already in the
        StripedHyena native layout (no 'backbone.' prefix; both
        embedding_layer.weight and unembed.weight present).
        """
        import pkgutil
        import yaml
        from evo.tokenizer import CharLevelTokenizer
        from stripedhyena.model import StripedHyena
        from stripedhyena.utils import dotdict

        cfg_bytes = pkgutil.get_data("evo", "configs/evo-1-131k-base_inference.yml")
        cfg = yaml.safe_load(cfg_bytes)
        cfg["rotary_emb_scaling_factor"] = 1.25
        cfg["max_sequence_len"] = 10240
        global_config = dotdict(cfg, Loader=yaml.FullLoader)

        snap = _resolve_snapshot(self.hf_id)
        bin_path = snap / "pytorch_model.bin"
        if not bin_path.exists():
            raise FileNotFoundError(
                f"microviridae pytorch_model.bin not found at {bin_path}"
            )
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)

        model = StripedHyena(global_config)
        model.load_state_dict(state_dict, strict=True)
        model.to_bfloat16_except_poles_residues()
        model = model.to(self.device).eval()

        self._model = model
        self._tokenizer = CharLevelTokenizer(512)

    def score(self, sequence: str) -> float:
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        from glmap.scoring.ar_likelihood import ARScore

        if self._model is None:
            self.load()
        if not sequence:
            raise ValueError("Evo1Loader: empty sequence is not scoreable")
        base_length = len(sequence)
        token_ids = self._tokenizer.tokenize(sequence)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        token_length = int(input_ids.shape[-1])

        with torch.inference_mode():
            out = self._model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out.logits

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


__all__ = ["Evo1Loader"]
