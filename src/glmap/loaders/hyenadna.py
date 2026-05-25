"""HyenaDNA loader.

The LongSafari/hyenadna-* checkpoints are not AutoModel-compatible; they ship
only `weights.ckpt` (a torch.save dict with a `state_dict` key) and
`config.json` (HyenaDNAModel kwargs). Loading goes through the
`standalone_hyenadna.py` module the authors publish in their repo, which
defines `HyenaDNAModel` (autoregressive, no LM head) and `CharacterTokenizer`.

AR scoring requires logits. HyenaDNA was trained AR with tied embedding/output
weights (the `tie_weights` method is commented out in the published
standalone — but the pretraining objective is next-token CE with the embedding
matrix as the output projection, which is the standard interpretation for these
LMBackbone-style models). We compute logits as `hidden_states @ W_embed^T`.

Resolution:
* hub repo:   `/data/yusen/software/.cache/huggingface/hub/models--LongSafari--{slug}/snapshots/{rev}/`
* contains:   `config.json` + `weights.ckpt`
* standalone: `/nvme-data3/.../models/modelsHFNoInfo/hyena-dna/standalone_hyenadna.py`
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
_HYENADNA_SRC = REPO_ROOT / "models" / "modelsHFNoInfo" / "hyena-dna"


def _ensure_standalone_importable() -> Any:
    """Inject the hyena-dna source dir on sys.path; return standalone module."""
    src = str(_HYENADNA_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import standalone_hyenadna as mod  # type: ignore[import-not-found]
    return mod


def _load_weights_into_scratch(
    scratch_dict: dict, pretrained_dict: dict, checkpointing: bool = False
) -> dict:
    """Replicate huggingface.py:load_weights inline so we don't trigger that
    file's top-level inference_single() side-effects on import."""
    import re

    def inject_substring(orig_str: str) -> str:
        # `.mixer` -> `.mixer.layer`, `.mlp` -> `.mlp.layer`
        s = re.sub(r"\.mixer", ".mixer.layer", orig_str)
        s = re.sub(r"\.mlp", ".mlp.layer", s)
        return s

    for key in list(scratch_dict.keys()):
        if "backbone" in key:
            key_loaded = "model." + key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except KeyError as exc:
                raise KeyError(
                    f"HyenaDNA weight load: scratch key {key!r} expected "
                    f"pretrained key {key_loaded!r} but it was missing"
                ) from exc
    return scratch_dict


def _resolve_snapshot(hf_id: str) -> Path:
    """Find the local HF cache snapshot for `LongSafari/hyenadna-...`."""
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            "/data/yusen/software/.cache/huggingface",
        )
    ) / "hub"
    repo_dir = cache_root / f"models--{hf_id.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        raise FileNotFoundError(
            f"HyenaDNA snapshot dir not found for {hf_id}: {snaps}. "
            "Pre-download the checkpoint into the HF cache."
        )
    children = sorted(snaps.iterdir())
    if not children:
        raise FileNotFoundError(f"No snapshot revisions under {snaps}")
    return children[-1]


def _ar_score_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    base_length: int,
) -> dict[str, Any]:
    """Standard AR shift-by-1: position i predicts token i+1.

    Returns the same field schema as `src.scoring.ar_likelihood.ARScore`.
    """
    shift_log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    shift_targets = input_ids[0, 1:]
    token_log_p = shift_log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
    token_log_probs_list = token_log_p.detach().cpu().tolist()
    sum_log_p = float(sum(token_log_probs_list))
    predictable_tokens = max(int(input_ids.shape[-1]) - 1, 1)
    ce_loss = -sum_log_p / predictable_tokens
    ell_per_base = sum_log_p / base_length
    bpb = -ell_per_base / math.log(2)
    return dict(
        base_length=base_length,
        token_length=int(input_ids.shape[-1]),
        token_length_no_special=int(input_ids.shape[-1]),
        special_tokens_count=0,
        predictable_tokens=predictable_tokens,
        sum_log_p=sum_log_p,
        ell_per_base=ell_per_base,
        bpb=bpb,
        ce_loss=ce_loss,
        token_log_probs=tuple(token_log_probs_list),
    )


@dataclass
class HyenaDNALoader:
    hf_id: str
    context_tokens: int
    device: str | torch.device = "cpu"
    trust_remote_code: bool = False
    torch_dtype: Any = None
    is_codon: bool = False
    branch: str = "ar"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self._tokenizer = None
        self._model = None
        self._embed_weight: torch.Tensor | None = None  # for tied-LM projection

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
        mod = _ensure_standalone_importable()
        snap = _resolve_snapshot(self.hf_id)
        cfg = json.loads((snap / "config.json").read_text())
        # config.json may flag the run as having been gradient-checkpointed,
        # which inserts an extra ".layer." in every backbone state_dict key.
        # Both `checkpoint_mixer` and `checkpoint_mlp` can independently set it.
        checkpointing = bool(cfg.get("checkpoint_mixer") or cfg.get("checkpoint_mlp"))
        # Drop output-only flags that confuse the constructor
        for k in ("return_hidden_state", "fused_mlp", "fused_dropout_add_ln"):
            cfg.pop(k, None)

        model = mod.HyenaDNAModel(**cfg, use_head=False)
        ckpt = torch.load(snap / "weights.ckpt", map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = _load_weights_into_scratch(
            model.state_dict(), state, checkpointing=checkpointing
        )
        model.load_state_dict(state)

        model = model.to(self.device).eval()
        self._model = model

        # Locate the word embedding to use as tied output projection.
        # In LMBackbone: backbone.embeddings.word_embeddings (nn.Embedding)
        embed = model.backbone.embeddings.word_embeddings.weight.detach()
        self._embed_weight = embed  # shape (vocab_size, d_model)

        # CharacterTokenizer with ACGTN + standard special tokens.
        # vocab ids: [CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5,
        # [UNK]=6, A=7, C=8, G=9, T=10, N=11. Matches config.vocab_size=12.
        # Published CharacterTokenizer (a) omits get_vocab() and (b) builds
        # `_vocab_str_to_int` *after* super().__init__(), but newer
        # transformers calls get_vocab() *inside* super().__init__() via
        # _add_tokens. Pre-seed the vocab and add a get_vocab() that
        # tolerates being called before the parent finishes.
        class _PatchedCharTok(mod.CharacterTokenizer):
            def __init__(self_inner, characters, model_max_length, **kwargs):
                self_inner._vocab_str_to_int = {
                    "[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3,
                    "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
                    **{ch: i + 7 for i, ch in enumerate(characters)},
                }
                self_inner._vocab_int_to_str = {
                    v: k for k, v in self_inner._vocab_str_to_int.items()
                }
                super().__init__(characters, model_max_length, **kwargs)

            def get_vocab(self_inner):
                return dict(getattr(self_inner, "_vocab_str_to_int", {}))

        self._tokenizer = _PatchedCharTok(
            characters=["A", "C", "G", "T", "N"],
            model_max_length=int(self.context_tokens),
        )

    def score(self, sequence: str) -> float:
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        from glmap.scoring.ar_likelihood import ARScore

        if self._model is None:
            self.load()
        if not sequence:
            raise ValueError("HyenaDNALoader: empty sequence is not scoreable")
        base_length = len(sequence)
        ids = self._tokenizer(sequence, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = ids.to(self.device)

        with torch.no_grad():
            hidden = self._model(ids)              # (1, T, d_model)
            logits = hidden @ self._embed_weight.t().to(hidden.dtype)  # (1, T, vocab)

        rec = _ar_score_from_logits(logits, ids, base_length)
        return ARScore(**rec)


__all__ = ["HyenaDNALoader"]
