"""PlasmidGPT loader (lingxusb/PlasmidGPT).

Why this isn't a stock HFCausalLMLoader: PlasmidGPT publishes
`pretrained_model.pt` (a torch-pickled GPT2LMHeadModel) plus a separate
`addgene_trained_dna_tokenizer.json` instead of the standard HF
config.json / model.safetensors / tokenizer.json layout. AutoModelForCausalLM.
from_pretrained doesn't see a config.json so it cannot load the repo;
we instead torch.load the .pt and build PreTrainedTokenizerFast from the
JSON file directly. Once both are in hand the model behaves exactly like
any HF GPT-2.

Notes from the live load:
  - GPT2LMHeadModel, config.n_positions = 2048, vocab_size = 30002.
  - Tokenizer registers no special tokens (pad/eos/bos/cls/unk/mask all None);
    add_special_tokens=True is a no-op so ar_score_forward's BERT-style
    wrapper assumption is harmless here.
  - BPE makes bp:token ratio sequence-dependent (32 bp/tok on poly-A,
    ~3.85 bp/tok on mixed ACGT). The 2048-token context comfortably
    covers the 128-1024 bp Stage 2 panel range regardless.

This loader satisfies the GLMLoader Protocol from src/loaders/base.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .base import Branch

DEFAULT_HF_ID = "lingxusb/PlasmidGPT"
DEFAULT_WEIGHT_FILENAME = "pretrained_model.pt"
DEFAULT_TOKENIZER_FILENAME = "addgene_trained_dna_tokenizer.json"


class PlasmidGPTLoader:
    """torch.load-based GPT-2 wrapper for lingxusb/PlasmidGPT."""

    hf_id: str = DEFAULT_HF_ID
    branch: Branch = "ar"
    context_tokens: int = 2048

    def __init__(
        self,
        device: str | torch.device = "cpu",
        weight_filename: str = DEFAULT_WEIGHT_FILENAME,
        tokenizer_filename: str = DEFAULT_TOKENIZER_FILENAME,
        weight_path: Path | None = None,
        tokenizer_path: Path | None = None,
    ) -> None:
        """Construct a lazy loader.

        Parameters
        ----------
        device :
            cpu / cuda / cuda:N.
        weight_filename, tokenizer_filename :
            Filenames inside the HF repo when fetching via hf_hub_download.
        weight_path, tokenizer_path :
            Optional local paths that override the HF fetch (useful for
            air-gapped runs or pinned weights).
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.weight_filename = weight_filename
        self.tokenizer_filename = tokenizer_filename
        self._weight_path_override = Path(weight_path) if weight_path else None
        self._tokenizer_path_override = Path(tokenizer_path) if tokenizer_path else None
        self._model: Any = None
        self._tokenizer: Any = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(f"{self.hf_id}: call load() before model")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError(f"{self.hf_id}: call load() before tokenizer")
        return self._tokenizer

    def load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast

        # Tokenizer
        tk_path = (
            self._tokenizer_path_override
            if self._tokenizer_path_override is not None
            else Path(hf_hub_download(self.hf_id, self.tokenizer_filename))
        )
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tk_path))

        # Weights: torch.load on the .pt yields a GPT2LMHeadModel instance
        # pickled with an older transformers version. Modern transformers
        # have added private attributes (e.g. `_output_attentions`,
        # `_attn_implementation_internal`) that the legacy pickle doesn't
        # carry, and accessing them via the new @property raises
        # AttributeError. Rather than playing whack-a-mole patching every
        # missing private name, we extract the state_dict (which is purely
        # tensor data and survives version drift) and reload it into a
        # freshly-instantiated GPT2LMHeadModel built from the current
        # transformers GPT2Config.
        pt_path = (
            self._weight_path_override
            if self._weight_path_override is not None
            else Path(hf_hub_download(self.hf_id, self.weight_filename))
        )
        legacy_model = torch.load(pt_path, weights_only=False, map_location="cpu")
        self._model = _rebuild_gpt2_from_legacy(legacy_model).to(self.device).eval()
        del legacy_model

        # Sanity: warn if config.n_positions disagrees with our advertised
        # context_tokens; trust the model.
        cfg_ctx = getattr(self._model.config, "n_positions", None) \
            or getattr(self._model.config, "max_position_embeddings", None)
        if cfg_ctx is not None and int(cfg_ctx) != self.context_tokens:
            # Update in place — the model's actual capacity is the truth.
            object.__setattr__(self, "context_tokens", int(cfg_ctx))

    def score(self, sequence: str) -> float:
        """Return forward AR ell per base (in nats)."""
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        """Delegate to src.scoring.ar_likelihood.ar_score_forward.

        ar_score_forward handles HF causal LM loss conventions correctly.
        For PlasmidGPT (BPE, no special tokens) it just produces token
        counts equal to the BPE encoding and a per-base normalization
        against len(sequence).
        """
        from glmap.scoring.ar_likelihood import ar_score_forward

        if self._model is None:
            self.load()
        return ar_score_forward(
            self._model, self._tokenizer, sequence, device=self.device
        )


# Keys we copy from the legacy GPT2Config dict into a freshly-built one.
# Anything not on this list (including private `_*` attrs that have changed
# between transformers versions) is left at the new GPT2Config default.
_GPT2_CONFIG_KEYS_TO_TRANSFER = (
    "vocab_size", "n_positions", "n_embd", "n_layer", "n_head",
    "n_inner", "activation_function", "resid_pdrop", "embd_pdrop",
    "attn_pdrop", "layer_norm_epsilon", "initializer_range",
    "summary_type", "summary_use_proj", "summary_activation",
    "summary_first_dropout", "summary_proj_to_labels",
    "scale_attn_weights", "scale_attn_by_inverse_layer_idx",
    "reorder_and_upcast_attn", "bos_token_id", "eos_token_id",
    "pad_token_id", "tie_word_embeddings",
)


def _rebuild_gpt2_from_legacy(legacy_model: Any) -> Any:
    """Re-instantiate a GPT2LMHeadModel from the current transformers
    package using the legacy model's tensor state_dict, leaving the new
    config at its modern defaults for everything we don't explicitly carry
    over."""
    from transformers import GPT2Config, GPT2LMHeadModel

    legacy_cfg = legacy_model.config
    cfg_kwargs: dict[str, Any] = {}
    for key in _GPT2_CONFIG_KEYS_TO_TRANSFER:
        if hasattr(legacy_cfg, key):
            try:
                cfg_kwargs[key] = getattr(legacy_cfg, key)
            except Exception:
                pass

    fresh_config = GPT2Config(**cfg_kwargs)
    fresh_model = GPT2LMHeadModel(fresh_config)
    # State dict shapes must match; legacy pickle was a GPT2LMHeadModel so
    # state_dict carries the same parameter tree.
    missing, unexpected = fresh_model.load_state_dict(
        legacy_model.state_dict(), strict=False
    )
    if missing or unexpected:
        # Don't raise — minor key drift between transformers GPT2 versions
        # is tolerable for inference, but surface it so the user can decide.
        print(
            f"[plasmidgpt] load_state_dict warning: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
        )
    return fresh_model


__all__ = [
    "PlasmidGPTLoader",
    "DEFAULT_HF_ID",
    "DEFAULT_WEIGHT_FILENAME",
    "DEFAULT_TOKENIZER_FILENAME",
]
