"""Pooled embedding extraction for all 123-model audit loaders.

This is the embedding analog of `src.scoring.{ar_likelihood, mlm_pseudo_ll}`:
it takes any loaded `*Loader` instance and a DNA sequence, and returns a
1-D numpy embedding vector. Used by `scripts/run_downstream_embed.py` to
feed a linear-probe downstream evaluation.

The pooling protocol per loader family:

  HF-route (HFCausalLM/HFMaskedLM and direct HF derivatives):
      mean-pool the last hidden state over content tokens.
      Returns a (D,) vector where D = hidden_size.

  megaDNA:
      3-layer mean-pool then concatenate, following lingxusb's GitHub
      issue #4 recipe:
          pooled_per_layer = [h.mean(dim=(0,1)) for h in hidden_states]
          embed = cat(pooled_per_layer, dim=-1)
      D = sum of per-layer dims (varies by megaDNA size).

  GenSLM / PlasmidGPT / GENERator / NTv3 / Botanic / PlantBiMoE / MutBERT /
  AIDO:
      All wrap a HF AutoModel; their underlying model accepts
      `output_hidden_states=True`. We mean-pool last_hidden_state.

  HyenaDNA / Evo1 / Evo2:
      Non-HF backbones; each has its own hidden-state convention.
      These need per-loader implementations below.

Conventions:
  - Returns float32 numpy array (compact for parquet, sufficient precision).
  - Mean pooling is over CONTENT tokens, not over batch — single-sequence
    forward each time keeps batch dim trivial.
  - For models that prepend [CLS] or similar special tokens, we exclude
    those positions from the mean. The exclusion is currently approximate
    (uses tokenizer.all_special_ids when available); refine per-loader as
    needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# ─────────────────────────── public API ────────────────────────────────


def compute_pooled_embedding(loader: Any, sequence: str) -> np.ndarray:
    """Dispatch on loader class name and return a 1-D float32 embedding.

    Caller must have already called `loader.load()`. Raises
    `NotImplementedError` for loader classes without an implementation
    (a deliberate fail-loud rather than silent fallback).
    """
    cls_name = type(loader).__name__

    if cls_name in _HF_HIDDEN_STATES_LOADERS:
        return _hf_mean_pool_last_hidden(loader, sequence)
    if cls_name == "MegaDNALoader":
        return _megadna_three_layer_concat(loader, sequence)
    if cls_name == "GenSLMLoader":
        return _genslm_mean_pool_last_hidden(loader, sequence)
    if cls_name == "PlasmidGPTLoader":
        return _plasmidgpt_mean_pool_last_hidden(loader, sequence)
    if cls_name == "HyenaDNALoader":
        return _hyenadna_mean_pool_last_hidden(loader, sequence)
    if cls_name == "Evo1Loader":
        return _evo1_mean_pool_last_hidden(loader, sequence)
    if cls_name == "Evo2Loader":
        return _evo2_mean_pool_last_hidden(loader, sequence)
    if cls_name == "CarbonCausalLMLoader":
        return _carbon_mean_pool_last_hidden(loader, sequence)
    if cls_name == "GENERatorLoader":
        return _generator_mean_pool_last_hidden(loader, sequence)
    if cls_name == "NTv3MaskedLMLoader":
        return _ntv3_mean_pool_last_hidden(loader, sequence)
    raise NotImplementedError(
        f"compute_pooled_embedding: loader class {cls_name!r} not yet "
        "supported. Add a per-loader implementation in "
        "src/scoring/embeddings.py before running downstream eval."
    )


# Loader class names that wrap a HF AutoModel and support
# `model(..., output_hidden_states=True)`. Adding a new HF-derived loader
# is as simple as appending to this set.
_HF_HIDDEN_STATES_LOADERS: set[str] = {
    "HFCausalLMLoader",
    "HFMaskedLMLoader",
    "BotanicLoader",
    "PlantBiMoELoader",
    "MutBERTLoader",
    "AIDOLoader",
    # NOTE: NTv3MaskedLMLoader is intentionally NOT here — its U-Net
    # requires input length divisible by `length_multiple` (128 or 32),
    # otherwise the downsampling stack produces output of size 0 and
    # the forward crashes. We use a dedicated `_ntv3_mean_pool_last_hidden`
    # that N-pads symmetrically to NTv3MaskedLMLoader.score_record, then
    # mean-pools over content positions only.
    # NOTE: GENERatorLoader is intentionally NOT here — its tokenizer
    # emits <oov> for any sequence whose length is not a multiple of 6,
    # and including <oov> positions in the mean-pool contaminates the
    # embedding (~14% of tokens on 41 bp 5mC probes). We use a dedicated
    # `_generator_mean_pool_last_hidden` that A-pads before tokenizing,
    # matching the scoring-path convention introduced with Carbon.
}


# ────────────────────────── HF generic path ────────────────────────────


def _model_input_device(loader: Any) -> torch.device:
    """Where input_ids should live before forward.

    For non-sharded models this is just `loader.device`. For models that
    OOM-fell-back to `device_map="auto"` in HFCausalLMLoader.load(),
    `loader.device` is still cuda:0 but the actual model is sharded
    across several devices; the embedding layer (first params) lives on
    whatever HF picked, and feeding input_ids on the wrong device raises
    "tensors on cuda:0 vs cpu" at the first index_select. Using
    `next(model.parameters()).device` is a no-op for the common case and
    self-corrects for the sharded case.
    """
    try:
        return next(loader.model.parameters()).device
    except (StopIteration, AttributeError):
        return loader.device


def _hf_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """Mean-pool the last hidden state over content (non-special) tokens.

    Works for any loader whose `.model` accepts `output_hidden_states=True`
    and exposes `.last_hidden_state` (or `.hidden_states[-1]`) on the
    returned ModelOutput.

    Some architectures (Mamba/SSM-based: Caduceus, PlantCAD2, …) do NOT
    accept an `attention_mask` kwarg in forward(). For those, we inspect
    the signature and skip the kwarg. SSM models naturally handle the
    full input as a sequence, so mean-pooling over all tokens (without
    mask weighting) is the correct fallback.
    """
    import inspect
    tokenizer = loader.tokenizer
    model = loader.model
    dev = _model_input_device(loader)

    enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    # Probe forward signature; skip kwargs the model can't take.
    try:
        forward_params = set(inspect.signature(model.forward).parameters.keys())
    except (TypeError, ValueError):
        forward_params = set()
    forward_kwargs: dict = {"output_hidden_states": True}
    if "attention_mask" in forward_params and attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask
    else:
        # SSM-style: no mask weighting in the pool either.
        attention_mask = None

    with torch.no_grad():
        out = model(input_ids, **forward_kwargs)

    last_hidden = _resolve_last_hidden(out)
    return _pool_mean_over_tokens(last_hidden, attention_mask).cpu().float().numpy()


def _resolve_last_hidden(out: Any) -> torch.Tensor:
    """Pull the (B, T, D) last hidden state from a HF ModelOutput, falling
    back across the common variants."""
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    raise RuntimeError(
        "compute_pooled_embedding: could not find last hidden state in model "
        f"output of type {type(out).__name__}"
    )


def _pool_mean_over_tokens(
    hidden: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """(B, T, D) → (D,). Mean over T, weighted by attention_mask if given,
    then squeeze the batch dim. Single-sequence forward → B=1."""
    if attention_mask is not None:
        # Under device_map="auto" the model can return `hidden` on a
        # different device than `attention_mask` (which was placed on the
        # input device). Snap mask to hidden's device so broadcasting
        # doesn't crash. No-op for single-device loads.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype).to(hidden.device)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
    else:
        pooled = hidden.mean(dim=1)
    return pooled.squeeze(0)


# ────────────────────────── per-loader paths ───────────────────────────


def _megadna_three_layer_concat(loader: Any, sequence: str) -> np.ndarray:
    """3-layer mean-pool concat per lingxusb/megaDNA GitHub issue #4.

    The MEGADNA model is a 3-level MEGABYTE multiscale transformer; each
    level has its own hidden dimension. We mean-pool each level over
    (batch, sequence) and concatenate.

    The model's forward supports `return_value="embedding"` which yields
    a list of 3 per-level hidden state tensors.
    """
    from glmap.loaders.megadna import encode_sequence

    ids = encode_sequence(sequence).to(loader.device)
    with torch.no_grad():
        # See megaDNA upstream: forward(..., return_value='embedding')
        # returns a list/tuple of per-level hidden state tensors.
        hidden_states = loader.model(ids, return_value="embedding")
    if not isinstance(hidden_states, (list, tuple)):
        raise RuntimeError(
            "MegaDNALoader embedding: expected list of 3 per-level hidden "
            f"states, got {type(hidden_states).__name__}"
        )
    pooled_per_layer = [h.detach().mean(dim=(0, 1)) for h in hidden_states]
    embed = torch.cat(pooled_per_layer, dim=-1)
    return embed.cpu().float().numpy()


def _genslm_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """GenSLM wraps a HF GPTNeoX with a codon tokenizer. Mirror
    GenSLMLoader.score_record's prep: ACGT-filter + right-trunc to a
    3-base boundary, space-join codons, then PreTrainedTokenizerFast →
    AR forward → mean-pool last hidden state."""
    from glmap.loaders.genslm import _clean_to_codons, _space_join_codons

    cleaned, n_codons = _clean_to_codons(sequence)
    if n_codons < 1:
        raise ValueError("GenSLM embedding: sequence yields 0 codons after cleaning")
    codon_str = _space_join_codons(cleaned)
    enc = loader.tokenizer(codon_str, add_special_tokens=False, return_tensors="pt")
    dev = _model_input_device(loader)
    ids = enc["input_ids"].to(dev)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(dev)
    with torch.no_grad():
        out = loader.model(ids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[-1]
    return _pool_mean_over_tokens(h, attn).cpu().float().numpy()


def _plasmidgpt_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """PlasmidGPT is a rebuilt GPT-2 with an Addgene BPE tokenizer."""
    tokenizer = loader.tokenizer
    enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(loader.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(loader.device)
    with torch.no_grad():
        out = loader.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    h = _resolve_last_hidden(out)
    return _pool_mean_over_tokens(h, attention_mask).cpu().float().numpy()


def _hyenadna_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """HyenaDNA's `HyenaDNAModel` forwards `(input_ids,) -> hidden_states`
    directly. We byte-encode the sequence via the single-nucleotide
    tokenizer the loader sets up, then mean-pool.
    """
    tokenizer = loader.tokenizer
    enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(loader.device)
    with torch.no_grad():
        # HyenaDNAModel.forward returns hidden states directly when called
        # without the tied LM head; the loader keeps the head attached, so
        # we extract the encoder output by calling the backbone.
        if hasattr(loader.model, "backbone"):
            h = loader.model.backbone(input_ids)
        else:
            # Fallback: assume model(input_ids) returns hidden states.
            h = loader.model(input_ids)
        if isinstance(h, tuple):
            h = h[0]
    pooled = h.mean(dim=1).squeeze(0)
    return pooled.cpu().float().numpy()


def _evo1_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """Evo1 uses StripedHyena directly (no HF wrapping). The model
    exposes `embed_input_ids` and forward; we extract hidden states from
    the final stripe block.
    """
    # Evo1 tokenizer is a single-byte CharLevelTokenizer; reuse loader's.
    tokenizer = loader.tokenizer
    input_ids = tokenizer.tokenize(sequence)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=loader.device).unsqueeze(0)
    with torch.no_grad():
        # StripedHyena exposes hidden_states via embed_input_ids -> blocks
        # forward; the simplest hook is to access model.embedding_layer + blocks.
        # We call forward and capture pre-final-norm hidden states via the
        # model's logit path: logits = unembed(hidden_states), so we can
        # invert this only if unembed is tied. The safest is to add a hook.
        hidden: list[torch.Tensor] = []
        def _hook(_module, _inp, out):
            hidden.append(out if isinstance(out, torch.Tensor) else out[0])
        # Last block in the StripedHyena layers list
        last_block = loader.model.blocks[-1]
        handle = last_block.register_forward_hook(_hook)
        try:
            _ = loader.model(input_ids)
        finally:
            handle.remove()
        if not hidden:
            raise RuntimeError("Evo1 embedding: forward hook captured no hidden state")
        h = hidden[-1]
    pooled = h.mean(dim=1).squeeze(0)
    return pooled.cpu().float().numpy()


def _evo2_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """Evo2 uses the evo2 package's `Evo2` wrapper around StripedHyena2.

    Evo2 exposes `embed_input_ids` directly. The model.model attribute is
    the StripedHyena2 backbone; we run input through it and grab the
    pre-unembed hidden state.
    """
    # Tokenize via loader's char-level tokenizer
    tokenizer = loader.tokenizer
    if hasattr(tokenizer, "tokenize"):
        ids = tokenizer.tokenize(sequence)
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(loader.device)
    else:
        enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(loader.device)
    with torch.no_grad():
        # Evo2 wrapper: model._model is StripedHyena2; call forward and
        # capture hidden state via hook on the final block.
        sh2 = loader._model if hasattr(loader, "_model") else loader.model
        hidden: list[torch.Tensor] = []
        def _hook(_module, _inp, out):
            hidden.append(out if isinstance(out, torch.Tensor) else out[0])
        last_block = sh2.blocks[-1]
        handle = last_block.register_forward_hook(_hook)
        try:
            _ = sh2(input_ids)
        finally:
            handle.remove()
        if not hidden:
            raise RuntimeError("Evo2 embedding: forward hook captured no hidden state")
        h = hidden[-1]
    pooled = h.mean(dim=1).squeeze(0)
    return pooled.cpu().float().numpy()


def _ntv3_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """NTv3 pooled embedding: N-pad to length_multiple, forward, then
    mean-pool over content positions only (skipping the padded N tail).
    Mirrors NTv3MaskedLMLoader.score_record's preprocessing exactly.

    Without the pad, the U-Net's downsampling stack collapses to size 0
    and the forward crashes (the bug that hit all 8 NTv3 models in the
    embed sweep). With the pad but no content-length slicing, the
    padded-N hidden states (which don't correspond to real bases) would
    contaminate the pooled embedding.
    """
    tokenizer = loader.tokenizer
    model = loader.model

    bp = len(sequence)
    if bp < 1:
        raise ValueError("NTv3 embedding: empty sequence is not embeddable")

    pad_to = ((bp + loader.length_multiple - 1) // loader.length_multiple) * loader.length_multiple
    n_pad = pad_to - bp
    scored_input = sequence + ("N" * n_pad) if n_pad > 0 else sequence

    if pad_to > loader.context_tokens:
        raise ValueError(
            f"{loader.hf_id}: probe {bp} bp padded to {pad_to} "
            f"(length_multiple={loader.length_multiple}) exceeds context_tokens="
            f"{loader.context_tokens}."
        )

    enc = tokenizer(scored_input, add_special_tokens=False, return_tensors="pt")
    dev = _model_input_device(loader)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = _resolve_last_hidden(out)   # (1, T, D)
    # NTv3 uses single-nucleotide tokenization (1 bp/tok), so the first
    # `bp` token positions are real bases and the trailing `n_pad`
    # positions are padded N's. Slice to content positions only.
    content_hidden = last_hidden[:, :bp, :]
    pooled = content_hidden.mean(dim=1).squeeze(0)
    return pooled.cpu().float().numpy()


def _generator_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """GENERator pooled embedding: A-pad to 6-multiple before tokenizing
    so the trailing partial 6-mer becomes a real DNA token instead of
    <oov>. Mirrors GENERatorLoader.score_record's preprocessing. Without
    the pad, raw tokenization emits one <oov> at the end whose hidden
    state would otherwise contaminate the mean — up to ~14% of the pool
    on 41 bp probes.
    """
    from glmap.loaders.generator import _right_pad_to_k6

    tokenizer = loader.tokenizer
    model = loader.model

    padded = _right_pad_to_k6(sequence)
    if not padded:
        raise ValueError("GENERator embedding: empty sequence is not embeddable")
    enc = tokenizer(padded, add_special_tokens=False, return_tensors="pt")
    dev = _model_input_device(loader)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = _resolve_last_hidden(out)
    return _pool_mean_over_tokens(last_hidden, attention_mask).cpu().float().numpy()


def _carbon_mean_pool_last_hidden(loader: Any, sequence: str) -> np.ndarray:
    """Carbon pooled embedding: mean of last hidden over DNA-token positions.

    Mirrors CarbonCausalLMLoader.score_record's preprocessing — right-pad
    with 'A' to a multiple of 6 and prepend `<dna>` to switch the tokenizer
    into 6-mer mode — then mean-pools the last hidden state over positions
    [1:T) (i.e. skipping the leading `<dna>` token at position 0). The
    closing `</dna>` is deliberately omitted so every non-skipped position
    corresponds to a real 6-mer DNA token.
    """
    from glmap.loaders.carbon import _right_pad_to_k6, DNA_OPEN_TAG

    tokenizer = loader.tokenizer
    model = loader.model

    padded = _right_pad_to_k6(sequence)
    if not padded:
        # Empty input → wrapped would be just "<dna>" (1 token) and the
        # post-skip slice [1:] would be empty → silent NaN embedding.
        # Fail loud instead so the per-sequence error fallback in
        # run_downstream_embed.embed_split logs it.
        raise ValueError("Carbon embedding: empty sequence is not embeddable")
    wrapped = DNA_OPEN_TAG + padded
    enc = tokenizer(wrapped, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(_model_input_device(loader))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    last_hidden = _resolve_last_hidden(out)   # (1, T, D)
    # Skip position 0 (= <dna> tag token).
    dna_hidden = last_hidden[:, 1:, :]
    pooled = dna_hidden.mean(dim=1).squeeze(0)
    return pooled.cpu().float().numpy()


__all__ = ["compute_pooled_embedding"]
