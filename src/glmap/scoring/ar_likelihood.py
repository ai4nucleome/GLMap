"""AR per-base log-likelihood scoring.

Implements the AR side of the scoring contract defined in phase_1.md section
"打分协议 § AR/Decoder":

    ell(x) = sum_t log p(x_t | x_<t) / L_x_in_bases       # forward only
    bpb(x) = -ell(x) / log(2)

`L_x_in_bases` is `len(sequence)`, **not** the number of tokens. BPE / k-mer
tokenizers shrink the token count below the base count, so per-base
normalization (the cross-tokenizer-comparable quantity) requires the raw
base length.

Reverse-complement: AR / MLM main signatures are forward-only by protocol.
The panel-level RC sanity check was retired (phase_1.md § Sanity Check #3,
2026-05-17) because non-overlapping k-mer tokenizers offset forward vs RC
token boundaries when the panel length is not divisible by k, so the
abs_diff is dominated by tokenizer artifact rather than model strand
handling. Callers may still run an RC pass for per-model diagnostics but
the result is not part of the matrix-level protocol.

Special tokens (e.g. BERT-style [CLS] / [SEP] wrappers on Mistral-DNA) are
counted toward `predictable_tokens` because HF's `model(input_ids, labels=...)
.loss` averages over all shift-by-1 positions. The resulting bias is small
when len(sequence) >> n_special_tokens; downstream code can subtract it
using `special_tokens_count` if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ARScore:
    """Result of a single forward AR scoring pass.

    Attributes
    ----------
    base_length :
        len(sequence) in bases.
    token_length :
        Number of tokens produced including special tokens (CLS/SEP/etc.).
    token_length_no_special :
        Number of tokens excluding special tokens (diagnostic only).
    special_tokens_count :
        token_length - token_length_no_special.
    predictable_tokens :
        Number of shift-by-1 positions that contributed to the HF causal-LM
        loss (= max(token_length - 1, 1)).
    sum_log_p :
        Sum over predictable positions of log p(x_t | x_<t), in nats.
    ell_per_base :
        sum_log_p / base_length, in nats per base. Phase 1 main signature.
    bpb :
        -ell_per_base / log(2), in bits per base. Phase 1 reporting scale.
    ce_loss :
        Raw HF causal-LM mean cross-entropy over predictable positions
        (= -sum_log_p / predictable_tokens, kept for parity with HF's
        `model(..., labels=ids).loss`).
    token_log_probs :
        Per-position log p(true_token | x_<t) at each of the T-1 shift
        positions, in nats. `sum(token_log_probs) == sum_log_p` exactly.
        Kept so downstream analyses (per-position heterozygosity,
        motif-level diagnostics, ModelMap-style resampling) can run
        without rescoring.
    """

    base_length: int
    token_length: int
    token_length_no_special: int
    special_tokens_count: int
    predictable_tokens: int
    sum_log_p: float
    ell_per_base: float
    bpb: float
    ce_loss: float
    token_log_probs: tuple[float, ...]


def ar_score_forward(
    model: Any,
    tokenizer: Any,
    sequence: str,
    device: str | torch.device = "cpu",
    base_length_override: int | None = None,
) -> ARScore:
    """Compute forward AR log-likelihood per base.

    `base_length_override` lets the caller report a different denominator
    for ell_per_base than `len(sequence)`. Used by loaders that prepend
    tokenizer-control prefixes to the raw DNA (e.g. Carbon's `<dna>`
    tag-mode switch): `sequence` becomes `"<dna>" + dna_padded` but only
    `len(dna_padded)` bases should enter the per-base denominator.
    Symmetric to NTv3's `content_length` parameter on the MLM side.

    HF causal-LM convention: `model(input_ids=ids, labels=ids).loss` is the
    mean cross-entropy over `n_tokens - 1` shift-by-1 predictable positions
    (the first token has no prior context). Multiply by that count to
    recover the un-normalized sum log p, then divide by `len(sequence)` to
    get per-base ell.

    The sequence is taken as-is; the caller is responsible for any cleaning
    (uppercase, ACGT-only enforcement, N handling). An empty sequence raises
    ValueError because per-base normalization would be ill-defined.
    """
    if not sequence:
        raise ValueError("ar_score_forward: empty sequence is not scoreable")
    base_length = int(base_length_override) if base_length_override is not None else len(sequence)
    if base_length <= 0:
        raise ValueError(
            f"ar_score_forward: base_length must be > 0 (got {base_length})"
        )

    # Cross-family comparability: tokenize WITHOUT special tokens. Different
    # tokenizers add different specials (NT family: <cls>; BERT-style:
    # [CLS]+[SEP]; BPE: <s>+</s>; HyenaDNA / Evo / GenSLM custom loaders:
    # none at all) and including their log p in sum_log_p makes the L
    # matrix entries incomparable across models.
    enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
    # When the loader fell back to device_map="auto" (HFCausalLMLoader's
    # OOM-fallback path on a single contested GPU), `device` is still the
    # caller's hint (typically cuda:0) but the model's embedding layer
    # may live elsewhere (cpu, or another shard). Snap the input to the
    # actual first-param device — a no-op for normal single-device loads.
    try:
        target_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        target_device = device
    input_ids = enc["input_ids"].to(target_device)
    token_length = int(input_ids.shape[-1])
    token_length_no_special = token_length        # by construction
    special_tokens_count = 0

    predictable_tokens = max(token_length - 1, 1)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # Standard *ForCausalLM exposes .logits. A few custom modeling configs
    # (e.g. some GENA-LM auto_map = BertForPreTraining variants accessed
    # via AR-path stubs) use .prediction_logits. Probe both.
    if hasattr(out, "logits") and out.logits is not None:
        logits = out.logits                             # (1, T, V)
    elif hasattr(out, "prediction_logits") and out.prediction_logits is not None:
        logits = out.prediction_logits
    else:
        raise AttributeError(
            f"ar_score_forward: model output of type {type(out).__name__} "
            "has neither .logits nor .prediction_logits."
        )

    # Standard AR shift-by-1: position i predicts token i+1.
    # Under device_map="auto", `logits` lives on the last layer's device,
    # which may differ from `input_ids`'s device (the embedding layer's).
    # gather() crashes on a device mismatch, so snap shift_targets onto
    # logits' device. No-op for the common single-device case.
    shift_log_probs = torch.log_softmax(logits[0, :-1], dim=-1)   # (T-1, V)
    shift_targets = input_ids[0, 1:].to(shift_log_probs.device)   # (T-1,)
    token_log_p = shift_log_probs.gather(1, shift_targets.unsqueeze(1)).squeeze(1)
                                                                  # (T-1,)
    token_log_probs_list = token_log_p.detach().cpu().tolist()
    sum_log_p = float(sum(token_log_probs_list))
    ce_loss = -sum_log_p / predictable_tokens
    ell_per_base = sum_log_p / base_length
    bpb = -ell_per_base / math.log(2)

    return ARScore(
        base_length=base_length,
        token_length=token_length,
        token_length_no_special=token_length_no_special,
        special_tokens_count=special_tokens_count,
        predictable_tokens=predictable_tokens,
        sum_log_p=sum_log_p,
        ell_per_base=ell_per_base,
        bpb=bpb,
        ce_loss=ce_loss,
        token_log_probs=tuple(token_log_probs_list),
    )


__all__ = ["ARScore", "ar_score_forward"]
