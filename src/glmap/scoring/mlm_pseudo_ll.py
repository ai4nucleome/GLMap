"""MLM stride pseudo-log-likelihood scoring.

Implements the MLM side of the scoring contract defined in phase_1.md
section "打分协议 § MLM / Encoder".

Two regimes
-----------
* **stride k = 1** — true (Salazar et al. 2020) leave-one-out
  pseudo-log-likelihood. For each content position p, one forward pass
  with EXACTLY that position masked; the rest of the sequence remains
  visible context. `content_position_count` forward passes total. Slow
  (≈ L× more forward passes than k=6) but the canonical PLL definition.

* **stride k >= 2** — k-pass stride approximation:
    for offset in 0..k-1:
      mask all content token positions p where p mod k == offset
      forward; accumulate log p(true_token | masked input) at masked positions
  k forward passes total. Each masked position has its k-1 nearest
  neighbours visible (no two masked tokens are within distance < k of
  each other), so the conditional approximates leave-one-out as k → 1.

The earlier implementation routed stride=1 through the k-pass branch,
which collapsed to a single "all-content-masked" forward pass — that is
not PLL (it conditions each token on zero visible context, and produced
NaN logits on the NT v1 family with 6-mer tokenizer). The true k=1
branch above is now the explicit special case.

Per-base output
---------------
`base_length` is `len(sequence)`, not the number of tokens. This
matches the AR convention and makes the per-base scalar comparable
across tokenizers.

Scope
-----
Non-overlapping tokenizers only (single-base, non-overlapping k-mer,
BPE). DNABERT overlapping k-mer requires special multi-position masking
to prevent target-base leakage and is out of scope for phase 0
(TODO Stage 1 supplement).

Content vs special tokens: positions whose token id is in
`tokenizer.all_special_ids` (CLS / SEP / PAD / MASK / UNK) are never
masked or scored; they are treated as the wrapping the model expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MLMScore:
    """Result of a single stride-PLL pass at a fixed stride k."""

    base_length: int
    token_length: int
    token_length_no_special: int
    stride: int
    content_position_count: int
    masked_position_count: int
    sum_log_p: float
    ell_per_base: float
    bpb: float
    token_log_probs: tuple[float, ...]   # per-content-position log p(true | masked),
                                          # ordered by absolute token position,
                                          # length = content_position_count


def stride_pll_forward(
    model: Any,
    tokenizer: Any,
    sequence: str,
    stride: int = 6,
    device: str | torch.device = "cpu",
    content_length: int | None = None,
) -> MLMScore:
    """Compute stride pseudo-log-likelihood per base.

    Parameters
    ----------
    model :
        HF AutoModelForMaskedLM-compatible module in eval mode. Caller is
        responsible for .eval() and device placement.
    tokenizer :
        HF AutoTokenizer-compatible. Must expose `mask_token_id` and
        `all_special_ids`.
    sequence :
        Cleaned uppercase ACGT[N] string. Length must be > 0.
    stride :
        Stride k. **k = 1** runs true leave-one-out PLL (one forward
        pass per content position, `content_position_count` total).
        **k >= 2** runs the k-pass stride approximation (k forward
        passes, each masking every k-th position). Primary signature
        uses k=6; supplementary sensitivity reports k=4 / 8 / 12. Must
        be a positive integer.
    device :
        Where to run forward passes.
    content_length :
        Optional opt-in. When set, the model still sees the full `sequence`
        (so any architectural length-alignment padding the caller appended
        — e.g. NTv3 U-Net right-pad with 'N' to a multiple of 2^num_downsamples
        — is included in the forward pass and contributes to attention
        context), but masking, log-p gather, and the returned `base_length`,
        `content_position_count`, `masked_position_count`, `sum_log_p`,
        `ell_per_base`, `bpb`, and `token_log_probs` are RESTRICTED to the
        first `content_length` content positions. This is what lets NTv3
        report a "per-original-probe" likelihood after architectural padding.

        ONLY VALID when the tokenizer is one-content-token-per-base
        (single-nucleotide / byte-level). Do NOT use with k-mer (NT v1/v2),
        BPE (DNABERT-2, GenomeOcean), or codon (GenSLM) tokenizers — the
        token-position-to-base mapping is not 1:1 there, so
        `content_positions[:content_length]` is meaningless.

        Default `None` preserves the historical behavior (score every
        content position). Existing call sites are API backward-compatible.

    Raises
    ------
    ValueError
        If sequence is empty, stride is non-positive, the tokenizer has no
        mask_token_id, or content_length is out of [1, content_position_count].
    """
    if not sequence:
        raise ValueError("stride_pll_forward: empty sequence is not scoreable")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if getattr(tokenizer, "mask_token_id", None) is None:
        raise ValueError(
            f"stride_pll_forward: tokenizer {type(tokenizer).__name__} "
            "has no mask_token_id; MLM scoring requires a [MASK] token"
        )
    if content_length is not None and content_length < 1:
        raise ValueError(
            f"content_length must be >= 1 when set, got {content_length}"
        )

    # Cross-family comparability: tokenize WITHOUT special tokens. Different
    # tokenizers add different specials (NT family: <cls>; BERT-style:
    # [CLS]+[SEP]; BPE: <s>+</s>; some none at all) and including their
    # positions in the PLL sum makes sum_log_p incomparable across models.
    # The `content_positions` filter below would already exclude specials
    # from the mask schedule, but the cleaner contract is to never inject
    # specials in the first place.
    enc = tokenizer(sequence, add_special_tokens=False, return_tensors="pt")
    # Snap to the model's actual first-param device — see ar_score_forward
    # comment. Robust to OOM-fallback device_map="auto" cases.
    try:
        target_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        target_device = device
    input_ids = enc["input_ids"].to(target_device)
    token_length = int(input_ids.shape[-1])
    token_length_no_special = token_length  # by construction; kept for schema parity

    # Defensive: even with add_special_tokens=False, a malformed tokenizer
    # could still inject specials. Filter on the actual ids that came through.
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    content_positions = [
        i
        for i, tid in enumerate(input_ids[0].tolist())
        if tid not in special_ids
    ]
    full_content_count = len(content_positions)
    if full_content_count == 0:
        raise ValueError(
            "stride_pll_forward: no content tokens after stripping special tokens"
        )

    # Apply the optional content window. Restrict to the first
    # `content_length` content positions; the remaining tail positions are
    # still tokenized and visible to the model (attention context), but
    # they are NOT masked and do NOT contribute to sum_log_p. The caller
    # must guarantee one-content-token-per-base for this slice to mean
    # "first N bases of the probe".
    if content_length is not None:
        if content_length > full_content_count:
            raise ValueError(
                f"content_length={content_length} exceeds content_position_count="
                f"{full_content_count}; caller must guarantee one content token "
                "per base and pad on the right with non-special tokens (e.g. 'N'), "
                "not the tokenizer's [PAD]."
            )
        content_positions = content_positions[:content_length]
        base_length = content_length
    else:
        base_length = len(sequence)
    content_position_count = len(content_positions)

    mask_id = int(tokenizer.mask_token_id)
    # Per-content-position log p, keyed by absolute token position. Each
    # content position is masked exactly once (k=1 path) or exactly once
    # across the k offsets (k>=2 path), so `position_log_p` ends up
    # dense (no overwrites, no gaps).
    position_log_p: dict[int, float] = {}
    masked_position_count = 0

    def _extract_mlm_logits(out):
        """Pick the MLM-head logits from a HF output object.

        BertForPreTraining (used by some GENA-LM checkpoints that auto_map
        AutoModel → BertForPreTraining instead of BertForMaskedLM) returns
        `prediction_logits` for the MLM head and `seq_relationship_logits`
        for the NSP head. Standard *ForMaskedLM returns `.logits`. Probe
        both — never co-exist, so the choice is unambiguous.
        """
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        if hasattr(out, "prediction_logits") and out.prediction_logits is not None:
            return out.prediction_logits
        raise AttributeError(
            f"stride_pll_forward: model output of type "
            f"{type(out).__name__} has neither .logits nor "
            ".prediction_logits — can't extract MLM head."
        )

    if stride == 1:
        # ───────────────── True leave-one-out PLL ────────────────── #
        # Salazar et al. 2020: for each content position, run one
        # forward pass with ONLY that position masked. Slow (L× more
        # forward passes than stride=6) but the canonical PLL
        # definition. Used as the gold-standard reference in the
        # stride-PLL sensitivity supplement.
        #
        # The earlier implementation routed stride=1 through the k-pass
        # branch below, which collapsed to a single all-content-masked
        # forward pass — that conditions each token on zero visible
        # context (≠ PLL) and produced NaN logits on the NT v1 6-mer
        # family. Fix: explicit single-mask loop here.
        for pos in content_positions:
            true_id = int(input_ids[0, pos].item())
            masked_input = input_ids.clone()
            masked_input[0, pos] = mask_id
            with torch.no_grad():
                out = model(input_ids=masked_input)
            logits = _extract_mlm_logits(out)
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            lp = float(log_probs[true_id].item())
            position_log_p[pos] = lp
            masked_position_count += 1
    else:
        # ──────────────── k-pass stride approximation (k >= 2) ───────────── #
        for offset in range(stride):
            # Mask every stride-th content position (indexed within the
            # content subsequence, not absolute token positions).
            positions_to_mask = [
                content_positions[i]
                for i in range(offset, content_position_count, stride)
            ]
            if not positions_to_mask:
                continue

            positions_tensor = torch.tensor(positions_to_mask, dtype=torch.long, device=target_device)
            true_ids = input_ids[0, positions_tensor]

            masked_input = input_ids.clone()
            masked_input[0, positions_tensor] = mask_id

            with torch.no_grad():
                out = model(input_ids=masked_input)
            logits = _extract_mlm_logits(out)

            # log-softmax + gather log p at true token ids. Under
            # device_map="auto", logits lives on the last layer's device
            # (potentially different from positions_tensor's). Snap to
            # logits.device for cross-device safety; no-op when single-device.
            pos_for_logits = positions_tensor.to(logits.device)
            true_ids_for_logits = true_ids.to(logits.device)
            log_probs = torch.log_softmax(logits[0, pos_for_logits], dim=-1)
            gathered = log_probs.gather(1, true_ids_for_logits.unsqueeze(1)).squeeze(1)
            gathered_list = gathered.detach().cpu().tolist()
            for pos, lp in zip(positions_to_mask, gathered_list):
                position_log_p[pos] = float(lp)
            masked_position_count += int(positions_tensor.numel())

    # Order per-content-position log p by absolute token position so
    # downstream analyses can align with the original sequence layout.
    token_log_probs_list = [position_log_p[p] for p in content_positions]
    sum_log_p = float(sum(token_log_probs_list))
    ell_per_base = sum_log_p / base_length
    bpb = -ell_per_base / math.log(2)

    return MLMScore(
        base_length=base_length,
        token_length=token_length,
        token_length_no_special=token_length_no_special,
        stride=stride,
        content_position_count=content_position_count,
        masked_position_count=masked_position_count,
        sum_log_p=sum_log_p,
        ell_per_base=ell_per_base,
        bpb=bpb,
        token_log_probs=tuple(token_log_probs_list),
    )


__all__ = ["MLMScore", "stride_pll_forward"]
