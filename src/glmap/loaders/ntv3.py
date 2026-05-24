"""NTv3 U-Net architecture loader.

InstaDeepAI's Nucleotide Transformer v3 family is a U-Net over MLM with
`num_downsamples` halving layers. The model requires input sequence length
divisible by 2^num_downsamples; otherwise the down/up-sample shapes mismatch
in the bottleneck and the forward errors out with a tensor-shape RuntimeError.
Upstream documentation:
  https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer_v3.md

  > Due to the U-Net architecture which uses downsampling and upsampling
  > layers, sequence lengths must be divisible by 2^num_downsamples. For
  > the main models, this is 128 for 7 downsamples.
  >
  > If your sequences don't meet this requirement, you can crop them to the
  > nearest valid length, or pad them with N tokens to the nearest valid
  > length. The models however were not trained on [PAD] tokens, so you
  > should not pad them with [PAD] tokens.

length_multiple values for the current family (audit-known):
  NTv3_8M_pre / NTv3_100M_pre / NTv3_650M_pre        : 128 (7 downsamples)
  NTv3_8M_pre_8kb / NTv3_100M_pre_8kb / NTv3_650M_pre_8kb : 128
  NTv3_5downsample_pre / NTv3_5downsample_pre_8kb    :  32 (5 downsamples)

Output column semantics for padded probes:
  sequence_length_bp     = original probe length (unpadded)
  token_length           = padded model-input length (e.g. 256 for a 156-bp
                           probe with length_multiple=128)
  content_position_count = sequence_length_bp (only unpadded bases
                           contribute to sum_log_p / ell_per_base / bpb)

Reviewers reading the parquet should NOT interpret `token_length >
sequence_length_bp` as a schema bug — it's the architectural pad.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .base import Branch


class NTv3MaskedLMLoader:
    """NTv3 U-Net MLM adapter: right-pads short inputs with 'N' to the U-Net
    alignment, scores only the unpadded original probe via
    `stride_pll_forward(..., content_length=bp)`.
    """

    branch: Branch = "mlm"

    def __init__(
        self,
        hf_id: str,
        context_tokens: int,
        length_multiple: int,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
    ) -> None:
        if length_multiple < 1:
            raise ValueError(
                f"length_multiple must be >= 1, got {length_multiple}"
            )
        self.hf_id = hf_id
        self.context_tokens = int(context_tokens)
        self.length_multiple = int(length_multiple)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None
        self._sanity_checked = False

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
        model = AutoModelForMaskedLM.from_pretrained(self.hf_id, **model_kwargs)
        self._model = model.to(self.device).eval()
        self._sanity_check_tokenizer()

    def _sanity_check_tokenizer(self) -> None:
        """Verify the 1-base-per-content-token invariant that
        `stride_pll_forward(content_length=...)` depends on.

        NTv3 ships a single-nucleotide tokenizer; if a future model in this
        family changes that (k-mer / BPE / etc.), `content_positions[:bp]`
        would silently slice the wrong window. Fail loud at load time
        instead.
        """
        if self._sanity_checked:
            return
        tok = self._tokenizer
        # "N" must tokenize to exactly one non-special token.
        n_ids = tok("N", add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        special_ids = set(getattr(tok, "all_special_ids", []) or [])
        n_content = [i for i in n_ids if i not in special_ids]
        if len(n_content) != 1:
            raise RuntimeError(
                f"{self.hf_id}: tokenizer('N') yielded {len(n_content)} content tokens; "
                f"NTv3MaskedLMLoader assumes one-content-token-per-base "
                f"(ids={n_ids}, content={n_content}). If this NTv3 ships a "
                "k-mer/BPE tokenizer, do NOT route it through this loader — "
                "content_length-based scoring would slice the wrong window."
            )
        # "ACGT" must tokenize to exactly 4 content tokens.
        acgt_ids = tok("ACGT", add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        acgt_content = [i for i in acgt_ids if i not in special_ids]
        if len(acgt_content) != 4:
            raise RuntimeError(
                f"{self.hf_id}: tokenizer('ACGT') yielded {len(acgt_content)} content "
                f"tokens (expected 4); ids={acgt_ids}. Same caveat as above."
            )
        self._sanity_checked = True

    def score(self, sequence: str, stride: int = 6) -> float:
        return self.score_record(sequence, stride=stride).ell_per_base

    def score_record(self, sequence: str, stride: int = 6):
        from glmap.scoring.mlm_pseudo_ll import stride_pll_forward

        if self._model is None:
            self.load()

        bp = len(sequence)
        if bp < 1:
            raise ValueError("NTv3MaskedLMLoader: empty sequence is not scoreable")

        # Right-pad with literal 'N' (a content base in the NT tokenizer) to
        # the next multiple of length_multiple. NOT the tokenizer's [PAD] —
        # NTv3 was not trained on [PAD] (upstream docs).
        pad_to = ((bp + self.length_multiple - 1) // self.length_multiple) * self.length_multiple
        n_pad = pad_to - bp
        if n_pad > 0:
            scored_input = sequence + ("N" * n_pad)
        else:
            scored_input = sequence

        # Guard: never silently truncate. Caller / panel build is responsible
        # for keeping probes <= context_tokens; the architectural N-pad must
        # not push us across that line.
        if pad_to > self.context_tokens:
            raise ValueError(
                f"{self.hf_id}: probe length {bp} bp padded to {pad_to} bp "
                f"(length_multiple={self.length_multiple}) exceeds context_tokens="
                f"{self.context_tokens}. Either crop the probe or use a longer-context "
                "NTv3 variant."
            )

        # If sequence was already aligned (n_pad == 0), skip content_length to
        # exercise the historical code path — keeps the default behavior
        # byte-identical to HFMaskedLMLoader for aligned inputs.
        content_length: int | None = bp if n_pad > 0 else None
        return stride_pll_forward(
            self.model,
            self.tokenizer,
            scored_input,
            stride=stride,
            device=self.device,
            content_length=content_length,
        )


__all__ = ["NTv3MaskedLMLoader"]
