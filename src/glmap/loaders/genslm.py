"""GenSLM loader (GPT-NeoX + codon tokenizer, PATRIC bacterial genomes).

Why this isn't a stock HFCausalLMLoader: GenSLM weights are PyTorch
Lightning checkpoints (state_dict keys prefixed with "model.") with no
HF repo. Architecture configs and codon tokenizer live in the project
tree under models/modelsHFNoInfo/genslm/.

This loader bypasses pytorch_lightning / deepspeed entirely:
  1. load GPT-NeoX HF config from architectures/neox/<config>.json
  2. load codon tokenizer from tokenizer_files/codon_wordlevel_69vocab.json
  3. AutoModelForCausalLM.from_config (gives a fresh GPTNeoXForCausalLM)
  4. torch.load the PTL .pt, strip "model." prefix, load_state_dict(strict=False)

Tokenization (per genslm.dataset.group_by_kmer):
    sequence is split into 3-base codons, joined by spaces, then passed
    through a word-level tokenizer with 64-codon + 5-special vocab.
    1 token = 3 bp. Native context = 2048 codon tokens = 6144 bp.

Cleaning:
    Non-ACGT bases (N, IUPAC ambiguity, lowercase) are dropped, and the
    tail is truncated to a 3-base boundary. This is consistent with the
    codon-aligned training data. Sequences with < 2 codons after cleaning
    are rejected (insufficient context for next-token prediction).

Matrix policy (phase_1.md § 单矩阵协议):
    is_codon_model=True is kept on the model side for downstream
    diagnostic loadings, but the single-matrix protocol (commit 5e59154)
    means GenSLM enters L on the full panel alongside nucleotide-tokenized
    models. The codon-vs-nucleotide systematic offset on noncoding probes
    is absorbed by ModelMap double-centering (row + col mean subtraction).
    The earlier three-matrix split (Q_pan / Q_coding_only /
    Q_nucleotide_only) that NaN-masked GenSLM on noncoding cells is retired.

Dependencies:
    transformers (>= 4.x; GPTNeoXForCausalLM is standard)
    tokenizers
    torch

Source: https://github.com/ramanathanlab/genslm (local clone at
models/modelsHFNoInfo/genslm). Weights distributed via Globus, symlinked
under models/modelsHFNoInfo/genslm/weights/.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .base import Branch

REPO_ROOT = Path(__file__).resolve().parents[2]
GENSLM_ROOT = REPO_ROOT / "models/modelsHFNoInfo/genslm"
GENSLM_PKG = GENSLM_ROOT / "genslm"
GENSLM_WEIGHTS_DIR = GENSLM_ROOT / "weights"
TOKENIZER_PATH = GENSLM_PKG / "tokenizer_files" / "codon_wordlevel_69vocab.json"
ARCHITECTURES_DIR = GENSLM_PKG / "architectures" / "neox"

# Mirror of genslm/inference.py MODELS dict, keyed by the bare names that
# appear in models/download_models_list.txt (matching the audit manifest
# model_id column).
GENSLM_MODELS: dict[str, dict[str, str]] = {
    "GenSLM-25M": {
        "config": "neox_25,290,752.json",
        "weights": "patric_25m_epoch01-val_loss_0.57_bias_removed.pt",
    },
    "GenSLM-250M": {
        "config": "neox_244,464,576.json",
        "weights": "patric_250m_epoch00_val_loss_0.48_attention_removed.pt",
    },
    "GenSLM-2.5B": {
        "config": "neox_2,533,931,008.json",
        "weights": "patric_2.5b_epoch00_val_los_0.29_bias_removed.pt",
    },
}

CODON_SIZE = 3
GENSLM_SEQ_LENGTH_TOKENS = 2048
GENSLM_SEQ_LENGTH_BP = GENSLM_SEQ_LENGTH_TOKENS * CODON_SIZE  # 6144


def _clean_to_codons(sequence: str) -> tuple[str, int]:
    """Drop non-ACGT bases, then truncate to a 3-base boundary.

    Returns (cleaned_sequence, n_codons). cleaned_sequence is always
    a multiple of 3 in length.
    """
    valid = "".join(b for b in sequence.upper() if b in "ACGT")
    n_codons = len(valid) // CODON_SIZE
    return valid[: n_codons * CODON_SIZE], n_codons


def _space_join_codons(cleaned: str) -> str:
    """`'ATGCGT...' -> 'ATG CGT ...'` (input must be ACGT-only, len % 3 == 0)."""
    return " ".join(cleaned[i : i + CODON_SIZE] for i in range(0, len(cleaned), CODON_SIZE))


class GenSLMLoader:
    """GPT-NeoX + codon tokenizer wrapper for GenSLM PATRIC checkpoints."""

    branch: Branch = "ar"
    context_tokens: int = GENSLM_SEQ_LENGTH_TOKENS

    def __init__(
        self,
        model_id: str,
        weights_dir: Path = GENSLM_WEIGHTS_DIR,
        config_dir: Path = ARCHITECTURES_DIR,
        tokenizer_path: Path = TOKENIZER_PATH,
        device: str | torch.device = "cpu",
    ) -> None:
        if model_id not in GENSLM_MODELS:
            raise ValueError(
                f"GenSLMLoader: unknown model_id {model_id!r}. "
                f"Choices: {sorted(GENSLM_MODELS)}"
            )
        self.hf_id = model_id
        info = GENSLM_MODELS[model_id]
        self.config_path = Path(config_dir) / info["config"]
        self.weight_path = Path(weights_dir) / info["weights"]
        self.tokenizer_path = Path(tokenizer_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self._model: Any = None
        self._tokenizer: Any = None
        self._missing_keys: list[str] = []
        self._unexpected_keys: list[str] = []

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
        for label, p in (
            ("config", self.config_path),
            ("weights", self.weight_path),
            ("tokenizer", self.tokenizer_path),
        ):
            if not p.exists():
                raise FileNotFoundError(
                    f"GenSLMLoader({self.hf_id}): {label} missing at {p}"
                )

        from tokenizers import Tokenizer
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            PreTrainedTokenizerFast,
        )

        tok = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(str(self.tokenizer_path))
        )
        # genslm's inference.py registers [PAD] as the pad token; mirror that
        # so HF causal-LM internals (loss masking, generate) behave the same.
        tok.add_special_tokens({"pad_token": "[PAD]"})
        self._tokenizer = tok

        config = AutoConfig.from_pretrained(str(self.config_path))
        model = AutoModelForCausalLM.from_config(config)

        ckpt = torch.load(
            self.weight_path, map_location=self.device, weights_only=False
        )
        # PTL checkpoints store params under "state_dict" with "model." prefix.
        # Some sanitized variants ship a bare state_dict; tolerate both.
        sd_raw = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        sd: dict[str, torch.Tensor] = {}
        for k, v in sd_raw.items():
            new_k = k[len("model."):] if k.startswith("model.") else k
            sd[new_k] = v
        result = model.load_state_dict(sd, strict=False)
        self._missing_keys = list(result.missing_keys)
        self._unexpected_keys = list(result.unexpected_keys)
        self._model = model.to(self.device).eval()

    def score(self, sequence: str) -> float:
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        """Compute per-base forward AR log-likelihood under the codon-AR convention.

        Token / base bookkeeping:
            n_codons     = len(cleaned_sequence) // 3
            n_tokens     = n_codons  (no auto-added specials at encode time)
            predictable  = max(n_tokens - 1, 1)
            sum_log_p    = -loss * predictable  (HF causal-LM mean CE
                                                 over `predictable` positions)
            ell_per_base = sum_log_p / len(cleaned_sequence)
            bpb          = -ell_per_base / log(2)

        The denominator uses **cleaned base count** (post-ACGT-filter,
        post-3-base truncation). This matches phase_1.md per-base
        normalization semantics: ell is normalized to what the model
        actually scored, not the raw input length.
        """
        from glmap.scoring.ar_likelihood import ARScore  # local import to break cycle

        if not sequence:
            raise ValueError("score_record: empty sequence is not scoreable")
        if self._model is None:
            self.load()

        cleaned, n_codons = _clean_to_codons(sequence)
        if n_codons < 2:
            raise ValueError(
                f"score_record: GenSLM needs >= 2 codons (6 ACGT bases) "
                f"to have at least one predictable position. Got "
                f"sequence of length {len(sequence)} -> {n_codons} codon(s)."
            )

        codon_str = _space_join_codons(cleaned)
        enc = self._tokenizer(
            codon_str, return_tensors="pt", add_special_tokens=False
        )
        ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        n_tokens = int(ids.shape[-1])
        if n_tokens != n_codons:
            raise RuntimeError(
                f"GenSLM tokenization mismatch: expected {n_codons} codon "
                f"tokens, got {n_tokens}. Likely UNK on non-canonical "
                f"codon — cleaning bug."
            )
        n_bases = len(cleaned)
        predictable = max(n_tokens - 1, 1)

        with torch.no_grad():
            logits = self._model(ids, attention_mask=attn).logits   # (1, T, V)

        # Standard AR shift-by-1: position i predicts token i+1.
        shift_log_probs = torch.log_softmax(logits[0, :-1], dim=-1)  # (T-1, V)
        shift_targets = ids[0, 1:]                                   # (T-1,)
        token_log_p = shift_log_probs.gather(
            1, shift_targets.unsqueeze(1)
        ).squeeze(1)                                                  # (T-1,)
        token_log_probs_list = token_log_p.detach().cpu().tolist()
        sum_log_p = float(sum(token_log_probs_list))
        ce_loss = -sum_log_p / predictable
        ell_per_base = sum_log_p / n_bases
        bpb = -ell_per_base / math.log(2)

        return ARScore(
            base_length=n_bases,
            token_length=n_tokens,
            token_length_no_special=n_tokens,
            special_tokens_count=0,
            predictable_tokens=predictable,
            sum_log_p=sum_log_p,
            ell_per_base=ell_per_base,
            bpb=bpb,
            ce_loss=ce_loss,
            token_log_probs=tuple(token_log_probs_list),
        )


__all__ = [
    "GenSLMLoader",
    "GENSLM_MODELS",
    "GENSLM_ROOT",
    "GENSLM_PKG",
    "GENSLM_WEIGHTS_DIR",
    "TOKENIZER_PATH",
    "ARCHITECTURES_DIR",
    "GENSLM_SEQ_LENGTH_TOKENS",
    "GENSLM_SEQ_LENGTH_BP",
]
