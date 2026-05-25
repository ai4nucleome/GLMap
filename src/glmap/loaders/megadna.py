"""megaDNA loader (lingxusb/megaDNA, MEGABYTE multiscale transformer).

Why this isn't a stock HFCausalLMLoader: megaDNA ships its weights as a
torch-pickled `MEGADNA` subclass of `MEGABYTE_pytorch.MEGABYTE`, not a
HuggingFace transformers checkpoint. There's no AutoModel registration,
no config.json, and no AutoTokenizer — the byte-level vocab is fixed at
6 ids and lives in the project README.

This loader:

- adds `models/modelsHFNoInfo/megaDNA/` to sys.path so torch's unpickler
  can resolve `megaDNA.megadna.MEGADNA`,
- `torch.load`s the 145M .pt into memory,
- exposes a `score_record(sequence)` API that mirrors the AR side of
  phase_1.md "打分协议": forward once, returns raw `sum_log_p` (nats, no
  length norm, no sign flip — matrix uses it as-is per ModelMap convention).
  `ell_per_base` and `bpb` are also reported for per-probe diagnostics but
  do not enter the L / Q / D matrices.

Vocabulary (per upstream README):

    index 0 -> "**"   (pad / sentinel)
    index 1 -> "A"
    index 2 -> "T"
    index 3 -> "C"
    index 4 -> "G"
    index 5 -> "#"    (end-of-sequence)

Non-ACGT bases (N, IUPAC ambiguity, lowercase) are mapped to the pad
token (0) by default; the caller is expected to clean probes upstream.

Dependencies (megaDNA repo `requirements.txt`):
    MEGABYTE_pytorch==0.2.1   <- exact version matters; later releases
                                 rename `EfficientAttentionConfig` and
                                 break the pickle.
    beartype, einops, tqdm, numpy, torch
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

from .base import Branch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHT_PATH = REPO_ROOT / "models/modelsHFNoInfo/megaDNA/megaDNA_phage_145M.pt"
DEFAULT_PACKAGE_PATH = REPO_ROOT / "models/modelsHFNoInfo/megaDNA"

# Vocabulary order from the project README (lingxusb/megaDNA).
MEGADNA_VOCAB: tuple[str, ...] = ("**", "A", "T", "C", "G", "#")
PAD_ID = 0
EOS_ID = 5
MEGADNA_NUCLEOTIDE_TO_TOKEN: dict[str, int] = {"A": 1, "T": 2, "C": 3, "G": 4}


def _ensure_package_importable(package_path: Path) -> None:
    """Add the megaDNA git clone to sys.path so torch.load can resolve
    `megaDNA.megadna.MEGADNA`."""
    if not package_path.exists():
        return
    p = str(package_path.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def encode_sequence(sequence: str, pad_unknown: bool = True) -> torch.Tensor:
    """Map an ACGT string to a (1, len(sequence)) LongTensor of token ids.

    Non-ACGT characters become PAD (id 0). If `pad_unknown` is False, a
    ValueError is raised on the first non-ACGT base; useful for strict
    debugging.
    """
    ids: list[int] = []
    for base in sequence:
        b = base.upper()
        if b in MEGADNA_NUCLEOTIDE_TO_TOKEN:
            ids.append(MEGADNA_NUCLEOTIDE_TO_TOKEN[b])
        elif pad_unknown:
            ids.append(PAD_ID)
        else:
            raise ValueError(f"encode_sequence: non-ACGT base {base!r} in sequence")
    return torch.tensor([ids], dtype=torch.long)


class MegaDNALoader:
    """torch.load-based wrapper for lingxusb/megaDNA (MEGABYTE multiscale)."""

    hf_id: str = "lingxusb/megaDNA"
    branch: Branch = "ar"
    # 131072 from user confirmation / project owner; matches override yaml.
    context_tokens: int = 131072

    def __init__(
        self,
        weight_path: Path = DEFAULT_WEIGHT_PATH,
        package_path: Path = DEFAULT_PACKAGE_PATH,
        device: str | torch.device = "cpu",
    ) -> None:
        self.weight_path = Path(weight_path)
        self.package_path = Path(package_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self._model: Any = None

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(f"{self.hf_id}: call load() before model")
        return self._model

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.weight_path.exists():
            raise FileNotFoundError(
                f"MegaDNALoader: weight file missing at {self.weight_path}. "
                "Run `git clone https://huggingface.co/lingxusb/megaDNA "
                f"{self.package_path}` and ensure megaDNA_phage_145M.pt is "
                "downloaded."
            )
        _ensure_package_importable(self.package_path)
        try:
            import megaDNA.megadna  # noqa: F401  (registers MEGADNA class for unpickle)
        except ImportError as exc:
            raise RuntimeError(
                "MegaDNALoader: cannot import megaDNA package. Verify the "
                f"clone at {self.package_path} and install pinned deps: "
                "`pip install MEGABYTE_pytorch==0.2.1 beartype einops`."
            ) from exc
        model = torch.load(self.weight_path, weights_only=False, map_location=self.device)
        self._model = model.to(self.device).eval()

    def score(self, sequence: str) -> float:
        """Return forward AR ell per base (in nats)."""
        return self.score_record(sequence).ell_per_base

    def score_record(self, sequence: str):
        """Compute per-base forward log-likelihood under the AR convention.

        Returns an `src.scoring.ar_likelihood.ARScore` so downstream code
        can consume the same record schema as HFCausalLMLoader.score_record.

        Token / base bookkeeping:
            tokens are bytes => n_tokens == n_bases
            predictable      = max(n_tokens - 1, 1)  (next-token prediction)
            sum_log_p        = -loss * predictable    (loss is mean CE over
                                                       predictable positions)
            ell_per_base     = sum_log_p / n_bases    (~ -loss for long seqs)
            bpb              = -ell_per_base / log(2)
        """
        from glmap.scoring.ar_likelihood import ARScore  # local import to break cycle

        if not sequence:
            raise ValueError("score_record: empty sequence is not scoreable")
        if self._model is None:
            self.load()

        ids = encode_sequence(sequence).to(self.device)
        n_bases = len(sequence)
        n_tokens = int(ids.shape[-1])
        predictable = max(n_tokens - 1, 1)
        with torch.no_grad():
            logits = self._model(ids, return_value="logits")        # (1, T, V)

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
            token_length_no_special=n_tokens,   # byte-level: no wrapping tokens
            special_tokens_count=0,
            predictable_tokens=predictable,
            sum_log_p=sum_log_p,
            ell_per_base=ell_per_base,
            bpb=bpb,
            ce_loss=ce_loss,
            token_log_probs=tuple(token_log_probs_list),
        )


__all__ = [
    "MegaDNALoader",
    "MEGADNA_VOCAB",
    "MEGADNA_NUCLEOTIDE_TO_TOKEN",
    "PAD_ID",
    "EOS_ID",
    "encode_sequence",
    "DEFAULT_WEIGHT_PATH",
    "DEFAULT_PACKAGE_PATH",
]
