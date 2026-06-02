"""Per-format readers for panel datasets.

Four supported formats (matching `format:` field in data/panel_sources.yaml):

  csv
      Plain CSV with named sequence + label columns. Used by all GUE and
      DFB datasets. Label policies handled here:
        positive_only       — keep rows where label == "1"
        keep_label          — keep rows where label in keep_labels
        keep_all            — keep both labels
        multiclass_species  — keep all, label encodes species

  fasta_pgb_binary
      PGB FASTA with header `>id|0` or `>id|1`. Used by poly_a, pro_seq,
      splicing. Label is the trailing |-separated token.

  fasta_pgb_binary_lenfilter
      Same as fasta_pgb_binary but with positive-only label policy AND
      length filtering applied here (kept rows must have 128 <= len <= 1024).
      Used by lncrna where positives have variable lengths > 1024.

  fasta_pgb_chromatin
      PGB FASTA with header `>chr:start-end_negative|0|0|...` (19 cell-type
      labels). Used by chromatin_access. Kept rows are those whose locus
      does NOT have the `_negative` suffix AND have at least one of the 19
      labels == 1.

All readers return list[ReaderResult]. Lazy generators were considered but
the downstream sampler needs a pool to draw from, so we materialize.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .main_panel import DatasetSpec


@dataclass(frozen=True)
class ReaderResult:
    sequence: str         # always uppercase, ACGT-only (validated at read time)
    raw_label: str        # original label string from the file (kept for species assignment)
    row_idx: int          # provenance


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _normalize_and_validate(seq: str) -> str | None:
    """Uppercase and require all bases ∈ {A,C,G,T}. Reject Ns and IUPAC ambiguous.

    Returns the normalized sequence or None if invalid. Centralizing this here
    keeps the sampler's pool clean so target_n is always reachable.
    """
    if not seq:
        return None
    u = seq.upper()
    for c in u:
        if c not in "ACGT":
            return None
    return u


def read_dataset(path: Path, spec: "DatasetSpec") -> list[ReaderResult]:
    fmt = spec.format
    if fmt == "csv":
        return _read_csv(path, spec)
    if fmt == "fasta_pgb_binary":
        return _read_fasta_pgb_binary(path, spec, length_filter=False)
    if fmt == "fasta_pgb_binary_lenfilter":
        return _read_fasta_pgb_binary(path, spec, length_filter=True)
    if fmt == "fasta_pgb_chromatin":
        return _read_fasta_pgb_chromatin(path, spec)
    raise ValueError(f"Unknown format: {fmt}")


# ─────────────────────── CSV ───────────────────────

def _read_csv(path: Path, spec: "DatasetSpec") -> list[ReaderResult]:
    seq_col = spec.seq_col or "sequence"
    label_col = spec.label_col or "label"
    keep: list[ReaderResult] = []
    skip_validate = spec.crop_to is not None
    with _open_text(path) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            label = (row.get(label_col) or "").strip()
            seq = (row.get(seq_col) or "").strip()
            if not seq:
                continue
            if not _csv_keep(label, spec):
                continue
            # When the spec crops the sequence later, validation has to happen
            # after the crop. Otherwise we'd over-reject (e.g. fungi 10000-bp
            # sample with one N might still center-crop into a valid 1024).
            if skip_validate:
                seq = seq.upper()
            else:
                norm = _normalize_and_validate(seq)
                if norm is None:
                    continue
                seq = norm
            keep.append(ReaderResult(sequence=seq, raw_label=label, row_idx=idx))
    return keep


def _csv_keep(label: str, spec: "DatasetSpec") -> bool:
    policy = spec.label_policy
    if policy == "positive_only":
        return label == "1"
    if policy == "keep_label":
        return spec.keep_labels is not None and label in spec.keep_labels
    if policy == "keep_all" or policy == "multiclass_species":
        return True
    raise ValueError(f"Unknown label_policy for CSV: {policy}")


# ─────────────────────── PGB FASTA binary ───────────────────────

def _read_fasta_pgb_binary(
    path: Path, spec: "DatasetSpec", length_filter: bool
) -> list[ReaderResult]:
    """Header format: `>id|label` where label is "0" or "1".

    With length_filter=True, also enforce 128 <= len(seq) <= 1024 (for lncRNA
    where positives have variable lengths).
    """
    keep: list[ReaderResult] = []
    LEN_MIN, LEN_MAX = 128, 1024
    for idx, (header, seq) in enumerate(_iter_fasta(path)):
        # Header tokens: id | label
        parts = header.split("|")
        if len(parts) < 2:
            continue
        label = parts[-1].strip()
        if spec.label_policy == "positive_only":
            if label != "1":
                continue
        elif spec.label_policy == "keep_all":
            pass
        else:
            raise ValueError(
                f"_read_fasta_pgb_binary got label_policy={spec.label_policy}, "
                f"expected positive_only / keep_all"
            )
        if length_filter and not (LEN_MIN <= len(seq) <= LEN_MAX):
            continue
        norm = _normalize_and_validate(seq)
        if norm is None:
            continue
        keep.append(ReaderResult(sequence=norm, raw_label=label, row_idx=idx))
    return keep


# ─────────────────────── PGB FASTA chromatin ───────────────────────

def _read_fasta_pgb_chromatin(path: Path, spec: "DatasetSpec") -> list[ReaderResult]:
    """Header format: `>chr:start-end_{negative,id}|l1|l2|...|l19`.

    Keep rows whose id-token does NOT end with `_negative` AND have ≥1 of
    the 19 cell-type labels == 1. Discards background regions.
    """
    keep: list[ReaderResult] = []
    for idx, (header, seq) in enumerate(_iter_fasta(path)):
        parts = header.split("|")
        if len(parts) < 2:
            continue
        id_token = parts[0]
        if id_token.endswith("_negative"):
            continue
        labels = parts[1:]
        # multi-label any-positive
        if not any(lab == "1" for lab in labels):
            continue
        # Combine all positive cell-type indices as raw_label (informational only)
        pos_idx = [i for i, lab in enumerate(labels) if lab == "1"]
        raw_label = ",".join(str(i) for i in pos_idx)
        norm = _normalize_and_validate(seq)
        if norm is None:
            continue
        keep.append(ReaderResult(sequence=norm, raw_label=raw_label, row_idx=idx))
    return keep


# ─────────────────────── FASTA iterator ───────────────────────

def _iter_fasta(path: Path):
    """Yield (header_without_gt, sequence) tuples."""
    cur_header: str | None = None
    cur_parts: list[str] = []
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if cur_header is not None:
                    yield cur_header, "".join(cur_parts)
                cur_header = line[1:]
                cur_parts = []
            else:
                cur_parts.append(line)
        if cur_header is not None:
            yield cur_header, "".join(cur_parts)


__all__ = ["ReaderResult", "read_dataset"]
