"""Schema contract for downstream embedding parquets.

Single source of truth for what counts as a valid embedding parquet.
Used by:
  - `scripts/run_downstream_embed.py:parquet_complete` (sweep resume
    integrity check)
  - `scripts/run_downstream_classify.py:load_embed_split` (per-pair fit)

Keeping the contract in a tiny CPU-only module (no torch / loader
imports) means the classify script — which is pure sklearn CPU work —
doesn't transitively pull in the embedding-extraction stack. The
"resume says complete ↔ classify accepts" invariant is enforced by
both consumers calling validate_embed_columns().
"""

from __future__ import annotations

from typing import Iterable


def validate_embed_columns(columns: Iterable[str]) -> list[str]:
    """Canonical embedding-column validator.

    Returns the list of `embed_*` columns sorted by their numeric suffix.
    Raises ValueError if:
      - no `embed_*` columns are present;
      - any `embed_*` column has no parseable integer suffix
        (e.g. `embed_garbage`);
      - the sorted suffixes don't form the dense range [0, 1, ..., D-1]
        (catches missing intermediate dims like embed_0 + embed_2 and
        duplicate semantic dims like embed_1 + embed_01 both → 1).

    The "dense [0..D-1]" contract matches what
    `scripts/run_downstream_embed.save_embed_parquet` writes today.
    """
    def _key(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"embed column {c!r} has no numeric suffix"
            ) from exc

    cols = sorted(
        (c for c in columns if c.startswith("embed_")),
        key=_key,
    )
    if not cols:
        raise ValueError(
            f"no `embed_*` columns found; got {list(columns)}"
        )
    suffixes = [_key(c) for c in cols]
    expected = list(range(len(suffixes)))
    if suffixes != expected:
        raise ValueError(
            f"embed_* suffixes must be the dense range [0..D-1]; "
            f"got sorted suffixes {suffixes} (D={len(suffixes)}, "
            f"expected {expected[:5]}{'...' if len(expected) > 5 else ''})"
        )
    return cols


__all__ = ["validate_embed_columns"]
