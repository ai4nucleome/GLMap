#!/usr/bin/env python3
"""Downstream-embedding sweep entry point.

Extract pooled embeddings for the model roster on the 6 downstream tasks:
dispatch ``scripts/downstream_tasks/run_downstream_embed.py`` per model across the GPU pool
with per-family env routing, writing
``results/analysis/embeddings/<slug>/<task>/{train,test}.parquet``
(6 tasks x train/test). Full-volume only — for a subsampled smoke, invoke
``run_downstream_embed.py --max-train N`` directly.

Thin CLI over the shared scheduler in ``scripts/score/sweep_engine.py``;
see its ``build_arg_parser`` for the full flag set (``--gpu-ids``,
``--only`` / ``--hf-ids``, ``--benchmark-dir``, ``--dry-run``, …).

Usage:
    python scripts/downstream_tasks/run_embed_sweep.py
    python scripts/downstream_tasks/run_embed_sweep.py --gpu-ids 0,5,6,7
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score.sweep_engine import run  # noqa: E402


def main() -> None:
    run("embed")


if __name__ == "__main__":
    main()
