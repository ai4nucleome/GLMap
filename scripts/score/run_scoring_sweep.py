#!/usr/bin/env python3
"""Scoring sweep entry point.

Score the model roster on the 10,000-probe panel: dispatch
``scripts/score/scoring_worker.py`` per model across the GPU pool with
per-family env routing, writing
``results/scores/AR_MLM_scores/<slug>/probes.parquet``. Afterwards run a
single ``scoring_worker.py --from-audit --strict-aggregate`` (CPU OK) to
build ``results/scores/matrices/``.

Thin CLI over the shared scheduler in ``scripts/score/sweep_engine.py``;
see its ``build_arg_parser`` for the full flag set (``--gpu-ids``,
``--only`` / ``--hf-ids`` / ``--branch``, ``--stride``, ``--out``,
``--scores-subdir``, ``--dry-run``, …).

Usage:
    python scripts/score/run_scoring_sweep.py
    python scripts/score/run_scoring_sweep.py --only evo --n-gpus 8 --force
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score.sweep_engine import run  # noqa: E402


def main() -> None:
    run("scoring")


if __name__ == "__main__":
    main()
