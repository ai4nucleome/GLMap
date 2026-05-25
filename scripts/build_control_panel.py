#!/usr/bin/env python3
"""Build the 10K control panel from an existing main panel.

The control panel has 3 disjoint subsets (random_ACGT / dinuc_shuffled /
motif_spiked) and is used for diagnostic null comparisons against the main
biological panel. It is NEVER merged into the main sequence-likelihood
matrix.

Reads:  out_panel/main_panel.parquet (default) — needed for dinuc_shuffled
                                                  subset which draws from
                                                  main-panel sequences
Writes: out_panel/control_panel.parquet
        out_panel/control_manifest.json

Run:
    python scripts/build_control_panel.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.panel.control_panel import build_control_panel, write_control_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-panel", type=Path, default=REPO_ROOT / "out_panel" / "main_panel.parquet")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "out_panel")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-random-acgt", type=int, default=3500)
    p.add_argument("--n-dinuc-shuffled", type=int, default=3500)
    p.add_argument("--n-motif-spiked", type=int, default=3000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.main_panel.exists():
        print(f"ERROR: main panel not found at {args.main_panel}", file=sys.stderr)
        print("Run scripts/build_panel.py first.", file=sys.stderr)
        return 1
    main_df = pd.read_parquet(args.main_panel)
    print(f"Loaded main panel: {len(main_df)} rows, "
          f"{main_df['functional_element'].nunique()} elements")
    ctrl = build_control_panel(
        main_df,
        seed=args.seed,
        n_random_acgt=args.n_random_acgt,
        n_dinuc_shuffled=args.n_dinuc_shuffled,
        n_motif_spiked=args.n_motif_spiked,
    )
    print(f"Built control panel: {len(ctrl)} rows")
    for subset in ["ctrl_random_ACGT", "ctrl_dinuc_shuffled", "ctrl_motif_spiked"]:
        n = (ctrl["functional_element"] == subset).sum()
        print(f"  {subset:<25} {n:>5d}")
    write_control_outputs(args.out_dir, ctrl, seed=args.seed)
    print(f"\nWrote: {args.out_dir}/control_panel.parquet")
    print(f"       {args.out_dir}/control_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
