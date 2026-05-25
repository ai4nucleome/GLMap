#!/usr/bin/env python3
"""Build a stratified 1000-probe subset of the main panel for the
MLM stride-PLL ablation experiment.

For experiment A of the k=1 vs k=6 ablation (see paper.md Fig S2a),
we need a smaller probe set so that re-scoring all 56 MLM models at
k=1 is tractable. The subset is stratified by ``functional_element``
with allocation proportional to each element's size in the full
panel, so the subset preserves the same compositional structure as
the full 10,000-probe panel.

Allocation logic
----------------
For each of the 14 functional_elements, allocate

    n_subset[elem] = round(N_subset * count_full[elem] / N_full)

then randomly sample exactly that many probes from the element
(without replacement). With N_subset = 1000 and the panel's
canonical element sizes (150 .. 1500), this yields:

    fungi_genome      150     (15.0% of subset)
    promoter          140     (14.0%)
    enhancer          140     (14.0%)
    yeast_genome      120     (12.0%)
    virus_species     110     (11.0%)
    splice_donor       60     ( 6.0%)
    splice_acceptor    60     ( 6.0%)
    virus_variants     60     ( 6.0%)
    chromatin_access   45     ( 4.5%)
    polyA              35     ( 3.5%)
    lncRNA             30     ( 3.0%)
    nascent_RNA        20     ( 2.0%)
    splicing_plant_acceptor  15  ( 1.5%)
    splicing_plant_donor     15  ( 1.5%)
    ------------------------------
    total            1000

Output
------
  out_panel/MLM_k1ablation_1000_main_panel.parquet    full ProbeRow schema
                                                      with N_subset = 1000
  out_panel/MLM_k1ablation_1000_manifest.json         sampling provenance:
                                                      seed, per-element counts,
                                                      timestamp, ablation purpose

The subset parquet has the same column schema as main_panel.parquet so
the existing scoring pipeline (run_phase1_scoring.py --panel <path>)
can read it without modification.

Usage
-----
  $PY scripts/build_k-stride-PPL_ablation_subset.py
  $PY scripts/build_k-stride-PPL_ablation_subset.py --n-subset 500 --seed 7
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--panel", type=Path,
                   default=REPO_ROOT / "out_panel" / "main_panel.parquet",
                   help="Source full panel parquet.")
    p.add_argument("--n-subset", type=int, default=1000,
                   help="Target subset size. Per-element counts are "
                        "proportional to the full panel composition "
                        "(rounded; tiny rounding drift may make the actual "
                        "total differ from this by 0-2). Default 1000.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility.")
    p.add_argument("--out-parquet", type=Path,
                   default=REPO_ROOT / "out_panel"
                   / "MLM_k1ablation_1000_main_panel.parquet",
                   help="Output parquet path for the subset (same schema "
                        "as the source panel).")
    p.add_argument("--out-manifest", type=Path,
                   default=REPO_ROOT / "out_panel"
                   / "MLM_k1ablation_1000_manifest.json",
                   help="Output JSON path for sampling manifest "
                        "(seed, per-element counts, timestamp).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.panel.exists():
        sys.exit(f"source panel parquet not found: {args.panel}")

    table = pq.read_table(args.panel)
    df = table.to_pandas()
    print(f"[ablation-subset] source panel: {len(df)} probes from "
          f"{args.panel.relative_to(REPO_ROOT)}", flush=True)

    # Per-element allocation, proportional to full panel composition.
    elements = sorted(df["functional_element"].unique())
    full_counts = {e: int((df["functional_element"] == e).sum()) for e in elements}
    N_full = sum(full_counts.values())

    # Allocate. Rounding can drift by 1-2; we accept that and report exact
    # totals in the manifest.
    target = {
        e: int(round(args.n_subset * full_counts[e] / N_full))
        for e in elements
    }
    print(f"[ablation-subset] per-element allocation (target n_subset = "
          f"{args.n_subset}):", flush=True)
    for e in elements:
        print(f"    {e:30s}  full = {full_counts[e]:>5d}  "
              f"→ subset = {target[e]:>4d}", flush=True)
    total = sum(target.values())
    print(f"    {'TOTAL':30s}  full = {N_full:>5d}  → subset = {total:>4d}",
          flush=True)

    # Sample within each element.
    rng = np.random.default_rng(args.seed)
    selected_rows = []
    for e in elements:
        elem_rows = df[df["functional_element"] == e]
        n = target[e]
        if n > len(elem_rows):
            sys.exit(f"target {n} > available {len(elem_rows)} for {e}")
        idx = rng.choice(len(elem_rows), size=n, replace=False)
        selected_rows.append(elem_rows.iloc[idx])

    subset_df = (
        # Preserve original probe_id sort order so the file looks orderly,
        # and any downstream column-index alignment is reproducible.
        sorted_concat(selected_rows)
    )
    print(f"[ablation-subset] subset rows: {len(subset_df)} "
          f"(probe_id unique: {subset_df['probe_id'].nunique()})", flush=True)

    # Write parquet (same schema as source).
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(args.out_parquet, index=False)
    print(f"[ablation-subset] wrote {args.out_parquet.relative_to(REPO_ROOT)}",
          flush=True)

    # Write manifest.
    manifest = {
        "ablation_purpose": (
            "MLM stride-PLL k=1 vs k=6 stability ablation; per-model "
            "Pearson r between k=1 and k=6 sum_log_p vectors on this "
            "stratified 1000-probe subset (paper.md Fig S2a)."
        ),
        "branch": "mlm_or_encoder",
        "source_panel": str(args.panel.relative_to(REPO_ROOT)),
        "source_panel_n_probes": int(N_full),
        "subset_n_probes_target": int(args.n_subset),
        "subset_n_probes_actual": int(len(subset_df)),
        "seed": int(args.seed),
        "per_element": {
            e: {
                "full_count": int(full_counts[e]),
                "subset_count": int(target[e]),
                "fraction_full": round(full_counts[e] / N_full, 4),
                "fraction_subset": round(target[e] / total, 4),
            }
            for e in elements
        },
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[ablation-subset] wrote {args.out_manifest.relative_to(REPO_ROOT)}",
          flush=True)


def sorted_concat(parts):
    """Concatenate parts and sort by probe_id."""
    import pandas as pd
    return pd.concat(parts, ignore_index=True).sort_values("probe_id").reset_index(drop=True)


if __name__ == "__main__":
    main()
