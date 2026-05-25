#!/usr/bin/env python3
"""Build the main biological panel (10K probes) from local benchmark sources.

Reads `data/panel_sources.yaml` — the single source of truth for which
dataset feeds which functional element, how labels are filtered, how
sequences are truncated, and how many probes each element gets.

Emits under --out-dir (default `out_panel/`):
    main_panel.parquet         10K rows × 11 fields (ProbeRow schema)
    panel_manifest.json        per-element / per-dataset emitted counts + seeds
    panel_summary.md           human-readable element × species cross-tab + GC stats

Run modes:
    full:   python scripts/build_panel.py
    fast:   python scripts/build_panel.py --fast    (1/10 size, ~ 1K probes)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.panel.main_panel import (  # noqa: E402
    build_main_panel,
    load_panel_config,
    write_panel_outputs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sources",
        type=Path,
        default=REPO_ROOT / "data" / "panel_sources.yaml",
        help="Panel config YAML (default: data/panel_sources.yaml)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "out_panel",
        help="Output directory (default: out_panel/)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Build at 1/10 scale (~1K probes) for quick iteration",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_panel_config(args.sources)

    if args.fast:
        # Scale every dataset's target_n by 1/10 (keep ≥1 per dataset)
        from dataclasses import replace
        scaled_elements = []
        for e in cfg.elements:
            new_ds = []
            for d in e.datasets:
                new_ds.append(replace(d, target_n=max(1, d.target_n // 10),
                                      balance_per_label=(max(1, d.balance_per_label // 10)
                                                        if d.balance_per_label else None)))
            scaled_elements.append(replace(e, n_probes=max(1, e.n_probes // 10),
                                           datasets=new_ds))
        cfg = replace(cfg, elements=scaled_elements,
                      total_probes=cfg.total_probes // 10)
        print(f"[fast mode] scaled to total_probes={cfg.total_probes}")

    print(f"Loaded config: {len(cfg.elements)} elements, target {cfg.total_probes} probes, seed {cfg.seed}")

    df, manifest = build_main_panel(cfg, repo_root=REPO_ROOT)

    print(f"Emitted {len(df)} probes across {df['functional_element'].nunique()} elements")
    # Per-element status print
    print("\nPer-element emission:")
    for elem_id, e in manifest["elements"].items():
        status = "✓" if e["n_probes_emitted"] >= e["n_probes_target"] * 0.95 else "⚠"
        print(f"  {status} {elem_id:<25} target={e['n_probes_target']:>5d}  emitted={e['n_probes_emitted']:>5d}")

    write_panel_outputs(args.out_dir, df, manifest)
    print(f"\nWrote: {args.out_dir}/main_panel.parquet  ({len(df)} rows)")
    print(f"       {args.out_dir}/panel_manifest.json")
    print(f"       {args.out_dir}/panel_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
