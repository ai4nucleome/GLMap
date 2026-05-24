"""Probe panel construction (Stage 2 of GOAL.md).

Two disjoint panels:
- main biological panel: 10,000 probes across 14 functional elements pulled
  directly from GUE / PGB / DFB benchmark files. Source-of-truth config is
  `data/panel_sources.yaml`. Schema (11 fields) is `main_panel.ProbeRow`.
- control panel: 10,000 synthetic probes across random_ACGT /
  dinucleotide_shuffled / motif_spiked, used for diagnostics only and never
  merged into the main sequence-likelihood matrix.

Build with `scripts/build_panel.py` and `scripts/build_control_panel.py`.
Once frozen, panels are immutable: probe_id is the cross-stage join key.

Modules:
  composition     GC fraction + dinuc/trinuc vector helpers
  readers         per-format dataset readers (CSV / 3 PGB FASTA variants)
  main_panel      orchestrator: config parsing + sampling + ProbeRow assembly
  control_panel   synthetic controls (3 subsets)
"""

from __future__ import annotations

__all__: list[str] = []
