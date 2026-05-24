"""Representation analysis helpers.

Module plan (matrix names per phase_1.md § 单矩阵协议:
L = raw sum_log_p, Q = double_center(clip(L)); analyses run on Q):
- pca_normalization.py     double-centering + truncated SVD: L → L_clipped → Q → SVD
- per_probe_heterozygosity Var_m(Q[m, x]) across the model population
- marginal_fst.py          legacy single-axis variance decomposition (phase 1)
- gc_axis_diagnostic.py    PC * probe-GC correlation; flag |r| > 0.7
- cross_branch_rank.py     Spearman rho + top-k overlap (AR vs MLM)
- mantel.py                Mantel + family-blocked + partial Mantel
- probe_bootstrap.py       bootstrap 95% CI for explained variance diagnostics
- metadata_baseline.py     B1: metadata-only embedding
- kmer_composition_baseline.py  B2: Procrustes / composition-control baseline
- phenotype_alignment.py   Mantel(D_lik, D_phenotype) for phase 2
"""

from __future__ import annotations

__all__: list[str] = []
