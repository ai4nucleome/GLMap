"""Single-matrix protocol construction (Stage 3 of GOAL.md).

For each branch (AR, MLM) we build one matrix per branch via the ModelMap
pipeline (`L → clip → Q → D`); every model × every probe enters L
unconditionally. The older three-matrix split (Q_pan / Q_coding_only /
Q_nucleotide_only) used to handle codon and mixed-modality models by
NaN-masking noncoding cells; that protocol was retired in commit
`5e59154` because clip + double-centering absorbs the codon-vs-nucleotide
offset (row mean subtracts each model's overall level, column mean
subtracts each probe's overall difficulty), and the systematic bias does
not pollute principal axes.

Codon-aware flags (`is_codon_model`, `mixed_modality`,
`valid_probe_classes`) from `data/audits/model_context_manifest.csv` are
still kept on the model side for downstream diagnostic loadings, but
they no longer gate matrix membership.

Never merge AR and MLM matrices (`log p(x)` and `PLL(x)` are different
probability objects); cross-branch analysis is rank-based only (see
`src/analysis/cross_branch_rank`).
"""

from __future__ import annotations

__all__: list[str] = []
