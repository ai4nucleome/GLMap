"""AR log-likelihood and MLM stride pseudo-log-likelihood scoring.

This package is the single implementation of the scoring contract defined in
phase_1.md section "打分协议". Phase 0 / 2 / 3 reuse the same code; do not
fork length normalization, stride PLL k, BPE adaptation, or codon adaptation
into stage-specific modules.

Module plan:
- ar_likelihood.py     forward sum_log_p + ell_per_base + bpb
- mlm_pseudo_ll.py     stride PLL (primary k=6, sensitivity k=4)
- rerun_stability.py   self-correlation >= 0.95 gate

(The earlier codon_handling.py / R_*_pan NaN-policy module was retired
when the three-matrix split moved to a single-matrix protocol in commit
5e59154; codon models now enter L with full coverage and the codon offset
on noncoding probes is absorbed by ModelMap double-centering.)
"""

from __future__ import annotations

__all__: list[str] = []
