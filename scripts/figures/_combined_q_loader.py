"""Shared loader for the combined-branch GLMap representation matrix.

The canonical strict-aggregate output (`out_phase1/matrices/`) writes
L_AR / L_MLM / Q_AR / Q_MLM / D_AR / D_MLM as PER-BRANCH matrices, per
the phase_1.md single-matrix protocol. For figures that adopt the
"combined GLMap" presentation (Option B in the Fig 2d / 2e design
discussion — see paper.md), this module assembles a 123-model
matrix on-the-fly and applies the canonical clip + double-center
pipeline once across both branches. The empirical justification for
showing AR + MLM in one centered-response GLMap is documented in
`docs/ar_mlm_merge_diagnostic.md`: raw scores are not treated as
commensurate probabilities, but branch effects are small in total
Q-space variance, do not dominate the leading PCs, and joint centering
preserves each branch's internal distance geometry.

Returned data
-------------
``CombinedGLMap`` carries everything Fig 2d (heatmap) and Fig 2e
(dendrogram) need, with rows aligned across L, Q, D and the per-row
metadata lists (hf_ids, branches, families, organizations):
  L                : (M=123, N=10000) raw sum_log_p, row-aligned with hf_ids
  L_clipped        : (M, N) after ``clip_lower(L, q=clip_q)``
  Q                : (M, N) after ``double_center(L_clipped)``
  D                : (M, M) pairwise squared Euclidean on Q
  hf_ids           : list[str], length M
  branches         : list[str], length M, each "ar_or_generative" or "mlm_or_encoder"
  families         : list[str], length M, from the audit
  organizations    : list[str], length M, from the audit
  param_counts     : list[int], length M, parameter counts from the audit
  probe_ids        : list[str], length N
  functional_elements : list[str], length N

The probe column order matches the canonical `out_panel/main_panel.parquet`
probe_id ordering (the same one used everywhere else in this repo).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.matrices.build import (  # noqa: E402
    clip_lower,
    double_center,
    pairwise_squared_distance,
)

FAMILY_CANONICAL = {
    "GenomeOceanV1.2": "GenomeOcean",
    "GENERatorv2": "GENERator",
    "Genosv2": "Genos",
}


def canonical_family(family: str) -> str:
    """Collapse checkpoint-version labels into paper-level model families."""
    return FAMILY_CANONICAL.get(family, family)


@dataclass
class CombinedGLMap:
    L: np.ndarray
    L_clipped: np.ndarray
    Q: np.ndarray
    D: np.ndarray
    hf_ids: list[str]
    branches: list[str]
    families: list[str]
    organizations: list[str]
    param_counts: list[int]
    probe_ids: list[str]
    functional_elements: list[str]
    clip_threshold: float


def load_combined_glmap(
    audit_path: Path = REPO_ROOT / "data/audits/models.json",
    scores_dir: Path = REPO_ROOT / "out_phase1/scores",
    panel_path: Path = REPO_ROOT / "out_panel/main_panel.parquet",
    clip_q: float = 0.02,
) -> CombinedGLMap:
    """Build the combined 123-model GLMap on-the-fly. Fail-fast on any
    data quality issue (missing parquet, probe_id misalignment, NaN)."""
    panel_df = pq.read_table(
        panel_path, columns=["probe_id", "functional_element"]
    ).to_pandas().sort_values("probe_id").reset_index(drop=True)
    probe_order = panel_df["probe_id"].tolist()
    elements = panel_df["functional_element"].tolist()

    audit = json.loads(audit_path.read_text())["models"]
    L_rows: list[np.ndarray] = []
    hf_ids: list[str] = []
    branches: list[str] = []
    families: list[str] = []
    organizations: list[str] = []
    param_counts: list[int] = []
    for m in audit:
        if m.get("branch") not in ("ar_or_generative", "mlm_or_encoder"):
            continue
        slug = m["hf_id"].replace("/", "__")
        pq_path = scores_dir / slug / "probes.parquet"
        if not pq_path.exists():
            sys.exit(
                f"[combined_q] FATAL: missing probes.parquet for {m['hf_id']}"
            )
        df = pq.read_table(
            pq_path, columns=["probe_id", "sum_log_p"]
        ).to_pandas().sort_values("probe_id").reset_index(drop=True)
        if df["probe_id"].tolist() != probe_order:
            sys.exit(
                f"[combined_q] FATAL: probe_id misalignment in {m['hf_id']}"
            )
        v = df["sum_log_p"].to_numpy()
        if np.isnan(v).any():
            sys.exit(
                f"[combined_q] FATAL: NaN sum_log_p in {m['hf_id']}"
            )
        L_rows.append(v)
        hf_ids.append(m["hf_id"])
        branches.append(m["branch"])
        families.append(canonical_family(m.get("family", "unknown")))
        organizations.append(m.get("organization") or "(unknown)")
        param_counts.append(int(m.get("param_count") or 0))

    L = np.stack(L_rows, axis=0)
    L_clipped, threshold = clip_lower(L, q=clip_q)
    Q, _, _, _ = double_center(L_clipped)
    D = pairwise_squared_distance(Q)
    return CombinedGLMap(
        L=L, L_clipped=L_clipped, Q=Q, D=D,
        hf_ids=hf_ids, branches=branches, families=families,
        organizations=organizations, param_counts=param_counts,
        probe_ids=probe_order, functional_elements=elements,
        clip_threshold=float(threshold),
    )


__all__ = ["CombinedGLMap", "FAMILY_CANONICAL", "canonical_family", "load_combined_glmap"]
