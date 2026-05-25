#!/usr/bin/env python3
"""Generate the table replacing the original Fig 4a (SVM scatter).

For each (anchor, partner, label) row in ``models/fig4a-svm.csv`` we
compute the GLMap pipeline distance between the partner model and the
shared anchor (``togethercomputer/evo-1-8k-base``):

  1. Load the AR-branch L matrix (all 67 AR models' per-probe sum_log_p
     sorted by probe_id) from ``out_phase1/scores/<slug>/probes.parquet``.
  2. Apply the ModelMap pipeline ``clip(q=0.02) + double_center`` →
     Q_AR ∈ R^{67 × 10000}. This is the same Q matrix that drives every
     other GLMap distance in the paper (Fig 4b, 5, 6, 7, etc.) — using
     it here keeps the lineage-distance values directly comparable.
  3. For each partner i: ``D[i] = ||Q_AR[anchor] - Q_AR[partner_i]||²``
     (squared Euclidean; matches ``src/matrices/build.py``).

Output
------
  tables/table_fig4a_evo_lineage_distances.tex   LaTeX booktabs.
  Markdown preview to stdout.

Schema (3 columns):
  Partner model | Derived from anchor (✓/✗) | GLMap distance to anchor

✓ corresponds to label = 1 in ``fig4a-svm.csv`` (partner is a direct
fine-tune / descendant of the anchor); ✗ corresponds to label = 0
(partner is a fresh / unrelated training run).

Usage
-----
  $PY scripts/tables/table_fig4a_evo_lineage_distances.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.matrices.build import clip_lower, double_center  # noqa: E402


CSV_PATH = REPO_ROOT / "models" / "fig4a-svm.csv"
AUDIT_PATH = REPO_ROOT / "data" / "audits" / "models.json"
SCORES_DIR = REPO_ROOT / "out_phase1" / "scores"
# Two LaTeX outputs, one per cosine-similarity variant.
TEX_OUT_RAWL = (REPO_ROOT / "tables"
                / "table_fig4a_evo_lineage_distances_cosine-rawL.tex")
TEX_OUT_Q    = (REPO_ROOT / "tables"
                / "table_fig4a_evo_lineage_distances_cosine-Q.tex")


# Architecture labels per the Evo family lineage. Used for the leftmost
# column of the lineage-distance table. evo1 / evo1.5 share the
# StripedHyena 1 backbone; evo2 family rebuilt on the StripedHyena 2
# backbone with FP8 attention.
ARCHITECTURE = {
    "togethercomputer/evo-1-8k-base":            "StripedHyena",
    "LongSafari/evo-1-8k-crispr":                "StripedHyena",
    "LongSafari/evo-1-8k-transposon":            "StripedHyena",
    "evo-design/evo-1-7b-131k-microviridae":     "StripedHyena",
    "evo-design/evo-1.5-8k-base":                "StripedHyena",
    "arcinstitute/evo2_7b_base":                 "StripedHyena 2",
    "arcinstitute/evo2_7b":                      "StripedHyena 2",
    "arcinstitute/evo2_7b_262k":                 "StripedHyena 2",
    "evo-design/evo-2-7b-8k-microviridae":       "StripedHyena 2",
}


def _load_csv():
    rows = []
    with CSV_PATH.open() as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 3 or not row[0].strip():
                continue
            anchor, partner, label = row[0].strip(), row[1].strip(), row[2].strip()
            rows.append((anchor, partner, int(label)))
    anchors = {a for a, _, _ in rows}
    if len(anchors) != 1:
        sys.exit(f"fig4a-svm.csv must share a single anchor; got {anchors}")
    return next(iter(anchors)), rows


def _load_all_L():
    """Load every audited model's sum_log_p into a combined L matrix.

    Combined (AR + MLM) so that the "Other AR" / "Other MLM" baseline
    rows in the lineage table can be computed on the same Q. Each
    model's branch (ar_or_generative / mlm_or_encoder) is returned
    alongside so the baseline rows can partition correctly.
    """
    audit = json.loads(AUDIT_PATH.read_text())["models"]
    L_rows, hf_ids, branches = [], [], []
    probe_order = None
    for m in audit:
        if m.get("branch") not in ("ar_or_generative", "mlm_or_encoder"):
            continue
        slug = m["hf_id"].replace("/", "__")
        p = SCORES_DIR / slug / "probes.parquet"
        if not p.exists():
            continue
        t = pq.read_table(p, columns=["probe_id", "sum_log_p"]).to_pandas()
        t = t.sort_values("probe_id").reset_index(drop=True)
        if probe_order is None:
            probe_order = t["probe_id"].tolist()
        elif t["probe_id"].tolist() != probe_order:
            sys.exit(f"probe_id order mismatch in {m['hf_id']}")
        vec = t["sum_log_p"].to_numpy()
        if np.isnan(vec).any():
            continue
        L_rows.append(vec)
        hf_ids.append(m["hf_id"])
        branches.append(m["branch"])
    if not L_rows:
        sys.exit("no models loaded — does out_phase1/scores/ exist?")
    return np.stack(L_rows, axis=0), hf_ids, branches


def _short(hf_id: str) -> str:
    """org/name → name (drops the org/ prefix)."""
    return hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id


def main() -> None:
    anchor, rows = _load_csv()
    print(f"[fig4a-table] anchor: {anchor}", flush=True)
    print(f"[fig4a-table] {len(rows)} partner rows from "
          f"{CSV_PATH.relative_to(REPO_ROOT)}", flush=True)

    L, hf_ids, branches = _load_all_L()
    print(f"[fig4a-table] combined L matrix: {L.shape}  "
          f"(AR={sum(b == 'ar_or_generative' for b in branches)}, "
          f"MLM={sum(b == 'mlm_or_encoder' for b in branches)})", flush=True)

    # GLMap pipeline (on combined 123-model matrix; pairwise distances
    # are invariant to whether column means were computed on AR-only or
    # the combined matrix — see /docs/, but clip threshold differs).
    L_clipped, threshold = clip_lower(L, q=0.02)
    Q, _, _, _ = double_center(L_clipped)
    print(f"[fig4a-table] clip threshold = {threshold:.2f}; Q shape {Q.shape}",
          flush=True)

    hf_index = {h: i for i, h in enumerate(hf_ids)}
    if anchor not in hf_index:
        sys.exit(f"anchor {anchor} not in combined L matrix")
    anchor_idx = hf_index[anchor]
    anchor_L = L[anchor_idx]
    anchor_L_clipped = L_clipped[anchor_idx]
    anchor_q = Q[anchor_idx]

    def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu == 0.0 or nv == 0.0:
            return float("nan")
        return float(np.dot(u, v) / (nu * nv))

    def _row_metrics(partner_idx: int) -> dict:
        """Compute distance / similarity metrics for one model vs the
        anchor. Returned keys:
          q_distance        : ||Q[anchor] - Q[partner]||²
                              (squared Euclidean on the GLMap pipeline
                               representation; canonical inter-model
                               distance used in Figs.~4b, 5, 6, 7, 8).
          cosine_sim_rawL   : cos(L[anchor], L[partner]) on the raw
                              pre-pipeline sum_log_p vectors. Values
                              uniformly close to 1 across DNA gLMs
                              because all entries are negative and
                              dominated by per-probe baseline
                              difficulty; discriminative signal is in
                              the residual (1 - cos).
          cosine_sim_Q      : cos(Q[anchor], Q[partner]) on the GLMap
                              representation (post clip + double-
                              center). Q is a high-dimensional residual,
                              so cosine on Q can be small or even
                              negative even for related models; range
                              roughly [-0.3, +0.4].
        """
        partner_L = L[partner_idx]
        partner_q = Q[partner_idx]
        return {
            "q_distance":      float(np.sum((anchor_q - partner_q) ** 2)),
            "cosine_sim_rawL": _cosine_similarity(anchor_L, partner_L),
            "cosine_sim_Q":    _cosine_similarity(anchor_q, partner_q),
        }

    # Drop anchor_L_clipped (used by previous metrics, no longer needed).
    del anchor_L_clipped

    # ── per-partner rows ──
    table_rows = []
    for _, partner, label in rows:
        if partner not in hf_index:
            sys.exit(f"partner {partner} not in combined L matrix")
        m = _row_metrics(hf_index[partner])
        table_rows.append({
            "partner": partner,
            "label": label,
            "architecture": ARCHITECTURE.get(partner, "?"),
            **m,
        })

    # Group by label (0 first, then 1); within group keep CSV order.
    table_rows.sort(key=lambda r: (r["label"], rows.index(
        next(row for row in rows if row[1] == r["partner"])
    )))

    # ── baseline summary rows ──
    # "Other AR" = AR branch excluding the anchor + the 8 listed partners.
    # "Other MLM" = all MLM models (anchor is not MLM, and partners are
    # all AR, so no exclusions needed).
    partner_set = {p for _, p, _ in rows}
    excluded_for_other_ar = {anchor} | partner_set
    other_ar_idx = [
        i for i, h in enumerate(hf_ids)
        if branches[i] == "ar_or_generative" and h not in excluded_for_other_ar
    ]
    other_mlm_idx = [
        i for i, h in enumerate(hf_ids) if branches[i] == "mlm_or_encoder"
    ]
    print(f"[fig4a-table] baselines: Other AR N={len(other_ar_idx)}, "
          f"Other MLM N={len(other_mlm_idx)}", flush=True)

    def _baseline_row(name: str, indices: list[int], arch: str) -> dict:
        metrics = [_row_metrics(i) for i in indices]
        q_d       = np.array([m["q_distance"]      for m in metrics])
        cs_rawL   = np.array([m["cosine_sim_rawL"] for m in metrics])
        cs_Q      = np.array([m["cosine_sim_Q"]    for m in metrics])
        return {
            "is_baseline": True,
            "name": name,
            "n": len(indices),
            "architecture": arch,
            "q_d_median":       float(np.median(q_d)),
            "q_d_mean":         float(np.mean(q_d)),
            "cs_rawL_median":   float(np.median(cs_rawL)),
            "cs_rawL_mean":     float(np.mean(cs_rawL)),
            "cs_Q_median":      float(np.median(cs_Q)),
            "cs_Q_mean":        float(np.mean(cs_Q)),
        }

    baseline_rows = [
        _baseline_row(
            f"Other AR models (excl. anchor + partners)", other_ar_idx,
            arch="StripedHyena/Hyena/Transformer/etc.",
        ),
        _baseline_row(
            f"Other MLM models", other_mlm_idx,
            arch="various (BERT, Mamba, etc.)",
        ),
    ]

    # ── Markdown preview ── #
    print()
    print("# Replacement for Fig 4a (Markdown preview)")
    print()
    print(f"Anchor: `{anchor}`  (Architecture: "
          f"{ARCHITECTURE.get(anchor, '?')})")
    print()
    print("| Architecture | Partner model | Derived | "
          "Euclidean distance | Cosine sim (raw L) | Cosine sim (Q) |")
    print("|---|---|:-:|---:|---:|---:|")
    for r in table_rows:
        mark = "✓" if r["label"] == 1 else "✗"
        print(f"| {r['architecture']} | `{_short(r['partner'])}` | "
              f"{mark} | "
              f"{r['q_distance']:.3e} | "
              f"{r['cosine_sim_rawL']:.4f} | "
              f"{r['cosine_sim_Q']:.4f} |")
    # baseline summary rows (median / mean across N models)
    for b in baseline_rows:
        print(
            f"| {b['architecture']} | "
            f"_{b['name']}_ (N={b['n']}) | — | "
            f"{b['q_d_median']:.3e} / {b['q_d_mean']:.3e} | "
            f"{b['cs_rawL_median']:.4f} / {b['cs_rawL_mean']:.4f} | "
            f"{b['cs_Q_median']:.4f} / {b['cs_Q_mean']:.4f} |"
        )
    print()
    print("(baseline rows show **median / mean** across N models)")
    print()

    # ── LaTeX booktabs ── #
    def _sci(x: float) -> str:
        """Format like '1.27\\!\\times\\!10^{8}' from 1.27e+08."""
        return f"{x:.2e}".replace("e+0", r"\!\times\!10^{") + "}"

    # The two output tables differ only in WHICH cosine column they
    # report. The Euclidean (GLMap-pipeline) distance is identical
    # across both. The caption text varies to describe the chosen
    # cosine variant correctly.
    CAPTION_COMMON = (
        r"Evo lineage distances. For each partner model in "
        r"\texttt{models/fig4a-svm.csv}, two complementary metrics "
        r"relate it to the shared anchor "
        r"\texttt{togethercomputer/evo-1-8k-base} (StripedHyena, evo1 "
        r"family). "
        r"\textbf{Euclidean distance} $= \| Q[\text{anchor}] - "
        r"Q[\text{partner}] \|^2$ (squared Euclidean), where $Q$ is the "
        r"combined-branch GLMap representation matrix "
        r"(\texttt{clip(q=0.02)} + double-centering applied to the L "
        r"matrix of all 123 audited models on the 10{,}000-probe panel; "
        r"\texttt{src/matrices/build.py}). This is the same pipeline "
        r"distance used in Figs.~2c, 5, 6, 7, and 8. "
    )
    CAPTION_RAWL_COSINE = (
        r"\textbf{Cosine similarity (raw L)} $= L[\text{anchor}] \cdot "
        r"L[\text{partner}] / (\|L[\text{anchor}]\| \, "
        r"\|L[\text{partner}]\|)$ is computed on the raw pre-pipeline "
        r"\texttt{sum\_log\_p} vectors and is scale-invariant. Values "
        r"are uniformly close to $1$ across DNA foundation models "
        r"because all \texttt{sum\_log\_p} entries are negative and "
        r"dominated by per-probe baseline difficulty; the discriminative "
        r"signal sits in the residual $(1-\cos)$, which shows the same "
        r"order-of-magnitude separation as the Euclidean distance. "
    )
    CAPTION_Q_COSINE = (
        r"\textbf{Cosine similarity (Q)} $= Q[\text{anchor}] \cdot "
        r"Q[\text{partner}] / (\|Q[\text{anchor}]\| \, "
        r"\|Q[\text{partner}]\|)$ is computed on the GLMap "
        r"representation $Q$ (post clip + double-centering). After "
        r"centering, $Q$ vectors live in a high-dimensional residual "
        r"space where the cosine of any two rows is typically small; "
        r"values here range from $\sim\!0.4$ (derived) down to "
        r"$\sim\!-0.3$ (unrelated). The sign and magnitude difference "
        r"reflects whether the partner's centered response pattern "
        r"co-varies with the anchor's (positive) or systematically "
        r"diverges from it (negative). "
    )
    CAPTION_TAIL = (
        r"The ``Derived from anchor'' column marks whether the partner "
        r"is a known descendant of the anchor (\checkmark{} = direct "
        r"fine-tune / weight-initialised from "
        r"\texttt{evo-1-8k-base}; \(\times\) = fresh training run, "
        r"e.g.\ the evo2 family).}"
    )

    def _make_table_tex(
        cosine_key: str,
        cosine_label: str,
        caption_cosine_text: str,
        latex_label: str,
    ) -> str:
        """Build the booktabs LaTeX for one cosine variant."""
        out = []
        out.append(r"\begin{table}[t]")
        out.append(r"  \centering")
        out.append(
            r"  \caption{" + CAPTION_COMMON + caption_cosine_text
            + CAPTION_TAIL
        )
        out.append(rf"  \label{{{latex_label}}}")
        out.append(r"  \small")
        out.append(r"  \begin{tabular}{l l c r r}")
        out.append(r"    \toprule")
        out.append(
            r"    Architecture & Partner model & Derived from anchor & "
            rf"Euclidean distance & {cosine_label} \\"
        )
        out.append(r"    \midrule")
        # Architecture column shown only on first row of each contiguous
        # group; subsequent rows in the same group leave column 1 blank.
        prev_arch = None
        for r in table_rows:
            partner_tt = _short(r["partner"]).replace("_", r"\_")
            arch = r["architecture"]
            arch_cell = arch if arch != prev_arch else ""
            prev_arch = arch
            mark = r"\checkmark" if r["label"] == 1 else r"\(\times\)"
            out.append(
                f"    {arch_cell} & \\texttt{{{partner_tt}}} & {mark} & "
                f"${_sci(r['q_distance'])}$ & {r[cosine_key]:.4f} \\\\"
            )
        out.append(r"    \bottomrule")
        out.append(r"  \end{tabular}")
        out.append(r"\end{table}")
        return "\n".join(out) + "\n"

    TEX_OUT_RAWL.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT_RAWL.write_text(_make_table_tex(
        cosine_key="cosine_sim_rawL",
        cosine_label="Cosine similarity (raw L)",
        caption_cosine_text=CAPTION_RAWL_COSINE,
        latex_label="tab:evo_lineage_distance_cosrawL",
    ))
    TEX_OUT_Q.write_text(_make_table_tex(
        cosine_key="cosine_sim_Q",
        cosine_label="Cosine similarity (Q)",
        caption_cosine_text=CAPTION_Q_COSINE,
        latex_label="tab:evo_lineage_distance_cosQ",
    ))
    print(f"# LaTeX written to:")
    print(f"#   {TEX_OUT_RAWL.relative_to(REPO_ROOT)}  "
          "(Euclidean distance + Cosine similarity on raw L)")
    print(f"#   {TEX_OUT_Q.relative_to(REPO_ROOT)}     "
          "(Euclidean distance + Cosine similarity on Q)")


if __name__ == "__main__":
    main()
