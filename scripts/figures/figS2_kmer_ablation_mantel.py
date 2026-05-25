#!/usr/bin/env python3
"""Figure S2: Mantel-style stability of pairwise model distances under
the k=6 → k=1 stride-PLL ablation.

Analogous to Fig 2c (split-half stability), but the perturbation
dimension is now the MLM scoring stride: instead of splitting probes
into two halves, we hold the probe set fixed and replace each model's
k=6 sum_log_p vector with its k=1 vector (the more expensive but
canonical pseudo-log-likelihood). For each ablation experiment we
compute the model × model pairwise squared-Euclidean distance matrix
under both k=6 and k=1, then plot every model-pair distance as a
single point (x = distance with k=6, y = distance with k=1) plus a
Pearson r annotation.

Crucially, the GLMap pipeline (clip + double-centering) is applied
**only on the subset of models and probes belonging to the ablation
experiment** — not on the full 123-model audit matrix. This isolates
the effect of the stride change from any noise contribution induced
by the surrounding (un-ablated) AR rows in the global matrix.

Two ablation experiments, two pipeline modes → 4 single-panel PDFs:

  Experiment A : 56 MLM models × 1000-probe stratified subset
    out: out_phase1/MLM_k1ablation_1000_scores/scores/<slug>/probes.parquet
    k=6 reference subset extracted from out_phase1/scores/<slug>/probes.parquet
    C(56, 2) = 1540 model pairs.

  Experiment B : 10 representative MLM models × full 10000-probe panel
    out: out_phase1/MLM_k1_ablation_full_scores/scores/<slug>/probes.parquet
    C(10, 2) = 45 model pairs.

Mode --no-pipeline switches off clip + double-center and computes the
distance directly on the raw L sub-matrix — the sanity check that
matches the no-pipeline arm of Fig 2c.

Usage
-----
  $PY scripts/figures/figS2_kmer_ablation_mantel.py --experiment A
  $PY scripts/figures/figS2_kmer_ablation_mantel.py --experiment B
  $PY scripts/figures/figS2_kmer_ablation_mantel.py --experiment A --no-pipeline
  $PY scripts/figures/figS2_kmer_ablation_mantel.py --all   # all 4 combos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402
from glmap.matrices.build import (  # noqa: E402
    clip_lower,
    double_center,
    pairwise_squared_distance,
)


# ─────────────────────── experiment specs ─────────────────────────── #


EXPERIMENTS = {
    "A": {
        "tag": "ExpA-56MLM-1000probes",
        "k1_dir": REPO_ROOT / "out_phase1/MLM_k1ablation_1000_scores/scores",
        "k6_dir": REPO_ROOT / "out_phase1/scores",
        "probe_filter_parquet": (
            REPO_ROOT / "out_panel/MLM_k1ablation_1000_main_panel.parquet"
        ),
        "expected_models": 56,
        "headline": "56 MLM models × 1000 probes (stratified subset)",
    },
    "B": {
        "tag": "ExpB-10MLM-10000probes",
        "k1_dir": REPO_ROOT / "out_phase1/MLM_k1_ablation_full_scores/scores",
        "k6_dir": REPO_ROOT / "out_phase1/scores",
        "probe_filter_parquet": None,    # full panel
        "expected_models": 10,
        "headline": "10 representative MLM × full 10000-probe panel",
    },
}


# ─────────────────────── data loading ─────────────────────────────── #


def _load_vec(parquet_path: Path, probe_filter: list[str] | None):
    if not parquet_path.exists():
        sys.exit(f"missing parquet: {parquet_path}")
    df = pq.read_table(
        parquet_path, columns=["probe_id", "sum_log_p"]
    ).to_pandas()
    if probe_filter is not None:
        df = df[df["probe_id"].isin(probe_filter)]
    return df.sort_values("probe_id").reset_index(drop=True)


def _build_subset_matrices(k1_dir: Path, k6_dir: Path,
                            probe_filter: list[str] | None,
                            nan_threshold: float = 0.05,
                            ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """For every slug present in k1_dir, load both its k=1 and k=6
    sum_log_p vectors on the chosen probe filter and stack them into
    matching L_k1, L_k6 matrices.

    Fail-fast on probe_id mis-alignment or missing parquets. Models
    whose k=1 parquet has >``nan_threshold`` fraction of NaN
    sum_log_p are reported and SKIPPED (separate ablation-scoring bug;
    see Fig S2 notes — affects NT v1 family with 6-mer tokenizer
    under --stride 1). Returns (L_k1, L_k6, kept_hf_ids, skipped_hf_ids).
    """
    slugs = sorted([d.name for d in k1_dir.iterdir() if d.is_dir()])
    L_k1_rows: list[np.ndarray] = []
    L_k6_rows: list[np.ndarray] = []
    kept_hf_ids: list[str] = []
    skipped_hf_ids: list[str] = []
    probe_order: list[str] | None = None

    for slug in slugs:
        k1_pq = k1_dir / slug / "probes.parquet"
        k6_pq = k6_dir / slug / "probes.parquet"
        if not k1_pq.exists():
            sys.exit(f"[figS2] FATAL: missing k=1 parquet for {slug}: {k1_pq}")
        if not k6_pq.exists():
            sys.exit(f"[figS2] FATAL: missing k=6 parquet for {slug}: {k6_pq}")

        df_k1 = _load_vec(k1_pq, probe_filter)
        df_k6 = _load_vec(k6_pq, probe_filter)

        # Align both rows to the probe_id intersection (k=6 may have a
        # larger set than k=1 in Exp A; we take whatever k=1 has).
        common = list(df_k1["probe_id"])
        df_k6 = df_k6[df_k6["probe_id"].isin(common)].sort_values(
            "probe_id"
        ).reset_index(drop=True)
        if df_k6["probe_id"].tolist() != df_k1["probe_id"].tolist():
            sys.exit(
                f"[figS2] FATAL: probe_id alignment failed for {slug}; "
                f"k=1 has {len(df_k1)} rows, aligned k=6 has {len(df_k6)}."
            )

        if probe_order is None:
            probe_order = df_k1["probe_id"].tolist()
        elif df_k1["probe_id"].tolist() != probe_order:
            sys.exit(
                f"[figS2] FATAL: probe_id order mismatch in {slug} vs "
                f"the first model. Re-score against the canonical panel."
            )

        v1 = df_k1["sum_log_p"].to_numpy()
        v6 = df_k6["sum_log_p"].to_numpy()
        if np.isnan(v6).any():
            sys.exit(f"[figS2] FATAL: NaN in k=6 sum_log_p for {slug}")

        # k=1 may have partial-NaN due to a separate ablation-scoring
        # bug (NT v1 / 6-mer tokenizer × --stride 1). Skip those.
        n_nan = int(np.isnan(v1).sum())
        if n_nan > nan_threshold * len(v1):
            print(f"  [skip-NaN] {slug}: {n_nan}/{len(v1)} NaN in k=1  "
                  f"(>{nan_threshold:.0%} threshold)", flush=True)
            skipped_hf_ids.append(slug.replace("__", "/"))
            continue
        if n_nan > 0:
            print(f"  [warn-NaN] {slug}: {n_nan}/{len(v1)} NaN in k=1  "
                  f"— replacing with k=6 fallback", flush=True)
            v1 = v1.copy()
            mask = np.isnan(v1)
            v1[mask] = v6[mask]

        L_k1_rows.append(v1)
        L_k6_rows.append(v6)
        kept_hf_ids.append(slug.replace("__", "/"))

    L_k1 = np.stack(L_k1_rows)
    L_k6 = np.stack(L_k6_rows)
    return L_k1, L_k6, kept_hf_ids, skipped_hf_ids


# ─────────────────── pipeline + distance ──────────────────────────── #


def _compute_distance_matrix(L_sub: np.ndarray, clip_q: float,
                              use_pipeline: bool) -> np.ndarray:
    if use_pipeline:
        Lc, _ = clip_lower(L_sub, q=clip_q)
        Q, _, _, _ = double_center(Lc)
        return pairwise_squared_distance(Q)
    return pairwise_squared_distance(L_sub)


def _upper_triangular_pairs(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


# ─────────────────── plot (same look as Fig 2c) ───────────────────── #


def _make_figure(
    d_k6: np.ndarray, d_k1: np.ndarray,
    r: float, p: float, slope: float, intercept: float,
    experiment: str, M: int, use_pipeline: bool,
    figsize: tuple[float, float], out_path: Path,
) -> None:
    """Single-panel Mantel scatter mirroring the Fig 2c design."""
    n_pairs = len(d_k6)
    local_rc = {
        **RCPARAMS,
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.4,
    }

    with plt.rc_context(local_rc):
        fig, ax = plt.subplots(figsize=figsize)

        lo = float(min(d_k6.min(), d_k1.min()))
        hi = float(max(d_k6.max(), d_k1.max()))
        pad = 0.04 * (hi - lo)
        line_lo, line_hi = lo - pad, hi + pad

        # y = x reference
        ax.plot([line_lo, line_hi], [line_lo, line_hi],
                color=PALETTE["neutral"], linestyle="--",
                linewidth=1.1, zorder=1, label="y = x")

        # Scatter
        ax.scatter(
            d_k6, d_k1,
            s=22 if n_pairs < 200 else 14,
            c=PALETTE["blue_main"], alpha=0.55 if n_pairs < 200 else 0.45,
            edgecolors="none", zorder=2,
        )

        # Best-fit (OLS)
        fit_x = np.array([line_lo, line_hi])
        fit_y = slope * fit_x + intercept
        ax.plot(
            fit_x, fit_y,
            color=PALETTE["red_strong"], linewidth=1.6, alpha=0.9,
            zorder=3, label=f"best fit (slope = {slope:.2f})",
        )

        # Stats annotation (upper-left)
        if p == 0:
            p_str = r"$p < 10^{-300}$"
        elif p < 1e-3:
            p_str = f"$p < 10^{{-{int(np.floor(-np.log10(p)))}}}$"
        else:
            p_str = f"$p = {p:.3g}$"
        ax.text(
            0.04, 0.96,
            f"Pearson $r$ = {r:.3f}\n"
            f"{p_str}\n"
            f"$N_{{\\mathrm{{pairs}}}}$ = {n_pairs:,}  ($M$ = {M})",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )

        # Title hierarchy: suptitle + grey axis title subtitle.
        pipeline_label = (
            "clip + double-centered"
            if use_pipeline else "raw L (no pipeline)"
        )
        fig.suptitle(
            "Pairwise model distances: stride PLL k=6 vs k=1",
            fontsize=14, y=0.995, x=0.55,
        )
        ax.set_title(
            f"{EXPERIMENTS[experiment]['headline']}  ·  {pipeline_label}",
            fontsize=11, color="#555", pad=8,
        )

        ax.set_xlabel("Pairwise distance (k = 6, default)")
        ax.set_ylabel("Pairwise distance (k = 1, ablation)")
        ax.set_xlim(line_lo, line_hi)
        ax.set_ylim(line_lo, line_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.7)
        ax.legend(loc="lower right", frameon=False, fontsize=11,
                  handlelength=2.4, borderaxespad=0.6)

        fig.tight_layout(pad=1.2)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


# ──────────────────────────── main ────────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", choices=list(EXPERIMENTS) + ["all"],
                   default="A",
                   help="Which ablation experiment to render. 'A' = 56 "
                        "MLM × 1000 probes; 'B' = 10 MLM × full panel; "
                        "'all' = produce both, each in both pipeline / "
                        "no-pipeline modes (4 PDFs total).")
    p.add_argument("--no-pipeline", action="store_true",
                   help="Skip clip + double-center; compute squared "
                        "Euclidean directly on the raw L sub-matrix. "
                        "Used only when --experiment is A or B "
                        "(ignored when --experiment all, which "
                        "produces both modes).")
    p.add_argument("--clip-q", type=float, default=0.02,
                   help="GLMap pipeline clip quantile (default 0.02).")
    p.add_argument("--out", dest="out_dir", type=Path,
                   default=REPO_ROOT / "figures",
                   help="Output directory.")
    p.add_argument("--figsize", type=str, default="6,6",
                   help='Inches, "W,H". Default "6,6".')
    return p.parse_args()


def _figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def _run_one(experiment: str, use_pipeline: bool, clip_q: float,
             out_dir: Path, figsize: tuple[float, float]) -> None:
    spec = EXPERIMENTS[experiment]
    print(f"\n[figS2] === {experiment} :: "
          f"{'with-pipeline' if use_pipeline else 'no-pipeline'} ===",
          flush=True)

    if spec["probe_filter_parquet"] is not None:
        probe_filter = pq.read_table(
            spec["probe_filter_parquet"], columns=["probe_id"]
        ).to_pandas()["probe_id"].tolist()
        print(f"[figS2] probe filter: {len(probe_filter)} probes from "
              f"{spec['probe_filter_parquet'].relative_to(REPO_ROOT)}",
              flush=True)
    else:
        probe_filter = None
        print(f"[figS2] probe filter: none (full panel)", flush=True)

    L_k1, L_k6, hf_ids, skipped_hf_ids = _build_subset_matrices(
        spec["k1_dir"], spec["k6_dir"], probe_filter
    )
    M = L_k1.shape[0]
    print(f"[figS2] subset matrix: M={M} models, N={L_k1.shape[1]} probes  "
          f"(expected M={spec['expected_models']}; skipped "
          f"{len(skipped_hf_ids)} NaN-affected)", flush=True)
    if skipped_hf_ids:
        print(f"[figS2] skipped models (k=1 NaN bug):", flush=True)
        for h in skipped_hf_ids:
            print(f"    - {h}", flush=True)

    D_k6 = _compute_distance_matrix(L_k6, clip_q=clip_q,
                                     use_pipeline=use_pipeline)
    D_k1 = _compute_distance_matrix(L_k1, clip_q=clip_q,
                                     use_pipeline=use_pipeline)
    d_k6 = _upper_triangular_pairs(D_k6)
    d_k1 = _upper_triangular_pairs(D_k1)

    r, p = pearsonr(d_k6, d_k1)
    slope, intercept = np.polyfit(d_k6, d_k1, deg=1)
    print(f"[figS2] Mantel: r = {r:.4f}, slope = {slope:.3f}, "
          f"N pairs = {len(d_k6)} (= C({M}, 2))", flush=True)

    pipeline_tag = "with-pipeline" if use_pipeline else "no-pipeline"
    out_path = out_dir / (
        f"FigS2-k1-vs-k6-mantel_{spec['tag']}_{pipeline_tag}.pdf"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_figure(
        d_k6, d_k1, r=float(r), p=float(p),
        slope=float(slope), intercept=float(intercept),
        experiment=experiment, M=M, use_pipeline=use_pipeline,
        figsize=figsize, out_path=out_path,
    )
    print(f"[done] wrote {out_path}", flush=True)


def main() -> None:
    args = parse_args()
    figsize = _figsize(args.figsize)
    if args.experiment == "all":
        # Produce all 4 PDFs.
        for exp in ("A", "B"):
            for use_pipeline in (True, False):
                _run_one(exp, use_pipeline=use_pipeline,
                         clip_q=args.clip_q, out_dir=args.out_dir,
                         figsize=figsize)
    else:
        _run_one(
            args.experiment, use_pipeline=not args.no_pipeline,
            clip_q=args.clip_q, out_dir=args.out_dir, figsize=figsize,
        )


if __name__ == "__main__":
    main()
