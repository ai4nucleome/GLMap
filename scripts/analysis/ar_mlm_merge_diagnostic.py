#!/usr/bin/env python3
"""AR / MLM merge diagnostic: scale + variance-decomposition checks.

Provides empirical justification for treating AR (autoregressive
``log p(x)``) and MLM (pseudo-log-likelihood ``PLL(x)``) scores within a
single combined-branch GLMap representation matrix. Raw AR likelihood and
MLM pseudo-likelihood are not directly comparable, but their clipped and
double-centered GLMap response profiles can be shown in one landscape
when branch effects are measured and shown not to dominate the
representation.

Inputs
------
  results/scores/AR_MLM_scores/<slug>/probes.parquet  — per-model sum_log_p
  data/audits/models.json                  — branches + families

Outputs (standalone PDFs under results/figures/)
----------------------------------------
  FigS2-a-raw_V_score_scale.pdf     raw V per-(model, probe) score distributions
  FigS2-b-Vd_row_std.pdf            V_d row-std distributions after centering
  FigS2-c-raw_V_metadata_eta2.pdf   eta^2 of branch/family/org in raw V
  FigS2-d-Vd_metadata_eta2.pdf      eta^2 of branch/family/org in V_d

Empirical checks
----------------
  1. Raw V scale sanity check (AR vs MLM marginal score scale).
  2. Post-pipeline V_d row scale check.
  3. Variance decomposition: branch vs family vs organization (eta^2),
     computed on both the raw V and the centered V_d.

Usage
-----
  $PY scripts/analysis/ar_mlm_merge_diagnostic.py
  $PY scripts/analysis/ar_mlm_merge_diagnostic.py --no-figure   # numbers only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402


# ─────────────────────────── tests ─────────────────────────── #


def _test1_raw_L_scale(L: np.ndarray, branches: np.ndarray) -> dict:
    ar = L[branches == "ar_or_generative"].ravel()
    mlm = L[branches == "mlm_or_encoder"].ravel()
    out = {
        "ar": {
            "median": float(np.median(ar)),
            "q25":    float(np.quantile(ar, 0.25)),
            "q75":    float(np.quantile(ar, 0.75)),
            "min":    float(ar.min()),
            "max":    float(ar.max()),
        },
        "mlm": {
            "median": float(np.median(mlm)),
            "q25":    float(np.quantile(mlm, 0.25)),
            "q75":    float(np.quantile(mlm, 0.75)),
            "min":    float(mlm.min()),
            "max":    float(mlm.max()),
        },
    }
    out["median_ratio"] = float(out["ar"]["median"] / out["mlm"]["median"])
    out["pass"] = 0.5 <= abs(out["median_ratio"]) <= 2.0
    return out


def _test2_Q_row_stats(Q: np.ndarray, branches: np.ndarray) -> dict:
    ar = Q[branches == "ar_or_generative"]
    mlm = Q[branches == "mlm_or_encoder"]
    ar_means = ar.mean(axis=1)
    mlm_means = mlm.mean(axis=1)
    ar_stds = ar.std(axis=1)
    mlm_stds = mlm.std(axis=1)
    out = {
        "ar": {
            "row_mean_mean": float(ar_means.mean()),
            "row_mean_std":  float(ar_means.std()),
            "row_std_mean":  float(ar_stds.mean()),
        },
        "mlm": {
            "row_mean_mean": float(mlm_means.mean()),
            "row_mean_std":  float(mlm_means.std()),
            "row_std_mean":  float(mlm_stds.mean()),
        },
    }
    out["std_ratio"] = float(out["ar"]["row_std_mean"] / out["mlm"]["row_std_mean"])
    out["pass"] = (
        abs(out["ar"]["row_mean_mean"]) < 5
        and abs(out["mlm"]["row_mean_mean"]) < 5
        and 0.5 <= abs(out["std_ratio"]) <= 2.0
    )
    return out


def _eta2(Q: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of row-space variance explained by a categorical label."""
    labels = np.asarray(labels)
    X = Q - Q.mean(axis=0, keepdims=True)
    total = float((X ** 2).sum())
    between = 0.0
    for lab in sorted(set(labels.tolist())):
        idx = labels == lab
        mu = X[idx].mean(axis=0)
        between += int(idx.sum()) * float((mu ** 2).sum())
    return between / total


def _test3_variance_decomposition(
    Q: np.ndarray,
    branches: np.ndarray,
    families: np.ndarray,
    organizations: np.ndarray,
) -> dict:
    branch_eta = _eta2(Q, branches)
    family_eta = _eta2(Q, families)
    org_eta = _eta2(Q, organizations)
    return {
        "branch_eta2": float(branch_eta),
        "family_eta2": float(family_eta),
        "organization_eta2": float(org_eta),
        "pass": branch_eta < 0.05 and branch_eta < family_eta and branch_eta < org_eta,
    }


# ─────────────────────────── figures ─────────────────────────── #


def _make_individual_figures(
    L: np.ndarray, Q: np.ndarray, branches: np.ndarray,
    t1: dict, t2: dict, t3: dict, t3_raw: dict,
    out_dir: Path,
) -> None:
    """Draw the four standalone Fig. S2 diagnostic panels as PDFs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa

    BRANCH_COLOR = {
        "ar_or_generative": PALETTE["blue_main"],
        "mlm_or_encoder":   PALETTE["red_strong"],
    }
    local_rc = {
        **RCPARAMS,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1.4,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    def _style(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _save(fig, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

    with plt.rc_context(local_rc):
        # FigS2-a: raw V score scale.
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        _style(ax)
        ar_L = L[branches == "ar_or_generative"].ravel()
        mlm_L = L[branches == "mlm_or_encoder"].ravel()
        bins = np.linspace(
            min(ar_L.min(), mlm_L.min()),
            max(ar_L.max(), mlm_L.max()),
            80,
        )
        ax.hist(ar_L, bins=bins, color=BRANCH_COLOR["ar_or_generative"],
                alpha=0.55, label=f"AR ({(branches=='ar_or_generative').sum()})",
                density=True)
        ax.hist(mlm_L, bins=bins, color=BRANCH_COLOR["mlm_or_encoder"],
                alpha=0.55, label=f"MLM ({(branches=='mlm_or_encoder').sum()})",
                density=True)
        ax.axvline(t1["ar"]["median"], color=BRANCH_COLOR["ar_or_generative"],
                   linestyle="--", linewidth=1.2)
        ax.axvline(t1["mlm"]["median"], color=BRANCH_COLOR["mlm_or_encoder"],
                   linestyle="--", linewidth=1.2)
        ax.set_xlabel(r"raw $V$ score per (model, probe)")
        ax.set_ylabel("density")
        ax.set_title(
            f"(a) Raw score scale before centering\n"
            f"median ratio AR/MLM = {t1['median_ratio']:.2f}"
        )
        ax.legend(loc="upper left", frameon=False)
        _save(fig, out_dir / "FigS2-a-raw_V_score_scale.pdf")

        # FigS2-b: V_d row standard deviations.
        fig, ax = plt.subplots(figsize=(4.8, 3.7))
        _style(ax)
        ar_Q = Q[branches == "ar_or_generative"]
        mlm_Q = Q[branches == "mlm_or_encoder"]
        ar_stds = ar_Q.std(axis=1)
        mlm_stds = mlm_Q.std(axis=1)
        max_std = max(ar_stds.max(), mlm_stds.max())
        bins = np.linspace(0, max_std * 1.05, 30)
        ax.hist(ar_stds, bins=bins, color=BRANCH_COLOR["ar_or_generative"],
                alpha=0.55, label="AR rows")
        ax.hist(mlm_stds, bins=bins, color=BRANCH_COLOR["mlm_or_encoder"],
                alpha=0.55, label="MLM rows")
        ax.set_xlabel(r"$V_d$ row std")
        ax.set_ylabel("count of models")
        ax.set_title(
            f"(b) Representation scale after clipping and centering\n"
            f"std ratio AR/MLM = {t2['std_ratio']:.2f}"
        )
        ax.legend(loc="upper right", frameon=False)
        _save(fig, out_dir / "FigS2-b-Vd_row_std.pdf")

        # FigS2-c/d: metadata variance explained (eta^2) in raw V and V_d.
        eta_labels = ["branch", "family", "organization"]
        eta_colors = [PALETTE["red_strong"], PALETTE["blue_secondary"], PALETTE["teal"]]
        for filename, panel, title, stats, figsize in (
            (
                "FigS2-c-raw_V_metadata_eta2.pdf",
                "(c)",
                r"Metadata variance explained in raw $V$",
                t3_raw,
                (4.2, 3.7),
            ),
            (
                "FigS2-d-Vd_metadata_eta2.pdf",
                "(d)",
                r"Metadata variance explained in $V_d$",
                t3,
                (5.2, 4.4),
            ),
        ):
            fig, ax = plt.subplots(figsize=figsize)
            _style(ax)
            vals = [
                100 * stats["branch_eta2"],
                100 * stats["family_eta2"],
                100 * stats["organization_eta2"],
            ]
            xs = np.arange(len(vals))
            ax.bar(xs, vals, color=eta_colors, alpha=0.9,
                   edgecolor="#222", linewidth=0.6)
            for x, val in zip(xs, vals):
                ax.text(x, val + 1.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels(eta_labels, rotation=18, ha="right")
            ax.set_ylabel(r"eta$^2$ (% of variance)")
            ax.set_ylim(0, max(vals) * 1.22)
            ax.set_title(f"{panel} {title}")
            _save(fig, out_dir / filename)


# ─────────────────────────── main ─────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clip-q", type=float, default=0.02)
    p.add_argument("--no-figure", action="store_true",
                   help="Print the diagnostic numbers only; skip the PDFs.")
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results/figures",
                   help="Directory for the FigS2-{a,b,c,d}*.pdf panels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("[ar-mlm-diag] loading combined GLMap …", flush=True)
    glmap = load_combined_glmap(clip_q=args.clip_q)
    branches = np.array(glmap.branches)
    M, N = glmap.Q.shape
    print(f"[ar-mlm-diag] V matrix: ({M}, {N})  "
          f"AR={int((branches=='ar_or_generative').sum())}, "
          f"MLM={int((branches=='mlm_or_encoder').sum())}", flush=True)

    print("[ar-mlm-diag] running scale + variance-decomposition checks …", flush=True)
    t1 = _test1_raw_L_scale(glmap.L, branches)
    print(f"  1. Raw V scale: AR/MLM median ratio = {t1['median_ratio']:.3f}",
          flush=True)
    t2 = _test2_Q_row_stats(glmap.Q, branches)
    print(f"  2. V_d row stats: std ratio = {t2['std_ratio']:.3f}", flush=True)
    t3 = _test3_variance_decomposition(
        glmap.Q,
        branches,
        np.array(glmap.families),
        np.array(glmap.organizations),
    )
    t3_raw = _test3_variance_decomposition(
        glmap.L,
        branches,
        np.array(glmap.families),
        np.array(glmap.organizations),
    )
    print(
        f"  3. eta² (V_d): branch={t3['branch_eta2']:.4f}, "
        f"family={t3['family_eta2']:.4f}, "
        f"organization={t3['organization_eta2']:.4f}",
        flush=True,
    )

    n_pass = sum([t1["pass"], t2["pass"], t3["pass"]])
    print(f"[ar-mlm-diag] {n_pass} / 3 checks pass", flush=True)

    if not args.no_figure:
        _make_individual_figures(
            glmap.L, glmap.Q, branches,
            t1, t2, t3, t3_raw,
            args.out_dir,
        )
        print(f"[done] wrote FigS2-{{a,b,c,d}} panel PDFs to {args.out_dir}",
              flush=True)


if __name__ == "__main__":
    main()
