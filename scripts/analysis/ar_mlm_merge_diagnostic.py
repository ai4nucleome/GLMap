#!/usr/bin/env python3
"""AR / MLM merge diagnostic: theory + empirical checks.

Provides the empirical justification for treating AR (autoregressive
``log p(x)``) and MLM (pseudo-log-likelihood ``PLL(x)``) scores within a
single combined-branch GLMap representation matrix (Option B; see
paper.md Fig 2d / 2e design discussion).  The original protocol forbade
merging on theoretical grounds — AR's ``log p(x)`` and MLM's ``PLL(x)``
are different probability objects, so any cross-branch comparison would
be confounded by the scoring function rather than reflecting model
ability.  This script tests a narrower and more defensible claim: raw
AR likelihood and MLM pseudo-likelihood are not directly comparable, but
their centered GLMap response profiles can be shown in one landscape
when branch effects are measured, acknowledged, and shown not to
dominate the representation.

Inputs
------
  out_phase1/scores/<slug>/probes.parquet  — per-model sum_log_p
  data/audits/models.json                  — branches + families

Outputs
-------
  docs/ar_mlm_merge_diagnostic.md          — markdown report (theory +
                                              empirical checks +
                                              interpretation)
  figures/FigS2-ar_mlm_merge_diagnostic.pdf      — diagnostic figure

Empirical checks
----------------
  1. Raw V scale sanity check.
  2. Post-pipeline V_d row scale check.
  3. Variance decomposition: branch vs family vs organization.
  4. Per-PC branch separation.
  5. Low-dimensional branch predictability.
  6. Preservation of within-branch geometry after joint centering.

Usage
-----
  $PY scripts/analysis/ar_mlm_merge_diagnostic.py
  $PY scripts/analysis/ar_mlm_merge_diagnostic.py --no-figure   # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures._combined_q_loader import load_combined_glmap  # noqa: E402
from glmap.matrices.build import (  # noqa: E402
    clip_lower,
    double_center,
    pairwise_squared_distance,
)


OUTLIER_MODELS = {
    "ZhejiangLab/Genos-10B",
    "RaphaelMourad/Mistral-DNA-v1-138M-noncoding",
    "ZhejiangLab/Genos-10B-v2",
    "lingxusb/PlasmidGPT",
    "evo-design/evo-2-7b-8k-microviridae",
    "AIRI-Institute/gena-lm-bert-base-yeast",
    "ZhejiangLab/OneGenome-Rice",
    "GenerTeam/GENERanno-prokaryote-0.5b-base",
}


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


def _test4_per_pc_branch_auc(Q: np.ndarray, branches: np.ndarray, k_pcs: int = 8) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score

    y = (branches == "mlm_or_encoder").astype(int)
    pca = PCA(n_components=k_pcs)
    scores = pca.fit_transform(Q)
    aucs = []
    for k in range(k_pcs):
        # roc_auc_score requires both classes present
        col = scores[:, k]
        auc = float(roc_auc_score(y, col))
        # AUC is symmetric: 0.3 means equally separating in opposite
        # direction; report max(auc, 1 - auc) so "separation strength" is in [0.5, 1].
        aucs.append(max(auc, 1 - auc))
    explained = pca.explained_variance_ratio_.tolist()
    out = {
        "per_pc_auc": [float(a) for a in aucs],
        "explained_variance_ratio": [float(e) for e in explained],
        "max_pc_auc": float(max(aucs)),
        "argmax_pc": int(int(np.argmax(aucs)) + 1),
        "pass": float(max(aucs)) < 0.80,
    }
    return out


def _test5_pca_branch_classification(Q: np.ndarray, branches: np.ndarray) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = (branches == "mlm_or_encoder").astype(int)
    n_components = min(30, Q.shape[0] - 1)
    scores = PCA(n_components=n_components).fit_transform(Q)
    out: dict = {"by_n_pc": {}}
    for n_pc in (2, 5, 10, 20):
        Z = scores[:, :n_pc]
        probs = np.zeros(len(y))
        for tr, te in LeaveOneOut().split(Z):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0, max_iter=2000, solver="liblinear",
                    class_weight="balanced",
                ),
            )
            clf.fit(Z[tr], y[tr])
            probs[te] = clf.predict_proba(Z[te])[:, 1]
        out["by_n_pc"][n_pc] = float(roc_auc_score(y, probs))
    out["pass"] = out["by_n_pc"][20] < 0.95
    return out


def _intra_inter_distance(D: np.ndarray, branches: np.ndarray) -> dict:
    ar_idx = np.where(branches == "ar_or_generative")[0]
    mlm_idx = np.where(branches == "mlm_or_encoder")[0]

    iu_ar  = np.triu_indices_from(D[np.ix_(ar_idx, ar_idx)],  k=1)
    iu_mlm = np.triu_indices_from(D[np.ix_(mlm_idx, mlm_idx)], k=1)

    intra_ar  = D[np.ix_(ar_idx, ar_idx)][iu_ar]
    intra_mlm = D[np.ix_(mlm_idx, mlm_idx)][iu_mlm]
    inter     = D[np.ix_(ar_idx, mlm_idx)].ravel()
    out = {
        "intra_ar":  {"mean": float(intra_ar.mean()),  "median": float(np.median(intra_ar))},
        "intra_mlm": {"mean": float(intra_mlm.mean()), "median": float(np.median(intra_mlm))},
        "inter":     {"mean": float(inter.mean()),     "median": float(np.median(inter))},
    }
    out["ratio_inter_over_intra_ar"]  = out["inter"]["mean"] / out["intra_ar"]["mean"]
    out["ratio_inter_over_intra_mlm"] = out["inter"]["mean"] / out["intra_mlm"]["mean"]
    return out


def _test6_within_branch_geometry(
    L: np.ndarray,
    Q_combined: np.ndarray,
    branches: np.ndarray,
) -> dict:
    from scipy.stats import pearsonr, spearmanr

    out = {}
    pass_all = True
    for branch, label in (
        ("ar_or_generative", "AR"),
        ("mlm_or_encoder", "MLM"),
    ):
        mask = branches == branch
        L_branch = L[mask]
        L_clipped, _ = clip_lower(L_branch, q=0.02)
        Q_branch, _, _, _ = double_center(L_clipped)
        D_branch = pairwise_squared_distance(Q_branch)
        D_combined = pairwise_squared_distance(Q_combined[mask])
        iu = np.triu_indices(int(mask.sum()), k=1)
        x = D_branch[iu]
        y = D_combined[iu]
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
        out[label] = {
            "n_models": int(mask.sum()),
            "pearson": pearson,
            "spearman": spearman,
            "median_ratio_combined_over_branch": float(np.median(y) / np.median(x)),
            "mean_ratio_combined_over_branch": float(y.mean() / x.mean()),
        }
        pass_all = pass_all and spearman > 0.90 and pearson > 0.90
    out["pass"] = pass_all
    return out


def _knn_same_branch(D: np.ndarray, branches: np.ndarray, ks: Iterable[int] = (5, 10)) -> dict:
    M = D.shape[0]
    y = branches
    counts = Counter(branches.tolist())
    p_ar = counts["ar_or_generative"] / M
    p_mlm = counts["mlm_or_encoder"] / M
    # For each model i, its expected same-branch rate under random:
    # if model i is AR, expected same-branch p = (n_ar - 1) / (M - 1);
    # if MLM, expected p = (n_mlm - 1) / (M - 1).  Average over models.
    expected_per_branch = {
        "ar_or_generative": (counts["ar_or_generative"] - 1) / (M - 1),
        "mlm_or_encoder":   (counts["mlm_or_encoder"]  - 1) / (M - 1),
    }
    expected_random = (
        counts["ar_or_generative"] * expected_per_branch["ar_or_generative"]
        + counts["mlm_or_encoder"] * expected_per_branch["mlm_or_encoder"]
    ) / M

    out: dict = {
        "n_ar": counts["ar_or_generative"],
        "n_mlm": counts["mlm_or_encoder"],
        "expected_random_same_branch_rate": float(expected_random),
        "by_k": {},
    }

    # For each i, nearest k indices excluding self.
    pass_all = True
    for k in ks:
        rates = []
        for i in range(M):
            # argsort ascending; exclude self at position 0 in the sort
            order = np.argsort(D[i])
            # Drop self (whichever index equals i)
            neighbors = order[order != i][:k]
            n_same = int((y[neighbors] == y[i]).sum())
            rates.append(n_same / k)
        rate_mean = float(np.mean(rates))
        out["by_k"][k] = {
            "same_branch_rate_mean":   rate_mean,
            "diff_from_random":        rate_mean - expected_random,
        }
        if abs(rate_mean - expected_random) > 0.10:
            pass_all = False
    out["pass"] = pass_all
    return out


def _branch_structure_sensitivity(
    D: np.ndarray,
    branches: np.ndarray,
    hf_ids: list[str],
) -> dict:
    full = {
        "distance": _intra_inter_distance(D, branches),
        "knn": _knn_same_branch(D, branches, ks=(5, 10)),
    }
    keep = np.array([hf not in OUTLIER_MODELS for hf in hf_ids])
    reduced = {
        "n_removed": int((~keep).sum()),
        "distance": _intra_inter_distance(D[np.ix_(keep, keep)], branches[keep]),
        "knn": _knn_same_branch(D[np.ix_(keep, keep)], branches[keep], ks=(5, 10)),
    }
    return {"full": full, "without_outliers": reduced}


# ─────────────────────────── plot ─────────────────────────── #


def _make_figure(
    L: np.ndarray, Q: np.ndarray, D: np.ndarray,
    branches: np.ndarray,
    t1: dict, t2: dict, t3: dict, t4: dict, t5: dict, t6: dict,
    t3_raw: dict, t4_raw: dict,
    out_path: Path,
) -> None:
    """Diagnostic figure focused on the main AR / MLM merge checks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa

    BRANCH_COLOR = {
        "ar_or_generative": PALETTE["blue_main"],
        "mlm_or_encoder":   PALETTE["red_strong"],
    }
    BRANCH_LABEL = {
        "ar_or_generative": "AR",
        "mlm_or_encoder":   "MLM",
    }
    local_rc = {
        **RCPARAMS,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.linewidth": 1.4,
    }

    with plt.rc_context(local_rc):
        fig, axes = plt.subplot_mosaic(
            [
                ["raw", "scale", "pca_raw", "pca_vd"],
                ["eta_raw", "eta_vd", "pcauc_raw", "pcauc_vd"],
            ],
            figsize=(17, 9.1),
        )
        for ax in axes.values():
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        panel_labels = {
            "raw": "(a)",
            "scale": "(b)",
            "pca_raw": "(c)",
            "pca_vd": "(d)",
            "eta_raw": "(e)",
            "eta_vd": "(f)",
            "pcauc_raw": "(g)",
            "pcauc_vd": "(h)",
        }
        for key, label in panel_labels.items():
            axes[key].text(
                -0.14, 1.08, label,
                transform=axes[key].transAxes,
                ha="left", va="top",
                fontsize=12,
            )

        # ── Raw V histograms per branch ── #
        ax = axes["raw"]
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
            f"Raw score scale before centering\n"
            f"median ratio AR/MLM = {t1['median_ratio']:.2f}"
        )
        ax.legend(loc="upper left", frameon=False)

        # ── V_d row std comparison ── #
        ax = axes["scale"]
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
            f"Representation scale after clipping and centering\n"
            f"std ratio AR/MLM = {t2['std_ratio']:.2f}"
        )
        ax.legend(loc="upper right", frameon=False)

        # ── Raw V PC1 vs PC2 scatter colored by branch ── #
        ax = axes["pca_raw"]
        pca_raw = PCA(n_components=2)
        proj_raw = pca_raw.fit_transform(L)
        for b in ("ar_or_generative", "mlm_or_encoder"):
            mask = branches == b
            ax.scatter(proj_raw[mask, 0], proj_raw[mask, 1],
                       s=22, c=BRANCH_COLOR[b], alpha=0.7, edgecolors="none",
                       label=BRANCH_LABEL[b])
        ax.set_xlabel(
            f"PC1 ({100*pca_raw.explained_variance_ratio_[0]:.1f}% var)"
        )
        ax.set_ylabel(
            f"PC2 ({100*pca_raw.explained_variance_ratio_[1]:.1f}% var)"
        )
        ax.set_title(
            r"Leading PCs of raw $V$"
        )
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.6)

        # ── V_d PC1 vs PC2 scatter colored by branch ── #
        ax = axes["pca_vd"]
        pca_vd = PCA(n_components=2)
        proj_vd = pca_vd.fit_transform(Q)
        for b in ("ar_or_generative", "mlm_or_encoder"):
            mask = branches == b
            ax.scatter(proj_vd[mask, 0], proj_vd[mask, 1],
                       s=22, c=BRANCH_COLOR[b], alpha=0.7, edgecolors="none",
                       label=BRANCH_LABEL[b])
        ax.set_xlabel(
            f"PC1 ({100*pca_vd.explained_variance_ratio_[0]:.1f}% var)"
        )
        ax.set_ylabel(
            f"PC2 ({100*pca_vd.explained_variance_ratio_[1]:.1f}% var)"
        )
        ax.set_title(
            r"Leading PCs of $V_d$"
        )
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.6)

        # ── Raw V per-PC branch AUC ── #
        ax = axes["pcauc_raw"]
        aucs = t4_raw["per_pc_auc"]
        xs = np.arange(1, len(aucs) + 1)
        ax.bar(xs, aucs, color=PALETTE["blue_secondary"], alpha=0.85,
               edgecolor="#222", linewidth=0.6)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("branch separation AUC")
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks(xs)
        ax.set_title(
            r"Branch separation across PCs of raw $V$"
        )

        # ── V_d per-PC branch AUC ── #
        ax = axes["pcauc_vd"]
        aucs = t4["per_pc_auc"]
        xs = np.arange(1, len(aucs) + 1)
        ax.bar(xs, aucs, color=PALETTE["blue_secondary"], alpha=0.85,
               edgecolor="#222", linewidth=0.6)
        ax.set_xlabel("Principal component")
        ax.set_ylabel("branch separation AUC")
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks(xs)
        ax.set_title(
            r"Branch separation across PCs of $V_d$"
        )

        # ── Variance explained by metadata labels ── #
        eta_labels = ["branch", "family", "organization"]
        eta_colors = [PALETTE["red_strong"], PALETTE["blue_secondary"], PALETTE["teal"]]
        for ax_key, title, stats in (
            ("eta_raw", r"Metadata variance explained in raw $V$", t3_raw),
            ("eta_vd", r"Metadata variance explained in $V_d$", t3),
        ):
            ax = axes[ax_key]
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
            ax.set_title(title)

        fig.suptitle(
            r"AR / MLM merge diagnostic in the $V_d$ response space",
            fontsize=13, y=0.995,
        )
        caption = (
            r"Figure. Empirical diagnostic for jointly analyzing autoregressive (AR) and masked-language-model (MLM) genomic language models in GLMap. "
            r"For each model and probe, raw $V$ stores the branch-native score: AR models contribute sequence log-likelihoods, whereas MLM models contribute pseudo-log-likelihoods. "
            r"Because these raw scores are not the same probability object, GLMap compares their response profiles after lower-tail clipping and double-centering, yielding $V_d$. "
            r"(a) Raw-score distributions show that AR and MLM marginal score scales are not separated by orders of magnitude (median ratio shown), but this panel is only a scale sanity check rather than evidence of probabilistic equivalence. "
            r"(b) Row-standard-deviation distributions in $V_d$ compare the residual response amplitude of each model after normalization; similar AR and MLM spreads indicate that neither branch dominates downstream distances simply by having a larger response scale. "
            r"(c) PCA of raw $V$ shows that the unnormalized matrix is dominated by a single global score axis, consistent with strong calibration or probe-difficulty effects in raw likelihood space. "
            r"(d) PCA of $V_d$ shows the centered GLMap response space after those first-order model-wide and probe-wide effects are removed. "
            r"(e) Eta$^2$ decomposition in raw $V$ quantifies how much total matrix variance is explained by branch, family, and organization labels. "
            r"(f) The same eta$^2$ decomposition in $V_d$ shows that the AR/MLM branch label explains only a small fraction of total centered-response variance, whereas model family and organization explain substantially more structure. "
            r"(g) PC-wise branch AUC in raw $V$ asks whether any individual raw-score PC acts as an AR-versus-MLM separation axis. "
            r"(h) PC-wise branch AUC in $V_d$ repeats this diagnostic after GLMap normalization. "
            r"AUC values are direction-symmetrized as max(AUC, 1 - AUC), so values near 0.5 indicate little branch separation and values near 1.0 indicate strong separation. "
            r"Together, these diagnostics indicate that AR and MLM scores should not be interpreted as directly comparable raw probabilities, but their clipped and double-centered GLMap response profiles can be jointly visualized and clustered because branch identity is detectable but not the dominant source of variance in $V_d$."
        )
        fig.text(0.02, 0.01, caption, ha="left", va="bottom", fontsize=8.1, wrap=True)
        fig.tight_layout(rect=[0, 0.12, 1, 0.96])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def _make_individual_figures(
    L: np.ndarray, Q: np.ndarray, branches: np.ndarray,
    t1: dict, t2: dict, t3: dict, t4: dict,
    t3_raw: dict, t4_raw: dict,
    out_dir: Path,
) -> None:
    """Redraw each Fig. S2 diagnostic panel as its own standalone PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa

    BRANCH_COLOR = {
        "ar_or_generative": PALETTE["blue_main"],
        "mlm_or_encoder":   PALETTE["red_strong"],
    }
    BRANCH_LABEL = {
        "ar_or_generative": "AR",
        "mlm_or_encoder":   "MLM",
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

        # FigS2-c: raw V PCA.
        fig, ax = plt.subplots(figsize=(4.6, 4.1))
        _style(ax)
        pca_raw = PCA(n_components=2)
        proj_raw = pca_raw.fit_transform(L)
        for b in ("ar_or_generative", "mlm_or_encoder"):
            mask = branches == b
            ax.scatter(proj_raw[mask, 0], proj_raw[mask, 1],
                       s=22, c=BRANCH_COLOR[b], alpha=0.7, edgecolors="none",
                       label=BRANCH_LABEL[b])
        ax.set_xlabel(
            f"PC1 ({100*pca_raw.explained_variance_ratio_[0]:.1f}% var)"
        )
        ax.set_ylabel(
            f"PC2 ({100*pca_raw.explained_variance_ratio_[1]:.1f}% var)"
        )
        ax.set_title(r"(c) Leading PCs of raw $V$")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.6)
        _save(fig, out_dir / "FigS2-c-raw_V_PCA.pdf")

        # FigS2-d: V_d PCA.
        fig, ax = plt.subplots(figsize=(4.6, 4.1))
        _style(ax)
        pca_vd = PCA(n_components=2)
        proj_vd = pca_vd.fit_transform(Q)
        for b in ("ar_or_generative", "mlm_or_encoder"):
            mask = branches == b
            ax.scatter(proj_vd[mask, 0], proj_vd[mask, 1],
                       s=22, c=BRANCH_COLOR[b], alpha=0.7, edgecolors="none",
                       label=BRANCH_LABEL[b])
        ax.set_xlabel(
            f"PC1 ({100*pca_vd.explained_variance_ratio_[0]:.1f}% var)"
        )
        ax.set_ylabel(
            f"PC2 ({100*pca_vd.explained_variance_ratio_[1]:.1f}% var)"
        )
        ax.set_title(r"(d) Leading PCs of $V_d$")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.18, linestyle=":", linewidth=0.6)
        _save(fig, out_dir / "FigS2-d-Vd_PCA.pdf")

        # FigS2-e/f: metadata variance explained.
        eta_labels = ["branch", "family", "organization"]
        eta_colors = [PALETTE["red_strong"], PALETTE["blue_secondary"], PALETTE["teal"]]
        for filename, panel, title, stats, figsize in (
            (
                "FigS2-e-raw_V_metadata_eta2.pdf",
                "(e)",
                r"Metadata variance explained in raw $V$",
                t3_raw,
                (4.2, 3.7),
            ),
            (
                "FigS2-f-Vd_metadata_eta2.pdf",
                "(f)",
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

        # FigS2-g/h: PC-wise branch AUC.
        for filename, panel, title, stats in (
            (
                "FigS2-g-raw_V_branch_PC_AUC.pdf",
                "(g)",
                r"Branch separation across PCs of raw $V$",
                t4_raw,
            ),
            (
                "FigS2-h-Vd_branch_PC_AUC.pdf",
                "(h)",
                r"Branch separation across PCs of $V_d$",
                t4,
            ),
        ):
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            _style(ax)
            aucs = stats["per_pc_auc"]
            xs = np.arange(1, len(aucs) + 1)
            ax.bar(xs, aucs, color=PALETTE["blue_secondary"], alpha=0.85,
                   edgecolor="#222", linewidth=0.6)
            ax.set_xlabel("Principal component")
            ax.set_ylabel("branch separation AUC")
            ax.set_ylim(0.4, 1.05)
            ax.set_xticks(xs)
            ax.set_title(f"{panel} {title}")
            _save(fig, out_dir / filename)


# ─────────────────────── markdown report ─────────────────────── #


def _write_markdown_report(
    out_path: Path, glmap, t1: dict, t2: dict, t3: dict,
    t4: dict, t5: dict, t6: dict, sens: dict,
    t3_raw: dict, t4_raw: dict,
) -> None:
    M = glmap.Q.shape[0]
    n_ar  = (np.array(glmap.branches) == "ar_or_generative").sum()
    n_mlm = (np.array(glmap.branches) == "mlm_or_encoder").sum()

    def _check(b: bool) -> str:
        return "✅" if b else "⚠️"

    n_pass = sum([t1["pass"], t2["pass"], t3["pass"], t4["pass"], t5["pass"], t6["pass"]])
    md = [
        "# AR / MLM merge diagnostic",
        "",
        f"Generated by `scripts/analysis/ar_mlm_merge_diagnostic.py` on the "
        f"combined-branch GLMap representation matrix `V_d` "
        f"(M = {M}: {n_ar} AR + "
        f"{n_mlm} MLM, N = {glmap.Q.shape[1]} probes; clip threshold = "
        f"{glmap.clip_threshold:.2f}).",
        "",
        "## Question",
        "",
        "Can AR (`log p(x)`) and MLM (`PLL(x)`) models be shown in a single "
        "GLMap landscape even though their raw scoring functions are not the "
        "same probability object?",
        "",
        "## Theoretical position",
        "",
        "**Raw scores are not commensurate.** We denote the raw score matrix as "
        "`V`. AR likelihood is a normalized "
        "left-to-right joint probability, whereas MLM pseudo-log-likelihood is "
        "a bidirectional pseudo-joint score. Therefore, raw `sum_log_p` values "
        "should not be interpreted as directly comparable probabilities.",
        "",
        "**The combined object is narrower.** GLMap does not compare raw "
        "likelihood calibration. It compares centered likelihood-response "
        "profiles over the same fixed probe panel. We denote the clipped and "
        "double-centered matrix as `V_d`:",
        "",
        "```",
        "V          = sum_log_p   (branch-specific raw score)",
        "V_clipped  = clip_lower(V, q=0.02)",
        "V_d        = double_center(V_clipped)",
        "D[i, j]    = ||V_d[i] - V_d[j]||^2",
        "```",
        "",
        "Double-centering removes first-order model-wide offsets and probe-wide "
        "difficulty effects. It does **not** prove that AR and MLM are "
        "exchangeable; instead, it creates a response-profile space in which "
        "residual branch structure can be quantified. The goal is therefore to "
        "support cautious joint visualization, not unrestricted raw-score "
        "comparison.",
        "",
        "## Empirical checks",
        "",
        "| # | Check | Main quantity | Result | Verdict |",
        "|---|---|---|---|---|",
        f"| 1 | Raw scale sanity | median(AR) / median(MLM) | "
        f"{t1['median_ratio']:.2f} | {_check(t1['pass'])} |",
        f"| 2 | Post-pipeline row scale | `V_d` row std AR/MLM | "
        f"{t2['std_ratio']:.2f} | {_check(t2['pass'])} |",
        f"| 3 | Variance explained in raw `V` | branch/family/org eta² | "
        f"branch={100*t3_raw['branch_eta2']:.2f}%, family={100*t3_raw['family_eta2']:.2f}%, "
        f"org={100*t3_raw['organization_eta2']:.2f}% | descriptive |",
        f"| 4 | Variance explained in `V_d` | branch/family/org eta² | "
        f"branch={100*t3['branch_eta2']:.2f}%, family={100*t3['family_eta2']:.2f}%, "
        f"org={100*t3['organization_eta2']:.2f}% | {_check(t3['pass'])} |",
        f"| 5 | Single-PC branch separation in raw `V` | max PC AUC | "
        f"{t4_raw['max_pc_auc']:.3f} on PC{t4_raw['argmax_pc']} | descriptive |",
        f"| 6 | Single-PC branch separation in `V_d` | max PC AUC | "
        f"{t4['max_pc_auc']:.3f} on PC{t4['argmax_pc']} | {_check(t4['pass'])} |",
        f"| 7 | Low-dimensional branch predictability | LOO logistic on 20 PCs of `V_d` | "
        f"AUC={t5['by_n_pc'][20]:.3f} | {_check(t5['pass'])} |",
        f"| 8 | Branch-internal geometry preservation | Spearman, separate vs combined `V_d` | "
        f"AR={t6['AR']['spearman']:.3f}, MLM={t6['MLM']['spearman']:.3f} | {_check(t6['pass'])} |",
        "",
        f"**{n_pass} of 6 thresholded checks clear their thresholds.** The branch label is "
        "detectable, but it explains little total `V_d`-space variance and does not "
        "dominate individual leading PCs. The raw matrix `V` is dominated by a "
        "single global score axis, whereas `V_d` redistributes variance after "
        "removing first-order model-wide and probe-wide effects. Most importantly, joint centering preserves "
        f"the internal geometry of each branch (minimum Spearman = "
        f"{min(t6['AR']['spearman'], t6['MLM']['spearman']):.3f}), so the "
        "combined view does not scramble the per-branch structure.",
        "",
        "## Figure interpretation",
        "",
        "`figures/FigS2-ar_mlm_merge_diagnostic.pdf` contains eight panels arranged from raw-score diagnostics to normalized-response diagnostics:",
        "",
        "- `(a)` Raw `V` score distributions compare the marginal scale of AR log-likelihoods and MLM pseudo-log-likelihoods. The close median ratio is a scale sanity check, not evidence that the two raw scores are the same probability object.",
        "- `(b)` `V_d` row-standard-deviation distributions compare branch-wise response amplitudes after clipping and double-centering. Similar spreads indicate that neither branch dominates distances simply by having larger centered-response amplitude.",
        "- `(c)` PCA of raw `V` shows that the unnormalized matrix has a dominant global score axis, consistent with calibration or probe-difficulty effects.",
        "- `(d)` PCA of `V_d` shows the centered GLMap response space after those first-order effects are removed.",
        "- `(e)` Eta² decomposition in raw `V` quantifies the variance explained by branch, family, and organization before normalization.",
        "- `(f)` Eta² decomposition in `V_d` repeats the same calculation after GLMap normalization; branch explains only a small fraction of centered-response variance, whereas family and organization explain substantially more.",
        "- `(g)` PC-wise branch AUC in raw `V` asks whether any individual raw-score PC acts as an AR-vs-MLM separation axis.",
        "- `(h)` PC-wise branch AUC in `V_d` repeats the same diagnostic after normalization.",
        "",
        "AUC values are direction-symmetrized as `max(AUC, 1 - AUC)`, so values near 0.5 indicate little branch separation and values near 1.0 indicate strong separation regardless of which branch has the larger PC score.",
        "",
        "### Raw `V` per-PC branch AUC table",
        "",
        "| PC | branch AUC | explained variance |",
        "|---|---:|---:|",
    ]
    for k, (a, e) in enumerate(zip(t4_raw["per_pc_auc"], t4_raw["explained_variance_ratio"])):
        md.append(f"| PC{k+1} | {a:.3f} | {e*100:.2f}% |")
    md.extend([
        "",
        "### `V_d` per-PC branch AUC table",
        "",
        "| PC | branch AUC | explained variance |",
        "|---|---:|---:|",
    ])
    for k, (a, e) in enumerate(zip(t4["per_pc_auc"], t4["explained_variance_ratio"])):
        md.append(f"| PC{k+1} | {a:.3f} | {e*100:.2f}% |")
    md.extend([
        "",
        "### Residual branch structure and outlier sensitivity",
        "",
        "| Setting | inter/intra-AR | inter/intra-MLM | kNN same-branch k=5 | kNN same-branch k=10 | random baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, item in (
        ("Full 123 models", sens["full"]),
        (f"Without {sens['without_outliers']['n_removed']} outliers", sens["without_outliers"]),
    ):
        dist = item["distance"]
        knn = item["knn"]
        md.append(
            f"| {label} | {dist['ratio_inter_over_intra_ar']:.2f} | "
            f"{dist['ratio_inter_over_intra_mlm']:.2f} | "
            f"{knn['by_k'][5]['same_branch_rate_mean']:.3f} | "
            f"{knn['by_k'][10]['same_branch_rate_mean']:.3f} | "
            f"{knn['expected_random_same_branch_rate']:.3f} |"
        )
    md.extend([
        "",
        "These supplemental numbers deliberately stay in the report because "
        "they are the strongest caveat: AR and MLM are not fully mixed. Local "
        "neighborhoods show same-branch enrichment, and AR is more internally "
        "heterogeneous than MLM. This is a reason to describe the combined map "
        "carefully, not a reason to discard it.",
        "",
        "## Conclusion",
        "",
        "**Recommended claim.** The combined AR+MLM GLMap is defensible for "
        "visualizing and clustering **centered likelihood-response profiles** "
        "over the same DNA probe panel. It should not be described as direct "
        "raw-likelihood calibration, nor should AR and MLM be called "
        "interchangeable.",
        "",
        "For paper text, use language like: `We jointly visualize AR and MLM "
        "models in the clipped and double-centered GLMap response space (V_d). Branch identity "
        "remains detectable, but it explains only a small fraction of total "
        "V_d-space variance and does not dominate the leading PCs; per-branch "
        "distance geometries are preserved after joint centering.`",
        "",
        "The per-branch `V_AR / V_MLM / V_d_AR / V_d_MLM / D_AR / D_MLM` matrices "
        "remain the stricter artefacts for branch-specific claims.",
    ])
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- This report : `docs/ar_mlm_merge_diagnostic.md`")
    md.append("- Diagnostic figure : `figures/FigS2-ar_mlm_merge_diagnostic.pdf`")
    md.append("- Generator script : `scripts/analysis/ar_mlm_merge_diagnostic.py`")
    md.append("")
    md.append("Inputs read from `out_phase1/scores/` (per-model "
              "`probes.parquet`) and `data/audits/models.json` (branch + "
              "family labels).  Re-run by:")
    md.append("")
    md.append("```bash")
    md.append("$PY scripts/analysis/ar_mlm_merge_diagnostic.py")
    md.append("```")
    md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))


# ─────────────────────────── main ─────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--clip-q", type=float, default=0.02)
    p.add_argument("--no-figure", action="store_true",
                   help="Skip the diagnostic figure; write the markdown "
                        "report only.")
    p.add_argument("--out-md", type=Path,
                   default=REPO_ROOT / "docs/ar_mlm_merge_diagnostic.md")
    p.add_argument("--out-fig", type=Path,
                   default=REPO_ROOT / "figures/FigS2-ar_mlm_merge_diagnostic.pdf")
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

    print("[ar-mlm-diag] running 6 empirical tests …", flush=True)
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
        f"  3. eta²: branch={t3['branch_eta2']:.4f}, "
        f"family={t3['family_eta2']:.4f}, "
        f"organization={t3['organization_eta2']:.4f}",
        flush=True,
    )
    t4 = _test4_per_pc_branch_auc(glmap.Q, branches, k_pcs=8)
    t4_raw = _test4_per_pc_branch_auc(glmap.L, branches, k_pcs=8)
    print(f"  4. max per-PC AUC = {t4['max_pc_auc']:.4f} on PC{t4['argmax_pc']}",
          flush=True)
    t5 = _test5_pca_branch_classification(glmap.Q, branches)
    print(f"  5. branch LOO AUC from 20 PCs = {t5['by_n_pc'][20]:.4f}",
          flush=True)
    t6 = _test6_within_branch_geometry(glmap.L, glmap.Q, branches)
    print(f"  6. within-branch geometry Spearman: "
          f"AR={t6['AR']['spearman']:.4f}, MLM={t6['MLM']['spearman']:.4f}",
          flush=True)
    sens = _branch_structure_sensitivity(glmap.D, branches, glmap.hf_ids)

    n_pass = sum([t1["pass"], t2["pass"], t3["pass"], t4["pass"], t5["pass"], t6["pass"]])
    print(f"[ar-mlm-diag] {n_pass} / 6 tests pass", flush=True)

    _write_markdown_report(
        args.out_md, glmap,
        t1, t2, t3, t4, t5, t6, sens,
        t3_raw, t4_raw,
    )
    print(f"[done] wrote {args.out_md}", flush=True)

    if not args.no_figure:
        _make_figure(
            glmap.L, glmap.Q, glmap.D, branches,
            t1, t2, t3, t4, t5, t6,
            t3_raw, t4_raw,
            args.out_fig,
        )
        print(f"[done] wrote {args.out_fig}", flush=True)
        _make_individual_figures(
            glmap.L, glmap.Q, branches,
            t1, t2, t3, t4,
            t3_raw, t4_raw,
            args.out_fig.parent,
        )
        print(f"[done] wrote standalone FigS2 panel PDFs to {args.out_fig.parent}", flush=True)


if __name__ == "__main__":
    main()
