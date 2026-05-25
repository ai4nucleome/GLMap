#!/usr/bin/env python3
"""Figure 4a: SVM separability of evo lineage relationships on GLMap.

Tests whether the GLMap L-vector DIFFERENCE between two models encodes
their lineage relationship to a fixed anchor. Pair-feature framing:
each row is an ordered pair (anchor, partner) with a binary label:
  label = 1  partner is a fine-tune / direct descendant of the anchor
  label = 0  partner has no direct lineage relation to the anchor
The feature vector for each pair is the per-probe signed difference:
  feature = L_partner - L_anchor                      ∈ ℝ^N (N=10,000)
This is the parsimonious "single-anchor" feature — for the current
fig4a-svm.csv whose anchor column is constant (togethercomputer/
evo-1-8k-base), the magnitude |diff| is a deterministic function of
diff and adds no new information, so we use diff only.

The LLaMA-2 sibling script (plot_able_llama2_dist.py) uses the
concatenation [|diff|, diff] because its CSV mixes multiple anchors
and the magnitude term is anchor-invariant. Re-introduce the concat
form here if you extend fig4a-svm.csv to multi-anchor data.

Inputs
------
  models/fig4a-svm.csv   CSV of (anchor, partner, label) rows. In the
                          current panel of 8 rows the anchor is fixed
                          to togethercomputer/evo-1-8k-base.
  out_phase1/scores/<slug>/probes.parquet   per-probe sum_log_p vectors
                          (panel size 10,000) for anchor + every partner.

Pipeline
--------
  1) Load the anchor and every partner's sum_log_p; cache to avoid re-
     reading the anchor parquet eight times.
  2) Build feature matrix F of shape (M=N_rows, N=10000), each row
     being the per-probe diff = L_partner - L_anchor.
  3) StandardScale columns of F per fold, run SVM (linear default,
     rbf optional) with a C grid sweep.
  4) Leave-one-out CV (M folds) on the FULL 10000-D feature matrix;
     report AUC using each fold's decision_function value.
  5) Visualization: PCA / UMAP to 2D, scatter colored by label,
     annotate partner names, fit a 2D-only SVM for the decision
     contour (illustrative; the reported AUC is the 20000-D LOO AUC).

Output
------
  figures/Fig4a-SVM_evo-lineage_LOO-AUC[_{kernel}][_{projection}].pdf

Methodology rationale
---------------------
  - Pair-feature = diff (signed): with a single fixed anchor, the
    magnitude |diff| is a deterministic function of diff, so the
    concatenation [|diff|, diff] used in the LLaMA-2 multi-anchor
    script does not add information here. We use diff only — the
    most parsimonious encoding (10000-D, half the size).
  - LOO CV (not k-fold): with M=8 pairs, k-fold would drop too many
    samples per fold; each held-out pair is one test trial.
  - StandardScaler per-column inside the LOO loop: avoids using the
    held-out pair's stats during training fold standardization.
  - Linear kernel default: minimizes overfitting given M=8 × D=10000
    overdetermined geometry. --kernel rbf available as a sensitivity
    option; γ='scale' (sklearn default).

Usage
-----
  $PY scripts/figures/fig4a_svm_evo_lineage.py
  $PY scripts/figures/fig4a_svm_evo_lineage.py \\
      --labels-csv models/fig4a-svm.csv \\
      --out figures \\
      --figsize 7,6 \\
      --kernel linear --projection pca
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402


# ─────────────────────────── data loading ─────────────────────────── #


def _load_label_csv(csv_path: Path) -> list[tuple[str, str, int]]:
    """Load fig4a-svm.csv as a list of (anchor, partner, label) rows.

    Schema (3 columns, no header):
      anchor   hf_id of the reference model
      partner  hf_id of the model being classified
      label    0 = unrelated to anchor; 1 = direct descendant
    """
    if not csv_path.exists():
        sys.exit(f"labels csv not found: {csv_path}")
    out: list[tuple[str, str, int]] = []
    with csv_path.open() as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 3 or not row[0].strip():
                continue
            anchor, partner = row[0].strip(), row[1].strip()
            try:
                label = int(row[2].strip())
            except ValueError:
                continue
            out.append((anchor, partner, label))
    if len(out) < 4:
        sys.exit(f"need at least 4 labeled pairs in {csv_path}; got {len(out)}")
    return out


def _read_L_vector(hf_id: str, scores_dir: Path,
                   probe_order_ref: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Load sum_log_p for one model, sorted by probe_id. Verify probe
    order against ``probe_order_ref`` if given."""
    slug = hf_id.replace("/", "__")
    pq_path = scores_dir / slug / "probes.parquet"
    if not pq_path.exists():
        sys.exit(f"missing probes.parquet for {hf_id} at {pq_path}")
    t = pq.read_table(pq_path, columns=["probe_id", "sum_log_p"]).to_pandas()
    t = t.sort_values("probe_id").reset_index(drop=True)
    order = t["probe_id"].tolist()
    if probe_order_ref is not None:
        assert order == probe_order_ref, f"probe_id order mismatch in {hf_id}"
    vec = t["sum_log_p"].to_numpy()
    n_nan = int(np.isnan(vec).sum())
    if n_nan > 0:
        sys.exit(f"{hf_id}: probes.parquet has {n_nan} NaN sum_log_p values")
    return vec, order


def _load_pair_features(
    rows: list[tuple[str, str, int]],
    scores_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """For each (anchor, partner, label) row, compute the pair feature
    ``diff = L_partner - L_anchor`` and stack into matrix F of shape
    (M=N_rows, N=10000).

    Returns (F, y, partner_names, common_anchor). All rows must share
    the same anchor; under that constraint, |diff| is a deterministic
    function of diff and adds no information — so we use diff only
    (vs. the [|diff|, diff] concat used in the LLaMA-2 sibling script
    where multiple anchors made |diff| anchor-invariant).
    """
    cache: dict[str, np.ndarray] = {}
    probe_order: list[str] | None = None

    def get_L(hf_id: str) -> np.ndarray:
        nonlocal probe_order
        if hf_id not in cache:
            vec, order = _read_L_vector(hf_id, scores_dir, probe_order)
            if probe_order is None:
                probe_order = order
            cache[hf_id] = vec
        return cache[hf_id]

    features: list[np.ndarray] = []
    labels: list[int] = []
    partners: list[str] = []
    anchors_seen: set[str] = set()

    for anchor, partner, label in rows:
        L_a = get_L(anchor)
        L_p = get_L(partner)
        # Signed per-probe difference (10000-D). Direction is informative:
        #   diff > 0 → partner gives higher likelihood than anchor on that probe
        #   diff < 0 → partner gives lower likelihood
        features.append(L_p - L_a)
        labels.append(label)
        partners.append(partner)
        anchors_seen.add(anchor)

    if len(anchors_seen) != 1:
        sys.exit(
            "fig4a-svm.csv must use a single shared anchor in column 1; "
            f"found {len(anchors_seen)} distinct anchors: {sorted(anchors_seen)}"
        )

    F = np.stack(features, axis=0)
    y = np.array(labels, dtype=int)
    anchor = next(iter(anchors_seen))
    return F, y, partners, anchor


# ───────────────────────── SVM (full-dim LOO) ─────────────────────── #


def _build_svm(kernel: str, C: float, gamma: str | float, seed: int):
    """Construct an SVM classifier for the requested kernel.

    - ``linear``: ``LinearSVC`` (squared_hinge, dual auto). The most
      stable choice for N=8 × D=10000 overdetermined geometry.
    - ``rbf``: ``SVC(kernel='rbf', gamma=...)``. ``gamma='scale'`` is
      the sklearn data-driven default (1 / (D × var(X))).
    """
    from sklearn.svm import LinearSVC, SVC
    if kernel == "linear":
        return LinearSVC(
            C=C, max_iter=20000, random_state=seed,
            dual="auto", loss="squared_hinge",
        )
    if kernel == "rbf":
        return SVC(
            kernel="rbf", C=C, gamma=gamma,
            random_state=seed,
        )
    raise ValueError(f"unsupported kernel={kernel!r}; expected 'linear' or 'rbf'")


def _loo_auc(
    L: np.ndarray, y: np.ndarray, c_grid: list[float],
    kernel: str = "linear", gamma: str | float = "scale",
    seed: int = 42,
) -> tuple[float, float, dict]:
    """Leave-one-out CV SVM; for each C, compute mean AUC across folds
    and the per-fold decision_function values; return best C and AUC.

    Per-fold pipeline: StandardScaler (fit on the 7 training folds) +
    SVM (linear or rbf). Returns (best_C, best_AUC, info_dict).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut

    info = {}
    best_C, best_auc = None, -np.inf
    for C in c_grid:
        decisions = np.full_like(y, fill_value=np.nan, dtype=float)
        loo = LeaveOneOut()
        for tr_idx, te_idx in loo.split(L):
            scaler = StandardScaler().fit(L[tr_idx])
            X_tr = scaler.transform(L[tr_idx])
            X_te = scaler.transform(L[te_idx])
            clf = _build_svm(kernel=kernel, C=C, gamma=gamma, seed=seed)
            clf.fit(X_tr, y[tr_idx])
            # decision_function: signed distance to margin (linear) or
            # to learned RBF surface; AUC works directly on this score.
            decisions[te_idx] = clf.decision_function(X_te)
        auc = float(roc_auc_score(y, decisions))
        info[C] = {"auc": auc, "decisions": decisions.copy()}
        if auc > best_auc:
            best_C, best_auc = C, auc
    return best_C, best_auc, info


# ───────────────────── 2D PCA + SVM contour ───────────────────────── #


def _pca_2d(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize columns then take top-2 PCs. Returns (embedding, var_ratio)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(L)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(Xs)
    return emb, pca.explained_variance_ratio_


def _umap_2d(
    L: np.ndarray, n_neighbors: int, min_dist: float, seed: int = 42,
) -> np.ndarray:
    """Standardize columns then UMAP to 2D.

    UMAP is nonlinear; the relative position of two points in 2D does
    NOT have a closed-form interpretation in the full L space. Style
    matches scripts/figures/panel_composition_figure.py (Fig 3) so
    Fig 4(a) and Fig 3 share a visual vocabulary.

    With N=8 samples, ``n_neighbors`` must be < N (UMAP requirement).
    Default 4 in the CLI; a value of 2-5 is the workable range here.
    """
    import umap
    from sklearn.preprocessing import StandardScaler
    if n_neighbors >= L.shape[0]:
        raise ValueError(
            f"UMAP n_neighbors {n_neighbors} must be < N_samples {L.shape[0]}"
        )
    Xs = StandardScaler().fit_transform(L)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(Xs)


def _fit_2d_svm(emb: np.ndarray, y: np.ndarray, C: float,
                kernel: str = "linear", gamma: str | float = "scale",
                seed: int = 42):
    """Fit an SVM in 2D PCA space for visualization only (linear or rbf)."""
    clf = _build_svm(kernel=kernel, C=C, gamma=gamma, seed=seed)
    clf.fit(emb, y)
    return clf


# ────────────────────────────── plot ──────────────────────────────── #

LABEL_NAMES = {0: "Independent (unrelated)", 1: "Correlated (derived)"}
LABEL_COLORS = {0: PALETTE["red_strong"], 1: PALETTE["blue_main"]}


def _short_name(hf_id: str) -> str:
    """Compact display name for annotation, e.g.
    arcinstitute/evo2_7b_base -> evo2_7b_base
    LongSafari/evo-1-8k-crispr -> evo-1-8k-crispr
    """
    return hf_id.split("/", 1)[-1]


def _fan_out_label_offsets(
    emb: np.ndarray, y: np.ndarray,
    radius_px: float = 14.0,
) -> dict[int, tuple[float, float]]:
    """Compute per-point label offsets (in points / 1/72 inch) by
    spreading labels in a fan around each lineage cluster's centroid.

    Why: 3 arcinstitute/evo2_7b variants and 2 LongSafari/evo-1-8k
    models cluster so tightly in PC1/PC2 space that text annotations
    overlap each other if all use the same (5, 5) offset. We honor
    the true coordinates but rotate each label around the cluster
    centroid so they fan outward and leader lines clarify which dot
    each label belongs to.

    Returns ``{i: (dx_pt, dy_pt)}`` for the i-th point's label offset.
    """
    offsets: dict[int, tuple[float, float]] = {}
    for label in np.unique(y):
        idxs = np.where(y == label)[0].tolist()
        if not idxs:
            continue
        centroid = emb[idxs].mean(axis=0)
        # Sort by polar angle around centroid so fan-out order is stable
        # (deterministic w.r.t. point geometry, not csv row order).
        def _angle(i: int) -> float:
            v = emb[i] - centroid
            return float(np.arctan2(v[1], v[0]))
        sorted_idxs = sorted(idxs, key=_angle)
        n = len(sorted_idxs)
        # Fan: cover a full 2π if n>=4; otherwise spread over a half-circle.
        arc = 2 * np.pi if n >= 4 else np.pi
        start = -arc / 2 if n >= 4 else 0.0
        for k, i in enumerate(sorted_idxs):
            theta = start + (k + 0.5) * (arc / n)
            offsets[i] = (radius_px * np.cos(theta), radius_px * np.sin(theta))
    return offsets


def _make_figure(
    emb: np.ndarray, y: np.ndarray, hf_ids: list[str],
    var_ratio: np.ndarray | None, loo_auc: float, best_C: float,
    kernel: str, gamma: str | float,
    projection: str,
    figsize: tuple[float, float],
    out_path: Path,
    anchor: str | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
) -> None:
    with plt.rc_context(RCPARAMS):
        fig, ax = plt.subplots(figsize=figsize)

        # 2D-space SVM only for the contour. Reported AUC comes from
        # full-dim LOO; mention this in the title to avoid confusion.
        clf2 = _fit_2d_svm(emb, y, C=best_C, kernel=kernel, gamma=gamma, seed=42)

        pad = 0.20
        x_lo, x_hi = emb[:, 0].min(), emb[:, 0].max()
        y_lo, y_hi = emb[:, 1].min(), emb[:, 1].max()
        x_pad = (x_hi - x_lo) * pad
        y_pad = (y_hi - y_lo) * pad
        xx, yy = np.meshgrid(
            np.linspace(x_lo - x_pad, x_hi + x_pad, 400),
            np.linspace(y_lo - y_pad, y_hi + y_pad, 400),
        )
        zz = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Shaded half-planes for the two predicted classes — only if the
        # decision boundary (z=0) actually crosses the visible region.
        # In nonlinear projections (e.g. UMAP) the 2D-trained SVM may
        # place all of the visible plane on one side of its margin,
        # so [zmin, 0, zmax] would not be monotonically increasing and
        # contourf would raise. Skip the fill in that case but still
        # draw the boundary contour at z=0 (matplotlib's contour
        # handles "no levels in range" gracefully).
        zmin, zmax = float(zz.min()), float(zz.max())
        if zmin < 0 < zmax:
            ax.contourf(xx, yy, zz, levels=[zmin, 0, zmax],
                        colors=[LABEL_COLORS[0], LABEL_COLORS[1]],
                        alpha=0.10, antialiased=True)
        # Margin contour at decision boundary.
        if zmin < 0 < zmax:
            ax.contour(xx, yy, zz, levels=[0], colors=["#333"],
                       linewidths=1.4, linestyles="--")

        # Scatter the 8 models.
        for label in (0, 1):
            m = y == label
            ax.scatter(
                emb[m, 0], emb[m, 1],
                s=110, c=LABEL_COLORS[label],
                edgecolors="white", linewidths=1.2,
                label=LABEL_NAMES[label], zorder=3,
            )

        # Annotate each dot with the short model name. Labels are
        # fan-positioned in 8 different directions around each lineage
        # cluster centroid (no leader lines) — small radius keeps the
        # text right next to its point while avoiding label overlap.
        label_offsets = _fan_out_label_offsets(emb, y)
        for i, ((x, yval), hf) in enumerate(zip(emb, hf_ids)):
            dx, dy = label_offsets[i]
            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"
            ax.annotate(
                _short_name(hf),
                xy=(x, yval), xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8, color="#222",
                ha=ha, va=va,
            )

        if projection == "pca":
            assert var_ratio is not None
            ax.set_xlabel(f"PC 1 ({var_ratio[0]*100:.1f}%)")
            ax.set_ylabel(f"PC 2 ({var_ratio[1]*100:.1f}%)")
            projection_tag = "2D PCA SVM"
        else:  # umap
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            if umap_n_neighbors is not None and umap_min_dist is not None:
                projection_tag = (
                    f"2D UMAP SVM (n_neighbors={umap_n_neighbors}, "
                    f"min_dist={umap_min_dist:g})"
                )
            else:
                projection_tag = "2D UMAP SVM"
        kernel_tag = (
            f"{kernel} SVM"
            + (f", γ={gamma}" if kernel == "rbf" else "")
        )
        anchor_short = (
            anchor.split("/", 1)[-1] if anchor and "/" in anchor else (anchor or "")
        )
        anchor_line = (
            f"anchor: {anchor_short}\n"
            if anchor_short else ""
        )
        ax.set_title(
            "SVM separability of evo lineage pairs on GLMap\n"
            + anchor_line
            + f"({kernel_tag}, LOO-CV AUC = {loo_auc:.3f}, best C = {best_C:g}; "
            + f"contour from {projection_tag}, illustrative only)"
        )
        ax.legend(
            loc="best", fontsize=9, title="Lineage label",
            title_fontsize=9, frameon=True, framealpha=0.9,
        )
        ax.grid(True, alpha=0.25, linestyle=":")

        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)


# ─────────────────────────────── main ─────────────────────────────── #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--labels-csv", type=Path,
                   default=REPO_ROOT / "models/fig4a-svm.csv",
                   help="CSV file with two columns: hf_id, lineage_label (0/1).")
    p.add_argument("--scores-dir", type=Path,
                   default=REPO_ROOT / "out_phase1/scores",
                   help="Directory containing <slug>/probes.parquet.")
    p.add_argument("--out", dest="out_dir", type=Path,
                   default=REPO_ROOT / "figures",
                   help="Output directory.")
    p.add_argument("--figsize", type=str, default="7,6",
                   help='Inches, "W,H". Default "7,6".')
    p.add_argument("--c-grid", type=str, default="0.01,0.1,1,10",
                   help="Comma-separated SVM C grid for LOO sweep.")
    p.add_argument("--kernel", type=str, default="linear",
                   choices=["linear", "rbf"],
                   help="SVM kernel. 'linear' (default, LinearSVC: most "
                   "stable for N=8 × D=10000 overdetermined geometry); "
                   "'rbf' (SVC(kernel='rbf')) draws a nonlinear "
                   "decision surface and is more sensitive to γ.")
    p.add_argument("--gamma", type=str, default="scale",
                   help='RBF γ. Either "scale" (default; 1/(D·var(X))), '
                   '"auto" (1/D), or a positive float (e.g. 0.001). '
                   'Ignored when --kernel=linear.')
    p.add_argument("--projection", type=str, default="pca",
                   choices=["pca", "umap"],
                   help="2D projection for the scatter + decision-"
                   "contour visualization. 'pca' (default): linear, "
                   "explained-variance interpretable; for the linear "
                   "kernel, the 2D contour directly projects the full-D "
                   "SVM hyperplane. 'umap': nonlinear, matches Fig 3 "
                   "style; the 2D contour is strictly illustrative "
                   "(no closed-form mapping back to full L space).")
    p.add_argument("--umap-n-neighbors", type=int, default=4,
                   dest="umap_n_neighbors",
                   help="UMAP n_neighbors. Must be < N_samples (=8 "
                   "here), so 2-5 is the workable range. Default 4.")
    p.add_argument("--umap-min-dist", type=float, default=0.5,
                   dest="umap_min_dist",
                   help="UMAP min_dist (default 0.5).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_gamma(s: str) -> str | float:
    """Pass 'scale' / 'auto' through; otherwise interpret as float."""
    if s in ("scale", "auto"):
        return s
    try:
        return float(s)
    except ValueError as e:
        raise ValueError(f"--gamma must be 'scale', 'auto', or a float; got {s!r}") from e


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    w, h = (float(x) for x in s.split(sep))
    return (w, h)


def main() -> None:
    args = parse_args()

    rows = _load_label_csv(args.labels_csv)
    print(f"[fig4a] {len(rows)} labeled pairs from "
          f"{args.labels_csv.relative_to(REPO_ROOT)}", flush=True)

    F, y, partners, anchor = _load_pair_features(rows, args.scores_dir)
    # Alias so the downstream code that uses L / hf_ids still works.
    L, hf_ids = F, partners
    print(f"[fig4a] anchor: {anchor}", flush=True)
    print(f"[fig4a] pair-feature matrix: {F.shape}  (label balance: "
          f"{int((y==0).sum())} unrelated / {int((y==1).sum())} derived)",
          flush=True)

    c_grid = [float(c) for c in args.c_grid.split(",")]
    gamma = _parse_gamma(args.gamma)
    best_C, loo_auc, info = _loo_auc(
        L, y, c_grid,
        kernel=args.kernel, gamma=gamma, seed=args.seed,
    )
    print(f"[fig4a] kernel={args.kernel}"
          + (f", γ={gamma}" if args.kernel == "rbf" else ""),
          flush=True)
    print(f"[fig4a] LOO-CV AUC sweep:",
          ", ".join(f"C={c:g}: {info[c]['auc']:.3f}" for c in c_grid),
          flush=True)
    print(f"[fig4a] best LOO AUC = {loo_auc:.3f}  at C = {best_C:g}", flush=True)

    if args.projection == "pca":
        emb, var_ratio = _pca_2d(L)
        print(f"[fig4a] PCA explained variance: PC1={var_ratio[0]*100:.1f}%, "
              f"PC2={var_ratio[1]*100:.1f}% (cumulative {sum(var_ratio[:2])*100:.1f}%)",
              flush=True)
    else:  # umap
        emb = _umap_2d(L, n_neighbors=args.umap_n_neighbors,
                       min_dist=args.umap_min_dist, seed=args.seed)
        var_ratio = None
        print(f"[fig4a] UMAP n_neighbors={args.umap_n_neighbors}, "
              f"min_dist={args.umap_min_dist:g}, seed={args.seed} "
              f"(layout is stochastic — fixed by --seed)", flush=True)

    # Filename encodes kernel + projection so different runs don't
    # overwrite each other; linear PCA stays untagged for backward
    # compatibility.
    parts = ["Fig4a-SVM_evo-lineage_LOO-AUC"]
    if args.kernel != "linear":
        parts.append(args.kernel)
    if args.projection != "pca":
        parts.append(args.projection)
    out_path = args.out_dir / ("_".join(parts) + ".pdf")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _make_figure(
        emb, y, hf_ids, var_ratio,
        loo_auc=loo_auc, best_C=best_C,
        kernel=args.kernel, gamma=gamma,
        projection=args.projection,
        figsize=_parse_figsize(args.figsize),
        out_path=out_path,
        anchor=anchor,
        umap_n_neighbors=(args.umap_n_neighbors if args.projection == "umap" else None),
        umap_min_dist=(args.umap_min_dist if args.projection == "umap" else None),
    )
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
