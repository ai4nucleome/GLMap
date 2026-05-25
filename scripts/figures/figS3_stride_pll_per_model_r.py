#!/usr/bin/env python3
"""Figure S3: per-model Pearson r between true k=1 PLL and stride k=6 PLL.

We canonical-score MLM models with stride pseudo-log-likelihood at
k=6 (one forward pass per stride offset), and want to demonstrate that
this is a robust approximation of the much more expensive true k=1 PLL
(L individual leave-one-out forward passes per sequence).  For each
MLM model that has both score variants on the same 1000-probe subset,
we compute a single Pearson r between its per-probe ``sum_log_p`` at
k=1 vs k=6.  Fig S3 visualizes the resulting distribution of r values
across the 51 models that completed the k=1 sweep.

Inputs
------
  out_phase1/MLM_k1ablation_1000_scores/scores/<slug>/probes.parquet
    51 / 56 MLM models successfully scored at stride=1 (true PLL) on
    the 1000-probe ablation subset.  Five PlantCAD2 / GENERanno /
    AIDO.DNA-7B checkpoints did not complete — they are excluded.

  out_phase1/scores/<slug>/probes.parquet
    Canonical k=6 stride PLL scores on the full 10,000-probe panel;
    we filter down to the 1000-probe ablation subset by probe_id.

  data/audits/models.json
    Branch / family metadata for per-axis annotation.

Outputs
-------
  figures/FigS3-stride_pll_per_model_r.pdf
    Two-panel figure:
      (a) Per-model Pearson r distribution — violin + jittered strip
          plot of the 51 r values, colored by family (top families
          named; long tail pooled into "Other").
      (b) Pooled scatter — every (model, probe) pair from the 51-model
          × 1000-probe ablation cross plotted as k=6 sum_log_p (x)
          vs k=1 sum_log_p (y), with overall pooled Pearson r in the
          annotation.

  out_phase1/figS3_per_model_r.json
    JSON record of the per-model r values + summary stats, so paper
    prose / supplementary tables can reference the exact numbers.

Usage
-----
  $PY scripts/figures/figS3_stride_pll_per_model_r.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.phase1_main_figure import PALETTE, RCPARAMS  # noqa: E402


K1_DIR = REPO_ROOT / "out_phase1/MLM_k1ablation_1000_scores/scores"
K6_DIR = REPO_ROOT / "out_phase1/scores"
PANEL_SUBSET = REPO_ROOT / "out_panel/MLM_k1ablation_1000_main_panel.parquet"
AUDIT_JSON = REPO_ROOT / "data/audits/models.json"

OTHER_COLOR = "#B8B8B8"


def _family_palette(families: list[str], min_count: int = 3) -> dict[str, str]:
    counts = Counter(families)
    ordered = [g for g, n in counts.most_common() if n >= min_count]
    palette = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )
    colors = {g: palette[i % len(palette)] for i, g in enumerate(ordered)}
    for g, n in counts.items():
        if n < min_count:
            colors[g] = OTHER_COLOR
    return colors


def _load_paired_scores(slug: str, probe_subset: list[str]) -> pd.DataFrame:
    """Inner-join the k=1 and k=6 parquets on probe_id, restricted to
    the 1000-probe ablation subset.  Returns a DataFrame with columns
    probe_id / sum_log_p_k1 / sum_log_p_k6.
    """
    k1_path = K1_DIR / slug / "probes.parquet"
    k6_path = K6_DIR / slug / "probes.parquet"
    if not k1_path.exists() or not k6_path.exists():
        return pd.DataFrame()
    k1 = pq.read_table(k1_path, columns=["probe_id", "sum_log_p"]).to_pandas()
    k1 = k1.rename(columns={"sum_log_p": "sum_log_p_k1"})
    k6 = pq.read_table(k6_path, columns=["probe_id", "sum_log_p"]).to_pandas()
    k6 = k6.rename(columns={"sum_log_p": "sum_log_p_k6"})
    paired = k1.merge(k6, on="probe_id", how="inner")
    paired = paired[paired["probe_id"].isin(probe_subset)].reset_index(drop=True)
    paired = paired.dropna(subset=["sum_log_p_k1", "sum_log_p_k6"])
    return paired


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-fig", type=Path,
                   default=REPO_ROOT / "figures/FigS3-stride_pll_per_model_r.pdf")
    p.add_argument("--out-json", type=Path,
                   default=REPO_ROOT / "out_phase1/figS3_per_model_r.json")
    p.add_argument("--figsize", type=str, default="13,5.8")
    return p.parse_args()


def _parse_figsize(s: str) -> tuple[float, float]:
    sep = "," if "," in s else "x"
    return tuple(float(x) for x in s.split(sep))


def main() -> None:
    args = parse_args()

    # ── load the 1000-probe ablation subset spec ── #
    panel_df = pq.read_table(
        PANEL_SUBSET, columns=["probe_id"]
    ).to_pandas()
    probe_subset = panel_df["probe_id"].tolist()
    print(f"[figS3] ablation subset has {len(probe_subset)} probes")

    # ── audit metadata ── #
    audit = json.loads(AUDIT_JSON.read_text())["models"]
    audit_by_id = {m["hf_id"]: m for m in audit}

    # ── find the models that ran k=1 successfully ── #
    completed = sorted(
        d.name for d in K1_DIR.iterdir()
        if d.is_dir() and (d / "probes.parquet").exists()
    )
    print(f"[figS3] {len(completed)} models with k=1 parquet on disk")

    # ── compute per-model Pearson r + collect pooled data ── #
    rows = []
    pooled_k1: list[float] = []
    pooled_k6: list[float] = []
    pooled_family: list[str] = []
    for slug in completed:
        hf_id = slug.replace("__", "/")
        if hf_id not in audit_by_id:
            print(f"[figS3] warning: {hf_id} not in audit, skipping")
            continue
        meta = audit_by_id[hf_id]
        if meta.get("branch") != "mlm_or_encoder":
            continue
        paired = _load_paired_scores(slug, probe_subset)
        if len(paired) < 100:
            print(f"[figS3] {hf_id}: only {len(paired)} paired probes, skipping")
            continue
        v1 = paired["sum_log_p_k1"].to_numpy()
        v6 = paired["sum_log_p_k6"].to_numpy()
        r_p = float(pearsonr(v6, v1).statistic)
        r_s = float(spearmanr(v6, v1).statistic)
        rows.append({
            "hf_id": hf_id,
            "slug": slug,
            "family": meta.get("family", "unknown"),
            "organization": meta.get("organization") or "(unknown)",
            "n_paired": int(len(paired)),
            "pearson_r": r_p,
            "spearman_r": r_s,
            "k1_mean": float(v1.mean()),
            "k6_mean": float(v6.mean()),
        })
        pooled_k1.extend(v1.tolist())
        pooled_k6.extend(v6.tolist())
        pooled_family.extend([meta.get("family", "unknown")] * len(v1))

    if not rows:
        sys.exit("[figS3] no models with paired k=1 / k=6 scores; aborting")

    df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    pooled_r = float(pearsonr(pooled_k6, pooled_k1).statistic)
    pooled_rho = float(spearmanr(pooled_k6, pooled_k1).statistic)

    # ── summary stats ── #
    r_vals = df["pearson_r"].to_numpy()
    summary = {
        "n_models": int(len(df)),
        "n_paired_per_model": int(df["n_paired"].iloc[0]),
        "pearson_r": {
            "mean":   float(r_vals.mean()),
            "median": float(np.median(r_vals)),
            "std":    float(r_vals.std(ddof=1)),
            "min":    float(r_vals.min()),
            "max":    float(r_vals.max()),
            "q05":    float(np.quantile(r_vals, 0.05)),
            "q95":    float(np.quantile(r_vals, 0.95)),
        },
        "pooled_pearson_r":  pooled_r,
        "pooled_spearman_r": pooled_rho,
    }
    print()
    print(f"[figS3] per-model Pearson r summary:")
    for k, v in summary["pearson_r"].items():
        print(f"    {k:>7s} = {v:.4f}")
    print(f"[figS3] pooled Pearson  r = {pooled_r:.4f}")
    print(f"[figS3] pooled Spearman ρ = {pooled_rho:.4f}")
    print()
    print(f"[figS3] lowest-r models:")
    for _, r in df.tail(5).iterrows():
        print(f"    {r['pearson_r']:.4f}  {r['hf_id']:50s}  family={r['family']}")

    # Persist
    out_payload = {
        "summary": summary,
        "per_model": df.to_dict(orient="records"),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_payload, indent=2))
    print(f"[done] wrote {args.out_json.relative_to(REPO_ROOT)}")

    # ── plot ── #
    figsize = _parse_figsize(args.figsize)
    local_rc = {
        **RCPARAMS,
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.4,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
    }
    fam_palette = _family_palette(df["family"].tolist())

    with plt.rc_context(local_rc):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize,
                                          gridspec_kw=dict(width_ratios=[1.0, 1.05]))

        # ── Panel (a): per-model r distribution ── #
        # Violin (background)
        parts = ax_a.violinplot(
            r_vals, positions=[0.0], widths=0.55,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("#D8D8D8")
            body.set_edgecolor("#888")
            body.set_alpha(0.55)
            body.set_linewidth(0.6)

        # Strip plot with family colors
        rng = np.random.default_rng(42)
        for _, r in df.iterrows():
            x_jitter = rng.uniform(-0.18, 0.18)
            color = fam_palette[r["family"]]
            ax_a.scatter(x_jitter, r["pearson_r"],
                         s=46, c=[color], alpha=0.88,
                         edgecolors="white", linewidths=0.6,
                         zorder=3)

        # Median + mean horizontal bars
        med = float(np.median(r_vals))
        mean = float(r_vals.mean())
        ax_a.plot([-0.30, 0.30], [med, med], color="black",
                  linewidth=1.8, zorder=4)
        ax_a.plot([-0.30, 0.30], [mean, mean], color=PALETTE["red_strong"],
                  linewidth=1.4, linestyle="--", zorder=4)

        # Annotation block on the right side
        annot = (
            f"$N$ = {summary['n_models']} models\n"
            f"median = {med:.3f}\n"
            f"mean = {mean:.3f}\n"
            f"min / max = {r_vals.min():.3f} / {r_vals.max():.3f}"
        )
        ax_a.text(0.5, np.median(r_vals), annot,
                  fontsize=10, va="center", ha="left", color="#333")

        ax_a.set_xticks([0.0])
        ax_a.set_xticklabels(["51 MLM models"])
        ax_a.set_xlim(-0.55, 1.05)
        # Symmetric padding so the violin sits visually centered
        y_lo = max(0.0, float(r_vals.min()) - 0.05)
        y_hi = min(1.005, float(r_vals.max()) + 0.02)
        ax_a.set_ylim(y_lo, y_hi)
        ax_a.set_ylabel("Per-model Pearson $r$  (true $k=1$ PLL vs stride $k=6$ PLL)")
        ax_a.set_title("Per-model $k=1$ vs $k=6$ PLL correlation",
                       loc="center", pad=8)
        ax_a.grid(True, axis="y", alpha=0.18, linestyle=":", linewidth=0.6)
        ax_a.set_axisbelow(True)
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.spines["left"].set_visible(True)
        ax_a.spines["bottom"].set_visible(True)
        ax_a.tick_params(axis="y", direction="out", length=4, width=1.0)
        ax_a.tick_params(axis="x", direction="out", length=3, width=1.0, pad=4)

        # Family legend (top families only)
        counts = Counter(df["family"])
        top_fams = [g for g, n in counts.most_common() if n >= 3]
        n_other = sum(n for g, n in counts.items() if n < 3)
        legend_handles = [
            Patch(facecolor=fam_palette[g], edgecolor="white",
                  linewidth=0.4, label=g)
            for g in top_fams
        ]
        if n_other:
            legend_handles.append(Patch(facecolor=OTHER_COLOR,
                                        label="Other"))
        legend_handles.append(Line2D([0], [0], color="black", linewidth=1.8,
                                     label="median"))
        legend_handles.append(Line2D([0], [0], color=PALETTE["red_strong"],
                                     linewidth=1.4, linestyle="--",
                                     label="mean"))
        ax_a.legend(handles=legend_handles,
                    loc="lower left", frameon=False, fontsize=8.5,
                    handlelength=1.2, labelspacing=0.22,
                    handletextpad=0.4)

        # ── Panel (b): pooled (model, probe) scatter ── #
        pooled_k1_arr = np.array(pooled_k1)
        pooled_k6_arr = np.array(pooled_k6)
        # Down-sample for visual clarity if pool > 30k points
        n_pool = len(pooled_k1_arr)
        if n_pool > 30000:
            idx = rng.choice(n_pool, size=30000, replace=False)
            x_show = pooled_k6_arr[idx]
            y_show = pooled_k1_arr[idx]
            n_show = 30000
        else:
            x_show, y_show, n_show = pooled_k6_arr, pooled_k1_arr, n_pool

        ax_b.scatter(x_show, y_show, s=2.0,
                     c=PALETTE["blue_main"], alpha=0.18,
                     edgecolors="none", rasterized=True)
        # y = x reference
        lo = float(min(pooled_k6_arr.min(), pooled_k1_arr.min()))
        hi = float(max(pooled_k6_arr.max(), pooled_k1_arr.max()))
        pad = 0.02 * (hi - lo)
        ax_b.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                  color=PALETTE["red_strong"], linestyle="--",
                  linewidth=1.2, alpha=0.8, label=r"$y = x$")
        # OLS best fit
        slope, intercept = np.polyfit(pooled_k6_arr, pooled_k1_arr, deg=1)
        ax_b.plot([lo - pad, hi + pad],
                  [slope * (lo - pad) + intercept, slope * (hi + pad) + intercept],
                  color="black", linewidth=1.0, alpha=0.65,
                  label=f"best fit (slope = {slope:.3f})")

        # Annotation
        annot_b = (
            f"$N$ = {n_pool:,} (model, probe) pairs\n"
            f"Pearson $r$ = {pooled_r:.4f}\n"
            f"Spearman $\\rho$ = {pooled_rho:.4f}"
        )
        ax_b.text(0.04, 0.96, annot_b,
                  transform=ax_b.transAxes, fontsize=10,
                  va="top", ha="left",
                  bbox=dict(boxstyle="round,pad=0.35",
                            facecolor="white", edgecolor="#CCC",
                            linewidth=0.6, alpha=0.95))

        ax_b.set_xlim(lo - pad, hi + pad)
        ax_b.set_ylim(lo - pad, hi + pad)
        ax_b.set_xlabel("Stride $k=6$ PLL score  ($\\sum \\log p$)")
        ax_b.set_ylabel("True $k=1$ PLL score  ($\\sum \\log p$)")
        ax_b.set_title("Pooled $k=1$ vs $k=6$ PLL scores",
                       loc="center", pad=8)
        ax_b.grid(True, alpha=0.18, linestyle=":", linewidth=0.6)
        ax_b.set_axisbelow(True)
        ax_b.legend(loc="lower right", frameon=False, fontsize=10)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        fig.tight_layout()

        args.out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_fig, dpi=300, bbox_inches="tight",
                    pad_inches=0.1)
        plt.close(fig)
        print(f"[done] wrote {args.out_fig.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
