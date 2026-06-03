#!/usr/bin/env python3
"""Phase 1 scoring: AR + MLM models x full main panel -> L → clip → Q → D (per branch).

Implements phase_1.md § 打分协议 + § Sequence-likelihood matrix on
data/panels/main_panel.parquet. Produces the canonical layout:

    results/scores/
      AR_MLM_scores/{model_id_slug}/
        {model_id_slug}.json              # per-model metadata
        probes.parquet
        # one row per panel probe, aligned by probe_id; columns include
        # sum_log_p, ell_per_base, bpb, token_length, scoring_error,
        # plus token_log_probs (per-token log p list; AR = T-1 floats,
        # MLM = content_position_count floats).
      matrices/
        V_AR.npy                          # (M_AR, N) raw sum_log_p, floor-clipped at the
                                            #   2nd-percentile (ModelMap)
        V_d_AR.npy                        # double-centered (row mean + col mean removed)
        D_AR.npy                          # (M_AR, M_AR) pairwise squared Euclidean on V_d
                                            #   approximates KL under small-divergence Taylor
        V_MLM.npy / V_d_MLM.npy / D_MLM.npy
        matrix_metadata.json              # ordered model_ids + probe_ids
                                          # + single-matrix protocol description

Convention per phase_1.md § 单矩阵协议 (ModelMap, raw nats, no length norm, no sign flip):
    sum_log_p_m(x)  = sum_t log p(x_t | x_<t)  (AR)  /  stride PLL k=6  (MLM)
    V[m, x]         = sum_log_p_m(x)           # negative (log p < 0), enters matrix
    V_clipped       = floor_clip(V, q=0.02)    # ModelMap convention
    V_d             = double_center(V_clipped) # main matrix for PCA / F_ST
    D               = pairwise_squared_distance(V_d)
ell_per_base = sum_log_p / base_length and bpb are written to probes.parquet
as cross-tokenizer-readable reports but are NOT used to build V / V_d / D.

Codon-model handling (commit 5e59154 retired the three-matrix split):
    Codon-tokenized models (GenSLM, Codon-NT) emit raw likelihood on every
    probe and enter the single L matrix alongside nucleotide-tokenized
    models. ModelMap's clip + double-center (row mean removes each model's
    overall level, column mean removes each probe's overall difficulty)
    absorbs the codon-vs-nucleotide systematic offset on noncoding probes.
    The `is_codon` flag is still emitted to model metadata for downstream
    diagnostic loadings but no longer gates matrix membership.

Resume:
    Each per-model probes.parquet acts as a checkpoint. Rerunning without
    --force will reuse existing files. The matrices step always re-aggregates
    from the parquets (cheap) so adding a new model + rerunning is enough.

Usage:
    python scripts/score/scoring_worker.py \\
        [--panel data/panels/main_panel.parquet] \\
        [--out results/scores] \\
        [--device cuda:6] \\
        [--stride 6] \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------- Model spec table ----------------------------- #


from glmap.loaders.dispatch import ModelSpec  # noqa: E402  (canonical, kw_only)
from glmap.pipeline import (  # scoring + aggregation building blocks
    BranchMatrices,
    _build_branch_matrices,
    _score_model,
    parquet_covers_panel,
)


# Phase 1 pilot model roster. Current count: 8 AR + 5 MLM = 13.
# GenSLM loaders (25M/250M/2.5B) are now wired in (see src/loaders/genslm.py);
# HyenaDNA and DNABERT 3..6 are still excluded from this DEFAULT — the
# former because its loader uses a separate dispatch path that the
# multi-env sweep handles rather than this script, the latter because
# of single-token overlap-mask leakage
# (phase_1.md supplement scope).
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    # AR branch
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-1M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-17M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-138M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="lingxusb/megaDNA", branch="ar", context_tokens=131072, loader_kind="megadna"),
    ModelSpec(hf_id="lingxusb/PlasmidGPT", branch="ar", context_tokens=2048, loader_kind="plasmidgpt"),
    ModelSpec(hf_id="GenSLM-25M", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    ModelSpec(hf_id="GenSLM-250M", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    ModelSpec(hf_id="GenSLM-2.5B", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    # MLM branch
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/agro-nucleotide-transformer-1b", branch="mlm", context_tokens=1024, trust_remote_code=True),
)


# NOTE: the three-matrix codon-NaN protocol (R_pan_DNA / R_coding_only /
# R_nucleotide_only) was retired in commit 5e59154 "matrix protocol: drop
# three-matrix split, build one L/Q/D per branch". The old NON_CODING_CLASSES
# / CODING_CLASS constants that gated that routing have been removed.


# --------------------------- Matrix assembly ----------------------------- #


def _save_branch_matrices(
    out_dir: Path,
    branch_label: str,
    matrices: BranchMatrices,
) -> dict:
    """Save V / V_d / D for one branch.

    File-naming convention:
      V_<branch>.npy     raw sum_log_p, floor-clipped at 2nd percentile
      V_d_<branch>.npy   double-centered V (consume in PCA / distance diagnostics)
      D_<branch>.npy     (M, M) pairwise squared Euclidean on V_d
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"V_{branch_label}.npy", matrices.L)
    np.save(out_dir / f"V_d_{branch_label}.npy", matrices.Q)
    np.save(out_dir / f"D_{branch_label}.npy", matrices.D)
    metadata: dict = {
        f"V_{branch_label}": {
            "shape": list(matrices.L.shape),
            "row_model_ids": matrices.model_ids,
            "col_probe_ids": matrices.probe_ids,
            "n_nan_cells": int(np.isnan(matrices.L).sum()),
            "clip_threshold": matrices.clip_threshold,
        },
        f"V_d_{branch_label}": {
            "shape": list(matrices.Q.shape),
            "row_model_ids": matrices.model_ids,
            "col_probe_ids": matrices.probe_ids,
            "n_nan_cells": int(np.isnan(matrices.Q).sum()),
            "note": "double-centered V (row mean + column mean removed). "
            "Consume here in PCA and distance diagnostics.",
        },
        f"D_{branch_label}": {
            "shape": list(matrices.D.shape),
            "row_model_ids": matrices.model_ids,
            "col_model_ids": matrices.model_ids,
            "n_nan_cells": int(np.isnan(matrices.D).sum()),
            "note": "pairwise squared Euclidean on V_d; approximates KL under "
            "small-divergence Taylor (ModelMap Sec. 6.1).",
        },
    }
    return metadata


# --------------------------------- main ----------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=REPO_ROOT / "data/panels/main_panel.parquet")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results/scores")
    p.add_argument(
        "--scores-subdir", dest="scores_subdir", type=str, default="AR_MLM_scores",
        help="Name of the per-model score subdirectory under --out "
             "(<out>/<scores-subdir>/<slug>/probes.parquet). Default "
             "'AR_MLM_scores' for the canonical tree; override e.g. "
             "'MLM_true-PLL_scores' for the k=1 true-PLL ablation so it "
             "lands in its own folder.")
    p.add_argument(
        "--device",
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="cuda:N or cpu. Default auto-picks cuda:0 when CUDA is available.",
    )
    p.add_argument("--stride", type=int, default=6,
                   help="MLM stride k (phase_1.md primary k=6).")
    p.add_argument("--force", action="store_true",
                   help="Re-score every model even if its parquet exists.")
    p.add_argument("--max-probes", type=int, default=None,
                   help="Debug: truncate panel to N probes before scoring.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated SUBSTRING filter on hf_id (for debug "
                        "and human use). NOTE: substrings can collide — e.g. "
                        "'evo2_7b' matches evo2_7b, evo2_7b_base, "
                        "evo2_7b_262k. For parallel sweeps where each "
                        "subprocess must score exactly one model, use "
                        "--hf-ids instead (exact match).")
    p.add_argument("--hf-ids", type=str, default=None,
                   help="Comma-separated list of EXACT hf_id strings. Each "
                        "must match the audit (or DEFAULT_MODELS) exactly. "
                        "Use this from run_sweep.py and other parallel "
                        "drivers to avoid the --only substring collisions.")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip the matrix-build step. Useful when "
                        "running parallel per-model jobs that should not race "
                        "on results/scores/matrices/. Run a final aggregate pass "
                        "without --only afterward.")
    p.add_argument("--strict-aggregate", action="store_true",
                   help="When aggregating, fail-fast if any model's parquet "
                        "is missing or has partial probe coverage. Without "
                        "this flag the matrix is built with NaN rows / cells "
                        "for incomplete models (a record of which models "
                        "actually contributed is always written to "
                        "out/matrices/scored_models_actual.json). Use the "
                        "strict mode for the final Stage 4 aggregate.")
    p.add_argument("--allow-missing", action="store_true",
                   help="When aggregating, DROP rows for models with no "
                        "parquet (or with partial / NaN-poisoned parquet) "
                        "instead of keeping NaN rows. Useful for a 'preview' "
                        "aggregate while a few stragglers are still running. "
                        "The resulting L/Q/D matrices have M_actual ≤ "
                        "len(specs); scored_models_actual.json records which "
                        "rows are present. Mutually exclusive with "
                        "--strict-aggregate.")
    p.add_argument("--from-audit", action="store_true",
                   help="Source the model roster from data/audits/models.json "
                        "(123 candidates as of 2026-05-20) instead of the "
                        "13-model DEFAULT_MODELS pilot set. The matrix "
                        "aggregation step also uses the audit-derived list, "
                        "so the L / Q / D matrices cover all scorable audit "
                        "models. This is the Stage 4 / phase 2 entry point.")
    p.add_argument("--audit-json", type=Path,
                   default=REPO_ROOT / "data" / "audits" / "models.json",
                   help="Audit JSON path (only consulted with --from-audit).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.panel.exists():
        raise SystemExit(f"--panel not found: {args.panel}")
    panel = pd.read_parquet(args.panel)
    required_cols = {"probe_id", "sequence", "functional_element", "GC_content"}
    missing_cols = required_cols - set(panel.columns)
    if missing_cols:
        raise SystemExit(f"panel parquet is missing columns: {missing_cols}")
    if not panel["probe_id"].is_unique:
        raise SystemExit("panel.probe_id is not unique; cannot align scores by probe_id")
    if args.max_probes is not None and args.max_probes < len(panel):
        panel = panel.head(args.max_probes).reset_index(drop=True)
        print(
            f"[panel] --max-probes={args.max_probes}; truncated to {len(panel)} probes",
            flush=True,
        )
    print(f"[panel] loaded {len(panel)} probes from {args.panel}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / args.scores_subdir).mkdir(parents=True, exist_ok=True)
    (args.out / "matrices").mkdir(parents=True, exist_ok=True)

    # Resolve the model roster.
    if args.from_audit:
        if not args.audit_json.exists():
            raise SystemExit(f"--from-audit but audit json not found: {args.audit_json}")
        from glmap.loaders.dispatch import specs_from_audit
        roster = specs_from_audit(audit_path=args.audit_json)
        roster_name = f"audit ({args.audit_json.name})"
    else:
        roster = list(DEFAULT_MODELS)
        roster_name = "DEFAULT_MODELS"

    if args.hf_ids and args.only:
        raise SystemExit("--hf-ids and --only are mutually exclusive")
    if args.hf_ids:
        wanted = {s.strip() for s in args.hf_ids.split(",") if s.strip()}
        roster_by_id = {s.hf_id: s for s in roster}
        missing = wanted - set(roster_by_id)
        if missing:
            raise SystemExit(
                f"--hf-ids referenced {len(missing)} hf_id(s) not in "
                f"{roster_name}: {sorted(missing)[:5]}..."
            )
        models_to_run = [roster_by_id[h] for h in wanted]
        print(
            f"[hf-ids] exact match: {len(models_to_run)}/{len(roster)} models "
            f"from {roster_name}: " + ", ".join(s.slug for s in models_to_run),
            flush=True,
        )
    elif args.only:
        only_tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        models_to_run = [
            s for s in roster if any(t in s.hf_id for t in only_tokens)
        ]
        print(
            f"[only] substring match: {len(models_to_run)}/{len(roster)} models "
            f"from {roster_name}: " + ", ".join(s.slug for s in models_to_run),
            flush=True,
        )
    else:
        models_to_run = list(roster)
        print(f"[roster] {len(models_to_run)} models from {roster_name}", flush=True)

    def _write_model_meta(spec: ModelSpec, meta_path: Path) -> None:
        """Per-model spec snapshot. Called on every iteration (both
        skipped-because-cached and freshly-scored) so newly-added spec
        fields (e.g. loader_kind, length_multiple) reflect the current
        dispatch on disk regardless of whether the parquet was rebuilt.
        Without this the per-model JSON would only update on re-score
        and stale metadata would diverge from the actual loader."""
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "hf_id": spec.hf_id,
            "branch": spec.branch,
            "context_tokens": spec.context_tokens,
            "trust_remote_code": spec.trust_remote_code,
            "is_codon": spec.is_codon,
            "loader_kind": spec.loader_kind,
            "length_multiple": spec.length_multiple,
            "slug": spec.slug,
            "stride_primary": args.stride if spec.branch == "mlm" else None,
            "scoring_protocol": (
                "AR forward sum_log_p (raw nats, no length norm, no sign flip; "
                "ModelMap convention)"
                if spec.branch == "ar"
                else f"MLM stride pseudo-log-likelihood, k={args.stride} "
                     "(raw nats, no length norm, no sign flip)"
            ),
        }, indent=2))

    for spec in models_to_run:
        score_dir = args.out / args.scores_subdir / spec.slug
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / "probes.parquet"
        # Per-model metadata snapshot now lives inside the model's own
        # score folder (alongside probes.parquet) rather than a sibling
        # models/ directory.
        meta_path = score_dir / f"{spec.slug}.json"

        if score_path.exists() and not args.force:
            # Resume-integrity check: file existing is necessary but not
            # sufficient. The check verifies that the cached parquet has
            # the full panel's probe_id set AND that every sum_log_p is
            # finite — a per-probe scoring failure would still write a
            # row with sum_log_p=NaN, which the earlier set-equality-only
            # check accepted as complete.
            ok, reason = parquet_covers_panel(score_path, set(panel["probe_id"]),
                                              n_panel=len(panel))
            if ok:
                print(
                    f"[skip] {spec.slug}: {score_path} {reason}",
                    flush=True,
                )
                # Refresh metadata even on skip so loader_kind / length_multiple
                # always reflect the current dispatch logic, not whatever the
                # spec was when the parquet was originally written.
                _write_model_meta(spec, meta_path)
                continue
            print(
                f"[resume] {spec.slug}: {score_path} {reason}; re-scoring.",
                flush=True,
            )

        loader = None
        try:
            df, loader = _score_model(
                spec=spec, panel=panel, device=args.device, stride=args.stride
            )
            df.to_parquet(score_path, index=False)
            print(
                f"[{spec.branch}] {spec.slug} wrote scores -> {score_path}",
                flush=True,
            )
        except Exception as exc:
            # Log the trace but don't kill the whole run; the matrices step
            # will fill the missing row with NaN.
            traceback.print_exc()
            print(
                f"[fail] {spec.slug}: scoring aborted with "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        finally:
            # Free GPU memory between models. `loader` may be None if the
            # try block raised before _score_model returned.
            if loader is not None:
                del loader
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        _write_model_meta(spec, meta_path)

    if args.skip_aggregate:
        print("[done] --skip-aggregate; skipping matrix build + report.", flush=True)
        return

    # ---- Aggregate matrices from per-model parquets ---- #
    # Aggregate over the full roster (audit-derived or DEFAULT_MODELS), not
    # the --only filter — so a multi-process parallel sweep can write each
    # model's parquet under --only and a final aggregate pass without --only
    # builds the matrices over everything on disk.
    if args.strict_aggregate and args.allow_missing:
        raise SystemExit(
            "--strict-aggregate and --allow-missing are mutually exclusive: "
            "the first fails on missing models, the second drops them."
        )

    ar_specs = [s for s in roster if s.branch == "ar"]
    mlm_specs = [s for s in roster if s.branch == "mlm"]

    ar_matrices, ar_actual = _build_branch_matrices(
        ar_specs, panel, args.out / args.scores_subdir, allow_missing=args.allow_missing,
    )
    mlm_matrices, mlm_actual = _build_branch_matrices(
        mlm_specs, panel, args.out / args.scores_subdir, allow_missing=args.allow_missing,
    )

    # Persist the explicit "which models actually contributed scores" record so
    # downstream analysis doesn't have to introspect NaN rows of the L matrix
    # to figure out which rows are real.
    actual_path = args.out / "matrices" / "scored_models_actual.json"
    actual_path.write_text(json.dumps({
        "ar": ar_actual,
        "mlm": mlm_actual,
        "n_ar_expected": len(ar_specs),
        "n_mlm_expected": len(mlm_specs),
        "n_ar_scored": len(ar_actual["scored_models"]),
        "n_mlm_scored": len(mlm_actual["scored_models"]),
    }, indent=2))

    n_missing = len(ar_actual["missing_models"]) + len(mlm_actual["missing_models"])
    n_partial = len(ar_actual["partial_models"]) + len(mlm_actual["partial_models"])
    if args.strict_aggregate and (n_missing or n_partial):
        raise SystemExit(
            f"--strict-aggregate: {n_missing} model(s) missing parquet, "
            f"{n_partial} model(s) with partial probe coverage. "
            f"See {actual_path} for details. Run scoring to completion "
            "before aggregating, or drop --strict-aggregate to accept "
            "NaN rows / cells."
        )

    matrices_meta: dict = {}
    matrices_meta.update(_save_branch_matrices(args.out / "matrices", "AR", ar_matrices))
    matrices_meta.update(_save_branch_matrices(args.out / "matrices", "MLM", mlm_matrices))
    matrices_meta["protocol"] = (
        "Single-matrix protocol per branch (ModelMap, commit 5e59154). "
        "L[m, x] = sum_log_p_m(x) in raw nats (no length normalization, no "
        "sign flip). L_clipped = floor_clip(L, q=0.02); Q = double_center("
        "L_clipped); D = pairwise_squared_distance(Q). Codon models "
        "(GenSLM, Codon-NT) emit raw likelihood on every probe and enter L "
        "alongside nucleotide-tokenized models; the codon-vs-nucleotide "
        "systematic offset on noncoding probes is absorbed by the row-mean "
        "and column-mean subtraction during double-centering. The earlier "
        "three-matrix split (R_pan_DNA / R_coding_only / R_nucleotide_only) "
        "is retired."
    )
    matrices_meta["scoring_stride_mlm"] = args.stride
    (args.out / "matrices" / "matrix_metadata.json").write_text(
        json.dumps(matrices_meta, indent=2)
    )

    print(f"[done] wrote outputs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
