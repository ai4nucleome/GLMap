#!/usr/bin/env python3
"""Phase 5 (downstream eval): linear probe AUC for each (model, task) pair.

Reads embeddings written by `scripts/run_downstream_embed.py`
(out_phase2/embeddings/<model>/<task>/{train,test}.parquet), fits an L2
logistic regression with C grid via 5-fold CV on train, evaluates on
test, and writes a result JSON per (model, task) to
out_phase2/downstream/<model>/<task>/result.json containing:

    {
      "model": <hf_id>,
      "task": <task_id>,
      "auc_test": float,
      "auc_cv_mean": float,
      "auc_cv_std": float,
      "best_C": float,
      "n_train": int,
      "n_test": int,
      "embed_dim": int,
      "n_classes": int,
      "multiclass": bool,
      "classifier": "l2_logistic",
    }

Binary tasks (n_classes == 2) — the current 6-task suite — use the
standard binary ROC-AUC. Multiclass tasks (n_classes > 2) would use
one-vs-rest **weighted** AUC (roc_auc_ovr_weighted scoring during CV,
average="weighted" at test time), but no task in the current suite
exercises that branch. The chosen average ("binary" or "ovr_weighted")
is written into the per-pair result.json as `auc_average` so future
multiclass additions can be audited.

Aggregate (every invocation, regardless of --hf-ids / --tasks filter):
    out_phase2/matrices/auc_matrix.npy    (M, T) float64
    out_phase2/matrices/auc_matrix_meta.json    {model_ids, task_ids,
                                                  aggregate_scope=
                                                  "all_existing_results",
                                                  generated_at_utc, ...}

The aggregate ALWAYS rescans the full out_phase2/downstream/ tree, even
when this invocation's per-pair fit was filtered. This way a partial
re-fit (e.g. `--hf-ids HuggingFaceBio/Carbon-3B` to refresh one model)
patches the matrix in place without orphaning the rest. Consumers
needing "this run only" results should read per-pair result.json.

Usage:
    $PY scripts/run_downstream_classify.py                       # all available
    $PY scripts/run_downstream_classify.py --hf-ids X,Y,Z        # subset
    $PY scripts/run_downstream_classify.py --tasks "5mC,Yeast"   # task subset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.io.embed_schema import validate_embed_columns   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings-dir", type=Path,
                   default=REPO_ROOT / "out_phase2/embeddings")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "out_phase2",
                   help="Output root; downstream/ and matrices/ written here.")
    p.add_argument("--hf-ids", type=str, default=None,
                   help="Comma-separated EXACT hf_id list to classify.")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated substring filter on task_id.")
    p.add_argument("--force", action="store_true",
                   help="Re-fit every (model, task) pair regardless of "
                        "cached result.json state.")
    p.add_argument("--invalidate-legacy-cache", action="store_true",
                   help="Treat cached result.json files that don't carry "
                        "a `fit_params` fingerprint as STALE and re-fit "
                        "them. Use this once after upgrading the "
                        "classify script from pre-fingerprint to "
                        "fingerprint-aware behavior. Without this flag, "
                        "legacy results are kept as-is to avoid mass "
                        "invalidating the 750+ already-fit cells.")
    p.add_argument("--no-retry-pipeline-errors", action="store_true",
                   help="Disables ONLY the pipeline-error auto-retry. "
                        "Default: cached load_fail/fit_fail records are "
                        "automatically re-fit (usually transient — loader "
                        "bug fixed, GPU contention cleared, etc.). With "
                        "this flag set, a load_fail/fit_fail record is "
                        "kept as-is **on a same-state, same-params re-run**. "
                        "Other cache-invalidation signals still apply: a "
                        "fresh embedding parquet (newer mtime) or a "
                        "different --c-grid/--n-cv/--seed will still "
                        "trigger a refit. To permanently exclude a "
                        "known-broken (model, task) pair across all "
                        "future runs, use --hf-ids / --tasks to filter "
                        "it out of discovery, or move its embedding "
                        "parquet directory out of out_phase2/embeddings/ "
                        "— deleting the cached result.json does NOT "
                        "freeze it, the pair would be re-discovered via "
                        "its embedding parquets and refit on the next "
                        "run.")
    p.add_argument("--c-grid", type=str, default="0.01,0.1,1,10,100",
                   help="Comma-separated L2 C grid for CV.")
    p.add_argument("--n-cv", type=int, default=5,
                   help="K-fold CV (default 5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="sklearn n_jobs.")
    return p.parse_args()


def load_embed_split(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Returns (X_filtered, y_filtered, n_raw, n_dropped).

    Drops rows where ANY embedding dimension is non-finite — these are
    the NaN-placeholder rows written by embed_split when a sequence's
    forward pass raised an exception. Reporting n_raw + n_dropped makes
    silent sample loss visible in the per-pair result.json.
    """
    df = pd.read_parquet(path)
    label_col = "label"
    # Schema validation is shared with parquet_complete() (sweep resume).
    # Both consumers call validate_embed_columns from src/io/embed_schema.py
    # so the load and resume layers enforce IDENTICAL contracts: if resume
    # accepts a parquet as complete, classify must accept; if classify
    # rejects, resume must too. The validator lives in a CPU-only module
    # so this script doesn't transitively pull in the embedding-extraction
    # stack (torch, loaders).
    try:
        embed_cols = validate_embed_columns(df.columns)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    X = df[embed_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)
    n_raw = int(len(df))
    mask = np.isfinite(X).all(axis=1)
    n_dropped = int(n_raw - int(mask.sum()))
    return X[mask], y[mask], n_raw, n_dropped


def fit_and_score(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    c_grid: list[float], n_cv: int, seed: int, n_jobs: int,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n_classes = int(np.unique(y_tr).size)
    multiclass = n_classes > 2
    scoring = "roc_auc_ovr_weighted" if multiclass else "roc_auc"
    solver = "lbfgs" if multiclass else "liblinear"

    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)

    # Wrap StandardScaler + LogisticRegression in a Pipeline so the scaler
    # is fit fresh on each CV training fold (no inner-CV leak from outer
    # train holdout into the scaler stats). Matches the methodology in
    # dna_foundation_benchmark/job_scripts/offline_linear_probe.py.
    def make_pipeline(C: float) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C, penalty="l2", solver=solver,
                max_iter=2000, random_state=seed,
            )),
        ])

    # CV grid: pick C with highest mean CV AUC
    cv_scores = {}
    for C in c_grid:
        scores = cross_val_score(
            make_pipeline(C), X_tr, y_tr,
            cv=cv, scoring=scoring, n_jobs=n_jobs,
        )
        cv_scores[C] = (float(scores.mean()), float(scores.std()))
    best_C = max(cv_scores, key=lambda C: cv_scores[C][0])
    cv_mean, cv_std = cv_scores[best_C]

    # Refit best-C pipeline on full train + eval on test. Scaler inside
    # the pipeline is re-fit on the full train, then applied to test.
    best_pipe = make_pipeline(best_C).fit(X_tr, y_tr)
    proba = best_pipe.predict_proba(X_te)
    if multiclass:
        auc_test = float(roc_auc_score(y_te, proba,
                                         multi_class="ovr",
                                         average="weighted"))
    else:
        classes = best_pipe.named_steps["clf"].classes_
        pos_idx = list(classes).index(1) if 1 in classes else 1
        auc_test = float(roc_auc_score(y_te, proba[:, pos_idx]))

    return {
        "auc_test": auc_test,
        "auc_cv_mean": cv_mean,
        "auc_cv_std": cv_std,
        "best_C": float(best_C),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "embed_dim": int(X_tr.shape[1]),
        "n_classes": int(n_classes),
        "multiclass": multiclass,
        "classifier": "l2_logistic",
        # Audit trail for the chosen AUC averaging — "binary" for the
        # current 6-task suite (n_classes==2), "ovr_weighted" for any
        # future multiclass task. Lets downstream consumers detect
        # which averaging convention produced a cell's value without
        # re-reading the docstring.
        "auc_average": "ovr_weighted" if multiclass else "binary",
    }


def main() -> None:
    args = parse_args()
    if not args.embeddings_dir.exists():
        raise SystemExit(f"embeddings dir not found: {args.embeddings_dir}")

    c_grid = [float(c) for c in args.c_grid.split(",")]

    # Discover (model, task) pairs that have BOTH train + test embedded
    pairs: list[tuple[str, str, Path, Path]] = []
    for model_dir in sorted(args.embeddings_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name
        if args.hf_ids:
            wanted = {h.strip().replace("/", "__") for h in args.hf_ids.split(",")}
            if model_slug not in wanted:
                continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if args.tasks:
                pats = [p.strip() for p in args.tasks.split(",")]
                if not any(p in task_name for p in pats):
                    continue
            tr = task_dir / "train.parquet"
            te = task_dir / "test.parquet"
            if not (tr.exists() and te.exists()):
                continue
            pairs.append((model_slug, task_name, tr, te))

    print(f"[downstream-classify] {len(pairs)} (model, task) pairs", flush=True)

    out_root = args.out / "downstream"
    out_root.mkdir(parents=True, exist_ok=True)

    # Parameter fingerprint stored in every newly-written result.json
    # so a `result.json` from an older command (different --c-grid /
    # --n-cv / --seed) is automatically invalidated. Existing results
    # without this fingerprint are treated as "accept" — we don't want
    # to mass-invalidate the 750+ already-fit cells unless the user
    # explicitly --force's.
    current_params = {
        "c_grid": c_grid,
        "n_cv": int(args.n_cv),
        "seed": int(args.seed),
    }

    # Categorize skip/fail records for the aggregate. tiny / class_gate
    # are deliberate (data didn't meet the gate); load_fail / fit_fail
    # are PIPELINE errors that warrant retry, not "skip-by-design".
    DELIBERATE_ERROR_TYPES = {"tiny", "class_gate"}
    PIPELINE_ERROR_TYPES = {"load_fail", "fit_fail"}

    def _write_skip_result(
        result_path: Path, model_slug: str, task_name: str,
        error_type: str, skip_reason: str,
        extra: dict | None = None, refit_reason: str | None = None,
    ) -> None:
        """All skip/fail paths route through here so EVERY result.json
        carries `fit_params` (lets a later run with different params
        invalidate this cell — e.g. a tiny task at n_cv=5 should be
        re-attempted at n_cv=3) and a structured `error_type` (lets
        the aggregate distinguish "deliberate gate" vs "pipeline
        error"). If invoked from the [refit] path, refit_reason
        propagates into the record so the trail is auditable."""
        payload: dict = {
            "model": model_slug.replace("__", "/"),
            "task": task_name.replace("__", "/"),
            "skip_reason": skip_reason,
            "error_type": error_type,
            "fit_params": current_params,
        }
        if refit_reason is not None:
            payload["refit_reason"] = refit_reason
        if extra:
            payload.update(extra)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2))

    for model_slug, task_name, tr_path, te_path in pairs:
        result_path = out_root / model_slug / task_name / "result.json"
        if (not args.force) and result_path.exists():
            refit_reason: str | None = None
            cached: dict | None = None
            # (a) Cache must be readable + structurally valid. A corrupt
            # JSON, missing both auc_test and skip_reason, or non-finite
            # auc_test all trigger a refit so the bad result eventually
            # gets fixed without --force.
            try:
                cached = json.loads(result_path.read_text())
            except Exception as exc:
                refit_reason = f"cached result.json unreadable ({type(exc).__name__})"
            else:
                if not isinstance(cached, dict):
                    refit_reason = "cached result.json is not a JSON object"
                elif "auc_test" not in cached and "skip_reason" not in cached:
                    refit_reason = (
                        "cached result.json has neither auc_test nor skip_reason"
                    )
                elif "auc_test" in cached:
                    try:
                        auc_cached = float(cached["auc_test"])
                        if not np.isfinite(auc_cached) or not (0.0 <= auc_cached <= 1.0):
                            refit_reason = f"cached auc_test out of range: {auc_cached}"
                    except (TypeError, ValueError):
                        refit_reason = "cached auc_test is non-numeric"

            # (b) Embedding parquet newer than cache → fresh embed run,
            # re-fit so the cached AUC reflects the new vectors.
            if refit_reason is None:
                r_mtime = result_path.stat().st_mtime
                tr_mtime = tr_path.stat().st_mtime if tr_path.exists() else 0.0
                te_mtime = te_path.stat().st_mtime if te_path.exists() else 0.0
                stalest_embed = max(tr_mtime, te_mtime)
                if stalest_embed > r_mtime:
                    refit_reason = (
                        f"embedding parquet newer than cached result "
                        f"({stalest_embed - r_mtime:.0f}s)"
                    )

            # (c) Parameters changed since the cache was written. Only
            # invalidates results that EXPLICITLY recorded a fingerprint;
            # older results (no fingerprint) are kept unless the user
            # passes --invalidate-legacy-cache.
            if refit_reason is None and cached is not None:
                if "fit_params" in cached:
                    old_params = cached["fit_params"]
                    if old_params != current_params:
                        refit_reason = (
                            f"fit params changed (was {old_params}, "
                            f"now {current_params})"
                        )
                elif args.invalidate_legacy_cache:
                    refit_reason = (
                        "legacy result.json (no fit_params fingerprint); "
                        "--invalidate-legacy-cache requested re-fit"
                    )

            # (d) Pipeline-error records (load_fail / fit_fail) auto-retry
            # by default — these are usually transient (the user fixed
            # a loader bug, GPU contention cleared, etc.) and the next
            # run almost always wants to re-attempt them.
            #
            # `--no-retry-pipeline-errors` is a SOFT opt-out: it disables
            # ONLY this branch. It does NOT override the prior checks
            # above — (a) corruption, (b) embedding mtime, (c) param
            # fingerprint still trigger a refit. Rationale: if the user
            # re-ran embed (loader bug fix) or changed --n-cv, they
            # almost certainly DO want the previously-failed cell to
            # be re-attempted; freezing it across all future runs
            # would surprise.
            #
            # To PERMANENTLY exclude a known-broken (model, task) pair,
            # filter it out of discovery via --hf-ids / --tasks, or move
            # its embedding parquet directory out of out_phase2/embeddings/.
            # Deleting the cached result.json does NOT freeze it — the
            # pair is re-discovered from its embedding parquets every
            # run (see the `pairs` loop earlier in main()) and would
            # land in the fresh-fit path with no cache to skip on.
            if (
                refit_reason is None
                and cached is not None
                and not args.no_retry_pipeline_errors
                and str(cached.get("error_type", "")).lower() in PIPELINE_ERROR_TYPES
            ):
                refit_reason = (
                    f"prior {cached['error_type']} record — auto-retrying "
                    "(pass --no-retry-pipeline-errors to keep failure)"
                )

            if refit_reason is not None:
                print(
                    f"  [refit] {model_slug}/{task_name}: {refit_reason}; "
                    "re-fitting.",
                    flush=True,
                )
            else:
                print(f"  [skip] {model_slug}/{task_name} done", flush=True)
                continue
        else:
            refit_reason = None   # fresh fit, no refit context

        try:
            X_tr, y_tr, n_train_raw, n_train_dropped = load_embed_split(tr_path)
            X_te, y_te, n_test_raw, n_test_dropped = load_embed_split(te_path)
        except Exception as exc:
            err_msg = f"load_fail: {type(exc).__name__}: {exc}"
            print(f"  [load fail] {model_slug}/{task_name}: {err_msg}",
                  flush=True)
            # Always overwrite the result.json (including the [refit]
            # case where the cached AUC would otherwise leak through to
            # the aggregate matrix as stale).
            _write_skip_result(
                result_path, model_slug, task_name,
                error_type="load_fail", skip_reason=err_msg,
                refit_reason=refit_reason,
            )
            continue

        # Surface large NaN-drop fractions so silent sample loss is visible.
        # A drop fraction > 0.05 typically means the embed step had model-
        # specific forward failures on whole batches of sequences.
        drop_frac_tr = n_train_dropped / max(n_train_raw, 1)
        drop_frac_te = n_test_dropped / max(n_test_raw, 1)
        if drop_frac_tr > 0.05 or drop_frac_te > 0.05:
            print(
                f"  [drop-warning] {model_slug}/{task_name} "
                f"train dropped {n_train_dropped}/{n_train_raw} "
                f"({drop_frac_tr:.1%}), test dropped "
                f"{n_test_dropped}/{n_test_raw} ({drop_frac_te:.1%}) — "
                f"AUC may be biased; investigate the embed step",
                flush=True,
            )

        # Aggregate-size gate.
        if len(X_tr) < 20 or len(X_te) < 5:
            reason = f"tiny: n_train={len(X_tr)}, n_test={len(X_te)}"
            print(f"  [tiny] {model_slug}/{task_name} ({reason}); skipping",
                  flush=True)
            _write_skip_result(
                result_path, model_slug, task_name,
                error_type="tiny", skip_reason=reason,
                refit_reason=refit_reason,
                extra={
                    "n_train_raw": n_train_raw,
                    "n_train_dropped": n_train_dropped,
                    "n_test_raw": n_test_raw,
                    "n_test_dropped": n_test_dropped,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                },
            )
            continue

        # Per-class minimum-count gate. StratifiedKFold(n_splits=args.n_cv)
        # requires each class to have >= n_cv samples in train; ROC-AUC on
        # test needs >= 1 positive AND >= 1 negative. Below either bound
        # `fit_and_score` would raise mid-CV, the user would see only a
        # generic "[fit fail]", and the aggregate AUC matrix would carry
        # a NaN cell with no recorded reason. Skip explicitly so the
        # result.json captures why.
        tr_classes, tr_counts = np.unique(y_tr, return_counts=True)
        te_classes, te_counts = np.unique(y_te, return_counts=True)
        min_tr_class = int(tr_counts.min()) if len(tr_counts) else 0
        n_classes_tr = int(len(tr_classes))
        n_classes_te = int(len(te_classes))
        gate_reason = None
        if n_classes_tr < 2:
            gate_reason = f"train has only {n_classes_tr} class after NaN-drop"
        elif n_classes_te < 2:
            gate_reason = f"test has only {n_classes_te} class after NaN-drop"
        elif set(int(c) for c in tr_classes) != set(int(c) for c in te_classes):
            # Multiclass case: roc_auc_score(..., multi_class="ovr") computes
            # per-class AUC and averages. If test is missing a class that
            # train has (e.g. train has {0,1,2}, NaN-drop leaves test with
            # {0,1}), the OVR step crashes on the missing class. Binary case
            # is already caught by the `n_classes_te < 2` branch above.
            tr_set = sorted(int(c) for c in tr_classes)
            te_set = sorted(int(c) for c in te_classes)
            gate_reason = (
                f"train classes {tr_set} != test classes {te_set} "
                f"(NaN-drop left test split missing classes; multiclass "
                "ROC-AUC would crash)"
            )
        elif min_tr_class < args.n_cv:
            gate_reason = (
                f"smallest train class has {min_tr_class} samples < "
                f"n_cv={args.n_cv} (StratifiedKFold would fail)"
            )
        if gate_reason is not None:
            print(f"  [class-gate] {model_slug}/{task_name}: {gate_reason}; "
                  "skipping", flush=True)
            _write_skip_result(
                result_path, model_slug, task_name,
                error_type="class_gate", skip_reason=gate_reason,
                refit_reason=refit_reason,
                extra={
                    "n_train_raw": n_train_raw,
                    "n_train_dropped": n_train_dropped,
                    "n_test_raw": n_test_raw,
                    "n_test_dropped": n_test_dropped,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                    "train_class_counts": {
                        int(c): int(n) for c, n in zip(tr_classes, tr_counts)
                    },
                    "test_class_counts": {
                        int(c): int(n) for c, n in zip(te_classes, te_counts)
                    },
                },
            )
            continue

        try:
            t0 = time.time()
            metrics = fit_and_score(
                X_tr, y_tr, X_te, y_te,
                c_grid=c_grid, n_cv=args.n_cv,
                seed=args.seed, n_jobs=args.n_jobs,
            )
            metrics["model"] = model_slug.replace("__", "/")
            metrics["task"] = task_name.replace("__", "/")
            # Provenance: raw vs post-NaN-drop counts so downstream
            # analyses can detect when AUC is on a heavily-filtered set.
            metrics["n_train_raw"] = n_train_raw
            metrics["n_train_dropped"] = n_train_dropped
            metrics["n_test_raw"] = n_test_raw
            metrics["n_test_dropped"] = n_test_dropped
            # Parameter fingerprint — consumed by the cache-validation
            # path to detect "result was fit under a different C grid /
            # CV setting / seed" and trigger an automatic refit.
            metrics["fit_params"] = current_params
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(metrics, indent=2))
            print(f"  [done] {model_slug}/{task_name}  "
                  f"AUC_test={metrics['auc_test']:.4f}  "
                  f"AUC_cv={metrics['auc_cv_mean']:.4f}  "
                  f"C*={metrics['best_C']:g}  "
                  f"drop={n_train_dropped}/{n_test_dropped}  "
                  f"{time.time()-t0:.1f}s",
                  flush=True)
        except Exception as exc:
            import traceback
            err_msg = f"fit_fail: {type(exc).__name__}: {exc}"
            print(f"  [fit fail] {model_slug}/{task_name}: {err_msg}",
                  flush=True)
            traceback.print_exc()
            # Critical: when this is a [refit] (cached result existed) and
            # the new fit fails, we MUST overwrite the cached JSON so the
            # aggregate doesn't keep reading the stale AUC. Same writer
            # path as tiny / class_gate / load_fail for uniformity.
            _write_skip_result(
                result_path, model_slug, task_name,
                error_type="fit_fail", skip_reason=err_msg,
                refit_reason=refit_reason,
                extra={
                    "n_train_raw": n_train_raw,
                    "n_train_dropped": n_train_dropped,
                    "n_test_raw": n_test_raw,
                    "n_test_dropped": n_test_dropped,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(X_te)),
                },
            )

    # ─────────── aggregate (M, T) AUC matrix ───────────
    # Scan every result.json under out_root/<model>/<task>/, build a
    # (M, T) matrix indexed by sorted model + task names. NaN where a
    # pair never produced a result. Written every run so the matrix
    # always reflects the latest set of result.json files on disk.
    #
    # IMPORTANT: scope is **all_existing_results** — we re-scan everything
    # under out_root regardless of --hf-ids / --tasks filters that scoped
    # the per-pair loop above. This matches the intended use ("final
    # matrix should include every model × task that has ever produced a
    # result"), so a subset re-fit (e.g. --hf-ids HuggingFaceBio/Carbon-3B
    # to refresh one model after fixing its embeddings) updates the
    # matrix in-place without orphaning the other 124 models. The meta
    # JSON records aggregate_scope so consumers can detect this.
    matrices_dir = args.out / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, float]] = {}
    corrupt_paths: list[tuple[str, str]] = []   # (path, reason)
    deliberate_paths: list[tuple[str, str]] = []   # tiny / class_gate
    error_paths: list[tuple[str, str]] = []   # load_fail / fit_fail
    unknown_skip_paths: list[tuple[str, str]] = []   # any other / missing error_type
    # Universe of (model_slug, task_slug) pairs seen on disk, regardless
    # of outcome. The matrix axes are derived from this set so a model
    # whose every task was class-gate-skipped still appears as a row
    # with all-NaN cells, instead of vanishing from the axes entirely.
    all_model_slugs: set[str] = set()
    all_task_slugs: set[str] = set()
    for model_dir in sorted(out_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            result_path = task_dir / "result.json"
            if not result_path.exists():
                continue
            # Record axis membership BEFORE outcome classification.
            all_model_slugs.add(model_dir.name)
            all_task_slugs.add(task_dir.name)
            try:
                m = json.loads(result_path.read_text())
            except Exception as exc:
                corrupt_paths.append((str(result_path), f"JSON decode: {exc}"))
                continue
            # Skip records — explicit three-way categorization on
            # `error_type`. Catching unknown / missing values into a
            # separate bucket means a typo or a future error_type we
            # haven't taught the aggregate about doesn't get silently
            # under-counted as deliberate. Legacy records without any
            # error_type also land here so they're visible (not lumped
            # into "skip by design").
            if "skip_reason" in m and "auc_test" not in m:
                err_type = str(m.get("error_type", "")).lower()
                if err_type in DELIBERATE_ERROR_TYPES:
                    deliberate_paths.append((str(result_path), str(m["skip_reason"])))
                elif err_type in PIPELINE_ERROR_TYPES:
                    error_paths.append((str(result_path), str(m["skip_reason"])))
                else:
                    # Unknown / missing error_type — surface separately
                    # so it doesn't disappear into "deliberate".
                    unknown_skip_paths.append((
                        str(result_path),
                        f"error_type={err_type!r}: {m['skip_reason']}",
                    ))
                continue
            if "auc_test" not in m:
                corrupt_paths.append((str(result_path), "missing auc_test"))
                continue
            try:
                auc = float(m["auc_test"])
            except (TypeError, ValueError) as exc:
                corrupt_paths.append((str(result_path), f"auc_test not numeric: {exc}"))
                continue
            if not (0.0 <= auc <= 1.0) or not np.isfinite(auc):
                corrupt_paths.append(
                    (str(result_path), f"auc_test out of [0,1] or non-finite: {auc}")
                )
                continue
            rows.setdefault(model_dir.name, {})[task_dir.name] = auc

    if corrupt_paths:
        print(
            f"[aggregate] WARN: {len(corrupt_paths)} corrupt/invalid result.json "
            f"files were skipped (first few):",
            flush=True,
        )
        for p, reason in corrupt_paths[:5]:
            print(f"  - {p}: {reason}", flush=True)

    # Always write meta JSON, even when no successful fits exist — the
    # skip / corrupt diagnostics are useful regardless. Axes are built
    # from the universe of (model, task) pairs that produced ANY
    # result.json (auc, skip, or corrupt); so a model with every task
    # class-gate-skipped still appears as a row of NaN cells, not as a
    # silently-missing row.
    from datetime import datetime, timezone
    model_ids = sorted(all_model_slugs)
    task_ids = sorted(all_task_slugs)
    auc_matrix = np.full((len(model_ids), len(task_ids)), np.nan, dtype=np.float64)
    for i, m_slug in enumerate(model_ids):
        for j, t_name in enumerate(task_ids):
            v = rows.get(m_slug, {}).get(t_name)
            if v is not None:
                auc_matrix[i, j] = v
    np.save(matrices_dir / "auc_matrix.npy", auc_matrix)

    meta = {
        "model_ids": [m.replace("__", "/") for m in model_ids],
        "task_ids": [t.replace("__", "/") for t in task_ids],
        "shape": list(auc_matrix.shape),
        "n_finite": int(np.isfinite(auc_matrix).sum()),
        "n_missing": int(np.isnan(auc_matrix).sum()),
        # Document that the matrix is built from EVERY result.json
        # under out_phase2/downstream/ regardless of this invocation's
        # filter scope. Consumers that need "results from this exact
        # run only" should look at per-pair result.json instead.
        "aggregate_scope": "all_existing_results",
        "command_hf_ids_filter": args.hf_ids or "",
        "command_tasks_filter": args.tasks or "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Provenance: corrupt result.json files that were skipped
        # during aggregation. n_corrupt > 0 should prompt manual review.
        "n_corrupt_skipped": len(corrupt_paths),
        "corrupt_examples": [
            {"path": p, "reason": r}
            for p, r in corrupt_paths[:10]
        ],
        # Deliberate-skip pairs (tiny / class_gate) — gate didn't admit
        # the data, NOT a pipeline error. Cell stays NaN; reason in
        # per-pair result.json.
        "n_deliberate_skips": len(deliberate_paths),
        "deliberate_skip_examples": [
            {"path": p, "reason": r}
            for p, r in deliberate_paths[:10]
        ],
        # Pipeline-error skips (load_fail / fit_fail) — embed parquet
        # was unreadable or the classifier crashed mid-fit. These cells
        # are bugs to investigate, not "skip by design"; separating
        # them from deliberate_skips lets the user grep for retries.
        "n_pipeline_errors": len(error_paths),
        "pipeline_error_examples": [
            {"path": p, "reason": r}
            for p, r in error_paths[:10]
        ],
        # Unknown/missing error_type — guard against silent under-
        # counting when a future error_type ships without aggregate
        # support, or when a legacy result.json lacks error_type
        # entirely. n_unknown_skips > 0 should prompt the user to
        # either teach aggregate the new type or fix the typo.
        "n_unknown_skips": len(unknown_skip_paths),
        "unknown_skip_examples": [
            {"path": p, "reason": r}
            for p, r in unknown_skip_paths[:10]
        ],
    }
    (matrices_dir / "auc_matrix_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    scope_note = ""
    if args.hf_ids or args.tasks:
        scopes_used = []
        if args.hf_ids:
            scopes_used.append("--hf-ids")
        if args.tasks:
            scopes_used.append("--tasks")
        scope_note = (
            "  [note] re-aggregated full out_phase2/downstream/ tree "
            "even though the per-pair fit was scoped by "
            + " + ".join(scopes_used)
        )
    summary_skips = (
        f"deliberate={meta['n_deliberate_skips']}, "
        f"pipeline_err={meta['n_pipeline_errors']}, "
        f"unknown={meta['n_unknown_skips']}, "
        f"corrupt={meta['n_corrupt_skipped']}"
    )
    if meta["n_unknown_skips"] > 0:
        print(
            f"[aggregate] WARN: {meta['n_unknown_skips']} skip records "
            f"have unknown/missing error_type — see meta.unknown_skip_examples",
            flush=True,
        )
    if rows:
        print(
            f"[aggregate] wrote auc_matrix.npy "
            f"({len(model_ids)} models × {len(task_ids)} tasks, "
            f"{meta['n_finite']} finite / {meta['n_missing']} missing; "
            f"skips: {summary_skips})"
            f"{scope_note}",
            flush=True,
        )
    else:
        print(
            f"[aggregate] no successful fits — wrote auc_matrix.npy "
            f"({len(model_ids)} models × {len(task_ids)} tasks, all NaN) "
            f"+ meta with {summary_skips}",
            flush=True,
        )


if __name__ == "__main__":
    main()
