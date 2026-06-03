#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 5 (downstream eval, stage 2 of 2): linear-probe AUC. Consumes the
# pooled embeddings written by step 4 and, for every (model, task) pair, fits
# an L2 logistic-regression probe (StandardScaler + 5-fold C-grid CV on train)
# and scores ROC-AUC on test:
#
#   scripts/downstream_tasks/run_classify_parallel.sh
#   → benchmark_perform_prediction/per_model_AUC_result_6tasks/<slug>/<task>/result.json
#   → benchmark_perform_prediction/all_model_AUC_6tasks/auc_matrix.npy   (123 × 6)
#
# CPU-only (sklearn). Cheap and re-runnable on its own: result.json is
# per-cell atomic + cache-resume, so this can be repeated after a C-grid
# change or to rebuild a race-corrupted auc_matrix.npy without redoing the
# GPU embedding sweep (step 4). Requires step 4 to have populated
# results/analysis/embeddings/ first.
#
# This is a thin numbered entry over the parallel fan-out worker; tune
# parallelism with N_WORKERS (see run_classify_parallel.sh for all knobs).
#
# Usage:
#     bash scripts/5_downstream_linearprobe_AUC.sh
#     N_WORKERS=12 bash scripts/5_downstream_linearprobe_AUC.sh
#
# Env overrides:
#     PY                python interpreter (default: env_python.base from the
#                       env-paths config below)
#     GLMAP_ENV_CONFIG  path to the micromamba env-paths YAML
#                       (default: env_paths.yaml)
#     N_WORKERS         parallel classify processes (default 8)
#     N_JOBS_PER_WORKER sklearn n_jobs per worker (default 8)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "===================================================================="
echo "Step 5 — linear-probe AUC over every (model, task) pair (CPU)"
echo "  Input : results/analysis/embeddings/   (from step 4)"
echo "  Output: results/analysis/benchmark_perform_prediction/"
echo "===================================================================="
bash scripts/downstream_tasks/run_classify_parallel.sh
