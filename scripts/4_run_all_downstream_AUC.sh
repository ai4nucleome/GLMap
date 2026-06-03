#!/usr/bin/env bash
# ⚠ Machine-specific paths: the per-family micromamba env locations live in
# env_paths.yaml (override with $GLMAP_ENV_CONFIG); GPU ids /
# runtime knobs are documented in models/env_routing.md. Adjust those for a
# new machine before running.
#
# Step 4 (downstream eval): the supervised-probe benchmark that GLMap is
# validated against. Two stages produce the entire downstream AUC tree:
#
#   1. Pooled-embedding extraction (GPU) across the 123-model audit roster
#        scripts/downstream_tasks/run_embed_sweep.py
#      → results/analysis/embeddings/<slug>/<task>/{train,test}.parquet
#
#   2. Linear-probe AUC (CPU; L2 logistic regression, 5-fold C-grid CV)
#        scripts/downstream_tasks/run_classify_parallel.sh
#      → benchmark_perform_prediction/per_model_AUC_result_6tasks/<slug>/<task>/result.json
#      → benchmark_perform_prediction/all_model_AUC_6tasks/auc_matrix.npy   (123 × 6)
#
# Needs: multiple micromamba envs + GPUs (see models/env_routing.md, stage 1
# only), the model weights, and the 6 downstream benchmark datasets (see
# data/README.md). Stage 2 is CPU-only (sklearn).
#
# Usage:
#     bash scripts/4_run_all_downstream_AUC.sh                    # full run
#     bash scripts/4_run_all_downstream_AUC.sh --gpu-ids 0,5,6,7  # pick GPUs (stage 1)
#     bash scripts/4_run_all_downstream_AUC.sh --only evo         # subset (embed sweep only)
#     bash scripts/4_run_all_downstream_AUC.sh --dry-run          # show routing, no run
#
# Any extra args are forwarded to the embedding sweep. With --dry-run the
# classify stage is skipped. Stage-2 parallelism is set via N_WORKERS (see
# run_classify_parallel.sh), not via these args.
#
# Env overrides:
#     PY                python interpreter (default: env_python.base from the
#                       env-paths config below)
#     GLMAP_ENV_CONFIG  path to the micromamba env-paths YAML
#                       (default: env_paths.yaml)
#     N_WORKERS         stage-2 parallel classify processes (default 8)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GLMAP_ENV_CONFIG="${GLMAP_ENV_CONFIG:-${REPO_ROOT}/env_paths.yaml}"
PY="${PY:-$(sed -n 's/^[[:space:]]*base:[[:space:]]*//p' "${GLMAP_ENV_CONFIG}" 2>/dev/null | head -1)}"
if [[ -z "${PY}" ]]; then
    echo "error: could not read env_python.base from ${GLMAP_ENV_CONFIG}" >&2
    echo "       set \$PY, or fix the config / \$GLMAP_ENV_CONFIG." >&2
    exit 1
fi

# Detect --dry-run among forwarded args so we can skip the classify stage.
DRY_RUN=0
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

echo "===================================================================="
echo "Step 1/2 — pooled-embedding sweep (123 models × 6 downstream tasks)"
echo "  Output: results/analysis/embeddings/<slug>/<task>/{train,test}.parquet"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
"${PY}" scripts/downstream_tasks/run_embed_sweep.py "$@"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo ""
    echo "[4_run_all] --dry-run: skipping the classify stage."
    exit 0
fi

echo ""
echo "===================================================================="
echo "Step 2/2 — linear-probe AUC over every (model, task) pair (CPU)"
echo "  Output: results/analysis/benchmark_perform_prediction/"
echo "===================================================================="
bash scripts/downstream_tasks/run_classify_parallel.sh

echo ""
echo "[4_run_all] done. downstream AUC built (embeddings + per_model_AUC + auc_matrix)."
