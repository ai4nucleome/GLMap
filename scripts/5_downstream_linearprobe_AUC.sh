#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 5 (downstream eval, stage 2 of 2): linear-probe AUC. Consumes the
# pooled embeddings written by step 4 and, for every (model, task) pair, fits
# an L2 logistic-regression probe (StandardScaler + 5-fold C-grid CV on train)
# and scores ROC-AUC on test:
#
#   → benchmark_perform_prediction/per_model_AUC_result_6tasks/<slug>/<task>/result.json
#   → benchmark_perform_prediction/all_model_AUC_6tasks/auc_matrix.npy   (123 × 6)
#
# CPU-only (sklearn). Cheap and re-runnable on its own: result.json is
# per-cell atomic + cache-resume, so this can be repeated after a C-grid
# change or to rebuild a race-corrupted auc_matrix.npy without redoing the
# GPU embedding sweep (step 4). Requires step 4 to have populated
# results/analysis/embeddings/ first.
#
# How it parallelises (scripts/downstream_tasks/run_downstream_classify.py is
# the per-(model,task) worker; this script just fans it out):
#   1. Read all model hf_ids from data/audits/models.json.
#   2. Round-robin assign to N_WORKERS processes via --hf-ids.
#   3. Each worker runs with --n-jobs N_JOBS_PER_WORKER (natural cap: a binary
#      task's cross_val_score parallelises 5 CV folds over single-threaded
#      liblinear, so extra jobs idle).
#   4. BLAS threads forced to 1 (OMP/OPENBLAS/MKL) to stop the thread
#      thrashing that hung the single-worker run.
#   5. result.json is per-cell atomic + cache-resume — workers never collide
#      on individual (model, task) writes.
#   6. The aggregate auc_matrix.npy IS race-prone (workers may write at
#      similar times), so a final single-process pass rebuilds it cleanly
#      from the cached cells (<1 min, skips every finished pair).
#
# Usage:
#     bash scripts/5_downstream_linearprobe_AUC.sh
#     N_WORKERS=12 bash scripts/5_downstream_linearprobe_AUC.sh
#     N_JOBS_PER_WORKER=4 bash scripts/5_downstream_linearprobe_AUC.sh
#
# Env overrides:
#     PY                 python interpreter (default: env_python.base from the
#                        env-paths config below)
#     GLMAP_ENV_CONFIG   env-paths YAML (default: env_paths.yaml)
#     N_WORKERS          parallel classify processes (default 8; safe on an
#                        80-free-core box)
#     N_JOBS_PER_WORKER  sklearn n_jobs per worker (default 8; >5 wasted, cv=5)
#     LOG_DIR            per-worker stdout/stderr dir
#                        (default scripts/logs/classify/parallel_<ts>)

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
N_WORKERS="${N_WORKERS:-8}"
N_JOBS_PER_WORKER="${N_JOBS_PER_WORKER:-8}"
LOG_DIR="${LOG_DIR:-scripts/logs/classify/parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "===================================================================="
echo "Step 5 — linear-probe AUC over every (model, task) pair (CPU)"
echo "  Input : results/analysis/embeddings/   (from step 4)"
echo "  Output: results/analysis/benchmark_perform_prediction/"
echo "===================================================================="
echo "[classify] reading audit..."
mapfile -t ALL_MODELS < <(
    "${PY}" - <<'PYLIST'
import json
from pathlib import Path
audit = json.loads(Path("data/audits/models.json").read_text())["models"]
for m in audit:
    print(m["hf_id"])
PYLIST
)

TOTAL=${#ALL_MODELS[@]}
if [[ "${TOTAL}" -eq 0 ]]; then
    echo "[classify] no models in audit — abort."
    exit 1
fi
echo "[classify] ${TOTAL} total models -> ${N_WORKERS} workers (~$(( (TOTAL + N_WORKERS - 1) / N_WORKERS )) models/worker)"
echo "[classify] per-worker --n-jobs ${N_JOBS_PER_WORKER}, BLAS threads = 1"
echo "[classify] log dir: ${LOG_DIR}"
echo ""

# Round-robin split into N chunks. Bash arrays-of-arrays are awkward;
# we keep N parallel comma-joined strings.
declare -a CHUNKS
for w in $(seq 0 $((N_WORKERS - 1))); do CHUNKS[$w]=""; done
for i in "${!ALL_MODELS[@]}"; do
    w=$(( i % N_WORKERS ))
    if [[ -z "${CHUNKS[$w]}" ]]; then
        CHUNKS[$w]="${ALL_MODELS[$i]}"
    else
        CHUNKS[$w]="${CHUNKS[$w]},${ALL_MODELS[$i]}"
    fi
done

declare -a PIDS=()
declare -a LABELS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
    chunk="${CHUNKS[$w]}"
    n_models=$(echo "${chunk}" | tr ',' '\n' | wc -l)
    log="${LOG_DIR}/worker${w}.log"
    echo "[worker ${w}] ${n_models} models -> ${log}"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        "${PY}" scripts/downstream_tasks/run_downstream_classify.py \
            --hf-ids "${chunk}" \
            --n-jobs "${N_JOBS_PER_WORKER}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
    LABELS+=("worker${w}(${n_models} models)")
done

echo ""
echo "[classify] ${#PIDS[@]} workers launched, PIDs: ${PIDS[*]}"
echo "[classify] tail -f ${LOG_DIR}/worker0.log  to watch progress"
echo "[classify] waiting..."

fail=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ok    ${LABELS[$i]} (pid=${PIDS[$i]})"
    else
        rc=$?
        echo "  FAIL  ${LABELS[$i]} (pid=${PIDS[$i]} exit=${rc})"
        fail=$((fail+1))
    fi
done

echo ""
echo "[classify] all workers finished. ${fail} failures."
echo "[classify] running ONE final aggregate pass to ensure clean auc_matrix.npy..."
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PY}" scripts/downstream_tasks/run_downstream_classify.py --n-jobs "${N_JOBS_PER_WORKER}" \
    > "${LOG_DIR}/final_aggregate.log" 2>&1

if [[ $? -eq 0 ]]; then
    echo "[classify] final aggregate ok"
else
    echo "[classify] WARN: final aggregate exit != 0; check ${LOG_DIR}/final_aggregate.log"
fi

echo ""
echo "[classify] summary"
echo "  result.json files now : $(find results/analysis/benchmark_perform_prediction/per_model_AUC_result_6tasks -name 'result.json' 2>/dev/null | wc -l)"
echo "  auc_matrix.npy        : $(ls -la results/analysis/benchmark_perform_prediction/all_model_AUC_6tasks/auc_matrix.npy 2>/dev/null || echo MISSING)"
echo "  log dir               : ${LOG_DIR}"

exit ${fail}
