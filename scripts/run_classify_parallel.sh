#!/usr/bin/env bash
# ⚠ WARNING: This script contains hardcoded paths (e.g. micromamba envs,
# Fan out scripts/run_downstream_classify.py across N parallel workers,
# splitting the 123-model audit set round-robin via --hf-ids.
#
# Strategy:
#   1. Read all model hf_ids from data/audits/models.json.
#   2. Round-robin assign to N workers.
#   3. Each worker runs with --n-jobs 8 (= natural cap, since
#      cross_val_score parallelises 5 CV folds and binary tasks use
#      liblinear which is single-threaded — extra n_jobs idle).
#   4. BLAS threads forced to 1 via OMP/OPENBLAS/MKL env to prevent
#      the thread thrashing that hung the single-worker run.
#   5. result.json is per-cell atomic and cache-resume — workers do
#      not collide on individual (model, task) writes.
#   6. The final aggregate auc_matrix.npy IS race-prone (8 workers
#      may write at similar times). Worst case: file gets corrupted
#      → run one more `run_downstream_classify.py` invocation with
#      no filter; it will skip every cached cell and rebuild the
#      matrix in <1 min.
#
# Usage:
#     bash scripts/run_classify_parallel.sh
#     N_WORKERS=12 bash scripts/run_classify_parallel.sh
#     N_JOBS_PER_WORKER=4 bash scripts/run_classify_parallel.sh
#
# Env overrides:
#     PY               python interpreter (default /nvme-data3/yusen/micomamba/bin/python)
#     N_WORKERS        parallel classify processes (default 8; safe on 80-free-core box)
#     N_JOBS_PER_WORKER  sklearn n_jobs per worker (default 8; >5 wasted since cv=5)
#     LOG_DIR          per-worker stdout/stderr dir (default scripts/logs/classify/parallel_<ts>)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-/nvme-data3/yusen/micomamba/bin/python}"
N_WORKERS="${N_WORKERS:-8}"
N_JOBS_PER_WORKER="${N_JOBS_PER_WORKER:-8}"
LOG_DIR="${LOG_DIR:-scripts/logs/classify/parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "[parallel-classify] reading audit..."
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
    echo "[parallel-classify] no models in audit — abort."
    exit 1
fi
echo "[parallel-classify] ${TOTAL} total models -> ${N_WORKERS} workers (~$(( (TOTAL + N_WORKERS - 1) / N_WORKERS )) models/worker)"
echo "[parallel-classify] per-worker --n-jobs ${N_JOBS_PER_WORKER}, BLAS threads = 1"
echo "[parallel-classify] log dir: ${LOG_DIR}"
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
        "${PY}" scripts/run_downstream_classify.py \
            --hf-ids "${chunk}" \
            --n-jobs "${N_JOBS_PER_WORKER}" \
        > "${log}" 2>&1 &
    PIDS+=($!)
    LABELS+=("worker${w}(${n_models} models)")
done

echo ""
echo "[parallel-classify] ${#PIDS[@]} workers launched, PIDs: ${PIDS[*]}"
echo "[parallel-classify] tail -f ${LOG_DIR}/worker0.log  to watch progress"
echo "[parallel-classify] waiting..."

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
echo "[parallel-classify] all workers finished. ${fail} failures."
echo "[parallel-classify] running ONE final aggregate pass to ensure clean auc_matrix.npy..."
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PY}" scripts/run_downstream_classify.py --n-jobs "${N_JOBS_PER_WORKER}" \
    > "${LOG_DIR}/final_aggregate.log" 2>&1

if [[ $? -eq 0 ]]; then
    echo "[parallel-classify] final aggregate ok"
else
    echo "[parallel-classify] WARN: final aggregate exit != 0; check ${LOG_DIR}/final_aggregate.log"
fi

echo ""
echo "[parallel-classify] summary"
echo "  result.json files now : $(find out_phase2/downstream -name 'result.json' 2>/dev/null | wc -l)"
echo "  auc_matrix.npy        : $(ls -la out_phase2/matrices/auc_matrix.npy 2>/dev/null || echo MISSING)"
echo "  log dir               : ${LOG_DIR}"

exit ${fail}
