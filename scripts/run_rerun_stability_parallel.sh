#!/usr/bin/env bash
# ⚠ WARNING: This script contains hardcoded paths (e.g. micromamba envs,
# Fan out scripts/run_rerun_stability.py across 8 GPUs for all 123 audited
# models (skipping those already done under out_phase1/stability/).
#
# Strategy:
#   1. Build the list of (still-undone) model hf_ids from data/audits/models.json.
#   2. Distribute round-robin across cuda:0..7 so model-size variation
#      (1M..40B params) is roughly balanced per GPU instead of piling all
#      the giants on cuda:7.
#   3. Launch 8 background python workers, each handling its own chunk
#      via `--hf-ids <comma_list>` on its assigned device.
#   4. After all 8 workers finish, print an aggregated summary.
#
# Each per-model JSON lands in out_phase1/stability/<slug>.json (resume-safe:
# pre-existing JSONs are skipped via --skip-done).
#
# Usage:
#     bash scripts/run_rerun_stability_parallel.sh
#     N_PROBES=20 bash scripts/run_rerun_stability_parallel.sh
#     N_GPUS=4 bash scripts/run_rerun_stability_parallel.sh
#
# Env overrides:
#     PY        python interpreter (default /nvme-data3/yusen/micomamba/bin/python)
#     LOG_DIR   per-worker stderr log directory (default /tmp/rerun_stability_logs)
#     N_PROBES  probes per model (default 10)
#     N_GPUS    number of GPUs to fan out across (default 8)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-/nvme-data3/yusen/micomamba/bin/python}"
LOG_DIR="${LOG_DIR:-/tmp/rerun_stability_logs}"
N_PROBES="${N_PROBES:-10}"
N_GPUS="${N_GPUS:-8}"
mkdir -p "${LOG_DIR}"

echo "[rerun_stability] gathering undone models from audit..."
mapfile -t REMAINING < <(
    "${PY}" - <<'PYLIST'
import json
from pathlib import Path
audit = json.loads(Path("data/audits/models.json").read_text())["models"]
done = {
    json.loads(p.read_text()).get("hf_id")
    for p in Path("out_phase1/stability").glob("*.json")
}
SUPERVISED = {"supervised_or_annotation"}
remaining = [
    m["hf_id"] for m in audit
    if m.get("branch") not in SUPERVISED
    and m["hf_id"] not in done
]
for hf_id in remaining:
    print(hf_id)
PYLIST
)

TOTAL=${#REMAINING[@]}
if [[ "${TOTAL}" -eq 0 ]]; then
    echo "[rerun_stability] nothing to do — all audit models already have stability JSONs."
    exit 0
fi
echo "[rerun_stability] ${TOTAL} models to process across ${N_GPUS} GPUs"
echo "                 ~$(( (TOTAL + N_GPUS - 1) / N_GPUS )) models per GPU"

declare -a CHUNK_0 CHUNK_1 CHUNK_2 CHUNK_3 CHUNK_4 CHUNK_5 CHUNK_6 CHUNK_7
for i in "${!REMAINING[@]}"; do
    gpu=$(( i % N_GPUS ))
    case "${gpu}" in
        0) CHUNK_0+=("${REMAINING[$i]}");;
        1) CHUNK_1+=("${REMAINING[$i]}");;
        2) CHUNK_2+=("${REMAINING[$i]}");;
        3) CHUNK_3+=("${REMAINING[$i]}");;
        4) CHUNK_4+=("${REMAINING[$i]}");;
        5) CHUNK_5+=("${REMAINING[$i]}");;
        6) CHUNK_6+=("${REMAINING[$i]}");;
        7) CHUNK_7+=("${REMAINING[$i]}");;
    esac
done

join_csv() {
    local IFS=,
    echo "$*"
}

declare -a pids=()
declare -a labels=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
    case "${gpu}" in
        0) chunk=("${CHUNK_0[@]}");;
        1) chunk=("${CHUNK_1[@]}");;
        2) chunk=("${CHUNK_2[@]}");;
        3) chunk=("${CHUNK_3[@]}");;
        4) chunk=("${CHUNK_4[@]}");;
        5) chunk=("${CHUNK_5[@]}");;
        6) chunk=("${CHUNK_6[@]}");;
        7) chunk=("${CHUNK_7[@]}");;
    esac
    if [[ "${#chunk[@]}" -eq 0 ]]; then
        echo "[gpu ${gpu}] no models assigned"
        continue
    fi
    ids_csv="$(join_csv "${chunk[@]}")"
    log="${LOG_DIR}/gpu${gpu}.log"
    echo "[gpu ${gpu}] launching ${#chunk[@]} models  log=${log}"
    "${PY}" scripts/run_rerun_stability.py \
        --device "cuda:${gpu}" \
        --hf-ids "${ids_csv}" \
        --n-probes "${N_PROBES}" \
        --skip-done \
        > "${log}" 2>&1 &
    pids+=($!)
    labels+=("cuda:${gpu}(${#chunk[@]} models)")
done

echo ""
echo "[rerun_stability] ${#pids[@]} workers running (PIDs ${pids[*]})"
echo "[rerun_stability] waiting..."

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  ok    ${labels[$i]} (pid=${pids[$i]})"
    else
        rc=$?
        echo "  FAIL  ${labels[$i]} (pid=${pids[$i]} exit=${rc})"
        fail=$((fail+1))
    fi
done

echo ""
echo "=== Aggregated rerun-stability summary ==="
"${PY}" - <<'PYAGG'
import json
from pathlib import Path
stab = Path("out_phase1/stability")
rows = []
for p in sorted(stab.glob("*.json")):
    rows.append(json.loads(p.read_text()))

n_pass = sum(1 for r in rows if r.get("passes_gate"))
n_fail_or_err = sum(1 for r in rows if not r.get("passes_gate"))

print(f"Total stability JSONs: {len(rows)}")
print(f"  PASS (r >= 0.95): {n_pass}")
print(f"  FAIL / ERROR    : {n_fail_or_err}")
print()

print("--- Failures / errors ---")
fails = [r for r in rows if not r.get("passes_gate")]
if not fails:
    print("(none)")
for r in fails:
    if "error" in r:
        print(f"  {r.get('hf_id','?'):60s} ERROR: {r['error'][:80]}")
    else:
        print(f"  {r['hf_id']:60s} r={r.get('pearson_r','?'):.4f}  "
              f"max_diff={r.get('max_abs_diff','?'):.2e}")

print()
print(f"--- Slowest 10 passes ---")
passes = [r for r in rows if r.get("passes_gate")]
print(f"{'hf_id':60s} {'branch':4s} {'pearson_r':>10s} {'max_diff':>11s} {'elapsed_s':>10s}")
for r in sorted(passes, key=lambda x: x.get("elapsed_seconds", 0), reverse=True)[:10]:
    print(f"{r['hf_id']:60s} {r['branch']:4s} "
          f"{r['pearson_r']:>10.6f} {r['max_abs_diff']:>11.2e} "
          f"{r['elapsed_seconds']:>10.1f}")
PYAGG

echo ""
echo "[done] parallel rerun-stability sweep complete; per-model JSONs in out_phase1/stability/"
exit ${fail}
