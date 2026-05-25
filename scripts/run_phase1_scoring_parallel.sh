#!/usr/bin/env bash
# ⚠ WARNING: This script contains hardcoded paths (e.g. micromamba envs,
# Fan out run_phase1_scoring.py across multiple GPUs, one model per GPU.
#
# Each model runs with --skip-aggregate so the per-model probes.parquet
# checkpoints are written but no two jobs race on out_phase1/matrices/.
# After all parallel jobs finish, a single aggregate pass on cpu re-reads
# every parquet and writes the matrices + report.
#
# Usage:
#     bash scripts/run_phase1_scoring_parallel.sh
#
# Override default model->GPU mapping by editing WAVE_1 / WAVE_2 below.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-/nvme-data3/yusen/micomamba/bin/python}"
LOG_DIR="${LOG_DIR:-/tmp/phase1_logs}"
mkdir -p "${LOG_DIR}"

# Wave 1: 8 GPUs, 8 models. Resume semantics skips models whose parquet
# already exists (negligible startup cost).
WAVE_1=(
    "Mistral-DNA-v1-1M-hg38              cuda:0"
    "Mistral-DNA-v1-17M-hg38             cuda:1"
    "Mistral-DNA-v1-138M-hg38            cuda:2"
    "lingxusb/megaDNA                    cuda:3"
    "lingxusb/PlasmidGPT                 cuda:4"
    "nucleotide-transformer-v2-50m       cuda:5"
    "nucleotide-transformer-v2-100m      cuda:6"
    "nucleotide-transformer-v2-250m      cuda:7"
)

# Wave 2: after wave 1 frees GPUs, reuse cuda:0 / cuda:1 for the two
# bigger MLM additions.
WAVE_2=(
    "nucleotide-transformer-v2-500m      cuda:0"
    "agro-nucleotide-transformer-1b      cuda:1"
)

run_wave() {
    local wave_name="$1"
    shift
    local -a jobs=("$@")
    declare -a pids=()
    declare -a names=()
    for line in "${jobs[@]}"; do
        read -r hf_filter gpu <<< "${line}"
        safe_name="$(echo "${hf_filter}" | tr '/' '_')"
        log="${LOG_DIR}/${safe_name}.log"
        echo "[${wave_name}] launch ${hf_filter}  on ${gpu}  log=${log}"
        "${PY}" scripts/run_phase1_scoring.py \
            --device "${gpu}" \
            --only "${hf_filter}" \
            --skip-aggregate \
            > "${log}" 2>&1 &
        pids+=($!)
        names+=("${hf_filter}@${gpu}")
    done
    echo "[${wave_name}] ${#pids[@]} jobs running (PIDs ${pids[*]})"
    local fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[${wave_name}] ok    ${names[$i]} (pid=${pids[$i]})"
        else
            rc=$?
            echo "[${wave_name}] FAIL  ${names[$i]} (pid=${pids[$i]} exit=${rc})"
            safe_name="$(echo "${names[$i]%@*}" | tr '/' '_')"
            echo "       tail of ${LOG_DIR}/${safe_name}.log:"
            tail -10 "${LOG_DIR}/${safe_name}.log" | sed 's/^/         /'
            fail=$((fail+1))
        fi
    done
    return ${fail}
}

run_wave "wave1" "${WAVE_1[@]}"
fail1=$?
if [[ ${fail1} -gt 0 ]]; then
    echo ""
    echo "[abort] wave 1 had ${fail1} failures; not launching wave 2."
    exit 1
fi

if [[ ${#WAVE_2[@]} -gt 0 ]]; then
    echo ""
    run_wave "wave2" "${WAVE_2[@]}"
    fail2=$?
    if [[ ${fail2} -gt 0 ]]; then
        echo ""
        echo "[abort] wave 2 had ${fail2} failures; matrices NOT aggregated."
        exit 1
    fi
fi

echo ""
echo "[aggregate] all per-model scoring done; running matrix build + report"
"${PY}" scripts/run_phase1_scoring.py --device cpu 2>&1 | tail -20
echo ""
echo "[done] parallel phase-1 scoring complete; outputs in out_phase1/"
