#!/usr/bin/env bash
# ⚠ WARNING: This script contains hardcoded paths (e.g. micromamba envs,
# Launch the MLM stride-PLL k=1 ablation experiments.
#
# Two experiments, separately launched:
#   A. ALL 56 MLM models × 1000-probe stratified subset × k=1
#      → out_phase1/MLM_k1ablation_1000_scores/<slug>/probes.parquet
#      → ~5-7 h on 8 GPUs
#   B. 10 representative MLM models × full 10000-probe panel × k=1
#      → out_phase1/MLM_k1_ablation_full_scores/<slug>/probes.parquet
#      → ~16-30 h on 8 GPUs
#
# Both experiments share the same down-the-line analysis (per-model
# Pearson r between k=1 and k=6 sum_log_p vectors, paper.md Fig S2a/b).
#
# Usage:
#   bash scripts/run_kmer_ablation.sh A         # launch experiment A only
#   bash scripts/run_kmer_ablation.sh B         # launch experiment B only
#   bash scripts/run_kmer_ablation.sh both      # A then B (B starts after A done)
#   bash scripts/run_kmer_ablation.sh --dry-run A
#   bash scripts/run_kmer_ablation.sh --dry-run B
#
# Override the stride for sensitivity sweeps (default --stride 1):
#   STRIDE=4 bash scripts/run_kmer_ablation.sh A      # k=4 ablation
#
# Override GPU pool:
#   GPU_IDS=4,5,6,7 bash scripts/run_kmer_ablation.sh A

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PY="${PY:-/nvme-data3/yusen/micomamba/bin/python}"
STRIDE="${STRIDE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# Pre-built 1000-probe subset for experiment A.
SUBSET_PANEL="${REPO_ROOT}/out_panel/MLM_k1ablation_1000_main_panel.parquet"
if [[ ! -f "${SUBSET_PANEL}" ]]; then
    echo "subset panel missing — run:"
    echo "  ${PY} scripts/build_k-stride-PPL_ablation_subset.py"
    exit 1
fi

# 10 representative MLM models for experiment B (one per family).
# Covers: NT v2 small + large, NTv3, DNABERT-2, GENA-LM, Caduceus,
# PlantCAD2, GROVER, MutBERT, AIDO.DNA.
EXP_B_MODELS=(
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
    "InstaDeepAI/NTv3_650M_pre"
    "zhihan1996/DNABERT-2-117M"
    "AIRI-Institute/gena-lm-bert-base"
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    "kuleshov-group/PlantCAD2-Medium-l48-d1024"
    "PoetschLab/GROVER"
    "JadenLong/MutBERT"
    "genbio-ai/AIDO.DNA-300M"
)
EXP_B_IDS_CSV="$(IFS=','; echo "${EXP_B_MODELS[*]}")"

# Parse flags / argv.
# DRY_RUN_ARGS holds either an empty array or `(--dry-run)`; expanded
# below as `"${DRY_RUN_ARGS[@]}"` so unset/empty yields zero arguments
# (no unquoted-variable splitting hazard).
DRY_RUN_ARGS=()
EXPERIMENT=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN_ARGS=(--dry-run) ;;
        A|B|both) EXPERIMENT="${arg}" ;;
        *)
            echo "unknown arg: ${arg}"
            echo "usage: bash scripts/run_kmer_ablation.sh [--dry-run] {A|B|both}"
            exit 1
            ;;
    esac
done
if [[ -z "${EXPERIMENT}" ]]; then
    echo "must specify experiment: A | B | both"
    echo "usage: bash scripts/run_kmer_ablation.sh [--dry-run] {A|B|both}"
    exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"

# ─────────────────────────── Experiment A ─────────────────────────── #
run_A() {
    local log_dir="scripts/logs/MLM_k1ablation_1000_${ts}"
    mkdir -p "${log_dir}"
    echo ""
    echo "===================================================================="
    echo "Experiment A — 56 MLM × 1000 probes × stride=${STRIDE}"
    echo "  Panel : ${SUBSET_PANEL#${REPO_ROOT}/}"
    echo "  Output: out_phase1/MLM_k1ablation_1000_scores/"
    echo "  Logs  : ${log_dir}"
    echo "  GPUs  : ${GPU_IDS}"
    echo "  Dry   : ${DRY_RUN_ARGS[*]:-(actual run)}"
    echo "===================================================================="
    "${PY}" scripts/run_sweep.py \
        --mode scoring \
        --branch mlm \
        --panel "${SUBSET_PANEL}" \
        --stride "${STRIDE}" \
        --out out_phase1/MLM_k1ablation_1000_scores \
        --gpu-ids "${GPU_IDS}" \
        --log-dir "${log_dir}" \
        "${DRY_RUN_ARGS[@]}"
}

# ─────────────────────────── Experiment B ─────────────────────────── #
run_B() {
    local log_dir="scripts/logs/MLM_k1_ablation_full_${ts}"
    mkdir -p "${log_dir}"
    echo ""
    echo "===================================================================="
    echo "Experiment B — 10 representative MLM × full 10000 panel × stride=${STRIDE}"
    echo "  Panel : out_panel/main_panel.parquet (default)"
    echo "  Output: out_phase1/MLM_k1_ablation_full_scores/"
    echo "  Logs  : ${log_dir}"
    echo "  GPUs  : ${GPU_IDS}"
    echo "  Models:"
    for m in "${EXP_B_MODELS[@]}"; do echo "    - ${m}"; done
    echo "  Dry   : ${DRY_RUN_ARGS[*]:-(actual run)}"
    echo "===================================================================="
    "${PY}" scripts/run_sweep.py \
        --mode scoring \
        --hf-ids "${EXP_B_IDS_CSV}" \
        --stride "${STRIDE}" \
        --out out_phase1/MLM_k1_ablation_full_scores \
        --gpu-ids "${GPU_IDS}" \
        --log-dir "${log_dir}" \
        "${DRY_RUN_ARGS[@]}"
}

case "${EXPERIMENT}" in
    A)   run_A ;;
    B)   run_B ;;
    both)
        run_A
        echo ""
        echo "[kmer-ablation] Experiment A finished; launching B..."
        run_B
        ;;
esac
