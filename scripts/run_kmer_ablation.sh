#!/usr/bin/env bash
# ⚠ WARNING: This script contains hardcoded paths (e.g. micromamba envs,
# GPU ids). Review before running on another machine.
#
# Launch the MLM stride-PLL k=1 ablation: re-score ALL 56 MLM models on a
# 1000-probe stratified subset of the panel at stride k=1 (the exact
# per-token PLL), so it can be compared against the primary k=6 stride PLL.
#
#   ALL 56 MLM models × 1000-probe stratified subset × k=1
#     → results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples/MLM_true-PLL_scores/<slug>/probes.parquet
#     → ~5-7 h on 8 GPUs
#
# Downstream analysis: per-model Pearson r between the k=1 and k=6
# sum_log_p vectors (Fig S2 / Fig S3).
#
# Usage:
#   bash scripts/run_kmer_ablation.sh             # run the ablation
#   bash scripts/run_kmer_ablation.sh --dry-run   # show routing, don't run
#
# Override the stride for sensitivity sweeps (default --stride 1):
#   STRIDE=4 bash scripts/run_kmer_ablation.sh      # k=4 ablation
#
# Override GPU pool:
#   GPU_IDS=4,5,6,7 bash scripts/run_kmer_ablation.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GLMAP_ENV_CONFIG="${GLMAP_ENV_CONFIG:-${REPO_ROOT}/env_paths.yaml}"
_cfg_base="$(sed -n 's/^[[:space:]]*base:[[:space:]]*//p' "${GLMAP_ENV_CONFIG}" 2>/dev/null | head -1)"
PY="${PY:-${_cfg_base:-/nvme-data3/yusen/micomamba/bin/python}}"
STRIDE="${STRIDE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# Pre-built 1000-probe stratified subset.
SUBSET_PANEL="${REPO_ROOT}/data/panels/MLM_k1ablation_1000_main_panel.parquet"
if [[ ! -f "${SUBSET_PANEL}" ]]; then
    echo "subset panel missing — run:"
    echo "  ${PY} scripts/build_k-stride-PPL_ablation_subset.py"
    exit 1
fi

# Parse flags. DRY_RUN_ARGS holds either an empty array or `(--dry-run)`;
# expanded below as `"${DRY_RUN_ARGS[@]}"` so unset/empty yields zero args.
DRY_RUN_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN_ARGS=(--dry-run) ;;
        *)
            echo "unknown arg: ${arg}"
            echo "usage: bash scripts/run_kmer_ablation.sh [--dry-run]"
            exit 1
            ;;
    esac
done

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="scripts/logs/MLM_k1ablation_1000_${ts}"
mkdir -p "${log_dir}"

echo ""
echo "===================================================================="
echo "MLM stride-PLL k=1 ablation — 56 MLM × 1000 probes × stride=${STRIDE}"
echo "  Panel : ${SUBSET_PANEL#${REPO_ROOT}/}"
echo "  Output: results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples/"
echo "  Logs  : ${log_dir}"
echo "  GPUs  : ${GPU_IDS}"
echo "  Dry   : ${DRY_RUN_ARGS[*]:-(actual run)}"
echo "===================================================================="
"${PY}" scripts/score/run_scoring_sweep.py \
    --branch mlm \
    --panel "${SUBSET_PANEL}" \
    --stride "${STRIDE}" \
    --out results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples \
    --scores-subdir MLM_true-PLL_scores \
    --gpu-ids "${GPU_IDS}" \
    --log-dir "${log_dir}" \
    "${DRY_RUN_ARGS[@]}"
