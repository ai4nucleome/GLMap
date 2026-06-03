#!/usr/bin/env bash
# ⚠ Machine-specific paths: micromamba env locations live in env_paths.yaml
# (override with $GLMAP_ENV_CONFIG); GPU ids in models/env_routing.md. Adjust
# those for a new machine before running.
#
# Launch the MLM representation-stability ablation: re-score ALL 56 MLM
# models on a 1000-probe stratified subset of the panel with the EXACT
# leave-one-out PLL (--method exact), so it can be compared against the
# primary k=6 stride PLL.
#
#   ALL 56 MLM models × 1000-probe stratified subset × exact PLL
#     → results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples/MLM_true-PLL_scores/<slug>/probes.parquet
#     → ~5-7 h on 8 GPUs
#
# Downstream analysis: per-model Pearson r between the exact (k=1) and the
# k=6 stride sum_log_p vectors (Fig S2 / Fig S3).
#
# Usage:
#   bash scripts/3_MLM_exactPLL_stridePLL_ablation.sh             # run the exact-PLL ablation
#   bash scripts/3_MLM_exactPLL_stridePLL_ablation.sh --dry-run   # show routing, don't run
#
# Default runs the exact PLL (METHOD=exact). For a k-stride sensitivity
# sweep instead:
#   METHOD=stride STRIDE=4 bash scripts/3_MLM_exactPLL_stridePLL_ablation.sh   # k=4 stride PLL
#
# Override GPU pool:
#   GPU_IDS=4,5,6,7 bash scripts/3_MLM_exactPLL_stridePLL_ablation.sh

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
METHOD="${METHOD:-exact}"      # 'exact' = true leave-one-out PLL; 'stride' for k-sensitivity
STRIDE="${STRIDE:-6}"          # only used when METHOD=stride
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

# Translate METHOD/STRIDE into scoring-worker args.
if [[ "${METHOD}" == "exact" ]]; then
    METHOD_ARGS=(--method exact)
    run_desc="exact leave-one-out PLL"
else
    METHOD_ARGS=(--method stride --stride "${STRIDE}")
    run_desc="stride PLL k=${STRIDE}"
fi

# Pre-built 1000-probe stratified subset.
SUBSET_PANEL="${REPO_ROOT}/data/panels/MLM_k1ablation_1000_main_panel.parquet"
if [[ ! -f "${SUBSET_PANEL}" ]]; then
    echo "subset panel missing — run:"
    echo "  ${PY} scripts/MLM_representation_stability/build_k-stride-PPL_ablation_subset.py"
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
            echo "usage: bash scripts/3_MLM_exactPLL_stridePLL_ablation.sh [--dry-run]"
            exit 1
            ;;
    esac
done

ts="$(date +%Y%m%d_%H%M%S)"
log_dir="scripts/logs/MLM_k1ablation_1000_${ts}"
mkdir -p "${log_dir}"

echo ""
echo "===================================================================="
echo "MLM stability ablation — 56 MLM × 1000 probes × ${run_desc}"
echo "  Panel : ${SUBSET_PANEL#${REPO_ROOT}/}"
echo "  Output: results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples/"
echo "  Logs  : ${log_dir}"
echo "  GPUs  : ${GPU_IDS}"
echo "  Dry   : ${DRY_RUN_ARGS[*]:-(actual run)}"
echo "===================================================================="
"${PY}" scripts/score/run_scoring_sweep.py \
    --branch mlm \
    --panel "${SUBSET_PANEL}" \
    "${METHOD_ARGS[@]}" \
    --out results/analysis/MLM_stride-PLL_vs_true-PLL_1000samples \
    --scores-subdir MLM_true-PLL_scores \
    --gpu-ids "${GPU_IDS}" \
    --log-dir "${log_dir}" \
    "${DRY_RUN_ARGS[@]}"
