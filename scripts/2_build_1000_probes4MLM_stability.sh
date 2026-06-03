#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 2 (data prep for the MLM representation-stability check): build a
# stratified 1,000-probe subset of the main panel for the k=1 vs k=6
# stride-PLL ablation (Fig S3). The subset keeps the same per-element
# composition as the full 10,000-probe panel so re-scoring all 56 MLM
# models at k=1 (the exact per-token PLL) stays tractable.
#   -> data/panels/MLM_k1ablation_1000_main_panel.parquet (+ manifest.json)
#
# CPU-only (pandas/numpy); needs data/panels/main_panel.parquet to exist
# (run scripts/0_build_10000_probes_dataset.sh first).
#
# Extra args are forwarded to the builder, e.g.
#   --n-subset 1000        subset size (default 1000)
#   --seed 42              sampling seed (default 42)
#   --panel / --out-parquet
#
# Usage:
#     bash scripts/2_build_1000_probes4MLM_stability.sh
#     bash scripts/2_build_1000_probes4MLM_stability.sh --n-subset 500 --seed 7
#
# Env overrides:
#     PY                python interpreter (default: env_python.base from
#                       env_paths.yaml)
#     GLMAP_ENV_CONFIG  env-paths YAML (default: env_paths.yaml)

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

echo "===================================================================="
echo "Building the 1,000-probe MLM stability subset"
echo "  Input : data/panels/main_panel.parquet"
echo "  Output: data/panels/MLM_k1ablation_1000_main_panel.parquet"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
exec "${PY}" scripts/MLM_representation_stability/build_k-stride-PPL_ablation_subset.py "$@"
