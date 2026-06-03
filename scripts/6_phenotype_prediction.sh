#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 6 (Figure 4): does the 10,000-probe GLMap signature predict a model's
# real downstream performance? Two parts:
#
#   1. COMPUTE (CPU; RidgeCV out-of-fold, random K-fold + family GroupKFold)
#        scripts/downstream_tasks/run_phenotype_prediction.py
#      features X = per-model likelihood signature V / V_d  (results/scores/)
#      target   y = downstream AUC matrix                   (from step 5)
#      → results/analysis/benchmark_perform_prediction/phenotype_prediction/
#          {predictions,metrics_by_seed,metrics_summary}.csv + config.json
#
#   2. PLOT / TABULATE (read the cached CSVs above; no refit)
#        scripts/figures/fig4a_downstream_auc_distribution.py  → Fig4a (AUC dist)
#        scripts/figures/fig4b_phenotype_prediction_scatter.py → Fig4b (pred scatter)
#        scripts/tables/table4_phenotype_prediction_metrics.py → table4_*.tex
#
# CPU-only (sklearn / matplotlib). Needs results/scores/ (step 1) and the
# downstream auc_matrix.npy (step 5) to exist first.
#
# Usage:
#     bash scripts/6_phenotype_prediction.sh
#     bash scripts/6_phenotype_prediction.sh --seeds 0,1,2   # extra args → compute step
#
# Any extra args are forwarded to the compute step (see
# run_phenotype_prediction.py --help for --seeds / --n-splits / --alphas / …).
#
# Env overrides:
#     PY                python interpreter (default: env_python.base from the
#                       env-paths config below)
#     GLMAP_ENV_CONFIG  path to the micromamba env-paths YAML
#                       (default: env_paths.yaml)

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
echo "Step 6.1 — phenotype prediction (RidgeCV: GLMap signature → AUC)"
echo "  Output: results/analysis/benchmark_perform_prediction/phenotype_prediction/"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
"${PY}" scripts/downstream_tasks/run_phenotype_prediction.py "$@"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
    echo "[6_phenotype] compute step failed (rc=${rc}); skipping plots." >&2
    exit "${rc}"
fi

echo ""
echo "===================================================================="
echo "Step 6.2 — Figure 4 panels + Table 4"
echo "  Output: results/figures/Fig4{a,b}-*.pdf , results/tables/table4_*.tex"
echo "===================================================================="
"${PY}" scripts/figures/fig4a_downstream_auc_distribution.py
# fig4b's --out is fixed (not split-keyed), so name the two splits apart to
# avoid overwrite: kfold keeps the canonical paper filename, the stricter
# family-held-out variant gets a suffix.
"${PY}" scripts/figures/fig4b_phenotype_prediction_scatter.py --split kfold
"${PY}" scripts/figures/fig4b_phenotype_prediction_scatter.py --split family_groupkfold \
    --out results/figures/Fig4b-phenotype_prediction_scatter_family-groupkfold.pdf
"${PY}" scripts/tables/table4_phenotype_prediction_metrics.py

echo ""
echo "[6_phenotype] done. phenotype_prediction/ + Fig4a/Fig4b + table4 built."
