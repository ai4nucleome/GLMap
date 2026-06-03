#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 7 (Fig3): the GLMap model map — project the 123 models into 2D from
# their likelihood-response signatures and draw the three panels. Two parts:
#
#   1. COMPUTE (CPU; t-SNE on V / V_d, MDS on sqrt(D))
#        scripts/model_map/run_fig3_model_map_embedding.py
#      → results/analysis/model_map/fig3_embedding_{V_tsne,Vd_tsne,D_mds}.csv
#        + fig3_embedding_config.json   (cached coordinates)
#
#   2. PLOT (read the cached V_d t-SNE coordinates; no re-embed)
#        scripts/figures/fig3a_model_map_family.py        → Fig3a (by family)
#        scripts/figures/fig3b_model_map_model_weight.py  → Fig3b (by params)
#        scripts/figures/fig3c_model_map_6task_mean.py    → Fig3c (by 6-task AUC)
#
# CPU-only (sklearn / matplotlib). Runs LATE in the pipeline despite being
# Fig3: the embedding enriches each model with its downstream mean AUC (the
# Fig3c overlay), so it needs BOTH results/scores/ (step 1) AND the
# downstream auc_matrix.npy produced by step 6 (5_/6_ downstream eval).
#
# Usage:
#     bash scripts/7_model_map.sh
#     bash scripts/7_model_map.sh --perplexity 15   # extra args → compute step
#
# Any extra args are forwarded to the embedding step (see
# run_fig3_model_map_embedding.py --help for --perplexity / --seed / …).
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
echo "Step 7.1 — GLMap model-map embedding (t-SNE on V/V_d, MDS on D)"
echo "  Output: results/analysis/model_map/fig3_embedding_*.csv"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
"${PY}" scripts/model_map/run_fig3_model_map_embedding.py "$@"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
    echo "[7_model_map] embedding step failed (rc=${rc}); skipping plots." >&2
    exit "${rc}"
fi

echo ""
echo "===================================================================="
echo "Step 7.2 — Fig3 panels (family / model weight / 6-task mean)"
echo "  Output: results/figures/Fig3{a,b,c}-*.pdf"
echo "===================================================================="
"${PY}" scripts/figures/fig3a_model_map_family.py
"${PY}" scripts/figures/fig3b_model_map_model_weight.py
"${PY}" scripts/figures/fig3c_model_map_6task_mean.py

echo ""
echo "[7_model_map] done. model_map embeddings + Fig3a/Fig3b/Fig3c built."
