#!/usr/bin/env bash
# ⚠ Machine-specific paths: the per-family micromamba env locations live in
# env_paths.yaml (override with $GLMAP_ENV_CONFIG); GPU ids /
# runtime knobs are documented in models/env_routing.md. Adjust those for a
# new machine before running.
#
# Step 0 of the GLMap reproduction: score all AR + MLM models on the
# 10,000-probe panel and build the per-branch GLMap matrices. Produces the
# entire results/scores/ tree:
#
#   1. Parallel scoring sweep across the 123-model audit roster
#        scripts/score/run_scoring_sweep.py
#      → results/scores/AR_MLM_scores/<slug>/probes.parquet   (per-model)
#
#   2. CPU aggregate pass over every per-model parquet
#        scripts/score/scoring_worker.py --from-audit --strict-aggregate
#      → results/scores/matrices/{V,V_d,D}_{AR,MLM}.npy + matrix_metadata.json
#
# Needs: multiple micromamba envs + GPUs (see models/env_routing.md), the
# frozen panel at data/panels/main_panel.parquet, and the model weights
# (scripts/download_models/ + models/setup_external_models.sh).
#
# Usage:
#     bash scripts/1_run_all_AR_MLM_scoring.sh                    # full run
#     bash scripts/1_run_all_AR_MLM_scoring.sh --gpu-ids 0,5,6,7  # pick GPUs
#     bash scripts/1_run_all_AR_MLM_scoring.sh --only evo         # subset (sweep only)
#     bash scripts/1_run_all_AR_MLM_scoring.sh --dry-run          # show routing, no run
#
# Any extra args are forwarded to the scoring sweep. With --dry-run the
# aggregate step is skipped. The aggregate is --strict-aggregate, so it
# expects the FULL roster to be scored; use it only after a full sweep.
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

# Detect --dry-run among forwarded args so we can skip the aggregate.
DRY_RUN=0
for arg in "$@"; do
    [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

echo "===================================================================="
echo "Step 1/2 — parallel scoring sweep (123 models × 10,000 probes)"
echo "  Output: results/scores/AR_MLM_scores/<slug>/probes.parquet"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
"${PY}" scripts/score/run_scoring_sweep.py "$@"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo ""
    echo "[1_run_all] --dry-run: skipping the aggregate step."
    exit 0
fi

echo ""
echo "===================================================================="
echo "Step 2/2 — aggregate per-model parquets into V/V_d/D matrices (CPU)"
echo "  Output: results/scores/matrices/"
echo "===================================================================="
"${PY}" scripts/score/scoring_worker.py --from-audit --strict-aggregate

echo ""
echo "[1_run_all] done. results/scores/ is built (AR_MLM_scores + matrices)."
