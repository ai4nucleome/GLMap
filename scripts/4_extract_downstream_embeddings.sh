#!/usr/bin/env bash
# ⚠ Machine-specific paths: the per-family micromamba env locations live in
# env_paths.yaml (override with $GLMAP_ENV_CONFIG); GPU ids /
# runtime knobs are documented in models/env_routing.md. Adjust those for a
# new machine before running.
#
# Step 4 (downstream eval, stage 1 of 2): pooled-embedding extraction.
# Dispatch every model in the 123-model audit roster across the GPU pool and,
# for each of the 6 downstream tasks, mean-pool the model's last hidden state
# over content tokens into one vector per sequence:
#
#   scripts/downstream_tasks/run_embed_sweep.py
#   → results/analysis/embeddings/<slug>/<task>/{train,test}.parquet
#                                  (columns embed_0..embed_{D-1}, label)
#
# Stage 2 (the linear-probe AUC that consumes these embeddings) is a separate
# CPU-only step: scripts/5_downstream_linearprobe_AUC.sh.
#
# Needs: multiple micromamba envs + GPUs (see models/env_routing.md), the
# model weights, and the 6 downstream benchmark datasets (see data/README.md).
#
# Usage:
#     bash scripts/4_extract_downstream_embeddings.sh                    # full run
#     bash scripts/4_extract_downstream_embeddings.sh --gpu-ids 0,5,6,7  # pick GPUs
#     bash scripts/4_extract_downstream_embeddings.sh --only evo         # subset
#     bash scripts/4_extract_downstream_embeddings.sh --dry-run          # show routing, no run
#
# All args are forwarded to the embedding sweep (see run_embed_sweep.py /
# sweep_engine.build_arg_parser for the full flag set). For a subsampled
# smoke, call run_downstream_embed.py --max-train N directly.
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
echo "Step 4 — pooled-embedding sweep (123 models × 6 downstream tasks)"
echo "  Output: results/analysis/embeddings/<slug>/<task>/{train,test}.parquet"
echo "  Args  : ${*:-(none)}"
echo "  Next  : scripts/5_downstream_linearprobe_AUC.sh"
echo "===================================================================="
"${PY}" scripts/downstream_tasks/run_embed_sweep.py "$@"
