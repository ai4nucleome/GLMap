#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 0 (prep, run before everything): audit the model roster. Reads the
# download manifest models/download_models_list.txt, fetches each model's HF
# config + tokenizer_config (from the shared HF cache), and emits the
# machine-readable registry the rest of the pipeline routes on:
#   -> data/audits/models.json   (per-model record: hf_id, family, branch,
#                                 architecture, param_count, context, tokenizer
#                                 type, score_protocol — primary artifact)
#   -> data/audits/models.md     (human-readable summary + table)
#
# data/audits/models.json is the ground truth consumed by the scoring /
# embedding sweeps (loader dispatch) and by the figures (family / branch /
# param_count metadata), so this must be (re)built whenever the model list
# changes — before step 1 scoring.
#
# CPU-only. Hits the HF cache for config.json / tokenizer_config.json; pass
# --skip-hf for a fully offline run (param/context then come from the
# data/audits/*overrides.yaml files only).
#
# Extra args are forwarded to scripts/audits/models.py, e.g.
#   --skip-hf            offline, no HF config fetches
#   --max-models 5       debug on a small subset
#   --out-dir DIR        write elsewhere (default data/audits/)
#
# Usage:
#     bash scripts/0_audit_model_dataset.sh
#     bash scripts/0_audit_model_dataset.sh --skip-hf
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
echo "Step 0 — model audit (download list -> data/audits/models.{json,md})"
echo "  Input : models/download_models_list.txt"
echo "  Output: data/audits/models.json + models.md"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
"${PY}" scripts/audits/models.py --markdown "$@"
