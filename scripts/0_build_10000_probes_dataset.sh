#!/usr/bin/env bash
# ⚠ Machine-specific paths: the base python comes from env_paths.yaml
# (override with $GLMAP_ENV_CONFIG).
#
# Step 0 (data prep): build the frozen 10,000-probe GLMap panel from the
# benchmark sources declared in scripts/panel_build/panel_sources.yaml.
#   -> data/panels/main_panel.parquet   (10,000 probes x 14 functional elements)
#
# This is CPU-only (pandas/pyarrow); no GPU or model weights needed, but the
# benchmark source datasets must be present (see scripts/panel_build/README.md
# and data/README.md for what to download).
#
# Extra args are forwarded to scripts/panel_build/build_panel.py, e.g.
#   --fast                 ~1K-probe smoke (1/10 size)
#   --out-dir DIR          write elsewhere (default data/panels/)
#   --sources FILE         alternative panel_sources.yaml
#
# Usage:
#     bash scripts/0_build_10000_probes_dataset.sh
#     bash scripts/0_build_10000_probes_dataset.sh --fast
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
echo "Building the 10,000-probe GLMap panel"
echo "  Config: scripts/panel_build/panel_sources.yaml"
echo "  Output: data/panels/main_panel.parquet"
echo "  Args  : ${*:-(none)}"
echo "===================================================================="
exec "${PY}" scripts/panel_build/build_panel.py "$@"
