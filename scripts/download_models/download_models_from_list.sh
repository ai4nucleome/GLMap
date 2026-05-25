#!/usr/bin/env bash
# Download HuggingFace models listed in download_models_list.txt.
# Requires: pip install -U huggingface_hub (provides the `hf` CLI).
#
# Usage:
#   bash scripts/download_models/download_models_from_list.sh
#   bash scripts/download_models/download_models_from_list.sh /path/to/list.txt
#
# Environment variables (optional):
#   HF_HOME      Override the huggingface_hub cache root.
#   HF_TOKEN     Authentication token (needed for gated models).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_FILE="${1:-${SCRIPT_DIR}/../../models/download_models_list.txt}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' command not found. Install: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Error: model list not found: $LIST_FILE" >&2
  exit 1
fi

mapfile -t lines < <(grep -vE '^\s*(#|$)' "$LIST_FILE" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
if [[ ${#lines[@]} -eq 0 ]]; then
  echo "Error: model list is empty or comment-only: $LIST_FILE" >&2
  exit 1
fi

echo "Model list: $LIST_FILE"
echo "Models: ${#lines[@]}"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "HF_HOME: $HF_HOME"
else
  echo "HF_HOME: not set; using huggingface_hub default cache location"
fi
echo ""

failed=()
n=0
for repo_id in "${lines[@]}"; do
  n=$((n + 1))
  # Skip entries that are not real HF repo IDs (e.g. GenSLM-* are local
  # weight names handled by setup_external_models.sh + manual download).
  if [[ "$repo_id" == GenSLM-* ]]; then
    echo ">>> [$n/${#lines[@]}] SKIP $repo_id (not an HF repo; see models/README.md)"
    continue
  fi
  echo ">>> [$n/${#lines[@]}] hf download ${repo_id} --exclude *.h5 tf_* *.joblib"
  if hf download "$repo_id" --exclude '*.h5' 'tf_*' '*.joblib'; then
    echo "    done: $repo_id"
  else
    echo "    FAILED: $repo_id" >&2
    failed+=("$repo_id")
  fi
  echo ""
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "The following models failed to download (${#failed[@]}):" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "All downloads completed successfully."
