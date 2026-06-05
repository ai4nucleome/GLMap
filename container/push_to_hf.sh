#!/bin/bash
# Upload the four GLMap scoring images to a Hugging Face repo.
# Prereq: `hf auth login` with a WRITE token.
# Usage:  REPO=<namespace>/GLMap-containers bash push_to_hf.sh
set -e
HF=${HF:-/nvme-data3/yusen/micomamba/bin/hf}
REPO=${REPO:?set REPO=<namespace>/GLMap-containers}
export HF_HUB_ENABLE_HF_TRANSFER=1          # fast, resumable uploads
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HF" repo create "$REPO" --repo-type model -y 2>/dev/null || true
for g in cu118 cu121 default evo; do
  echo "==== upload bio-$g.sif -> $REPO ===="
  "$HF" upload "$REPO" "$HERE/bio-$g.sif" "bio-$g.sif" --repo-type model
done
echo "ALL UPLOADED -> https://huggingface.co/$REPO"
