#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Upload the four GLMap scoring .sif images to a Hugging Face *dataset* repo.
#
# Uses hf_transfer for fast, resumable, chunked uploads (robust on flaky / CN
# links — far better than pushing to a US container registry).
#
# ── one-time setup ───────────────────────────────────────────────────────────
#   1. A HuggingFace token with WRITE access:  hf auth login
#   2. (optional, faster) ensure hf_transfer is installed:  pip install hf_transfer
#
# ── run ──────────────────────────────────────────────────────────────────────
#        REPO=<namespace>/GLMap-containers bash push_to_hf.sh
#   e.g. REPO=Tim419/GLMap-containers      bash push_to_hf.sh
#
#   Re-run any time — hf upload resumes, only sending what is missing.
#
# ── download (users) ─────────────────────────────────────────────────────────
#        hf download <namespace>/GLMap-containers bio-cu118.sif \
#            --repo-type dataset --local-dir .
#        apptainer run --nv ... bio-cu118.sif ...
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HF="${HF:-/nvme-data3/yusen/micomamba/bin/hf}"
REPO="${REPO:?set REPO=<namespace>/GLMap-containers}"
export HF_HUB_ENABLE_HF_TRANSFER=1
HERE="$(cd "$(dirname "$0")" && pwd)"

"$HF" repo create "$REPO" --repo-type dataset -y 2>/dev/null || true
for g in cu118 cu121 default evo; do
  sif="${HERE}/bio-${g}.sif"
  [ -f "$sif" ] || { echo "!! missing $sif — skipping"; continue; }
  echo "==== upload bio-${g}.sif -> ${REPO} (dataset) ===="
  "$HF" upload "$REPO" "$sif" "bio-${g}.sif" --repo-type dataset
done
echo "ALL UPLOADED -> https://huggingface.co/datasets/${REPO}"
