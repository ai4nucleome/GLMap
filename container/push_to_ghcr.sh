#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Push the four GLMap scoring .sif images to GitHub Container Registry (GHCR).
#
# Uses the `oras` CLI (chunked + retrying uploads — far more robust than
# `apptainer push` on flaky / proxied links) and tags the blobs with the SIF
# media types so `apptainer pull oras://...` works on the other end.
#
# ── one-time setup ───────────────────────────────────────────────────────────
#   1. GitHub → Settings → Developer settings → Personal access token (classic)
#      with scope:  write:packages  (+ read:packages, delete:packages optional)
#
#   2. Get the oras CLI (single static binary):
#        curl -sL https://github.com/oras-project/oras/releases/download/v1.2.0/oras_1.2.0_linux_amd64.tar.gz \
#          | tar xz oras && chmod +x oras && sudo mv oras /usr/local/bin/   # or keep in PATH
#
#   3. Log in (token stored in ~/.docker/config.json; oras + apptainer both read it):
#        echo <YOUR_TOKEN> | oras login ghcr.io -u <github-username> --password-stdin
#
# ── run ──────────────────────────────────────────────────────────────────────
#        ORG=ai4nucleome TAG=v1 bash push_to_ghcr.sh
#
# Tip: if a connection still drops, just re-run — oras resumes already-pushed
# blobs, so each retry only uploads what is missing.
#
# ── after pushing ────────────────────────────────────────────────────────────
#   * GHCR packages are PRIVATE by default. On GitHub → your packages →
#     each package → Package settings → Change visibility → Public, and
#     "Connect repository" → ai4nucleome/GLMap.
#   * Pull:           apptainer pull oras://ghcr.io/ORG/glmap-bio-cu118:v1
#   * Faster in CN:   apptainer pull oras://m.daocloud.io/ghcr.io/ORG/glmap-bio-cu118:v1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ORG="${ORG:-ai4nucleome}"
TAG="${TAG:-v1}"
ORAS="${ORAS:-oras}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CFG_TYPE="application/vnd.sylabs.sif.config.v1+json"
LAYER_TYPE="application/vnd.sylabs.sif.layer.v1.sif"

for g in cu118 cu121 default evo; do
  sif="${HERE}/bio-${g}.sif"
  ref="ghcr.io/${ORG}/glmap-bio-${g}:${TAG}"
  [ -f "$sif" ] || { echo "!! missing $sif — skipping"; continue; }
  echo "==== push ${sif} -> oras://${ref} ===="
  "$ORAS" push "$ref" \
      --config /dev/null:"${CFG_TYPE}" \
      "${sif}:${LAYER_TYPE}" \
      --disable-path-validation
  echo "    done: oras://${ref}"
done
echo "ALL PUSHED -> https://github.com/orgs/${ORG}/packages"
