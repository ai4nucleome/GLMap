#!/bin/bash
# Push the four GLMap scoring images to GitHub Container Registry.
# Prereq: apptainer registry login -u <user> --password-stdin oras://ghcr.io
set -e
APPT=${APPT:-/nvme-data3/yusen/micomamba/envs/apptainer/bin/apptainer}
ORG=ai4nucleome
TAG=${TAG:-v1}
HERE="$(cd "$(dirname "$0")" && pwd)"
for g in cu118 cu121 default evo; do
  sif="$HERE/bio-$g.sif"
  ref="oras://ghcr.io/$ORG/glmap-bio-$g:$TAG"
  echo "==== push $sif -> $ref ===="
  "$APPT" push "$sif" "$ref"
done
echo "ALL PUSHED"
