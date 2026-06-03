#!/usr/bin/env bash
# setup_external_models.sh — clone the 8 upstream repos that cannot be loaded
# via standard HuggingFace transformers (torch.load .pt, custom architectures,
# or non-HF model packages).
#
# These repos are NOT bundled in the GLMap release to respect upstream licenses
# and keep the repository size manageable (~946 MB uncompressed). Running this
# script clones them into models/modelsHFNoInfo/ at the commit SHAs that were
# used for the paper's 123-model scoring run.
#
# Usage:
#   cd /path/to/GLMap-code-public
#   bash models/setup_external_models.sh
#
# After cloning, see models/env_routing.md for which micromamba environment
# each family requires.

# Note: no `set -e` — one repo failing (network, deleted upstream, bad
# SHA) should not abort the remaining clones. Failures are collected and
# reported at the end.
set -uo pipefail

DEST="${1:-models/modelsHFNoInfo}"
mkdir -p "$DEST"

failed=()

clone_at() {
    local name="$1" url="$2" sha="$3"
    if [ -d "$DEST/$name/.git" ]; then
        echo "[skip] $name already cloned"
        return 0
    fi

    local tmpdir
    tmpdir=$(mktemp -d)
    if ! git clone --quiet "$url" "$tmpdir/$name" \
        || ! git -C "$tmpdir/$name" checkout --quiet "$sha"; then
        echo "[FAILED] $name ($url @ $sha)" >&2
        failed+=("$name")
        rm -rf "$tmpdir"
        return 1
    fi

    if [ -d "$DEST/$name" ]; then
        # Directory already exists with non-git files (e.g. the megaDNA
        # weight downloaded before this clone). Merge the clone — code +
        # .git — into it without overwriting existing files.
        echo "[clone] $name @ $sha (merging into existing dir)"
        cp -rn "$tmpdir/$name/." "$DEST/$name/"
    else
        echo "[clone] $name @ $sha"
        mv "$tmpdir/$name" "$DEST/$name"
    fi
    rm -rf "$tmpdir"
    echo "[done] $name"
}

clone_at evo           https://github.com/evo-design/evo.git               6856bba
clone_at evo2          https://github.com/ArcInstitute/evo2.git            3a4d1d0
clone_at genslm        https://github.com/ramanathanlab/genslm.git         6622c47
clone_at hyena-dna     https://github.com/HazyResearch/hyena-dna.git       d553021
clone_at megaDNA        https://github.com/lingxusb/megaDNA.git              cb2f5ab
clone_at ModelGenerator https://github.com/genbio-ai/ModelGenerator.git     c562a20
clone_at PlantBiMoE     https://github.com/HUST-Keep-Lin/PlantBiMoE.git     e3b6d53
clone_at PlantCaduceus  https://github.com/kuleshov-group/PlantCaduceus.git  f0d18ac

echo ""
if [ ${#failed[@]} -gt 0 ]; then
    echo "WARNING: ${#failed[@]} repo(s) failed to clone:" >&2
    printf '  - %s\n' "${failed[@]}" >&2
    echo "Re-run the script to retry the failed ones (completed repos are skipped)." >&2
    exit 1
fi

echo "All 8 repos cloned into $DEST/"
echo "Next steps:"
echo "  1. See models/env_routing.md for per-family environment setup."
echo "  2. Download HuggingFace models: bash scripts/0_download_models_from_list.sh"
