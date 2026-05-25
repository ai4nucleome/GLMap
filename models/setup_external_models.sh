#!/usr/bin/env bash
# setup_external_models.sh — clone the 9 upstream repos that cannot be loaded
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
# After cloning, see docs/env_routing.md for which micromamba environment
# each family requires.

set -euo pipefail

DEST="${1:-models/modelsHFNoInfo}"
mkdir -p "$DEST"

clone_at() {
    local name="$1" url="$2" sha="$3"
    if [ -d "$DEST/$name" ]; then
        echo "[skip] $name already exists"
        return
    fi
    echo "[clone] $name from $url @ $sha"
    git clone --quiet "$url" "$DEST/$name"
    git -C "$DEST/$name" checkout --quiet "$sha"
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
clone_at PlasmidGPT     https://github.com/lingxusb/PlasmidGPT.git          5578c91

echo ""
echo "All 9 repos cloned into $DEST/"
echo "Next steps:"
echo "  1. See docs/env_routing.md for per-family environment setup."
echo "  2. Download HuggingFace models: bash scripts/download_models/download_models_from_list.sh"
