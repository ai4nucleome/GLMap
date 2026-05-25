"""Stage 0 gate test: every src/ subpackage imports cleanly.

The audit scripts live outside src/ (they're standalone scripts under
scripts/audits/), but the shared `common` module is also import-checked
here so future refactors catch syntax breakage early without re-running
the slow benchmark scan.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_AUDIT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "audits"
if str(_AUDIT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AUDIT_SCRIPTS_DIR))

PACKAGES = [
    "glmap",
    "glmap.loaders",
    "glmap.loaders.base",
    "glmap.loaders.megadna",
    "glmap.loaders.plasmidgpt",
    "glmap.scoring",
    "glmap.panel",
    "glmap.matrices",
    "glmap.analysis",
    "glmap.figures",
]


def test_subpackages_import() -> None:
    """Every package and shipped stub module imports without error."""
    for name in PACKAGES:
        importlib.import_module(name)


def test_loader_protocol_runtime_check() -> None:
    """MegaDNA / PlasmidGPT stubs satisfy the GLMLoader protocol shape.

    isinstance(...) against a runtime_checkable Protocol only verifies that
    the required attributes exist (not their types or that load()/score()
    return correct values). This is what we want at the Stage 0 gate: the
    contract is wired up; Stage 1 fills in real behavior.
    """
    from glmap.loaders.base import GLMLoader
    from glmap.loaders.megadna import MegaDNALoader
    from glmap.loaders.plasmidgpt import PlasmidGPTLoader

    megadna = MegaDNALoader()
    plasmid = PlasmidGPTLoader(
        weight_path="/nonexistent/pretrained_model.pt",
        tokenizer_path="/nonexistent/tokenizer.json",
    )
    assert isinstance(megadna, GLMLoader)
    assert isinstance(plasmid, GLMLoader)
    assert megadna.hf_id == "lingxusb/megaDNA"
    assert megadna.context_tokens == 131072
    assert plasmid.hf_id == "lingxusb/PlasmidGPT"
    # PlasmidGPT real n_positions = 2048 (verified by loading model.config);
    # the earlier 1024 estimate was a GPT-2 default-based guess.
    assert plasmid.context_tokens == 2048


def test_audit_module_imports() -> None:
    """Audit common module imports cleanly (catches syntax errors without
    re-running the full audit)."""
    import common as audit

    assert callable(audit.infer_branch)
    assert callable(audit.infer_context_from_name)
    assert callable(audit.load_context_overrides)


def test_infer_context_from_name_power_of_two() -> None:
    """Regression: NTv3_*_8kb and NTv3_*_131kb resolve to power-of-two,
    not the prior x1000 fallback."""
    import common as audit

    cases = {
        "InstaDeepAI/NTv3_650M_post_131kb": 131072,
        "InstaDeepAI/NTv3_100M_pre_8kb": 8192,
        "InstaDeepAI/NTv3_8M_pre_8kb": 8192,
        "InstaDeepAI/NTv3_5downsample_post_131kb": 131072,
        "arcinstitute/evo2_7b_262k": 262144,
        "togethercomputer/evo-1-131k-base": 131072,
        "LongSafari/hyenadna-tiny-1k-seqlen": 1024,
        "LongSafari/hyenadna-large-1m-seqlen": 1_048_576,
        # Non-power-of-two HyenaDNA checkpoints retain decimal sizing.
        "LongSafari/hyenadna-medium-160k-seqlen": 160_000,
        "LongSafari/hyenadna-medium-450k-seqlen": 450_000,
        # Names without context markers must remain unresolved.
        "RaphaelMourad/Mistral-DNA-v1-138M-hg38": None,
    }
    for hf_id, expected in cases.items():
        got, _src = audit.infer_context_from_name(hf_id)
        assert got == expected, f"{hf_id}: expected {expected}, got {got}"
