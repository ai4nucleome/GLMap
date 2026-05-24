"""Loader dispatch: map an audit entry → ModelSpec → concrete loader.

The heavy per-family loaders (and their torch / transformers /
family-specific dependencies) are imported **lazily** inside
``get_loader()``. This module's top-level imports are pure standard
library plus :mod:`dataclasses`. As a result, ``import glmap`` never
triggers ``import torch`` — the core install (``pip install
ai4nucleome-glmap``) stays usable for analysis, matrix loading, and
figures even without the ``scoring`` extra installed.

The mapping from audit row → ``ModelSpec`` is the authoritative dispatch
table for the paper's 123-model collection; ``specs_from_audit()`` walks
``data/audits/models.json`` and yields one ``ModelSpec`` per scorable
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import json

__all__ = [
    "ModelSpec",
    "audit_entry_to_spec",
    "specs_from_audit",
    "specs_from_hf_ids",
    "get_loader",
]

# Audit-side branch label (used in data/audits/models.json) → loader-side
# branch label (used by every loader's `.branch` attribute and by the
# AR/MLM scoring dispatch).
_AUDIT_BRANCH_TO_SCORE_BRANCH = {
    "ar_or_generative": "ar",
    "mlm_or_encoder": "mlm",
}


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    """Minimal config for one scorable model.

    All fields are keyword-only to prevent positional-argument confusion
    (e.g. swapping ``branch`` and ``loader_kind`` silently).


    Attributes
    ----------
    hf_id
        Hugging Face identifier or local path used by the loader to find
        weights and tokenizer.
    branch
        Either ``"ar"`` or ``"mlm"`` — matches each loader's ``.branch``
        attribute and the AR/MLM scoring dispatch. See ``audit_branch``
        below for the raw audit value.
    context_tokens
        Maximum input length the loader should accept, in tokens.
    loader_kind
        Which loader class to instantiate. See :func:`get_loader` for the
        full dispatch table.
    trust_remote_code
        Forwarded to ``AutoModel.from_pretrained`` and friends. Defaults
        to ``True`` because most DNA models on the Hub require it; the
        family-specific dispatch table overrides to ``False`` for the
        handful of loaders that don't (megaDNA, PlasmidGPT, GenSLM,
        HyenaDNA, AIDO.DNA, Evo-1, Evo-2).
    is_codon
        Whether the tokenizer is codon-based (3-mer non-overlapping).
        Only ``True`` for GenSLM. Downstream diagnostics may use this to
        report codon-vs-nucleotide loadings separately.
    length_multiple
        NTv3-specific. The U-Net architecture requires input lengths
        divisible by ``2 ** num_downsamples`` (128 for the main NTv3
        family, 32 for the ``5downsample`` variants). ``None`` for every
        other ``loader_kind``.
    family
        Optional human-readable family label (e.g. ``"NT"``, ``"Evo1"``,
        ``"PlantCAD2"``) carried through from the audit.
    tokenizer_type
        Optional tokenizer category from the audit (e.g.
        ``"single_nucleotide"``, ``"non_overlapping_6mer"``,
        ``"byte_pair_encoding"``).
    audit_branch
        Raw audit branch value (e.g. ``"ar_or_generative"``,
        ``"mlm_or_encoder"``, ``"supervised_or_annotation"``). Preserved
        verbatim for audit-traceability.
    extra
        Free-form dict for downstream extensions (e.g. third-party
        ``loader_kind`` values).
    """

    hf_id: str
    branch: Literal["ar", "mlm"]
    context_tokens: int
    loader_kind: str = "hf"
    trust_remote_code: bool = True
    is_codon: bool = False
    length_multiple: Optional[int] = None
    family: Optional[str] = None
    tokenizer_type: Optional[str] = None
    audit_branch: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """A filesystem-safe identifier (HF id with ``/`` replaced)."""
        return self.hf_id.replace("/", "__")


def audit_entry_to_spec(entry: dict) -> Optional[ModelSpec]:
    """Convert one ``models.json`` audit entry to a :class:`ModelSpec`.

    Returns ``None`` for entries that are not scorable as language models
    (e.g. supervised-only or annotation-only models without an LM head).

    The dispatch table is encoded inline below; see the table in
    :func:`get_loader` for the runtime loader class each ``loader_kind``
    maps to.
    """
    hf_id = entry["hf_id"]
    audit_branch = entry.get("branch", "")
    ctx = int(entry.get("context_tokens") or 0)
    family = entry.get("family")
    tokenizer_type = entry.get("tokenizer_type")

    def _spec(branch, *, trust_remote_code=False, is_codon=False,
              loader_kind="hf", length_multiple=None):
        return ModelSpec(
            hf_id=hf_id,
            branch=branch,
            context_tokens=ctx,
            loader_kind=loader_kind,
            trust_remote_code=trust_remote_code,
            is_codon=is_codon,
            length_multiple=length_multiple,
            family=family,
            tokenizer_type=tokenizer_type,
            audit_branch=audit_branch,
        )

    # Family-specific routing. Order matters — earlier rules win.
    if hf_id == "lingxusb/megaDNA":
        return _spec("ar", loader_kind="megadna")
    if hf_id == "lingxusb/PlasmidGPT":
        return _spec("ar", loader_kind="plasmidgpt")
    if hf_id.startswith("GenSLM-"):
        return _spec("ar", is_codon=True, loader_kind="genslm")
    if hf_id.startswith("living-models/Botanic0"):
        return _spec("mlm", trust_remote_code=True, loader_kind="botanic")
    if hf_id == "plant-llms/PlantBiMoE":
        return _spec("mlm", trust_remote_code=True, loader_kind="plantbimoe")
    if hf_id.startswith("JadenLong/MutBERT"):
        return _spec("mlm", trust_remote_code=True, loader_kind="mutbert")
    if hf_id.startswith("LongSafari/hyenadna-"):
        return _spec("ar", loader_kind="hyenadna")
    if hf_id.startswith("genbio-ai/AIDO.DNA"):
        return _spec("mlm", loader_kind="aido")
    if hf_id.startswith("arcinstitute/evo2_") or hf_id == "evo-design/evo-2-7b-8k-microviridae":
        return _spec("ar", loader_kind="evo2")
    if hf_id in {
        "togethercomputer/evo-1-8k-base",
        "togethercomputer/evo-1-131k-base",
        "LongSafari/evo-1-8k-crispr",
        "LongSafari/evo-1-8k-transposon",
        "evo-design/evo-1.5-8k-base",
        "evo-design/evo-1-7b-131k-microviridae",
    }:
        return _spec("ar", loader_kind="evo1")
    if hf_id.startswith("GenerTeam/GENERator-"):
        # NB: distinct from GENERanno-* which are single-nucleotide MLM.
        return _spec("ar", trust_remote_code=True, loader_kind="generator")
    if hf_id.startswith("HuggingFaceBio/Carbon-"):
        return _spec("ar", trust_remote_code=True, loader_kind="carbon")
    if hf_id.startswith("InstaDeepAI/NTv3_"):
        # U-Net MLM: input length must be divisible by 2 ** num_downsamples.
        # 128 for main models, 32 for the *5downsample* variants.
        length_multiple = 32 if "5downsample" in hf_id else 128
        return _spec("mlm", trust_remote_code=True, loader_kind="ntv3",
                     length_multiple=length_multiple)

    # Generic HF fallback. Only accept entries whose audit branch is one
    # of the two LM-scorable values; supervised / annotation models have
    # no LM head and cannot be scored, return None.
    if audit_branch == "ar_or_generative":
        score_branch = "ar"
    elif audit_branch == "mlm_or_encoder":
        score_branch = "mlm"
    else:
        return None

    return _spec(
        score_branch,
        trust_remote_code=True,
        is_codon=bool(entry.get("is_codon_model", False)),
        loader_kind="hf",
    )


def specs_from_audit(audit_path: Optional[Path | str] = None) -> list[ModelSpec]:
    """Load every scorable model from ``data/audits/models.json``.

    Unscorable entries (audit ``branch`` neither ``ar_or_generative`` nor
    ``mlm_or_encoder``) are silently dropped.

    Parameters
    ----------
    audit_path
        Optional explicit path to ``models.json``. If ``None``, the path
        is resolved via :func:`glmap._data_resolver.resolve_data_path`
        (which checks ``$GLMAP_DATA_DIR``, package-bundled data, and the
        repo root in turn).
    """
    if audit_path is None:
        from glmap._data_resolver import resolve_data_path
        audit_path = resolve_data_path("data/audits/models.json")
    audit_path = Path(audit_path)
    payload = json.loads(audit_path.read_text())
    out: list[ModelSpec] = []
    for entry in payload.get("models", []):
        spec = audit_entry_to_spec(entry)
        if spec is not None:
            out.append(spec)
    return out


def specs_from_hf_ids(
    hf_ids: list[str],
    audit_path: Optional[Path | str] = None,
) -> list[ModelSpec]:
    """Build ``ModelSpec`` objects for a specific subset of HF ids.

    The order in ``hf_ids`` is preserved. Ids not found in the audit, or
    not scorable as an LM, are skipped silently.
    """
    if audit_path is None:
        from glmap._data_resolver import resolve_data_path
        audit_path = resolve_data_path("data/audits/models.json")
    audit_path = Path(audit_path)
    by_id = {
        e["hf_id"]: e
        for e in json.loads(audit_path.read_text()).get("models", [])
    }
    out: list[ModelSpec] = []
    for hf_id in hf_ids:
        entry = by_id.get(hf_id)
        if entry is None:
            continue
        spec = audit_entry_to_spec(entry)
        if spec is not None:
            out.append(spec)
    return out


def get_loader(
    spec: ModelSpec,
    *,
    device: str = "cpu",
    torch_dtype=None,
    load: bool = True,
    **kwargs,
):
    """Instantiate the appropriate loader for ``spec``.

    Heavy dependencies (torch, transformers, family-specific packages)
    are imported lazily inside this function. ``import glmap`` alone
    never triggers them.

    Parameters
    ----------
    spec
        The :class:`ModelSpec` to dispatch on.
    device
        Torch device string (``"cpu"`` / ``"cuda:0"`` / ``"cuda"``).
    torch_dtype
        Optional ``torch.dtype``. Forwarded to every loader that accepts
        it: ``hf``, ``botanic``, ``plantbimoe``, ``mutbert``, ``hyenadna``,
        ``aido``, ``evo1``, ``evo2``, ``generator``, ``carbon``, ``ntv3``.
        The three legacy ``torch.load``-style loaders (``megadna``,
        ``plasmidgpt``, ``genslm``) do not expose a dtype knob; passing
        ``torch_dtype`` for them is silently ignored. Defaults to
        ``None`` (each loader picks its own dtype, usually FP32 / FP16).
    load
        If ``True`` (the default), call ``.load()`` on the returned
        loader so it is immediately ready for scoring. Pass ``load=False``
        to defer loading (useful when constructing many specs ahead of
        time, or when controlling GPU memory manually).
    **kwargs
        Additional keyword arguments forwarded to the loader constructor
        (loader-specific; see each loader class).

    Returns
    -------
    A loader instance with a ``.score(sequence, ...)`` / ``.score_record(
    sequence, ...)`` interface.
    """
    kind = spec.loader_kind

    # Loaders that accept ``torch_dtype`` get it forwarded; the three
    # legacy .pt-style loaders (megadna / plasmidgpt / genslm) don't take
    # it and are constructed without that kwarg.
    if kind == "hf" and spec.branch == "ar":
        from glmap.loaders.huggingface import HFCausalLMLoader
        loader = HFCausalLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "hf" and spec.branch == "mlm":
        from glmap.loaders.huggingface import HFMaskedLMLoader
        loader = HFMaskedLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "megadna":
        from glmap.loaders.megadna import MegaDNALoader
        loader = MegaDNALoader(device=device, **kwargs)
    elif kind == "plasmidgpt":
        from glmap.loaders.plasmidgpt import PlasmidGPTLoader
        loader = PlasmidGPTLoader(device=device, **kwargs)
    elif kind == "genslm":
        from glmap.loaders.genslm import GenSLMLoader
        loader = GenSLMLoader(spec.hf_id, device=device, **kwargs)
    elif kind == "botanic":
        from glmap.loaders.custom_mlm import BotanicLoader
        loader = BotanicLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "plantbimoe":
        from glmap.loaders.custom_mlm import PlantBiMoELoader
        loader = PlantBiMoELoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "mutbert":
        from glmap.loaders.custom_mlm import MutBERTLoader
        loader = MutBERTLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "hyenadna":
        from glmap.loaders.hyenadna import HyenaDNALoader
        loader = HyenaDNALoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "aido":
        from glmap.loaders.aido import AIDOLoader
        loader = AIDOLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "evo1":
        from glmap.loaders.evo1_loader import Evo1Loader
        loader = Evo1Loader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "evo2":
        from glmap.loaders.evo2_loader import Evo2Loader
        loader = Evo2Loader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "generator":
        from glmap.loaders.generator import GENERatorLoader
        loader = GENERatorLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "carbon":
        from glmap.loaders.carbon import CarbonCausalLMLoader
        loader = CarbonCausalLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    elif kind == "ntv3":
        from glmap.loaders.ntv3 import NTv3MaskedLMLoader
        if spec.length_multiple is None:
            raise ValueError(
                f"{spec.hf_id}: loader_kind='ntv3' requires length_multiple "
                "(128 for 7-downsample main models, 32 for 5downsample variants)"
            )
        loader = NTv3MaskedLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            length_multiple=spec.length_multiple,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown loader_kind={kind!r} branch={spec.branch!r} for {spec.hf_id!r}"
        )

    if load:
        loader.load()
    return loader
