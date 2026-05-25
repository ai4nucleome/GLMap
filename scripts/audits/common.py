"""Shared helpers for the audit pipeline (models.py + benchmarks.py).

Three roles:

1. **Constants and paths** that both audit scripts share.
2. **Pure inference functions** that classify a HuggingFace repo id into
   family / branch / architecture / tokenizer / parameter count / context
   window. These are the single source of truth — when adding a new model
   family, edit only this module.
3. **Small I/O helpers** for fetching HF config files, loading the
   `context_overrides.yaml` priority table, and writing JSON + Markdown
   reports.

Nothing in this module loads model weights or scans benchmark files; those
are owned by models.py and benchmarks.py respectively.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import HfApi, hf_hub_download
except Exception:  # pragma: no cover
    HfApi = None
    hf_hub_download = None

try:
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_LIST_PATH = REPO_ROOT / "models/download_models_list.txt"
AUDIT_DIR = REPO_ROOT / "data/audits"
CONTEXT_OVERRIDES_PATH = AUDIT_DIR / "context_overrides.yaml"
PARAM_OVERRIDES_PATH = AUDIT_DIR / "param_overrides.yaml"

# Benchmark roots (scanned by benchmarks.py).
BENCHMARK_ROOTS: dict[str, Path] = {
    "gue": REPO_ROOT / "data/GUE",
    "pgb": REPO_ROOT / "data/PGB",
    "dnabert_s_eval": REPO_ROOT / "data/dnabert-s_eval",
    "dna_foundation_benchmark": REPO_ROOT / "data/dna_foundation_benchmark",
    # `data/genomic-benchmarks/` is intentionally excluded: its files carry
    # BED coordinates (id,region,start,end,strand), not DNA sequences, so
    # the audit can't compute the length / tokenization stats this catalog
    # is meant to populate. Resolving those coordinates against hg38 belongs
    # in panel construction, not the benchmark inventory.
}


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------


def clean_model_list(path: Path = MODELS_LIST_PATH) -> list[str]:
    """Read download_models_list.txt, deduplicating and stripping comments."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def repo_basename(hf_id: str) -> str:
    """'lingxusb/megaDNA' -> 'megaDNA'. Bare IDs returned unchanged."""
    return hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id


def repo_organization(hf_id: str) -> str:
    """'lingxusb/megaDNA' -> 'lingxusb'. Bare IDs return empty string."""
    r = hf_id.lower()
    if (
        "evo2_" in r
        or "evo-2" in r
        or "evo-1" in r
        or "evo-1.5" in r
    ):
        return "arcinstitute"
    return hf_id.split("/", 1)[0] if "/" in hf_id else ""


# ---------------------------------------------------------------------------
# Family / branch / architecture / tokenizer / paradigm inference
# ---------------------------------------------------------------------------


def infer_family(hf_id: str) -> str:
    """Map a repo id to a coarse family label.

    Order matters: longer / more specific substrings come first so e.g.
    DNABERT-2 wins over DNABERT, and Agro-NT wins over Nucleotide Transformer.
    """
    r = hf_id.lower()
    rules: tuple[tuple[str, str], ...] = (
        ("agro-nucleotide-transformer", "AgroNT"),
        ("dnabert-s", "DNABERT-S"),
        ("dnabert-2", "DNABERT-2"),
        ("dnabert2", "DNABERT-2"),
        ("dna_bert", "DNABERT"),
        ("dnabert", "DNABERT"),
        ("ntv3", "NTv3"),
        ("nucleotide-transformer-v2", "NTv2"),
        ("nucleotide-transformer", "NT"),
        ("gena-lm", "GENA-LM"),
        ("moderngena", "ModernGENA"),
        ("grover", "GROVER"),
        ("hyenadna", "HyenaDNA"),
        ("plantcad", "PlantCaduceus"),
        ("caduceus", "Caduceus"),
        ("plantbimoe", "PlantBiMoE"),
        ("evo2", "Evo2"),
        ("evo-2", "Evo2"),
        ("evo-1", "Evo1"),
        ("evo-1.5", "Evo1"),
        ("megadna", "MegaDNA"),
        ("generator-v2", "GENERator"),
        ("generator", "GENERator"),
        ("generanno", "GENERanno"),
        ("carbon", "Carbon"),
        ("plasmidgpt", "PlasmidGPT"),
        ("genomeocean", "GenomeOcean"),
        ("genos", "Genos"),
        ("onegenome", "OneGenome"),
        ("aido", "AIDO.DNA"),
        ("hybridna", "HybriDNA"),
        ("botanic", "Botanic0"),
        ("mistral-dna", "Mistral-DNA"),
        ("modernbert-dna", "ModernBERT-DNA"),
        ("jamba-dna", "Jamba-DNA"),
        ("genslm", "GenSLM"),
    )
    for needle, family in rules:
        if needle in r:
            return family
    return repo_basename(hf_id).split("-", 1)[0]


def infer_branch(hf_id: str) -> str:
    """Return one of: ar_or_generative / mlm_or_encoder /
    supervised_or_annotation / manual_unknown.

    Family-specific notes:
      - NT family (NT v1/v2, NTv3): all MLM encoders. The historical special
        case for `ntv3 + generative` returning AR was wrong — `NTv3_generative`
        is MDLM (masked discrete language model, diffusion-on-tokens), so it's
        still an MLM under the hood and is scored via stride PLL.
      - GENERanno: `-cds-annotator` checkpoints are token-classification heads
        (caught above by the `annotator` keyword). The remaining `-base`
        checkpoints are `GenerannoForMaskedLM` (fill-mask) so they belong in
        the MLM branch, NOT AR.
      - Botanic0 ships a custom AutoModel whose forward returns masked-LM
        logits; the scoring path uses BotanicLoader + stride PLL, so it is
        treated as MLM/encoder for audit and paper metadata.
    """
    r = hf_id.lower()
    if "annotator" in r or "artificial-detector" in r:
        return "supervised_or_annotation"
    if (
        "dnabert" in r
        or "dna_bert" in r
        or "nucleotide-transformer" in r
        or "ntv3" in r
    ):
        return "mlm_or_encoder"
    mlm_markers = (
        "gena-lm",
        "moderngena",
        "grover",
        "caduceus",
        "plantcad",
        "plantbimoe",
        "aido.dna",
        "modernbert-dna",
        "botanic",
        "generanno",  # GenerannoForMaskedLM (fill-mask); -annotator caught above
        "d3lm",       # D3LMForMaskedLM (discrete diffusion DNA LM trained with masked fill objective)
        "mutbert",    # RoPEBertForMaskedLM (MutBERT — probabilistic genome representation with mutation data)
    )
    if any(m in r for m in mlm_markers):
        return "mlm_or_encoder"
    ar_markers = (
        "hyenadna",
        "evo",
        "megadna",
        "plasmidgpt",
        "generator",  # GENERator AR foundation (distinct from GENERanno MLM)
        "carbon",     # HuggingFaceBio/Carbon-* Llama-style causal LM with hybrid BPE+6mer tokenizer
        "genomeocean",
        "genos",
        "onegenome",
        "mistral-dna",
        "jamba-dna",
        "hybridna",
        "genslm",
    )
    if any(m in r for m in ar_markers):
        return "ar_or_generative"
    return "manual_unknown"


_BRANCH_TO_PARADIGM = {
    "ar_or_generative": "ntp",
    "mlm_or_encoder": "mlm",
    "supervised_or_annotation": "supervised",
    "manual_unknown": "unknown",
}


def training_paradigm(branch: str) -> str:
    return _BRANCH_TO_PARADIGM.get(branch, "unknown")


_SCORE_PROTOCOL_BY_BRANCH = {
    "ar_or_generative": "ar_likelihood_candidate",
    "mlm_or_encoder": "mlm_pll_candidate",
    "supervised_or_annotation": "unsupported_supervised_head",
    "manual_unknown": "manual_unknown",
}


def infer_score_protocol(hf_id: str, branch: str) -> str:
    if "dnabert-s" in hf_id.lower():
        return "embedding_or_mlm_manual"
    return _SCORE_PROTOCOL_BY_BRANCH.get(branch, "manual_unknown")


# Architecture rules: hf_id substring -> architecture label. First match wins.
_ARCH_RULES: tuple[tuple[str, str], ...] = (
    ("ntv3", "unet_transformer"),
    ("megadna", "transformer_decoder"),
    ("plasmidgpt", "transformer_decoder"),
    ("plantbimoe", "bimamba_moe"),
    ("plantcad", "bimamba"),
    ("caduceus", "bimamba"),
    ("modernbert-dna", "modernbert_encoder"),
    ("moderngena", "modernbert_encoder"),
    ("hybridna", "mamba_hybrid"),
    ("jamba-dna", "mamba_hybrid"),
    ("hyenadna", "hyena"),
    ("evo", "striped_hyena"),
    ("dnabert-2", "transformer_encoder_bpe"),
    ("dnabert2", "transformer_encoder_bpe"),
    ("dnabert-s", "transformer_encoder"),
    ("dna_bert", "transformer_encoder"),
    ("dnabert", "transformer_encoder"),
    ("agro-nucleotide-transformer", "transformer_encoder"),
    ("nucleotide-transformer", "transformer_encoder"),
    ("gena-lm", "transformer_encoder_bpe"),
    ("grover", "transformer_encoder_bpe"),
    ("aido", "transformer_encoder"),
    ("d3lm", "discrete_diffusion_esm_encoder"),
    ("mutbert", "rope_bert_encoder"),
    ("mistral-dna", "transformer_decoder"),
    ("genslm", "transformer_decoder"),
    ("genomeocean", "transformer_decoder"),
    ("genos", "transformer_decoder"),
    ("generator", "transformer_decoder"),
    ("generanno", "transformer_decoder"),
    ("botanic", "transformer_encoder"),
    ("onegenome", "transformer_decoder"),
)

_HF_MODEL_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("striped_hyena", "striped_hyena"),
    ("stripedhyena", "striped_hyena"),
    ("hyena", "hyena"),
    ("bimamba", "bimamba"),
    ("mamba", "mamba"),
    ("jamba", "mamba_hybrid"),
    ("modernbert", "modernbert_encoder"),
    ("bert", "transformer_encoder"),
    ("roberta", "transformer_encoder"),
    ("electra", "transformer_encoder"),
    ("gpt_neox", "transformer_decoder"),
    ("gpt2", "transformer_decoder"),
    ("gptj", "transformer_decoder"),
    ("llama", "transformer_decoder"),
    ("mistral", "transformer_decoder"),
    ("mixtral", "transformer_decoder_moe"),
    ("qwen", "transformer_decoder"),
)


def infer_architecture(hf_id: str, config: dict[str, Any] | None) -> tuple[str, str]:
    """Return (architecture, source). source is debug-only provenance."""
    r = hf_id.lower()
    for needle, arch in _ARCH_RULES:
        if needle in r:
            return arch, "hf_id_pattern"
    if config:
        mt = str(config.get("model_type", "")).lower()
        for needle, arch in _HF_MODEL_TYPE_RULES:
            if needle in mt:
                return arch, "hf_config_model_type"
        archs = config.get("architectures") or []
        if archs:
            cls = str(archs[0]).lower()
            for needle, arch in _HF_MODEL_TYPE_RULES:
                if needle in cls:
                    return arch, "hf_config_architectures"
    return "unknown", "unknown"


def infer_tokenizer_type(hf_id: str, config: dict[str, Any] | None = None) -> str:
    r = hf_id.lower()
    m = re.search(r"dna_bert_(\d+)", r)
    if m:
        return f"overlapping_{m.group(1)}mer"
    if "3mer" in r or "codon" in r or "genslm" in r:
        return "non_overlapping_3mer_or_codon"
    # GENERanno (GenerTeam/GENERanno-*) ships a single-nucleotide tokenizer
    # (vocab includes A/C/G/T as individual tokens; empirical 1 bp/tok); the
    # earlier blanket "generator/generanno -> 6mer" rule misclassified it.
    if "generanno" in r:
        return "single_nucleotide_or_byte"
    if "agro-nucleotide-transformer" in r or "generator" in r:
        return "non_overlapping_6mer"
    if "huggingfacebio/carbon" in r:
        # HybridDNATokenizer: Qwen3 BPE base + fixed 6-mer DNA mode when wrapped
        # in <dna>...</dna>. Vocab = 155,776 (base BPE + 4^6 6-mers + specials).
        # We treat the DNA branch as the operative tokenization for our protocol.
        return "non_overlapping_6mer"
    if "botanic" in r:
        # Botanic0 ships a DNATokenizer with vocab=4107 (= 4^6 + 11 specials);
        # empirical 5.88 bp/tok on 1 kb probes confirms the 6-mer split.
        return "non_overlapping_6mer"
    if "d3lm" in r:
        # D3LM EsmTokenizer ships vocab=4111 (= 4^6 6-mers + 6 specials + 5
        # single-base fallbacks ACGTN); non-overlapping 6-mer split.
        return "non_overlapping_6mer"
    if "nucleotide-transformer" in r:
        return "non_overlapping_6mer_like"
    byte_markers = (
        "ntv3",
        "hyenadna",
        "caduceus",
        "plantcad",
        "plantbimoe",
        "evo",
        "megadna",
        "aido.dna",
        # HybriDNA ships HybriDNATokenizer with vocab=12; empirical 1.00 bp/tok
        # across random / GC-rich / poly-A probes confirms single-nucleotide.
        "hybridna",
        "mutbert",   # MutBERT vocab=9 (ACGT + 5 specials), 1 bp/tok
    )
    if any(m in r for m in byte_markers):
        return "single_nucleotide_or_byte"
    bpe_markers = (
        "gena",
        "grover",
        "plasmidgpt",
        "mistral-dna",
        "modernbert-dna",
        "dnabert-2",
        "dnabert2",
        "genomeocean",
    )
    if any(m in r for m in bpe_markers):
        return "bpe_or_wordpiece"
    if config:
        mt = str(config.get("model_type", "")).lower()
        if "bert" in mt:
            return "wordpiece_or_bpe"
        if any(x in mt for x in ("gpt", "llama", "mistral", "mixtral", "mamba", "jamba")):
            return "bpe_or_byte"
    return "manual_unknown"


def tokenizer_name_from_config(tokenizer_config: dict[str, Any] | None) -> str | None:
    """Extract the `tokenizer_class` string (e.g. 'EsmTokenizer')."""
    if not tokenizer_config:
        return None
    cls = tokenizer_config.get("tokenizer_class")
    if cls:
        return str(cls)
    auto_map = tokenizer_config.get("auto_map") or {}
    if isinstance(auto_map, dict):
        for value in auto_map.values():
            if isinstance(value, str):
                return value.split(".", 1)[-1]
            if isinstance(value, (list, tuple)) and value:
                first = value[0]
                if isinstance(first, str):
                    return first.split(".", 1)[-1]
    return None


# ---------------------------------------------------------------------------
# Context window (tokens) inference
# ---------------------------------------------------------------------------

CONTEXT_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "max_sequence_length",
    "max_seq_length",
    "seq_length",
    "sequence_length",
    "model_max_length",
    "max_seqlen",
    "max_context_length",
)
# NOTE: `max_length` is intentionally NOT in this list. In HF configs it is
# almost always the GenerationConfig default (cap on generated tokens), not
# the model's input capacity — e.g. InstaDeepAI/NTv3_generative ships
# `max_length: 20` for sampling but its real context is 1M tokens.


def _is_useful_context_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return 0 < value < 10_000_000


def _find_context_values(payload: Any, prefix: str = "") -> dict[str, int]:
    """Walk a nested config dict and return every plausible context-length
    value, keyed by the dotted path that contained it."""
    out: dict[str, int] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            path = f"{prefix}.{k}" if prefix else k
            if k in CONTEXT_KEYS and _is_useful_context_value(v):
                out[path] = int(v)
            if isinstance(v, (dict, list)):
                out.update(_find_context_values(v, path))
    elif isinstance(payload, list):
        for idx, v in enumerate(payload):
            if isinstance(v, (dict, list)):
                out.update(_find_context_values(v, f"{prefix}[{idx}]"))
    return out


def infer_context_from_name(hf_id: str) -> tuple[int | None, str]:
    """Pull a context length out of suffix shorthands like _8kb / _131k / _1m."""
    r = hf_id.lower()
    has_marker = (
        "seqlen" in r or "context" in r or "evo" in r or "kb" in r or "kbp" in r
    )
    if has_marker:
        # Power-of-two shorthands take priority — DNA model cards almost
        # always intend the rounded power even when written as kbp/kb.
        for token, value in (
            ("262k", 262_144),
            ("131k", 131_072),
            ("65k", 65_536),
            ("32k", 32_768),
            ("16k", 16_384),
            ("8k", 8_192),
            ("4k", 4_096),
            ("2k", 2_048),
            ("1k", 1_024),
            ("1m", 1_048_576),
        ):
            if token in r:
                return value, "model_name_power_of_two"
        for token, value in (("450k", 450_000), ("160k", 160_000)):
            if token in r:
                return value, "model_name"
    patterns = (
        (r"seqlen-(\d+)k", 1_000),
        (r"seqlen-(\d+)m", 1_000_000),
        (r"(\d+)k-seqlen", 1_000),
        (r"(\d+)m-seqlen", 1_000_000),
    )
    for pat, mult in patterns:
        m = re.search(pat, r)
        if m:
            return int(m.group(1)) * mult, "model_name"
    return None, ""


def choose_context(
    config: dict[str, Any] | None,
    tokenizer_config: dict[str, Any] | None,
    name_context: int | None,
    name_source: str,
    override: dict[str, Any] | None,
) -> tuple[int | None, str]:
    """Resolve final context_tokens with the priority chain:
    override > config field > tokenizer_config field > name."""
    if override:
        return int(override["context_tokens"]), f"override:{override.get('source', 'override')}"
    if config:
        cfg_hits = _find_context_values(config)
        for priority_key in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_sequence_length",
            "max_seq_length",
            "seq_length",
        ):
            for path, value in cfg_hits.items():
                if path == priority_key or path.endswith(f".{priority_key}"):
                    return value, f"config:{path}"
        if cfg_hits:
            path, value = next(iter(cfg_hits.items()))
            return value, f"config:{path}"
    if tokenizer_config:
        for path, value in _find_context_values(tokenizer_config).items():
            return value, f"tokenizer_config:{path}"
    if name_context:
        return name_context, name_source or "model_name"
    return None, ""


# Empirical bp:token ratios for families whose tokenizer is BPE/wordpiece
# but whose DNA training corpus pins the ratio tightly enough to give a
# meaningful single number. Without an entry here the BPE branch returns
# None (the conservative default) because BPE ratios drift on out-of-
# distribution input.
_FAMILY_BPE_BP_PER_TOKEN: dict[str, float] = {
    # AIRI-Institute DNA BPE. Exact ratio 8.7890625 = 9000/1024 keeps
    # 1024-tok / 4096-tok / 512-tok variants on the same integer grid
    # (9000 / 36000 / 4500 bp respectively).
    "GENA-LM": 9000.0 / 1024.0,
    "ModernGENA": 9000.0 / 1024.0,
    # PoetschLab GROVER BPE. Empirical 3.98–4.00 bp/tok flat across random /
    # GC-rich / poly-A 1 kb probes.
    "GROVER": 4.0,
    # RaphaelMourad's DNA BPE (vocab 4096). Same tokenizer shared between
    # Mistral-DNA and ModernBERT-DNA per the HF model cards. Empirical
    # 3.97–4.02 bp/tok.
    "Mistral-DNA": 4.0,
    "ModernBERT-DNA": 4.0,
    # JGI GenomeOcean BPE (vocab 4096). Empirical 3.98–5.99 bp/tok depending
    # on probe complexity; 5.0 is the working average matching the model
    # card's quoted usable context per tokens.
    "GenomeOcean": 5.0,
    # zhihan1996 DNABERT-2 / DNABERT-S share the same 4096-vocab DNA BPE.
    # Empirical 3.97–4.02 bp/tok flat across complexity classes.
    "DNABERT-2": 4.0,
    "DNABERT-S": 4.0,
    # PlasmidGPT Addgene BPE (vocab 30000, much larger than the 4 kb DNA-BPE
    # baseline). Empirical 3.98 bp/tok on mixed ACGT but 5.99 bp/tok on low-
    # complexity probes — BPE picks long merges in poly-A / GC-rich regions.
    # 5.0 averages the three probe classes, matching the GenomeOcean choice.
    "PlasmidGPT": 5.0,
}


def estimate_context_bp(
    context_tokens: int | None,
    tokenizer_type: str,
    family: str | None = None,
) -> int | None:
    """Convert token-length to bp-length for tokenizers where the ratio is
    well-defined.

    Returns the raw capacity — does NOT subtract for special tokens. The
    [CLS]/[SEP] count varies across architectures (BERT uses 2, GPT/AR
    models 0–1, NTv3 custom), so a blanket `-2` would be wrong for half
    the catalog. Downstream callers can reserve those positions themselves.

    BPE / wordpiece tokenizers fall back to a per-family empirical ratio
    (see `_FAMILY_BPE_BP_PER_TOKEN`); when the family has no known ratio,
    None is returned and the audit reports the BPE bp limit as unknown.
    """
    if context_tokens is None or context_tokens <= 0:
        return None
    m = re.search(r"overlapping_(\d+)mer", tokenizer_type)
    if m and "non_overlapping" not in tokenizer_type:
        # Overlapping k-mer: N tokens cover N + (k-1) bp.
        return context_tokens + int(m.group(1)) - 1
    if "6mer" in tokenizer_type:
        return context_tokens * 6
    if "3mer" in tokenizer_type or "codon" in tokenizer_type:
        return context_tokens * 3
    if "single_nucleotide" in tokenizer_type or "byte" in tokenizer_type:
        return context_tokens
    if "bpe" in tokenizer_type or "wordpiece" in tokenizer_type:
        ratio = _FAMILY_BPE_BP_PER_TOKEN.get(family or "")
        if ratio is not None:
            return int(context_tokens * ratio)
    return None


# ---------------------------------------------------------------------------
# Codon / modality flags
# ---------------------------------------------------------------------------

_MIXED_MODALITY_MARKERS = ("glm2", "lucaone")


def infer_modality(hf_id: str, tokenizer_type: str) -> tuple[bool, bool, str]:
    """Return (is_codon_model, mixed_modality, valid_probe_classes).

    These flags are recorded on the model side for downstream diagnostics
    and per-element loading reports. They originally encoded the retired
    three-matrix protocol routing (codon → Q_coding_only, mixed-modality
    → coding_only_pending_manual); under the current single-matrix protocol
    (commit 5e59154) every model enters L on the full panel and the flags
    no longer gate matrix membership. They remain useful to:
      - identify codon vs nucleotide models on PC loadings,
      - flag mixed-modality models that need explicit DNA-only scoring
        protocols before they can enter L at all.
    """
    r = hf_id.lower()
    is_codon = (
        tokenizer_type == "non_overlapping_3mer_or_codon"
        or "codon" in tokenizer_type
        or "genslm" in r
    )
    mixed = any(m in r for m in _MIXED_MODALITY_MARKERS)
    if mixed:
        return is_codon, True, "coding_only_pending_manual"
    if is_codon:
        return True, False, "coding_only"
    return False, False, "coding_and_noncoding"


# ---------------------------------------------------------------------------
# Parameter count inference
# ---------------------------------------------------------------------------

_PARAM_NAME_RE = re.compile(
    r"(?<![A-Za-z\d])(\d+(?:\.\d+)?)\s*([mb])(?![A-Za-z])",
    flags=re.IGNORECASE,
)

_HYENADNA_PRESET: dict[str, int] = {
    "tiny": 1_700_000,
    "small": 6_600_000,
    "medium": 13_200_000,
    "large": 53_000_000,
}

# Non-HF models that ship as torch-pickled .pt files outside the HF
# transformers convention. For these we cannot trust the HF config.json
# estimator (no config) and we deliberately avoid filename-based guesses
# — the only reliable answer is to load the weights and count them.
# See _count_local_params below.
_NONHF_PACKAGE_DIRS: dict[str, Path] = {
    "lingxusb/megaDNA": REPO_ROOT / "models/modelsHFNoInfo/megaDNA",
}


def _name_to_params(hf_id: str) -> int | None:
    r = hf_id.lower()
    matches = list(_PARAM_NAME_RE.finditer(r))
    if not matches:
        return None
    best = 0
    for m in matches:
        value = float(m.group(1))
        unit = m.group(2).lower()
        n = int(value * (1_000_000_000 if unit == "b" else 1_000_000))
        if n > best:
            best = n
    return best or None


def _count_loaded_module_params(obj: Any) -> int | None:
    """Sum numel() over a torch.nn.Module, a state_dict, or a nested dict
    of tensors. Returns None if the object doesn't expose tensors."""
    import torch

    if hasattr(obj, "parameters"):
        try:
            return int(sum(p.numel() for p in obj.parameters()))
        except Exception:
            return None
    if isinstance(obj, dict):
        total = 0
        any_tensor = False
        stack: list[Any] = [obj]
        while stack:
            item = stack.pop()
            if isinstance(item, torch.Tensor):
                total += item.numel()
                any_tensor = True
            elif isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return int(total) if any_tensor else None
    return None


def _count_megadna_params() -> int | None:
    """Load lingxusb/megaDNA from the local clone and count parameters.

    The .pt file pickles a `megaDNA.megadna.MEGADNA` subclass, so the
    package directory must be on sys.path before torch.load fires the
    unpickler. CPU-only — we only need numel, no compute."""
    pkg = _NONHF_PACKAGE_DIRS["lingxusb/megaDNA"]
    weight = pkg / "megaDNA_phage_145M.pt"
    if not weight.exists():
        return None
    try:
        import sys
        import torch

        p = str(pkg.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
        try:
            import megaDNA.megadna  # noqa: F401
        except ImportError:
            return None
        obj = torch.load(weight, weights_only=False, map_location="cpu")
        return _count_loaded_module_params(obj)
    except Exception:
        return None


def _count_plasmidgpt_params() -> int | None:
    """Load lingxusb/PlasmidGPT pretrained_model.pt and count parameters.

    PlasmidGPT distributes weights only through HF, not bundled locally,
    so we fetch via huggingface_hub (cached after the first call)."""
    if hf_hub_download is None:
        return None
    try:
        import torch

        path = hf_hub_download(
            repo_id="lingxusb/PlasmidGPT",
            filename="pretrained_model.pt",
            local_files_only=False,
        )
        obj = torch.load(path, weights_only=False, map_location="cpu")
        return _count_loaded_module_params(obj)
    except Exception:
        return None


_LOCAL_PARAM_LOADERS: dict[str, "callable[[], int | None]"] = {
    "lingxusb/megaDNA": _count_megadna_params,
    "lingxusb/PlasmidGPT": _count_plasmidgpt_params,
}


def _count_local_params(hf_id: str) -> int | None:
    """Authoritative ground-truth source for non-HF models: instantiate the
    weights and sum tensor numel(). Returns None for HF-loadable models or
    when the local files are unavailable."""
    loader = _LOCAL_PARAM_LOADERS.get(hf_id)
    if loader is None:
        return None
    return loader()


def _hf_cache_dir() -> Path:
    """Resolve the HF Hub cache root, honoring HF_HOME / HUGGINGFACE_HUB_CACHE."""
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _find_cached_weight_file(hf_id: str) -> Path | None:
    """Return a locally-readable weight file for the repo, or None.

    Looks in the standard HF Hub cache at
    `HF_HOME/hub/models--<org>--<repo>/snapshots/`, picking the newest
    snapshot's `model.safetensors` or `pytorch_model.bin`. The audit-time
    network is unreliable; this lets the param-count pipeline succeed
    entirely offline whenever the weights are already cached (which is
    the case for every model in the current catalog)."""
    repo_dir = _hf_cache_dir() / f"models--{hf_id.replace('/', '--')}"
    snaps_dir = repo_dir / "snapshots"
    if not snaps_dir.exists():
        return None
    snapshots = sorted(
        (p for p in snaps_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap in snapshots:
        for filename in ("model.safetensors", "pytorch_model.bin"):
            f = snap / filename
            if f.exists():
                return f.resolve()
    return None


def _count_safetensors_file(path: Path) -> int | None:
    """Sum element counts across all tensors in a local .safetensors file.

    Uses `safetensors.safe_open` with `get_slice`, which only reads the
    file's header (a few KB). The tensor data itself is never materialized,
    so this is cheap even for multi-GB files."""
    try:
        from safetensors import safe_open
    except Exception:
        return None
    try:
        total = 0
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                n = 1
                for dim in shape:
                    n *= int(dim)
                total += n
        return total if total > 0 else None
    except Exception:
        return None


def _count_pytorch_bin_file(path: Path) -> int | None:
    """Sum element counts across the tensors in a locally-cached
    pytorch_model.bin state_dict. Heavier than safetensors (full pickle
    materialization), but the only option for older repos."""
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        return _count_loaded_module_params(obj)
    except Exception:
        return None


def _count_hf_weight_file_params(hf_id: str) -> int | None:
    """Last-resort param counter for HF models when the cheap
    `model_info.safetensors.total` metadata was unavailable.

    Order:
      1. Local cache hit — `model.safetensors` -> read header via
         `safetensors.safe_open` (no network).
      2. Local cache hit — `pytorch_model.bin` -> torch.load + sum numel.
      3. Network download (`pytorch_model.bin`) via huggingface_hub, then
         the same torch.load count. Skipped silently if the mirror /
         gated-repo path refuses the file.

    Returns None when the repo is gated, has no weight file at all
    (code-only utility repos), or all paths fail. Does not raise.
    """
    local = _find_cached_weight_file(hf_id)
    if local is not None:
        if local.name == "model.safetensors":
            n = _count_safetensors_file(local)
            if n is not None:
                return n
        elif local.name == "pytorch_model.bin":
            n = _count_pytorch_bin_file(local)
            if n is not None:
                return n

    if hf_hub_download is None:
        return None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = Path(hf_hub_download(hf_id, filename, local_files_only=False))
        except Exception:
            continue
        if filename == "model.safetensors":
            n = _count_safetensors_file(path)
        else:
            n = _count_pytorch_bin_file(path)
        if n is not None:
            return n
    return None


def _count_hf_safetensors_params(hf_id: str, max_retries: int = 3) -> int | None:
    """Ground-truth param count for HF models via the safetensors metadata
    that the Hub exposes on model_info. This is essentially free — the API
    response carries `safetensors.total` (sum of element counts across all
    safetensors shards) without us downloading any weight file.

    Retries on transient API failures: bulk audits make ~132 sequential
    model_info calls, which is enough to hit rate-limit / connection-reset
    spikes from the Hub. We back off briefly and try again before giving up.

    Returns None when the model has no safetensors files (older pytorch_model.bin
    repos), or when all retries fail."""
    if HfApi is None:
        return None
    import time

    for attempt in range(max_retries):
        try:
            info = HfApi().model_info(hf_id, files_metadata=False)
            break
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5 * (2**attempt))
    safetensors = getattr(info, "safetensors", None)
    if safetensors is None:
        return None
    total = getattr(safetensors, "total", None)
    if isinstance(total, int) and total > 0:
        return total
    return None


def _hyenadna_preset_params(hf_id: str) -> int | None:
    r = hf_id.lower()
    if "hyenadna" not in r:
        return None
    for token, value in _HYENADNA_PRESET.items():
        if f"hyenadna-{token}" in r:
            return value
    return None


def _config_estimate_params(config: dict[str, Any] | None) -> int | None:
    """Rule-of-thumb 12·L·d² for vanilla transformers. Accurate within ~30%
    for decoder transformers; meaningless for state-space hybrids — caller
    can ignore by checking the source field."""
    if not config:
        return None
    n_layers = (
        config.get("num_hidden_layers")
        or config.get("n_layer")
        or config.get("n_layers")
        or config.get("num_layers")
    )
    hidden = (
        config.get("hidden_size")
        or config.get("d_model")
        or config.get("n_embd")
        or config.get("n_embed")
    )
    if not n_layers or not hidden:
        return None
    try:
        return int(12 * int(n_layers) * (int(hidden) ** 2))
    except (TypeError, ValueError):
        return None


def infer_param_count(
    hf_id: str,
    config: dict[str, Any] | None,
    overrides: dict[str, int],
) -> tuple[int | None, str]:
    """Priority: override > name token > hyenadna preset > config estimate."""
    if hf_id in overrides:
        return int(overrides[hf_id]), "override"
    # Ground truth #1: load local non-HF weights (megaDNA / PlasmidGPT).
    n = _count_local_params(hf_id)
    if n is not None:
        return n, "weights_loaded"
    # Ground truth #2: HF Hub `safetensors.total` metadata — exact tensor
    # element count, fetched as a small JSON, no weight download.
    n = _count_hf_safetensors_params(hf_id)
    if n is not None:
        return n, "hf_safetensors_metadata"
    # Ground truth #3: read tensor shapes from a cached/downloaded weight
    # file (safetensors header or pytorch_model.bin). Placed above the name
    # / config-estimate paths so loading-based truth always beats the
    # 12*L*d^2 formula.
    n = _count_hf_weight_file_params(hf_id)
    if n is not None:
        return n, "hf_weight_file_loaded"
    # Below this line are guesses, kept only as fallbacks for repos where
    # no weight file is reachable at all.
    n = _name_to_params(hf_id)
    if n is not None:
        return n, "hf_id_name"
    n = _hyenadna_preset_params(hf_id)
    if n is not None:
        return n, "hyenadna_preset"
    n = _config_estimate_params(config)
    if n is not None:
        return n, "hf_config_estimate"
    return None, "unknown"


# ---------------------------------------------------------------------------
# HF I/O
# ---------------------------------------------------------------------------


def fetch_hf_json(repo_id: str, filename: str) -> dict[str, Any] | None:
    """Try to fetch a small JSON file (config.json / tokenizer_config.json)
    via huggingface_hub. Returns None on any failure — caller decides how to
    proceed without it."""
    if hf_hub_download is None:
        return None
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            etag_timeout=10,
            local_files_only=False,
        )
    except Exception:
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def load_context_overrides(path: Path = CONTEXT_OVERRIDES_PATH) -> dict[str, dict[str, Any]]:
    """Read the highest-priority override yaml for context_tokens."""
    if not path.exists() or _yaml is None:
        return {}
    payload = _yaml.safe_load(path.read_text()) or {}
    out: dict[str, dict[str, Any]] = {}
    for hf_id, entry in payload.items():
        if not isinstance(entry, dict) or "context_tokens" not in entry:
            continue
        value = entry["context_tokens"]
        if isinstance(value, int) and value > 0:
            out[hf_id] = {
                "context_tokens": value,
                "source": str(entry.get("source", "manual_override")),
                "note": str(entry.get("note", "")),
            }
    return out


def load_param_overrides(path: Path = PARAM_OVERRIDES_PATH) -> dict[str, int]:
    """Optional manual fill-in for param_count. Same shape as context overrides."""
    if not path.exists() or _yaml is None:
        return {}
    payload = _yaml.safe_load(path.read_text()) or {}
    out: dict[str, int] = {}
    for hf_id, entry in payload.items():
        if isinstance(entry, dict) and isinstance(entry.get("param_count"), int):
            out[hf_id] = int(entry["param_count"])
    return out


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_markdown(
    path: Path,
    title: str,
    summary_lines: list[str],
    sections: list[tuple[str, str]] | None = None,
) -> None:
    """Write a markdown report: title + summary bullets + optional sections.

    Each section is a (heading, body) pair. Body is rendered as-is (caller
    is responsible for it being valid markdown — typically a table from
    pandas.DataFrame.to_markdown).
    """
    parts: list[str] = [f"# {title}", ""]
    for line in summary_lines:
        parts.append(f"- {line}")
    parts.append("")
    if sections:
        for heading, body in sections:
            parts.append(f"## {heading}")
            parts.append("")
            parts.append(body)
            parts.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def percentile_from_counts(counts: Counter[int], q: float) -> int | None:
    """Quantile from a length->count Counter without materializing the
    expanded list. Used by benchmarks.py for length statistics."""
    if not counts:
        return None
    total = sum(counts.values())
    target = q * total
    cum = 0
    for length in sorted(counts):
        cum += counts[length]
        if cum >= target:
            return int(length)
    return int(max(counts))


__all__ = [
    "REPO_ROOT",
    "MODELS_LIST_PATH",
    "AUDIT_DIR",
    "CONTEXT_OVERRIDES_PATH",
    "PARAM_OVERRIDES_PATH",
    "BENCHMARK_ROOTS",
    "clean_model_list",
    "repo_basename",
    "repo_organization",
    "infer_family",
    "infer_branch",
    "training_paradigm",
    "infer_score_protocol",
    "infer_architecture",
    "infer_tokenizer_type",
    "tokenizer_name_from_config",
    "infer_context_from_name",
    "choose_context",
    "estimate_context_bp",
    "infer_modality",
    "infer_param_count",
    "fetch_hf_json",
    "load_context_overrides",
    "load_param_overrides",
    "write_json",
    "write_markdown",
    "percentile_from_counts",
]
