#!/usr/bin/env python3
"""Phase 1 scoring: AR + MLM models x full main panel -> L → clip → Q → D (per branch).

Implements phase_1.md § 打分协议 + § Sequence-likelihood matrix on
out_panel/main_panel.parquet. Produces the canonical layout:

    out_phase1/
      probes/main_panel.parquet           # copy of input
      probes/control_panel.parquet        # copy of input (if available)
      models/{model_id_slug}.json         # per-model metadata
      scores/{model_id_slug}/probes.parquet
        # one row per panel probe, aligned by probe_id; columns include
        # sum_log_p, ell_per_base, bpb, token_length, scoring_error,
        # plus token_log_probs (per-token log p list; AR = T-1 floats,
        # MLM = content_position_count floats).
      matrices/
        L_AR.npy                          # (M_AR, N) raw sum_log_p, floor-clipped at the
                                            #   2nd-percentile (ModelMap)
        Q_AR.npy                          # double-centered (row mean + col mean removed)
        D_AR.npy                          # (M_AR, M_AR) pairwise squared Euclidean on Q
                                            #   approximates KL under small-divergence Taylor
        L_MLM.npy / Q_MLM.npy / D_MLM.npy
        matrix_metadata.json              # ordered model_ids + probe_ids
                                          # + single-matrix protocol description
      stability/{model_id_slug}.json      # rerun pearson r + max diff
      reports/phase1.md                   # gate summary + per-class GC / ell

Convention per phase_1.md § 单矩阵协议 (ModelMap, raw nats, no length norm, no sign flip):
    sum_log_p_m(x)  = sum_t log p(x_t | x_<t)  (AR)  /  stride PLL k=6  (MLM)
    L[m, x]         = sum_log_p_m(x)           # negative (log p < 0), enters matrix
    L_clipped       = floor_clip(L, q=0.02)    # ModelMap convention
    Q               = double_center(L_clipped) # main matrix for PCA / F_ST
    D               = pairwise_squared_distance(Q)
ell_per_base = sum_log_p / base_length and bpb are written to probes.parquet
as cross-tokenizer-readable reports but are NOT used to build L / Q / D.

Codon-model handling (commit 5e59154 retired the three-matrix split):
    Codon-tokenized models (GenSLM, Codon-NT) emit raw likelihood on every
    probe and enter the single L matrix alongside nucleotide-tokenized
    models. ModelMap's clip + double-center (row mean removes each model's
    overall level, column mean removes each probe's overall difficulty)
    absorbs the codon-vs-nucleotide systematic offset on noncoding probes.
    The `is_codon` flag is still emitted to model metadata for downstream
    diagnostic loadings but no longer gates matrix membership.

Resume:
    Each per-model probes.parquet acts as a checkpoint. Rerunning without
    --force will reuse existing files. The matrices step always re-aggregates
    from the parquets (cheap) so adding a new model + rerunning is enough.

Usage:
    python scripts/run_phase1_scoring.py \\
        [--panel out_panel/main_panel.parquet] \\
        [--control out_panel/control_panel.parquet] \\
        [--out out_phase1] \\
        [--device cuda:6] \\
        [--stride 6] \\
        [--rerun-probes 10] \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glmap.loaders.huggingface import HFCausalLMLoader, HFMaskedLMLoader  # noqa: E402


# --------------------------------------------------------------------- #
# Resume integrity (shared by run_phase1_scoring + run_sweep)
# --------------------------------------------------------------------- #

def parquet_covers_panel(
    score_path: Path, panel_ids: set[str], n_panel: int | None = None
) -> tuple[bool, str]:
    """Return (covers, reason). True means the cached per-model parquet is
    a complete, finite-valued scoring of the panel:

      1. file exists and is readable as parquet,
      2. probe_id column is unique,
      3. row count equals len(panel),
      4. probe_id set equals the panel's,
      5. sum_log_p column exists and every cell is finite.

    Any failure leaves the model in the queue. This is stricter than the
    earlier "probe_id set equality only" check, which would have accepted
    a parquet whose rows were all NaN (the per-probe failure path still
    writes the row with probe_id + sum_log_p=NaN).
    """
    import numpy as _np
    import pandas as _pd

    if n_panel is None:
        n_panel = len(panel_ids)
    if not score_path.exists():
        return False, "file missing"
    try:
        df = _pd.read_parquet(score_path, columns=["probe_id", "sum_log_p"])
    except Exception as exc:
        return False, f"unreadable parquet ({exc})"
    if not df["probe_id"].is_unique:
        return False, "probe_id not unique"
    if len(df) != n_panel:
        return False, f"row count {len(df)} != panel {n_panel}"
    if set(df["probe_id"]) != panel_ids:
        return False, "probe_id set mismatch"
    n_finite = int(_np.isfinite(df["sum_log_p"].to_numpy(dtype=float)).sum())
    if n_finite != n_panel:
        return False, f"only {n_finite}/{n_panel} sum_log_p are finite"
    return True, f"complete ({n_panel} probes, all finite)"


# ----------------------------- Model spec table ----------------------------- #


from glmap.loaders.dispatch import ModelSpec  # noqa: E402  (canonical, kw_only)


# Phase 1 pilot model roster. Current count: 8 AR + 5 MLM = 13.
# GenSLM loaders (25M/250M/2.5B) are now wired in (see src/loaders/genslm.py);
# HyenaDNA and DNABERT 3..6 are still excluded from this DEFAULT — the
# former because its loader uses a separate dispatch path that the
# multi-env sweep handles in scripts/run_rerun_stability.py rather than
# this script, the latter because of single-token overlap-mask leakage
# (phase_1.md supplement scope).
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    # AR branch
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-1M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-17M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="RaphaelMourad/Mistral-DNA-v1-138M-hg38", branch="ar", context_tokens=256),
    ModelSpec(hf_id="lingxusb/megaDNA", branch="ar", context_tokens=131072, loader_kind="megadna"),
    ModelSpec(hf_id="lingxusb/PlasmidGPT", branch="ar", context_tokens=2048, loader_kind="plasmidgpt"),
    ModelSpec(hf_id="GenSLM-25M", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    ModelSpec(hf_id="GenSLM-250M", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    ModelSpec(hf_id="GenSLM-2.5B", branch="ar", context_tokens=2048, is_codon=True, loader_kind="genslm"),
    # MLM branch
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", branch="mlm", context_tokens=2048, trust_remote_code=True),
    ModelSpec(hf_id="InstaDeepAI/agro-nucleotide-transformer-1b", branch="mlm", context_tokens=1024, trust_remote_code=True),
)


# NOTE: the three-matrix codon-NaN protocol (R_pan_DNA / R_coding_only /
# R_nucleotide_only) was retired in commit 5e59154 "matrix protocol: drop
# three-matrix split, build one L/Q/D per branch". The old NON_CODING_CLASSES
# / CODING_CLASS constants that gated that routing have been removed.


# ------------------------------- Per-model -------------------------------- #


def _score_ar_one_probe(loader: HFCausalLMLoader, probe: dict) -> dict:
    """Score one probe with an AR loader. Returns the full record dict on
    success or a NaN-filled record with a populated scoring_error on failure."""
    try:
        rec = loader.score_record(probe["sequence"])
        return {
            "probe_id": probe["probe_id"],
            "functional_element": probe["functional_element"],
            "sequence_length_bp": rec.base_length,
            "token_length": rec.token_length,
            "token_length_no_special": rec.token_length_no_special,
            "special_tokens_count": rec.special_tokens_count,
            "predictable_tokens": rec.predictable_tokens,
            "sum_log_p": rec.sum_log_p,
            "ell_per_base": rec.ell_per_base,
            "bpb": rec.bpb,
            "ce_loss": rec.ce_loss,
            "token_log_probs": list(rec.token_log_probs),
            "scoring_error": "",
        }
    except Exception as exc:
        return {
            "probe_id": probe["probe_id"],
            "functional_element": probe["functional_element"],
            "sequence_length_bp": len(probe["sequence"]),
            "token_length": None,
            "token_length_no_special": None,
            "special_tokens_count": None,
            "predictable_tokens": None,
            "sum_log_p": float("nan"),
            "ell_per_base": float("nan"),
            "bpb": float("nan"),
            "ce_loss": float("nan"),
            "token_log_probs": None,
            "scoring_error": f"{type(exc).__name__}: {exc}",
        }


def _score_mlm_one_probe(
    loader: HFMaskedLMLoader, probe: dict, stride: int
) -> dict:
    try:
        rec = loader.score_record(probe["sequence"], stride=stride)
        return {
            "probe_id": probe["probe_id"],
            "functional_element": probe["functional_element"],
            "sequence_length_bp": rec.base_length,
            "token_length": rec.token_length,
            "token_length_no_special": rec.token_length_no_special,
            "stride": rec.stride,
            "content_position_count": rec.content_position_count,
            "masked_position_count": rec.masked_position_count,
            "sum_log_p": rec.sum_log_p,
            "ell_per_base": rec.ell_per_base,
            "bpb": rec.bpb,
            "token_log_probs": list(rec.token_log_probs),
            "scoring_error": "",
        }
    except Exception as exc:
        return {
            "probe_id": probe["probe_id"],
            "functional_element": probe["functional_element"],
            "sequence_length_bp": len(probe["sequence"]),
            "token_length": None,
            "token_length_no_special": None,
            "stride": stride,
            "content_position_count": None,
            "masked_position_count": None,
            "sum_log_p": float("nan"),
            "ell_per_base": float("nan"),
            "bpb": float("nan"),
            "token_log_probs": None,
            "scoring_error": f"{type(exc).__name__}: {exc}",
        }


def _score_model(
    spec: ModelSpec,
    panel: pd.DataFrame,
    device: str,
    stride: int,
    progress_every: int = 100,
):
    """Load `spec`, score every probe in `panel`. Returns (DataFrame, loader).

    The loader is returned in `.eval()` state so the caller can run rerun-
    stability without re-loading weights.
    """
    if spec.loader_kind == "megadna":
        from glmap.loaders.megadna import MegaDNALoader
        loader = MegaDNALoader(device=device)
    elif spec.loader_kind == "plasmidgpt":
        from glmap.loaders.plasmidgpt import PlasmidGPTLoader
        loader = PlasmidGPTLoader(device=device)
    elif spec.loader_kind == "genslm":
        from glmap.loaders.genslm import GenSLMLoader
        loader = GenSLMLoader(spec.hf_id, device=device)
    elif spec.loader_kind == "hf" and spec.branch == "ar":
        loader = HFCausalLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "hf" and spec.branch == "mlm":
        loader = HFMaskedLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "botanic":
        from glmap.loaders.custom_mlm import BotanicLoader
        loader = BotanicLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "hyenadna":
        from glmap.loaders.hyenadna import HyenaDNALoader
        loader = HyenaDNALoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "aido":
        from glmap.loaders.aido import AIDOLoader
        loader = AIDOLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "evo2":
        from glmap.loaders.evo2_loader import Evo2Loader
        loader = Evo2Loader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "evo1":
        from glmap.loaders.evo1_loader import Evo1Loader
        loader = Evo1Loader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "plantbimoe":
        from glmap.loaders.custom_mlm import PlantBiMoELoader
        loader = PlantBiMoELoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "mutbert":
        from glmap.loaders.custom_mlm import MutBERTLoader
        loader = MutBERTLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
        )
    elif spec.loader_kind == "generator":
        from glmap.loaders.generator import GENERatorLoader
        loader = GENERatorLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "carbon":
        from glmap.loaders.carbon import CarbonCausalLMLoader
        loader = CarbonCausalLMLoader(
            hf_id=spec.hf_id,
            context_tokens=spec.context_tokens,
            device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "ntv3":
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
        )
    else:
        raise ValueError(
            f"unknown loader_kind={spec.loader_kind!r} branch={spec.branch!r} "
            f"for {spec.hf_id}"
        )

    print(f"[{spec.branch}] loading {spec.hf_id}", flush=True)
    t_load = time.time()
    loader.load()
    print(f"[{spec.branch}] loaded in {time.time() - t_load:.1f}s", flush=True)

    probes = panel.to_dict("records")
    rows: list[dict] = []
    t_score = time.time()
    for i, probe in enumerate(probes, start=1):
        if spec.branch == "ar":
            rows.append(_score_ar_one_probe(loader, probe))
        else:
            rows.append(_score_mlm_one_probe(loader, probe, stride=stride))
        if i % progress_every == 0 or i == len(probes):
            err_count = sum(1 for r in rows if r["scoring_error"])
            # Surface a sample of the actual exception so a "100/100 errors"
            # run is visible at glance instead of looking like normal progress.
            sample_err = ""
            if err_count > 0:
                sample_err = next(
                    (r["scoring_error"] for r in rows if r["scoring_error"]),
                    "",
                )
                sample_err = f"  first_err={sample_err[:120]}"
            print(
                f"[{spec.branch}] {spec.slug} {i}/{len(probes)} "
                f"errors={err_count} elapsed={time.time() - t_score:.1f}s"
                f"{sample_err}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    return df, loader


# ------------------------------ Rerun gate -------------------------------- #


def _rerun_stability(
    loader,
    panel: pd.DataFrame,
    spec: ModelSpec,
    stride: int,
    n_probes: int,
) -> dict:
    """Score the first `n_probes` probes twice; report Pearson r + max diff.

    phase_0.md gate is r >= 0.95. With HF eval() + no_grad we expect
    bit-identical results (r = 1.0, diff = 0.0). Anomalies surface dropout
    leaks or unstable kernel paths.
    """
    if n_probes <= 0:
        return {"n_probes": 0, "pearson_r": 1.0, "max_abs_diff": 0.0, "passes_gate": True}

    sub = panel.head(n_probes).to_dict("records")
    if spec.branch == "ar":
        run1 = [_score_ar_one_probe(loader, p)["ell_per_base"] for p in sub]
        run2 = [_score_ar_one_probe(loader, p)["ell_per_base"] for p in sub]
    else:
        run1 = [_score_mlm_one_probe(loader, p, stride)["ell_per_base"] for p in sub]
        run2 = [_score_mlm_one_probe(loader, p, stride)["ell_per_base"] for p in sub]

    arr1 = np.array(run1, dtype=np.float64)
    arr2 = np.array(run2, dtype=np.float64)
    finite = np.isfinite(arr1) & np.isfinite(arr2)
    if finite.sum() < 2:
        # Pearson is undefined; only "pass" if both runs match bit-for-bit on the finite slice.
        identical = bool(np.all(arr1[finite] == arr2[finite])) if finite.any() else False
        pearson = 1.0 if identical else 0.0
        max_diff = (
            float(np.abs(arr1[finite] - arr2[finite]).max()) if finite.any() else float("inf")
        )
    else:
        a = arr1[finite] - arr1[finite].mean()
        b = arr2[finite] - arr2[finite].mean()
        denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if denom == 0.0:
            pearson = 1.0 if np.all(arr1[finite] == arr2[finite]) else 0.0
        else:
            pearson = float((a * b).sum() / denom)
        max_diff = float(np.abs(arr1[finite] - arr2[finite]).max())

    return {
        "n_probes": int(finite.sum()),
        "pearson_r": pearson,
        "max_abs_diff": max_diff,
        "passes_gate": pearson >= 0.95,
        "run1_values": arr1.tolist(),
        "run2_values": arr2.tolist(),
    }


# --------------------------- Matrix assembly ----------------------------- #


@dataclass
class BranchMatrices:
    """ModelMap-style L / Q / D triple for one branch.

    All models in the branch are stacked into a single matrix on the full
    panel (no codon / coding-only split, no per-row NaN policy). Whatever
    `sum_log_p` the model emits on each probe enters the matrix — codon
    models on noncoding probes contribute as-is.

    L  : raw `sum_log_p` floor-clipped at 2nd percentile (no length norm,
         no sign flip — cells are negative log-likelihoods).
    Q  : double-centered L (row mean + column mean removed). Consumed
         by PCA / distance diagnostics.
    D  : (M, M) pairwise squared Euclidean distance on Q. Approximates
         KL under small-divergence Taylor (ModelMap Sec. 6.1).
    """

    branch: str                              # "ar" | "mlm"
    model_ids: list[str]
    probe_ids: list[str]
    L: np.ndarray
    Q: np.ndarray
    D: np.ndarray
    clip_threshold: float


def _build_branch_matrices(
    specs: list[ModelSpec],
    panel: pd.DataFrame,
    scores_dir: Path,
    allow_missing: bool = False,
) -> tuple["BranchMatrices", dict]:
    """Aggregate per-model `sum_log_p` into a single matrix per branch, then
    apply the ModelMap clip + double-center + pairwise-distance pipeline.

    No codon-vs-nucleotide split. No coding-only column subset. Every model
    × every probe enters L; whatever the model scores on each probe is
    taken at face value.

    allow_missing=True drops any spec whose parquet is missing or contains
    a non-finite sum_log_p column from the L matrix entirely (instead of
    leaving its row NaN). Returned `model_ids` then reflects the kept set,
    not the input spec list. Mutually exclusive with --strict-aggregate at
    the caller level.
    """
    from glmap.matrices.build import build_L_Q_D

    N = len(panel)
    probe_ids = panel["probe_id"].tolist()

    panel_probe_set = set(probe_ids)
    panel_n = len(probe_ids)

    def _safe_read_sum_log_p(score_path: Path) -> tuple[np.ndarray | None, str]:
        """Read probe_id + sum_log_p, align to panel order, return
        (aligned_array_or_None, reason). Treats any read-time failure
        (parquet corrupt, missing column, duplicate probe_id collapsing
        on reindex, extra/missing probes vs panel) as "model unavailable"
        rather than aborting the whole aggregate. Returned array is
        (N,) float64 or None on failure; reason is a short diagnostic
        string.

        Mirrors the EXACT contract of parquet_covers_panel() (used by
        the resume layer in run_sweep.py): row count must equal panel
        N AND probe_id set must equal panel's. Otherwise a stale
        parquet from a different panel build could silently be re-
        indexed onto the current panel — losing the extra probes
        without warning."""
        try:
            df = pd.read_parquet(score_path, columns=["probe_id", "sum_log_p"])
        except Exception as exc:
            return None, f"parquet read failed: {type(exc).__name__}: {exc}"
        if "probe_id" not in df.columns or "sum_log_p" not in df.columns:
            return None, "missing probe_id or sum_log_p column"
        if df["probe_id"].duplicated().any():
            n_dup = int(df["probe_id"].duplicated().sum())
            return None, f"{n_dup} duplicated probe_id rows; reindex would silently drop"
        if len(df) != panel_n:
            return None, (
                f"row count {len(df)} != panel N={panel_n}; parquet from a "
                f"different panel build — re-score against the current panel"
            )
        parquet_probe_set = set(df["probe_id"])
        if parquet_probe_set != panel_probe_set:
            extra = parquet_probe_set - panel_probe_set
            missing = panel_probe_set - parquet_probe_set
            return None, (
                f"probe_id set mismatch vs panel "
                f"({len(extra)} extra, {len(missing)} missing); "
                f"parquet from a different panel build"
            )
        try:
            aligned = df.set_index("probe_id").reindex(probe_ids)
        except Exception as exc:
            return None, f"reindex failed: {type(exc).__name__}: {exc}"
        return aligned["sum_log_p"].to_numpy(dtype=np.float64), "ok"

    # First pass: figure out which specs actually have a complete (all-finite)
    # parquet on disk. Used both for the missing/partial diagnostic record
    # and (when allow_missing=True) for filtering specs out of L.
    # spec_status: (spec, has_parquet, n_finite, aligned_or_None, reason)
    spec_status: list[tuple[ModelSpec, bool, int, np.ndarray | None, str]] = []
    corrupt_models: list[tuple[str, str]] = []  # (hf_id, reason)
    for spec in specs:
        score_path = scores_dir / spec.slug / "probes.parquet"
        if not score_path.exists():
            spec_status.append((spec, False, 0, None, "no parquet"))
            continue
        aligned, reason = _safe_read_sum_log_p(score_path)
        if aligned is None:
            corrupt_models.append((spec.hf_id, reason))
            spec_status.append((spec, False, 0, None, reason))
            continue
        n_finite = int(np.isfinite(aligned).sum())
        spec_status.append((spec, True, n_finite, aligned, "ok"))

    if allow_missing:
        kept_specs = [s for (s, has, n_fin, _a, _r) in spec_status if has and n_fin == N]
    else:
        kept_specs = specs

    M = len(kept_specs)
    kept_slugs = {s.slug for s in kept_specs}

    L_raw = np.full((M, N), np.nan, dtype=np.float64)
    missing_models: list[str] = []
    partial_models: list[tuple[str, int]] = []   # (hf_id, n_scored)
    scored_models: list[str] = []
    row_i_of_kept = {s.slug: i for i, s in enumerate(kept_specs)}
    for spec, has_parquet, n_finite, aligned, _reason in spec_status:
        if not has_parquet:
            missing_models.append(spec.hf_id)
            continue
        if n_finite < N:
            partial_models.append((spec.hf_id, n_finite))
            if not allow_missing:
                # Keep the NaN-row behavior: still write the finite cells
                # into the global L_raw at this spec's original index. But
                # under allow_missing=True we already dropped this spec at
                # filter time, so don't try to write.
                pass
        if spec.slug not in kept_slugs:
            # Dropped under allow_missing=True
            continue
        # `aligned` was computed in the first pass and reused here — no
        # second parquet read needed. Only probe_id + sum_log_p are
        # tracked; token_log_probs (per-token arrays, typically 100×-
        # 1000× larger) never enters memory.
        assert aligned is not None  # guaranteed by has_parquet branch
        finite = np.isfinite(aligned)
        L_raw[row_i_of_kept[spec.slug], finite] = aligned[finite]
        scored_models.append(spec.hf_id)

    if missing_models:
        print(
            f"[matrices] WARN: {len(missing_models)} models have no parquet; "
            f"rows stay NaN:",
            flush=True,
        )
        for hf_id in missing_models[:10]:
            print(f"  - {hf_id}", flush=True)
    if partial_models:
        print(
            f"[matrices] WARN: {len(partial_models)} models have partial probe "
            f"coverage (less than {N}); affected probes stay NaN:",
            flush=True,
        )
        for hf_id, n in partial_models[:10]:
            print(f"  - {hf_id}: {n}/{N}", flush=True)
    if corrupt_models:
        print(
            f"[matrices] WARN: {len(corrupt_models)} models have CORRUPT "
            f"parquets that aborted the per-model read; treated as missing:",
            flush=True,
        )
        for hf_id, reason in corrupt_models[:10]:
            print(f"  - {hf_id}: {reason}", flush=True)

    result = build_L_Q_D(L_raw)
    # IMPORTANT: model_ids must align row-for-row with L. Under
    # allow_missing=True we filtered specs → kept_specs at the top,
    # and L_raw was sized M = len(kept_specs); returning [s.hf_id for s
    # in specs] would mis-label rows (and bleed into matrix_metadata.json
    # / downstream figure labels).
    return BranchMatrices(
        branch=kept_specs[0].branch if kept_specs else (specs[0].branch if specs else ""),
        model_ids=[s.hf_id for s in kept_specs],
        probe_ids=probe_ids,
        L=result["L_clipped"],
        Q=result["Q"],
        D=result["D"],
        clip_threshold=result["clip_threshold"],
    ), {
        "scored_models": scored_models,
        "missing_models": missing_models,
        "partial_models": partial_models,
        "corrupt_models": corrupt_models,
    }


def _save_branch_matrices(
    out_dir: Path,
    branch_label: str,
    matrices: BranchMatrices,
) -> dict:
    """Save L / Q / D for one branch.

    File-naming convention:
      L_<branch>.npy   raw sum_log_p, floor-clipped at 2nd percentile
      Q_<branch>.npy   double-centered L (consume in PCA / distance diagnostics)
      D_<branch>.npy   (M, M) pairwise squared Euclidean on Q
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"L_{branch_label}.npy", matrices.L)
    np.save(out_dir / f"Q_{branch_label}.npy", matrices.Q)
    np.save(out_dir / f"D_{branch_label}.npy", matrices.D)
    metadata: dict = {
        f"L_{branch_label}": {
            "shape": list(matrices.L.shape),
            "row_model_ids": matrices.model_ids,
            "col_probe_ids": matrices.probe_ids,
            "n_nan_cells": int(np.isnan(matrices.L).sum()),
            "clip_threshold": matrices.clip_threshold,
        },
        f"Q_{branch_label}": {
            "shape": list(matrices.Q.shape),
            "row_model_ids": matrices.model_ids,
            "col_probe_ids": matrices.probe_ids,
            "n_nan_cells": int(np.isnan(matrices.Q).sum()),
            "note": "double-centered L (row mean + column mean removed). "
            "Consume here in PCA and distance diagnostics.",
        },
        f"D_{branch_label}": {
            "shape": list(matrices.D.shape),
            "row_model_ids": matrices.model_ids,
            "col_model_ids": matrices.model_ids,
            "n_nan_cells": int(np.isnan(matrices.D).sum()),
            "note": "pairwise squared Euclidean on Q; approximates KL under "
            "small-divergence Taylor (ModelMap Sec. 6.1).",
        },
    }
    return metadata


# ----------------------------- Report writer ------------------------------ #


def _write_report(
    path: Path,
    panel: pd.DataFrame,
    ar_matrices: BranchMatrices | None,
    mlm_matrices: BranchMatrices | None,
    stability: dict[str, dict],
    stride: int,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 1 Scoring Report")
    lines.append("")
    lines.append("Generated by `scripts/run_phase1_scoring.py`. Implements "
                 "phase_1.md § 打分协议 + § 单矩阵协议 on the frozen Stage 2 "
                 "main panel (10,000 probes × 14 functional elements × 4 "
                 "species groups; see `data/panel_sources.yaml`).")
    lines.append("")
    lines.append("## Panel")
    lines.append("")
    lines.append(f"- Total probes: {len(panel)}")
    lines.append(f"- Class breakdown:")
    lines.append("")
    cb = panel.groupby("functional_element").size().rename("count").to_frame()
    cb["mean_GC"] = panel.groupby("functional_element")["GC_content"].mean().round(4)
    lines.append(cb.to_markdown())
    lines.append("")

    for label, matrices in (("AR", ar_matrices), ("MLM", mlm_matrices)):
        lines.append(f"## {label} branch")
        lines.append("")
        if matrices is None or matrices.L.size == 0:
            lines.append("(no models)")
            lines.append("")
            continue
        lines.append(f"- Models (rows of L_{label}):")
        for mid in matrices.model_ids:
            lines.append(f"  - `{mid}`")
        lines.append("")
        lines.append(
            f"- L_{label} shape: `{matrices.L.shape}` "
            f"(NaN cells: {int(np.isnan(matrices.L).sum())}, "
            f"clip threshold: {matrices.clip_threshold:.3f} nats)"
        )
        lines.append("")

        # Per-class mean L (ModelMap convention: raw sum log-likelihood,
        # higher = easier for the model; values are negative nats).
        per_class = []
        for cls in sorted(set(panel["functional_element"].tolist())):
            mask = (panel["functional_element"].values == cls)
            sub = matrices.L[:, mask]
            with np.errstate(invalid="ignore"):
                per_class.append({
                    "class": cls,
                    "n_probes": int(mask.sum()),
                    "mean_L": float(np.nanmean(sub)) if np.isfinite(sub).any() else float("nan"),
                    "std_L": float(np.nanstd(sub)) if np.isfinite(sub).any() else float("nan"),
                })
        lines.append("- Per-class mean L (raw sum log-likelihood in nats, higher = easier):")
        lines.append("")
        lines.append(pd.DataFrame(per_class).round(4).to_markdown(index=False))
        lines.append("")

    lines.append("## Rerun stability (phase_0.md gate: pearson r >= 0.95)")
    lines.append("")
    if stability:
        rows = []
        for hf_id, rep in stability.items():
            rows.append({
                "model": hf_id,
                "n_probes": rep["n_probes"],
                "pearson_r": round(rep["pearson_r"], 6),
                "max_abs_diff": rep["max_abs_diff"],
                "passes_gate": rep["passes_gate"],
            })
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
    else:
        lines.append("(no stability data recorded)")
    lines.append("")
    lines.append(f"## Scoring parameters")
    lines.append("")
    lines.append(f"- MLM stride k (primary): {stride}")
    lines.append("- AR signature: forward only (RC sanity check retired in phase_1.md § Sanity Check)")
    lines.append("- Matrix protocol: L → clip(q=0.02) → Q (double-center) → D (pairwise sq Euclidean)")
    lines.append("- Codon-model handling: raw likelihood on full panel; codon-vs-nucleotide offset "
                 "on noncoding probes absorbed by ModelMap double-centering (commit 5e59154 retired "
                 "the three-matrix split)")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# --------------------------------- main ----------------------------------- #


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, default=REPO_ROOT / "out_panel/main_panel.parquet")
    p.add_argument("--control", type=Path,
                   default=REPO_ROOT / "out_panel/control_panel.parquet")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "out_phase1")
    p.add_argument(
        "--device",
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="cuda:N or cpu. Default auto-picks cuda:0 when CUDA is available.",
    )
    p.add_argument("--stride", type=int, default=6,
                   help="MLM stride k (phase_1.md primary k=6).")
    p.add_argument("--rerun-probes", type=int, default=10,
                   help="Number of probes used for the rerun-stability gate.")
    p.add_argument("--force", action="store_true",
                   help="Re-score every model even if its parquet exists.")
    p.add_argument("--max-probes", type=int, default=None,
                   help="Debug: truncate panel to N probes before scoring.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated SUBSTRING filter on hf_id (for debug "
                        "and human use). NOTE: substrings can collide — e.g. "
                        "'evo2_7b' matches evo2_7b, evo2_7b_base, "
                        "evo2_7b_262k. For parallel sweeps where each "
                        "subprocess must score exactly one model, use "
                        "--hf-ids instead (exact match).")
    p.add_argument("--hf-ids", type=str, default=None,
                   help="Comma-separated list of EXACT hf_id strings. Each "
                        "must match the audit (or DEFAULT_MODELS) exactly. "
                        "Use this from run_sweep.py and other parallel "
                        "drivers to avoid the --only substring collisions.")
    p.add_argument("--skip-aggregate", action="store_true",
                   help="Skip the matrix-build + report step. Useful when "
                        "running parallel per-model jobs that should not race "
                        "on out_phase1/matrices/. Run a final aggregate pass "
                        "without --only afterward.")
    p.add_argument("--strict-aggregate", action="store_true",
                   help="When aggregating, fail-fast if any model's parquet "
                        "is missing or has partial probe coverage. Without "
                        "this flag the matrix is built with NaN rows / cells "
                        "for incomplete models (a record of which models "
                        "actually contributed is always written to "
                        "out/matrices/scored_models_actual.json). Use the "
                        "strict mode for the final Stage 4 aggregate.")
    p.add_argument("--allow-missing", action="store_true",
                   help="When aggregating, DROP rows for models with no "
                        "parquet (or with partial / NaN-poisoned parquet) "
                        "instead of keeping NaN rows. Useful for a 'preview' "
                        "aggregate while a few stragglers are still running. "
                        "The resulting L/Q/D matrices have M_actual ≤ "
                        "len(specs); scored_models_actual.json records which "
                        "rows are present. Mutually exclusive with "
                        "--strict-aggregate.")
    p.add_argument("--from-audit", action="store_true",
                   help="Source the model roster from data/audits/models.json "
                        "(123 candidates as of 2026-05-20) instead of the "
                        "13-model DEFAULT_MODELS pilot set. The matrix "
                        "aggregation step also uses the audit-derived list, "
                        "so the L / Q / D matrices cover all scorable audit "
                        "models. This is the Stage 4 / phase 2 entry point.")
    p.add_argument("--audit-json", type=Path,
                   default=REPO_ROOT / "data" / "audits" / "models.json",
                   help="Audit JSON path (only consulted with --from-audit).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.panel.exists():
        raise SystemExit(f"--panel not found: {args.panel}")
    panel = pd.read_parquet(args.panel)
    required_cols = {"probe_id", "sequence", "functional_element", "GC_content"}
    missing_cols = required_cols - set(panel.columns)
    if missing_cols:
        raise SystemExit(f"panel parquet is missing columns: {missing_cols}")
    if not panel["probe_id"].is_unique:
        raise SystemExit("panel.probe_id is not unique; cannot align scores by probe_id")
    if args.max_probes is not None and args.max_probes < len(panel):
        panel = panel.head(args.max_probes).reset_index(drop=True)
        print(
            f"[panel] --max-probes={args.max_probes}; truncated to {len(panel)} probes",
            flush=True,
        )
    print(f"[panel] loaded {len(panel)} probes from {args.panel}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probes").mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out / "probes" / "main_panel.parquet", index=False)
    if args.control.exists():
        control = pd.read_parquet(args.control)
        control.to_parquet(args.out / "probes" / "control_panel.parquet", index=False)
        print(f"[panel] copied control panel ({len(control)} probes)", flush=True)

    (args.out / "models").mkdir(parents=True, exist_ok=True)
    (args.out / "scores").mkdir(parents=True, exist_ok=True)
    (args.out / "stability").mkdir(parents=True, exist_ok=True)
    (args.out / "matrices").mkdir(parents=True, exist_ok=True)

    stability_reports: dict[str, dict] = {}

    # Resolve the model roster. --from-audit lazy-imports the audit-→ModelSpec
    # router from run_rerun_stability so we don't duplicate dispatch logic
    # (and so there's no circular import — run_rerun_stability already
    # imports ModelSpec from this module).
    if args.from_audit:
        if not args.audit_json.exists():
            raise SystemExit(f"--from-audit but audit json not found: {args.audit_json}")
        from glmap.loaders.dispatch import specs_from_audit
        roster = specs_from_audit(audit_path=args.audit_json)
        roster_name = f"audit ({args.audit_json.name})"
    else:
        roster = list(DEFAULT_MODELS)
        roster_name = "DEFAULT_MODELS"

    if args.hf_ids and args.only:
        raise SystemExit("--hf-ids and --only are mutually exclusive")
    if args.hf_ids:
        wanted = {s.strip() for s in args.hf_ids.split(",") if s.strip()}
        roster_by_id = {s.hf_id: s for s in roster}
        missing = wanted - set(roster_by_id)
        if missing:
            raise SystemExit(
                f"--hf-ids referenced {len(missing)} hf_id(s) not in "
                f"{roster_name}: {sorted(missing)[:5]}..."
            )
        models_to_run = [roster_by_id[h] for h in wanted]
        print(
            f"[hf-ids] exact match: {len(models_to_run)}/{len(roster)} models "
            f"from {roster_name}: " + ", ".join(s.slug for s in models_to_run),
            flush=True,
        )
    elif args.only:
        only_tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        models_to_run = [
            s for s in roster if any(t in s.hf_id for t in only_tokens)
        ]
        print(
            f"[only] substring match: {len(models_to_run)}/{len(roster)} models "
            f"from {roster_name}: " + ", ".join(s.slug for s in models_to_run),
            flush=True,
        )
    else:
        models_to_run = list(roster)
        print(f"[roster] {len(models_to_run)} models from {roster_name}", flush=True)

    def _write_model_meta(spec: ModelSpec, meta_path: Path) -> None:
        """Per-model spec snapshot. Called on every iteration (both
        skipped-because-cached and freshly-scored) so newly-added spec
        fields (e.g. loader_kind, length_multiple) reflect the current
        dispatch on disk regardless of whether the parquet was rebuilt.
        Without this the per-model JSON would only update on re-score
        and stale metadata would diverge from the actual loader."""
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({
            "hf_id": spec.hf_id,
            "branch": spec.branch,
            "context_tokens": spec.context_tokens,
            "trust_remote_code": spec.trust_remote_code,
            "is_codon": spec.is_codon,
            "loader_kind": spec.loader_kind,
            "length_multiple": spec.length_multiple,
            "slug": spec.slug,
            "stride_primary": args.stride if spec.branch == "mlm" else None,
            "scoring_protocol": (
                "AR forward sum_log_p (raw nats, no length norm, no sign flip; "
                "ModelMap convention)"
                if spec.branch == "ar"
                else f"MLM stride pseudo-log-likelihood, k={args.stride} "
                     "(raw nats, no length norm, no sign flip)"
            ),
        }, indent=2))

    for spec in models_to_run:
        score_dir = args.out / "scores" / spec.slug
        score_dir.mkdir(parents=True, exist_ok=True)
        score_path = score_dir / "probes.parquet"
        stability_path = args.out / "stability" / f"{spec.slug}.json"
        meta_path = args.out / "models" / f"{spec.slug}.json"

        if score_path.exists() and not args.force:
            # Resume-integrity check: file existing is necessary but not
            # sufficient. The check verifies that the cached parquet has
            # the full panel's probe_id set AND that every sum_log_p is
            # finite — a per-probe scoring failure would still write a
            # row with sum_log_p=NaN, which the earlier set-equality-only
            # check accepted as complete.
            ok, reason = parquet_covers_panel(score_path, set(panel["probe_id"]),
                                              n_panel=len(panel))
            if ok:
                print(
                    f"[skip] {spec.slug}: {score_path} {reason}",
                    flush=True,
                )
                if stability_path.exists():
                    stability_reports[spec.hf_id] = json.loads(
                        stability_path.read_text()
                    )
                else:
                    print(
                        f"[skip] {spec.slug}: no stability JSON found; "
                        "pass --force to re-score or run rerun-stability "
                        "separately.",
                        flush=True,
                    )
                # Refresh metadata even on skip so loader_kind / length_multiple
                # always reflect the current dispatch logic, not whatever the
                # spec was when the parquet was originally written.
                _write_model_meta(spec, meta_path)
                continue
            print(
                f"[resume] {spec.slug}: {score_path} {reason}; re-scoring.",
                flush=True,
            )

        loader = None
        try:
            df, loader = _score_model(
                spec=spec, panel=panel, device=args.device, stride=args.stride
            )
            df.to_parquet(score_path, index=False)
            print(
                f"[{spec.branch}] {spec.slug} wrote scores -> {score_path}",
                flush=True,
            )

            stab = _rerun_stability(
                loader=loader, panel=panel, spec=spec,
                stride=args.stride, n_probes=args.rerun_probes,
            )
            stability_reports[spec.hf_id] = stab
            stability_path.write_text(json.dumps(stab, indent=2))
            print(
                f"[{spec.branch}] {spec.slug} rerun pearson_r={stab['pearson_r']:.6f} "
                f"max_diff={stab['max_abs_diff']:.2e} gate={stab['passes_gate']}",
                flush=True,
            )
        except Exception as exc:
            # Log the trace but don't kill the whole run; the matrices step
            # will fill the missing row with NaN.
            traceback.print_exc()
            print(
                f"[fail] {spec.slug}: scoring aborted with "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        finally:
            # Free GPU memory between models. `loader` may be None if the
            # try block raised before _score_model returned.
            if loader is not None:
                del loader
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

        _write_model_meta(spec, meta_path)

    if args.skip_aggregate:
        print("[done] --skip-aggregate; skipping matrix build + report.", flush=True)
        return

    # ---- Aggregate matrices from per-model parquets ---- #
    # Aggregate over the full roster (audit-derived or DEFAULT_MODELS), not
    # the --only filter — so a multi-process parallel sweep can write each
    # model's parquet under --only and a final aggregate pass without --only
    # builds the matrices over everything on disk.
    if args.strict_aggregate and args.allow_missing:
        raise SystemExit(
            "--strict-aggregate and --allow-missing are mutually exclusive: "
            "the first fails on missing models, the second drops them."
        )

    ar_specs = [s for s in roster if s.branch == "ar"]
    mlm_specs = [s for s in roster if s.branch == "mlm"]

    ar_matrices, ar_actual = _build_branch_matrices(
        ar_specs, panel, args.out / "scores", allow_missing=args.allow_missing,
    )
    mlm_matrices, mlm_actual = _build_branch_matrices(
        mlm_specs, panel, args.out / "scores", allow_missing=args.allow_missing,
    )

    # Persist the explicit "which models actually contributed scores" record so
    # downstream analysis doesn't have to introspect NaN rows of the L matrix
    # to figure out which rows are real.
    actual_path = args.out / "matrices" / "scored_models_actual.json"
    actual_path.write_text(json.dumps({
        "ar": ar_actual,
        "mlm": mlm_actual,
        "n_ar_expected": len(ar_specs),
        "n_mlm_expected": len(mlm_specs),
        "n_ar_scored": len(ar_actual["scored_models"]),
        "n_mlm_scored": len(mlm_actual["scored_models"]),
    }, indent=2))

    n_missing = len(ar_actual["missing_models"]) + len(mlm_actual["missing_models"])
    n_partial = len(ar_actual["partial_models"]) + len(mlm_actual["partial_models"])
    if args.strict_aggregate and (n_missing or n_partial):
        raise SystemExit(
            f"--strict-aggregate: {n_missing} model(s) missing parquet, "
            f"{n_partial} model(s) with partial probe coverage. "
            f"See {actual_path} for details. Run scoring to completion "
            "before aggregating, or drop --strict-aggregate to accept "
            "NaN rows / cells."
        )

    matrices_meta: dict = {}
    matrices_meta.update(_save_branch_matrices(args.out / "matrices", "AR", ar_matrices))
    matrices_meta.update(_save_branch_matrices(args.out / "matrices", "MLM", mlm_matrices))
    matrices_meta["protocol"] = (
        "Single-matrix protocol per branch (ModelMap, commit 5e59154). "
        "L[m, x] = sum_log_p_m(x) in raw nats (no length normalization, no "
        "sign flip). L_clipped = floor_clip(L, q=0.02); Q = double_center("
        "L_clipped); D = pairwise_squared_distance(Q). Codon models "
        "(GenSLM, Codon-NT) emit raw likelihood on every probe and enter L "
        "alongside nucleotide-tokenized models; the codon-vs-nucleotide "
        "systematic offset on noncoding probes is absorbed by the row-mean "
        "and column-mean subtraction during double-centering. The earlier "
        "three-matrix split (R_pan_DNA / R_coding_only / R_nucleotide_only) "
        "is retired."
    )
    matrices_meta["scoring_stride_mlm"] = args.stride
    (args.out / "matrices" / "matrix_metadata.json").write_text(
        json.dumps(matrices_meta, indent=2)
    )

    # ---- Report ---- #
    _write_report(
        path=args.out / "reports" / "phase1.md",
        panel=panel,
        ar_matrices=ar_matrices if ar_specs else None,
        mlm_matrices=mlm_matrices if mlm_specs else None,
        stability=stability_reports,
        stride=args.stride,
    )
    print(f"[done] wrote outputs to {args.out}", flush=True)

    failed_gates = [
        hf_id for hf_id, rep in stability_reports.items() if not rep["passes_gate"]
    ]
    if failed_gates:
        print(
            f"[gate] WARN: rerun stability < 0.95 for {len(failed_gates)} model(s): "
            + ", ".join(failed_gates),
            flush=True,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
