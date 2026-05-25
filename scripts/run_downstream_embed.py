#!/usr/bin/env python3
"""Phase 5 (downstream eval): per-model × per-task pooled embedding extraction.

For each scorable model in `data/audits/models.json` and each of the
selected 6 downstream tasks, this script loads the model, walks the
task's train.csv + test.csv from dna_foundation_benchmark/data_processed/,
extracts a pooled embedding per sequence via
`src.scoring.embeddings.compute_pooled_embedding`, and writes the
embeddings to:

    out_phase2/embeddings/<model_slug>/<task_name>/{train,test}.parquet

Each parquet has columns:
    embed_0, embed_1, ..., embed_{D-1}, label

Resume is per (model, task, split): an existing complete parquet is
skipped unless --force is passed.

The 6 task selection is (panel-aligned + length + species diverse):

    iDNA_ABF/5mC                          41 bp   5K   methylation
    mouse/mouse_TFBS_3                   101 bp   3K   TFBS, mouse
    enhancers/enhancer                   200 bp  15K   enhancer  (panel ✓)
    iPro-WAEL/Promoter_Arabidopsis_TATA  251 bp   4K   plant promoter
    prom/promoter_tata_300bps            300 bp   6K   human promoter (panel ✓)
    EMP/Yeast_H4                         500 bp  15K   fungi histone (panel ✓)

Usage:
    $PY scripts/run_downstream_embed.py --hf-ids zhihan1996/DNABERT-2-117M
    $PY scripts/run_downstream_embed.py --from-audit
    $PY scripts/run_downstream_embed.py --from-audit --only "DNABERT,GROVER" --force
"""

from __future__ import annotations

import argparse
import json
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

# Re-use the audit-derived spec list + per-spec loader dispatch from
# the likelihood-scoring runner so we have a single source of truth for
# how each of the 123 models is loaded.
from glmap.loaders.dispatch import ModelSpec, specs_from_audit  # noqa: E402
from scripts.run_phase1_scoring import _score_model  # noqa: E402
from glmap.scoring.embeddings import compute_pooled_embedding  # noqa: E402


# ─────────────────────── task definitions ─────────────────────────


@dataclass(frozen=True)
class DownstreamTask:
    task_id: str            # e.g. "EMP/Yeast_H4"
    rel_path: str           # path under data_processed/, e.g. "EMP/Yeast_H4"
    max_bp: int             # nominal sequence length cap (informational)
    n_labels: int           # 2 for binary, 3+ for multiclass

    @property
    def name(self) -> str:
        return self.rel_path.replace("/", "__")


# Locked 6-task suite, panel-aligned + diverse (see module docstring).
TASKS: tuple[DownstreamTask, ...] = (
    DownstreamTask("iDNA_ABF/5mC",                          "iDNA_ABF/5mC",                          41,  2),
    DownstreamTask("mouse/mouse_TFBS_3",                    "mouse/mouse_TFBS_3",                   101,  2),
    DownstreamTask("enhancers/enhancer",                    "enhancers/enhancer",                   200,  2),
    DownstreamTask("iPro-WAEL/Promoter_Arabidopsis_TATA",   "iPro-WAEL/Promoter_Arabidopsis_TATA",  251,  2),
    DownstreamTask("prom/promoter_tata_300bps",             "prom/promoter_tata_300bps",            300,  2),
    DownstreamTask("EMP/Yeast_H4",                          "EMP/Yeast_H4",                         500,  2),
)


def _resolve_seq_col(df: pd.DataFrame) -> str:
    for cand in ("Sequence", "sequence"):
        if cand in df.columns:
            return cand
    raise KeyError(
        f"Expected 'Sequence' or 'sequence' column in CSV; got {list(df.columns)}"
    )


def _resolve_label_col(df: pd.DataFrame) -> str:
    for cand in ("Label", "label"):
        if cand in df.columns:
            return cand
    raise KeyError(
        f"Expected 'Label' or 'label' column in CSV; got {list(df.columns)}"
    )


def load_task_split(
    benchmark_dir: Path, task: DownstreamTask, split: str
) -> pd.DataFrame:
    """split ∈ {'train', 'test'}. Returns a DataFrame with columns
    ['sequence', 'label']."""
    path = benchmark_dir / "data_processed" / task.rel_path / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"task split not found: {path}")
    df = pd.read_csv(path)
    sc = _resolve_seq_col(df)
    lc = _resolve_label_col(df)
    out = pd.DataFrame({
        "sequence": df[sc].astype(str).str.upper(),
        "label": df[lc].astype(int),
    })
    return out


# ─────────────────────── embedding extraction ─────────────────────


def embed_split(loader, df: pd.DataFrame, progress_every: int = 200) -> np.ndarray:
    """Run `compute_pooled_embedding` on every row of `df`. Returns an
    (N, D) float32 numpy array. Catches and reports per-sequence
    failures via NaN rows so the parquet is still writable; downstream
    classifier can then drop NaN rows or fail-fast."""
    embeds: list[np.ndarray] = []
    n_err = 0
    first_dim: int | None = None
    t0 = time.time()
    for i, seq in enumerate(df["sequence"].to_numpy()):
        try:
            v = compute_pooled_embedding(loader, str(seq))
        except Exception as exc:
            n_err += 1
            if n_err <= 3:
                print(f"  [err {i}] {type(exc).__name__}: {exc}", flush=True)
            embeds.append(None)  # placeholder
            continue
        if first_dim is None:
            first_dim = int(v.shape[-1])
        elif int(v.shape[-1]) != first_dim:
            raise RuntimeError(
                f"embedding dim mismatch at i={i}: got {v.shape[-1]} vs {first_dim}"
            )
        embeds.append(v.astype(np.float32))
        if (i + 1) % progress_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  embed {i+1}/{len(df)}  errs={n_err}  rate={rate:.1f} seq/s",
                  flush=True)
    if first_dim is None:
        raise RuntimeError("all sequences failed embedding; can't infer dim")
    # Fill NaN rows for failures
    out = np.full((len(df), first_dim), np.nan, dtype=np.float32)
    for i, v in enumerate(embeds):
        if v is not None:
            out[i] = v
    return out


def save_embed_parquet(arr: np.ndarray, labels: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f"embed_{i}": arr[:, i] for i in range(arr.shape[1])}
    cols["label"] = labels.astype(np.int64)
    pd.DataFrame(cols).to_parquet(path, index=False)


# Re-export for back-compat; canonical home is now src/io/embed_schema.py
# (a CPU-only module — keeping the classify script transitively light).
from glmap.io.embed_schema import validate_embed_columns   # noqa: E402,F401


def parquet_complete(path: Path, expected_n: int) -> bool:
    """Cheap resume check: file exists, row count matches, embed_* schema
    is dense [0..D-1], and at least 95% of rows have ALL embedding
    dimensions finite (matches the row-drop logic used by
    run_downstream_classify.py:load_embed_split).

    Earlier versions only checked `embed_0`. That missed parquets where
    embed_0 was finite but, say, embed_17 had NaN — the classifier would
    silently drop those rows and the resume check would falsely accept the
    parquet as complete. Schema validation is shared with load_embed_split
    via `validate_embed_columns()` so the two layers stay in lockstep:
    if resume says "complete", classify must accept; if classify rejects,
    resume must too."""
    if not path.exists():
        return False
    try:
        # Read all columns once (parquet is columnar, this is one file
        # open + per-column page reads). For 1024-dim embeddings on 10K
        # rows this is ~40 MB; in sweep startup we do ~1500 such reads
        # totaling minutes, acceptable.
        df = pd.read_parquet(path)
    except Exception:
        return False
    if "label" not in df.columns:
        return False
    if len(df) != expected_n:
        return False
    try:
        embed_cols = validate_embed_columns(df.columns)
    except ValueError:
        return False
    arr = df[embed_cols].to_numpy()
    # A row is "valid" only if every embed dim is finite — same predicate
    # the downstream classifier uses to drop rows.
    valid_per_row = np.isfinite(arr).all(axis=1)
    finite_frac = float(valid_per_row.mean())
    return finite_frac >= 0.95


# ─────────────────────── runner ────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-audit", action="store_true",
                     help="Iterate all scorable models from "
                          "data/audits/models.json.")
    src.add_argument("--hf-ids", type=str,
                     help="Comma-separated EXACT hf_id list to score.")
    p.add_argument("--audit-json", type=Path,
                   default=REPO_ROOT / "data/audits/models.json")
    p.add_argument("--benchmark-dir", type=Path,
                   default=REPO_ROOT / "dna_foundation_benchmark",
                   help="Vendored eval suite root (must contain data_processed/).")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "out_phase2",
                   help="Output root; embeddings under out/embeddings/.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated substring filter on hf_id.")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated substring filter on task_id (default: all 6).")
    p.add_argument("--force", action="store_true",
                   help="Re-embed every (model, task, split) even when parquet "
                        "is complete on disk.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-train", type=int, default=None,
                   help="Subsample train split to N rows (deterministic; uses head). "
                        "Test split is never subsampled.")
    return p.parse_args()


def resolve_roster(args) -> list[ModelSpec]:
    if args.hf_ids:
        all_specs = specs_from_audit(audit_path=args.audit_json)
        wanted = {h.strip() for h in args.hf_ids.split(",") if h.strip()}
        roster = [s for s in all_specs if s.hf_id in wanted]
        missing = wanted - {s.hf_id for s in roster}
        if missing:
            raise SystemExit(
                f"[downstream] --hf-ids: {len(missing)} not found in audit: "
                f"{sorted(missing)[:5]}"
            )
        return roster
    return specs_from_audit(audit_path=args.audit_json)


def resolve_tasks(only: str | None) -> list[DownstreamTask]:
    if not only:
        return list(TASKS)
    patterns = [p.strip() for p in only.split(",") if p.strip()]
    out = [t for t in TASKS if any(p in t.task_id for p in patterns)]
    if not out:
        raise SystemExit(f"[downstream] --tasks {only!r} matched 0 tasks")
    return out


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"benchmark dir not found: {args.benchmark_dir}")

    roster = resolve_roster(args)
    if args.only:
        patterns = [p.strip() for p in args.only.split(",") if p.strip()]
        roster = [s for s in roster if any(p in s.hf_id for p in patterns)]
    tasks = resolve_tasks(args.tasks)

    print(f"[downstream] {len(roster)} models × {len(tasks)} tasks",
          flush=True)

    embeddings_dir = args.out / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    for spec in roster:
        slug = spec.slug
        print(f"\n[{spec.branch}] === {spec.hf_id}", flush=True)

        # Plan: skip the model if every (task, split) is already done
        all_done = True
        plan: list[tuple[DownstreamTask, str, Path, int]] = []
        for task in tasks:
            for split in ("train", "test"):
                df = load_task_split(args.benchmark_dir, task, split)
                expected_n = len(df) if (split != "train" or args.max_train is None) \
                                     else min(len(df), args.max_train)
                target = embeddings_dir / slug / task.name / f"{split}.parquet"
                if (not args.force) and parquet_complete(target, expected_n):
                    print(f"  [skip] {task.name}/{split} complete ({expected_n} rows)",
                          flush=True)
                    continue
                plan.append((task, split, target, expected_n))
                all_done = False
        if all_done:
            print(f"  [model done] all {len(tasks)} tasks × 2 splits complete; "
                  "skipping load.", flush=True)
            continue

        # Load model once per spec
        t_load = time.time()
        try:
            # Re-use _score_model's loader build path: it calls loader.load()
            # before scoring; we mimic the build but skip the scoring loop.
            loader = _build_loader_only(spec, device=args.device)
        except Exception as exc:
            print(f"  [load fail] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            continue
        print(f"  loaded in {time.time() - t_load:.1f}s", flush=True)

        for task, split, target, expected_n in plan:
            df = load_task_split(args.benchmark_dir, task, split)
            if split == "train" and args.max_train is not None and len(df) > args.max_train:
                df = df.head(args.max_train)
            print(f"  [{task.name}/{split}] {len(df)} sequences "
                  f"-> {target}", flush=True)
            try:
                t0 = time.time()
                embeds = embed_split(loader, df)
                save_embed_parquet(embeds, df["label"].to_numpy(), target)
                print(f"    done in {time.time() - t0:.1f}s, "
                      f"dim={embeds.shape[1]}", flush=True)
            except Exception as exc:
                print(f"    [fail] {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

        # Free GPU memory between models
        del loader
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def _build_loader_only(spec: ModelSpec, device: str):
    """Build and load the loader for one spec, without doing any scoring.

    Mirrors the dispatch in `scripts.run_phase1_scoring._score_model` but
    stops at `loader.load()`. We import the loader factories lazily so
    the env-specific deps don't get pulled in for models we won't run.
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
        from glmap.loaders.huggingface import HFCausalLMLoader
        loader = HFCausalLMLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens,
            device=device, trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "hf" and spec.branch == "mlm":
        from glmap.loaders.huggingface import HFMaskedLMLoader
        loader = HFMaskedLMLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens,
            device=device, trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "botanic":
        from glmap.loaders.custom_mlm import BotanicLoader
        loader = BotanicLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "hyenadna":
        from glmap.loaders.hyenadna import HyenaDNALoader
        loader = HyenaDNALoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "aido":
        from glmap.loaders.aido import AIDOLoader
        loader = AIDOLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "evo2":
        from glmap.loaders.evo2_loader import Evo2Loader
        loader = Evo2Loader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "evo1":
        from glmap.loaders.evo1_loader import Evo1Loader
        loader = Evo1Loader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "plantbimoe":
        from glmap.loaders.custom_mlm import PlantBiMoELoader
        loader = PlantBiMoELoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "mutbert":
        from glmap.loaders.custom_mlm import MutBERTLoader
        loader = MutBERTLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
        )
    elif spec.loader_kind == "generator":
        from glmap.loaders.generator import GENERatorLoader
        loader = GENERatorLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "carbon":
        from glmap.loaders.carbon import CarbonCausalLMLoader
        loader = CarbonCausalLMLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens, device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    elif spec.loader_kind == "ntv3":
        from glmap.loaders.ntv3 import NTv3MaskedLMLoader
        if spec.length_multiple is None:
            raise ValueError(
                f"{spec.hf_id}: loader_kind='ntv3' requires length_multiple"
            )
        loader = NTv3MaskedLMLoader(
            hf_id=spec.hf_id, context_tokens=spec.context_tokens,
            length_multiple=spec.length_multiple, device=device,
            trust_remote_code=spec.trust_remote_code,
        )
    else:
        raise ValueError(
            f"unknown loader_kind={spec.loader_kind!r} branch={spec.branch!r} "
            f"for {spec.hf_id}"
        )
    loader.load()
    return loader


if __name__ == "__main__":
    main()
