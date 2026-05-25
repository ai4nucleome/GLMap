"""Full-roster parallel sweep across the audit-derived roster (123 models as of 2026-05-20).

For each scorable model in `data/audits/models.json`, look up the correct
micromamba env + runtime knobs per docs/env_routing.md, then dispatch a
per-model subprocess on a pool of GPUs. Two modes are supported:

  --mode stability (default)
      Dispatches `scripts/run_rerun_stability.py --hf-ids <hf_id> --n-probes N`.
      N probes are scored twice and Pearson r is checked against the 0.95 gate.
      Output: out_phase1/stability/<slug>.json.
      Resume criterion: skip when stability JSON exists with passes_gate=True.

  --mode scoring
      Dispatches `scripts/run_phase1_scoring.py --from-audit
      --hf-ids=<hf_id> --skip-aggregate`. Uses --hf-ids (exact match)
      not --only (substring) so collision-prone prefixes like
      'arcinstitute/evo2_7b' do not match evo2_7b_base / evo2_7b_262k
      and cause parallel subprocesses to race on the same parquet.
      Each subprocess writes only its own
      out_phase1/scores/<slug>/probes.parquet. After the parallel sweep,
      run a single aggregate pass (`run_phase1_scoring.py --from-audit
      --strict-aggregate`) to build out_phase1/matrices/{L,Q,D}_{AR,MLM}.npy.
      Resume criterion: skip when out_phase1/scores/<slug>/probes.parquet
      has the panel's full probe_id set AND every sum_log_p is finite
      (a per-probe failure writes the row with sum_log_p=NaN; that does
      not count as done). --force propagates to the child so the worker
      re-scores even if its own parquet exists.

Usage
-----
    # Stability gate sweep (Stage 1 entry; lightweight, ~3 probes per model)
    python scripts/run_sweep.py                        # 3 probes
    python scripts/run_sweep.py --n-probes 10

    # Full panel scoring sweep (Stage 4 / phase 2 entry; needs 10K-probe
    # frozen panel + all audit-listed models, currently 123). Run this,
    # then run a final aggregate:
    python scripts/run_sweep.py --mode scoring
    python scripts/run_phase1_scoring.py --from-audit --strict-aggregate  # CPU OK, builds L/Q/D

    python scripts/run_sweep.py --only evo             # substring filter on hf_id
    python scripts/run_sweep.py --n-gpus 8 --force     # rerun everything
    python scripts/run_sweep.py --gpu-ids 0,5,6,7      # use physical GPU ids
    python scripts/run_sweep.py --max-gpus-per-model 1 # only 1-GPU models
    python scripts/run_sweep.py --dry-run              # show routing decisions, exit

Scheduling
----------
GPUs are a flat pool. A task with `gpus_needed=k` waits until k
contiguous-by-availability GPUs are free, then is launched with
`CUDA_VISIBLE_DEVICES=<those gpus>` + `--device cuda:0`. Big-GPU tasks
(Genos-10B-v2: 4 GPUs; evo2_20b: 4 GPUs; evo2_40b{,_base}: 8 GPUs) are
sorted to the front so they don't get starved by the long tail of
small ones.

This script itself runs in the base micomamba env (it only imports
stdlib); it does NOT do scoring, only orchestrates the per-env
subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------- #

ENV_PYTHON = {
    "base":      "/nvme-data3/yusen/micomamba/bin/python",
    "caduceus":  "/nvme-data3/yusen/micomamba/envs/caduceus/bin/python",
    "PlantCAD":  "/nvme-data3/yusen/micomamba/envs/PlantCAD/bin/python",
    "dnabert2":  "/nvme-data3/yusen/micomamba/envs/dnabert2/bin/python",
    "gf":        "/nvme-data3/yusen/micomamba/envs/gf/bin/python",
    "evo":       "/nvme-data3/yusen/micomamba/envs/evo/bin/python",
    "evo2":      "/nvme-data3/yusen/micomamba/envs/evo2/bin/python",
    "hyena-dna": "/nvme-data3/yusen/micomamba/envs/hyena-dna/bin/python",
}

_PLANTCAD_LIB    = "/nvme-data3/yusen/micomamba/envs/PlantCAD/lib"
_EVO2_LIB        = "/nvme-data3/yusen/micomamba/envs/evo2/lib"
_EVO2_TORCH_LIB  = "/nvme-data3/yusen/micomamba/envs/evo2/lib/python3.12/site-packages/torch/lib"


@dataclass
class RouteSpec:
    """How to launch one model.

    env              : which micromamba env's python to invoke
    gpus_needed      : how many GPUs to allocate from the pool
    ld_library_path  : paths to prepend to LD_LIBRARY_PATH for the subprocess
    extra_env        : additional env vars (e.g. HF_HUB_OFFLINE)
    """

    env: str
    gpus_needed: int = 1
    ld_library_path: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)


def route_model(
    hf_id: str,
    evo2_40b_gpus: int | None = None,
) -> RouteSpec:
    """Map hf_id → RouteSpec. Mirrors docs/env_routing.md table.

    `evo2_40b_gpus` overrides the GPU allocation for any 40b evo2 model
    (both `arcinstitute/evo2_40b` and `arcinstitute/evo2_40b_base`). When
    None, defaults are: 40b_base → 4 GPUs (8K context, fits comfortably);
    40b (full 1M context) → 8 GPUs (activation memory needs the headroom).
    """

    # Caduceus 6 — original env
    if hf_id.startswith("kuleshov-group/caduceus-"):
        return RouteSpec(env="caduceus")

    # PlantBiMoE — same PlantCAD env, but the upstream HF repo ships
    # `tokenization_plantbimoe.py` inside a `plantbimoe/` subdirectory
    # while our cached snapshot has a flat copy at the repo root. Online
    # mode keeps re-hitting hf-mirror.com for the root path, getting a
    # 404, and rewriting the `.no_exist/.../tokenization_plantbimoe.py`
    # negative-cache marker, which then blocks even the flat copy.
    # Force offline so transformers reads the local snapshot directly.
    if hf_id == "plant-llms/PlantBiMoE":
        return RouteSpec(
            env="PlantCAD", ld_library_path=[_PLANTCAD_LIB],
            extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )

    # PlantCAD2 / HybriDNA / Jamba — mamba_ssm 2.x stack. All share the
    # PlantCAD env + LD_LIBRARY_PATH for env's newer libstdc++.
    # CUDA_VISIBLE_DEVICES remap (+ --device cuda:0) avoids the mamba_ssm 2.x
    # Triton kernels' hard-coded cuda:0.
    if (
        hf_id.startswith("kuleshov-group/PlantCAD2")
        or hf_id.startswith("Mishamq/HybriDNA-")
        or hf_id == "RaphaelMourad/Jamba-DNA-v1-114M-hg38"
    ):
        return RouteSpec(env="PlantCAD", ld_library_path=[_PLANTCAD_LIB])

    # HyenaDNA × 7
    if hf_id.startswith("LongSafari/hyenadna-"):
        return RouteSpec(env="hyena-dna")

    # DNABERT-2 / DNABERT-S — dedicated env w/o triton or flash_attn
    if hf_id in ("zhihan1996/DNABERT-2-117M", "zhihan1996/DNABERT-S"):
        return RouteSpec(env="dnabert2")

    # GenomeOcean × 4 — gf env with upgraded transformers
    if hf_id.startswith("DOEJGI/GenomeOcean"):
        return RouteSpec(env="gf")

    # NT v2 50m-3mer specifically — needs upgraded transformers
    if hf_id == "InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species":
        return RouteSpec(env="gf")

    # Evo 1.x (incl. microviridae fine-tune) — evo-model package path
    if hf_id in (
        "togethercomputer/evo-1-8k-base",
        "togethercomputer/evo-1-131k-base",
        "LongSafari/evo-1-8k-crispr",
        "LongSafari/evo-1-8k-transposon",
        "evo-design/evo-1.5-8k-base",
        "evo-design/evo-1-7b-131k-microviridae",
    ):
        return RouteSpec(
            env="evo",
            extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )

    # Evo 2.x — evo2 package, needs torch/lib on LD_LIBRARY_PATH
    if hf_id.startswith("arcinstitute/evo2_") or hf_id == "evo-design/evo-2-7b-8k-microviridae":
        # 40B bf16 weights = 80 GB. evo2 uses naive pipeline-parallel at
        # batch=1 inference, so more shards = more inter-GPU activation
        # transfer overhead with NO compute speed-up (the layers run
        # sequentially across cards either way). So we want the minimum
        # GPU count that fits weights + activations + KV cache.
        #   - evo2_40b_base (8K context): activations ~12 GB total →
        #     4 cards × ~25 GB each fits comfortably, ~15% faster than 8.
        #   - evo2_40b (1M context):     activations balloon → 8 cards
        #     needed for activation memory headroom.
        # `--evo2-40b-gpus N` (passed through `evo2_40b_gpus`) overrides
        # both variants when set; useful for diagnostics or for forcing
        # the older 8-card behaviour back when needed.
        if "40b" in hf_id:
            if evo2_40b_gpus is not None:
                gpus = evo2_40b_gpus
            elif hf_id.endswith("/evo2_40b"):
                gpus = 8                                # 1M context
            else:
                gpus = 4                                # _base (8K context)
        elif "20b" in hf_id:
            gpus = 4
        else:
            gpus = 1
        return RouteSpec(
            env="evo2",
            gpus_needed=gpus,
            ld_library_path=[_EVO2_LIB, _EVO2_TORCH_LIB],
        )

    # Genos-10B / Genos-10B-v2 — both ~10B params, fp32 won't fit a single
    # 44GB card, HFCausalLMLoader's OOM fallback to device_map="auto" with
    # only one visible GPU causes partial CPU offload (256-byte buffer
    # warning) → every embed forward fails. Genos-10B was previously left
    # at the default gpus=1, hitting this exact path. Bump both to gpus=4
    # so device_map distributes layers cleanly across multiple GPUs
    # without CPU offload. Scoring still works at gpus=1 (the cached
    # parquet from the previous sweep is fine — the OOD per-tok numbers
    # are genuine model behavior, not a numeric artifact; see Genos
    # investigation notes) but embed needs the multi-GPU route.
    if hf_id in ("ZhejiangLab/Genos-10B", "ZhejiangLab/Genos-10B-v2"):
        return RouteSpec(env="base", gpus_needed=4)

    # HuggingFaceBio/Carbon — HybridDNATokenizer instantiation calls
    # `AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")` internally,
    # which hits hf-mirror.com for `added_tokens.json` (a file Qwen3-4B-Base
    # does not ship). With proxy unreachable from worker subprocesses this
    # blows up at load time. Force offline so transformers reads the local
    # Qwen tokenizer snapshot directly without metadata HEAD calls.
    if hf_id.startswith("HuggingFaceBio/Carbon-"):
        return RouteSpec(
            env="base",
            extra_env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )

    # Default: base env, single GPU
    return RouteSpec(env="base")


# --------------------------------------------------------------------- #
# GPU pool + scheduler
# --------------------------------------------------------------------- #

@dataclass
class GPUPool:
    gpu_ids: int | list[int]

    def __post_init__(self) -> None:
        if isinstance(self.gpu_ids, int):
            if self.gpu_ids < 1:
                raise ValueError("GPUPool requires at least one GPU")
            ids = list(range(self.gpu_ids))
        else:
            ids = list(self.gpu_ids)
            if not ids:
                raise ValueError("GPUPool requires at least one GPU id")
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate GPU ids in pool: {ids}")
        self.gpu_ids = ids
        self.busy: dict[int, bool] = {gpu_id: False for gpu_id in ids}

    @property
    def n_gpus(self) -> int:
        return len(self.gpu_ids)

    def acquire(self, n: int) -> list[int] | None:
        free = [gpu_id for gpu_id, is_busy in self.busy.items() if not is_busy]
        if len(free) < n:
            return None
        chosen = free[:n]
        for g in chosen:
            self.busy[g] = True
        return chosen

    def release(self, gpus: list[int]) -> None:
        for g in gpus:
            self.busy[g] = False


def parse_gpu_ids(value: str) -> list[int]:
    """Parse a comma-separated physical CUDA id list, e.g. '0,5,6,7'."""

    ids: list[int] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(
                f"GPU ids must be non-negative integers, got {token!r} in {value!r}"
            )
        ids.append(int(token))
    if not ids:
        raise ValueError("GPU id list is empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate GPU ids in {value!r}")
    return ids


def resolve_pool_gpu_ids(gpu_ids_arg: str | None, n_gpus_arg: int | None) -> tuple[list[int], str]:
    """Resolve the physical GPU ids used by the scheduler.

    Precedence:
      1. --gpu-ids
      2. outer CUDA_VISIBLE_DEVICES if it is a comma-separated integer list
      3. default physical range 0..N-1

    --n-gpus optionally caps how many ids from that list are used.
    """

    source = "default physical range"
    if gpu_ids_arg:
        try:
            available = parse_gpu_ids(gpu_ids_arg)
        except ValueError as exc:
            raise SystemExit(f"--gpu-ids: {exc}") from exc
        source = "--gpu-ids"
    else:
        env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_value:
            try:
                available = parse_gpu_ids(env_value)
            except ValueError as exc:
                raise SystemExit(
                    "CUDA_VISIBLE_DEVICES is set but is not a comma-separated "
                    f"integer GPU id list ({env_value!r}): {exc}. Pass "
                    "--gpu-ids explicitly to override."
                ) from exc
            source = "CUDA_VISIBLE_DEVICES"
        else:
            available = list(range(n_gpus_arg if n_gpus_arg is not None else 8))

    if n_gpus_arg is None:
        n_use = len(available)
    else:
        if n_gpus_arg < 1:
            raise SystemExit("--n-gpus must be >= 1")
        if n_gpus_arg > len(available):
            raise SystemExit(
                f"--n-gpus={n_gpus_arg} asks for more GPUs than available from "
                f"{source}: {available}"
            )
        n_use = n_gpus_arg

    return available[:n_use], source


def build_command(
    hf_id: str,
    route: RouteSpec,
    gpus: list[int],
    n_probes: int,
    panel: str | None,
    mode: str = "stability",
    force: bool = False,
    stride: int | None = None,
    out_dir: str | None = None,
    audit_path: str | None = None,
    benchmark_dir: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Construct argv + env for subprocess.Popen.

    mode='stability' (default) — dispatches to scripts/run_rerun_stability.py
        for the rerun-stability gate; --n-probes is the per-model sample
        size (typically 3-10 probes, just enough to compute Pearson r).
        force is not propagated (stability worker re-runs unconditionally).
    mode='scoring' — dispatches to scripts/run_phase1_scoring.py with
        --from-audit + --hf-ids=<hf_id> + --skip-aggregate so each parallel
        subprocess writes only its own out_phase1/scores/<slug>/probes.parquet.
        force=True propagates as --force to the child, so the worker also
        ignores its own existing parquet (the parent's resume check already
        gates this at scheduler entry, but the child re-checks per spec —
        without --force the child would no-op on a cached parquet).
        A final aggregate pass on cpu (run_sweep emits the command, doesn't
        run it inline) builds out_phase1/matrices/{L,Q,D}_{AR,MLM}.npy.
    mode='embed' — dispatches to scripts/run_downstream_embed.py with
        --hf-ids=<hf_id> (--from-audit/--hf-ids are mutually exclusive in
        the embed script), which extracts pooled embeddings for all 6
        selected downstream tasks (see that script's task list) and writes
        them to out_phase2/embeddings/<slug>/<task>/{train,test}.parquet.
        force=True propagates as --force.
    """

    py = ENV_PYTHON[route.env]
    if mode == "stability":
        args = [
            py,
            "scripts/run_rerun_stability.py",
            "--hf-ids", hf_id,
            "--device", "cuda:0",                      # always cuda:0 inside the masked process
            "--n-probes", str(n_probes),
        ]
        if audit_path:
            args.extend(["--audit", audit_path])
        if panel:
            args.extend(["--panel", panel])
    elif mode == "scoring":
        # --hf-ids (exact match) NOT --only (substring) — the audit has
        # 15 collision-prone substrings like 'evo2_7b' that match multiple
        # full hf_ids and would cause parallel subprocesses to race on the
        # same parquet.
        args = [
            py,
            "scripts/run_phase1_scoring.py",
            "--from-audit",
            "--hf-ids", hf_id,
            "--skip-aggregate",
            "--device", "cuda:0",
        ]
        if audit_path:
            args.extend(["--audit-json", audit_path])
        if panel:
            args.extend(["--panel", panel])
        if force:
            args.append("--force")
        # MLM stride pass-through for the k=1 vs k=6 ablation.  When not
        # set, run_phase1_scoring.py uses its own default (k=6 per
        # phase_1.md primary).
        if stride is not None:
            args.extend(["--stride", str(stride)])
        # Output dir pass-through.  Necessary when running an ablation
        # that must not overwrite the canonical out_phase1/scores/ tree.
        if out_dir is not None:
            args.extend(["--out", out_dir])
    elif mode == "embed":
        # Downstream-task pooled embedding extraction (6 tasks × 2 splits =
        # 12 parquets per model). Loader is loaded once per model and
        # reused across all 12 calls; output paths under out_phase2/.
        # --from-audit and --hf-ids are mutually exclusive in the embed
        # script; pass --hf-ids (exact-match) to select one model.
        #
        # Sweep embed mode is FULL-VOLUME ONLY: we never pass --max-train
        # to the child, and the parent's resume check
        # (`parquet_complete(path, expected_n=full_csv_len)`) assumes the
        # child wrote full-sized parquets. If you need a subsampled
        # smoke, call `run_downstream_embed.py --max-train N` directly —
        # do NOT route through sweep, since sweep would then loop
        # forever re-deciding the partial parquet is "incomplete".
        args = [
            py,
            "scripts/run_downstream_embed.py",
            "--hf-ids", hf_id,
            "--device", "cuda:0",
        ]
        if audit_path:
            args.extend(["--audit-json", audit_path])
        if benchmark_dir:
            args.extend(["--benchmark-dir", benchmark_dir])
        if force:
            args.append("--force")
    else:
        raise ValueError(
            f"unknown mode={mode!r}; expected 'stability', 'scoring', or 'embed'"
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if route.ld_library_path:
        existing = env.get("LD_LIBRARY_PATH", "")
        prepend = ":".join(route.ld_library_path)
        env["LD_LIBRARY_PATH"] = prepend + (":" + existing if existing else "")
    # Sweep operates on pre-cached models only (audit + cache are populated
    # before the sweep is launched). When the HF_ENDPOINT proxy at
    # localhost:7890 is unreachable from worker subprocesses, transformers /
    # huggingface_hub still emits HEAD probes for tokenizer_config.json /
    # added_tokens.json / model.safetensors metadata even when the file is
    # already in cache — these probes fail with ProxyError and abort load.
    # Default the entire sweep to offline; per-route extra_env below can
    # override (none currently do — they only ever set offline ON).
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    for k, v in route.extra_env.items():
        env[k] = v
    return args, env


def classify_log(log_path: Path, rc: int, mode: str = "stability") -> str:
    """Tag the result of one model run by reading its log tail.

    The three sweep modes print different verdict markers:
      stability worker (run_rerun_stability.py) → "PASS: ..." / "FAIL: ..."
      scoring   worker (run_phase1_scoring.py)  → "[done] --skip-aggregate ..."
                                                   ("wrote scores -> ..." per probe)
      embed     worker (run_downstream_embed.py)→ no explicit success banner;
                                                   each (task, split) prints
                                                   "done in <t>s, dim=<D>".
                                                   We treat the run as DONE
                                                   when at least one "done in"
                                                   line is present AND no
                                                   Traceback.
    """

    if rc != 0 and rc != -signal.SIGTERM:
        # Subprocess crashed before printing a verdict
        return "CRASH"
    try:
        text = log_path.read_text()
    except OSError:
        return "?"

    if mode == "scoring":
        # Stronger failure signal than the success signal: if anything looks
        # like a Python traceback / explicit ERROR, fail before declaring DONE.
        if "Traceback" in text or "ERROR:" in text:
            return "ERR"
        if "[done] --skip-aggregate" in text or "wrote scores ->" in text:
            return "DONE"
        return "?"

    if mode == "embed":
        # Same failure-first policy. run_downstream_embed prints one
        # "done in Ns, dim=D" line per (task, split). We require at
        # least one before declaring DONE.
        if "Traceback" in text or "[load fail]" in text or "[fail]" in text:
            return "ERR"
        if "done in" in text and " dim=" in text:
            return "DONE"
        return "?"

    # stability mode (default — backward compatible)
    if "PASS:" in text:
        return "PASS"
    if "FAIL:" in text:
        return "FAIL"
    if "ERROR:" in text or "Traceback" in text:
        return "ERR"
    return "?"


def run_sweep(
    tasks: list[tuple[str, RouteSpec]],
    pool: GPUPool,
    n_probes: int,
    panel: str | None,
    log_dir: Path,
    poll_interval: float = 2.0,
    order: str = "small-first",
    mode: str = "stability",
    force: bool = False,
    stride: int | None = None,
    out_dir: str | None = None,
    panel_ids: set[str] | None = None,
    embed_expected_n: dict | None = None,
    parquet_complete_fn=None,
    audit_path: str | None = None,
    benchmark_dir: str | None = None,
) -> list[tuple[str, str, int, float]]:
    """Drive the parallel sweep. Returns list of (hf_id, status, rc, elapsed_sec).

    `order`:
      "small-first" (default) — all 1-GPU tasks fill the pool first 8-way
        parallel for fast results turnaround; multi-GPU tasks (Genos / evo2
        20b / 40b / 40b_base) get scheduled at the end, when no single-GPU
        task remains to compete for slots.
      "big-first" — multi-GPU tasks go first, blocking the pool but
        guaranteeing big-job latency isn't pushed to the very end.

    `panel_ids` (scoring mode only): the panel's probe_id set. After a
    scoring child exits with classify_log()=='DONE', the parquet is
    re-validated with parquet_covers_panel() — this catches the case
    where per-probe failures wrote sum_log_p=NaN rows; the worker still
    prints '[done] --skip-aggregate' in that scenario, so classify_log
    alone would mark the model as DONE while the parquet is unusable.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import (only relevant for scoring mode); kept inside the function
    # so the stability path stays import-free.
    parquet_covers_panel = None
    if mode == "scoring" and panel_ids is not None:
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.run_phase1_scoring import parquet_covers_panel as _pcp
        parquet_covers_panel = _pcp

    if order == "small-first":
        # gpus_needed ASC: 1-GPU first, then 4-GPU, then 8-GPU
        pending = sorted(tasks, key=lambda t: t[1].gpus_needed)
    elif order == "big-first":
        pending = sorted(tasks, key=lambda t: -t[1].gpus_needed)
    else:
        raise ValueError(f"unknown order={order!r}")
    running: list[tuple[subprocess.Popen, list[int], str, Path, float]] = []
    results: list[tuple[str, str, int, float]] = []
    t0 = time.time()
    n_total = len(pending)

    def _shutdown(_sig=None, _frame=None) -> None:
        print(f"\n[sweep] caught signal — terminating {len(running)} running subprocesses")
        for popen, _g, hf_id, _l, _ts in running:
            try:
                popen.terminate()
            except ProcessLookupError:
                pass
        for popen, _g, _h, _l, _ts in running:
            try:
                popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                popen.kill()
        sys.exit(130)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while pending or running:
        # Reap completed
        still_running = []
        for popen, gpus, hf_id, log_path, t_start in running:
            rc = popen.poll()
            if rc is None:
                still_running.append((popen, gpus, hf_id, log_path, t_start))
                continue
            pool.release(gpus)
            elapsed = time.time() - t_start
            status = classify_log(log_path, rc, mode=mode)
            # Scoring-mode integrity gate: classify_log says DONE iff the
            # worker printed the success banner. But the worker also writes
            # rows with sum_log_p=NaN when individual probes fail, and still
            # prints "[done] --skip-aggregate". Re-check the parquet against
            # the panel so the summary doesn't falsely show DONE for a
            # parquet that would later be rejected by --strict-aggregate.
            if (
                mode == "scoring"
                and status == "DONE"
                and parquet_covers_panel is not None
                and panel_ids is not None
            ):
                slug = hf_id.replace("/", "__")
                score_path = REPO_ROOT / "out_phase1" / "scores" / slug / "probes.parquet"
                ok, reason = parquet_covers_panel(
                    score_path, panel_ids, n_panel=len(panel_ids)
                )
                if not ok:
                    status = "PARTIAL"
                    print(
                        f"[done] {hf_id:60s} log says DONE but parquet integrity "
                        f"check FAILED: {reason}",
                        flush=True,
                    )
            # Embed-mode integrity gate (parallel to the scoring one above).
            # run_downstream_embed.py prints "done in" per task even when
            # individual sequences failed and were written as NaN rows.
            # Re-verify the 12 parquets against expected row counts so the
            # summary doesn't falsely show DONE for an output that resume
            # will subsequently re-run.
            if (
                mode == "embed"
                and status == "DONE"
                and parquet_complete_fn is not None
                and embed_expected_n is not None
            ):
                slug = hf_id.replace("/", "__")
                model_dir = REPO_ROOT / "out_phase2" / "embeddings" / slug
                missing_or_bad: list[str] = []
                for (task_name, split), n_exp in embed_expected_n.items():
                    path = model_dir / task_name / f"{split}.parquet"
                    if not parquet_complete_fn(path, n_exp):
                        missing_or_bad.append(f"{task_name}/{split}")
                if missing_or_bad:
                    status = "PARTIAL"
                    print(
                        f"[done] {hf_id:60s} log says DONE but parquet integrity "
                        f"check FAILED: {len(missing_or_bad)}/12 parquets "
                        f"incomplete (first: {missing_or_bad[0]})",
                        flush=True,
                    )
            results.append((hf_id, status, rc, elapsed))
            n_done = len(results)
            print(
                f"[done {n_done:3d}/{n_total}] {hf_id:60s} {status:5s} "
                f"gpus={gpus} elapsed={elapsed:.0f}s rc={rc}",
                flush=True,
            )
        running = still_running

        # Launch what we can
        new_pending: list[tuple[str, RouteSpec]] = []
        progress = False
        for hf_id, route in pending:
            gpus = pool.acquire(route.gpus_needed)
            if gpus is None:
                new_pending.append((hf_id, route))
                continue
            args, env = build_command(hf_id, route, gpus, n_probes, panel,
                                      mode=mode, force=force,
                                      stride=stride, out_dir=out_dir,
                                      audit_path=audit_path,
                                      benchmark_dir=benchmark_dir)
            slug = hf_id.replace("/", "__")
            log_path = log_dir / f"{slug}.log"
            log_path.write_text("")  # truncate
            popen = subprocess.Popen(
                args,
                env=env,
                stdout=open(log_path, "ab"),
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
            t_now = time.time() - t0
            print(
                f"[start {t_now:>5.0f}s] {hf_id:60s} env={route.env:10s} gpus={gpus}",
                flush=True,
            )
            running.append((popen, gpus, hf_id, log_path, time.time()))
            progress = True
        pending = new_pending

        if not progress and running:
            time.sleep(poll_interval)
        elif not progress and not running:
            break

    return results


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

SUPERVISED_BRANCH = {"supervised_or_annotation"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path,
                   default=REPO_ROOT / "data/audits/models.json",
                   help="Path to the models audit JSON.")
    p.add_argument("--benchmark-dir", type=Path,
                   default=REPO_ROOT / "data" / "dna_foundation_benchmark",
                   help="Benchmark data root for embed mode (must contain data_processed/). "
                        "Download from: huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset")
    p.add_argument("--stability-dir", type=Path,
                   default=REPO_ROOT / "out_phase1/stability",
                   help="Where stability JSONs live (used for --skip-done lookup).")
    p.add_argument("--n-probes", type=int, default=3,
                   help="Probes per model (default 3 for smoke; bump to 1000 for Stage 2).")
    p.add_argument("--n-gpus", type=int, default=None,
                   help="Number of GPUs to use from --gpu-ids / CUDA_VISIBLE_DEVICES "
                   "/ the default physical range. Default: all ids from --gpu-ids "
                   "or CUDA_VISIBLE_DEVICES, otherwise 8.")
    p.add_argument("--gpu-ids", type=str, default=None,
                   help="Comma-separated physical CUDA ids to use, e.g. 0,5,6,7. "
                   "If omitted, an outer CUDA_VISIBLE_DEVICES integer list is "
                   "honored; otherwise the pool defaults to physical 0..7.")
    p.add_argument("--panel", type=str, default=None,
                   help="Override panel parquet for the worker (default: worker default).")
    p.add_argument("--stride", type=int, default=None,
                   help="MLM stride pass-through to run_phase1_scoring.py "
                        "(scoring mode only). Default: child uses its own "
                        "default (k=6 per phase_1.md primary). Set to 1 "
                        "for the k=1 ablation experiment.")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory pass-through to "
                        "run_phase1_scoring.py (scoring mode only). "
                        "Default: child writes to out_phase1/scores. Set "
                        "to e.g. out_phase1/MLM_k1ablation_1000_scores "
                        "for an ablation that must not overwrite the "
                        "canonical k=6 scoring tree. Resume-skip in this "
                        "sweep is also redirected to this dir.")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated SUBSTRING filters on hf_id; only models matching "
                   "at least one substring run. Use --hf-ids for exact match.")
    p.add_argument("--hf-ids", dest="hf_ids", type=str, default=None,
                   help="Comma-separated EXACT hf_id list. Sweep restricts to "
                        "models whose hf_id exactly equals one of these "
                        "entries. Applied AFTER --branch / --only. Use this "
                        "(not --only) when you need to pin down a specific "
                        "set whose hf_ids overlap as substrings (e.g. "
                        "'gena-lm-bert-base' is a substring of "
                        "'gena-lm-bert-base-athaliana').")
    p.add_argument("--branch", type=str, default=None,
                   help="Filter candidates by audit `branch`. Accepts "
                        "'ar' / 'mlm' (short aliases) or the canonical "
                        "'ar_or_generative' / 'mlm_or_encoder'. Useful for "
                        "ablation sweeps that should only touch one branch "
                        "(e.g. --branch mlm + --stride 1 for the MLM k=1 "
                        "ablation).")
    p.add_argument("--evo2-40b-gpus", type=int, default=None,
                   help="Override the GPU allocation for any evo2 40B "
                        "variant (both `arcinstitute/evo2_40b` and "
                        "`arcinstitute/evo2_40b_base`). Default: 40b_base "
                        "→ 4 GPUs (8K context, fits comfortably); 40b "
                        "(1M context) → 8 GPUs (activation memory "
                        "headroom). Use e.g. 8 to force both back to "
                        "the old conservative behaviour.")
    p.add_argument("--max-gpus-per-model", type=int, default=None,
                   help="Skip models whose route needs more than this many GPUs. "
                   "Use 1 to run all single-GPU models, or 4 to run everything "
                   "except the current 8-GPU evo2_40b models.")
    p.add_argument("--force", action="store_true",
                   help="Rerun every model even if its prior run looks done. "
                   "stability mode: ignore passing stability JSONs. "
                   "scoring mode: ignore existing parquets AND propagate "
                   "--force to the child so the worker re-scores its own "
                   "parquet too.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print routing decisions + counts, exit without launching.")
    p.add_argument("--log-dir", type=Path,
                   default=Path("/tmp/sweep_logs"),
                   help="Where per-model stdout/stderr logs land.")
    p.add_argument("--order", type=str, default="small-first",
                   choices=["small-first", "big-first"],
                   help="Schedule single-GPU tasks first (default) so 8-way "
                   "parallelism kicks in immediately, or multi-GPU tasks first "
                   "to avoid pushing their latency to the end.")
    p.add_argument("--mode", type=str, default="stability",
                   choices=["stability", "scoring", "embed"],
                   help="stability (default): dispatch run_rerun_stability.py "
                   "for the rerun-stability gate (N probes, twice, Pearson r). "
                   "scoring: dispatch run_phase1_scoring.py --from-audit "
                   "--hf-ids <hf_id> --skip-aggregate so each subprocess writes "
                   "its out_phase1/scores/<slug>/probes.parquet; a final "
                   "aggregate pass (--from-audit --strict-aggregate, CPU OK) "
                   "builds out_phase1/matrices/. Scoring mode is the "
                   "Stage 4 / phase 2 entry point. "
                   "embed: dispatch run_downstream_embed.py "
                   "--hf-ids <hf_id>; each subprocess writes out_phase2/"
                   "embeddings/<slug>/<task>/{train,test}.parquet for the 6 "
                   "downstream tasks (~48K seqs/model). Phase 5 entry point. "
                   "**Full-volume only** — sweep never passes --max-train, "
                   "and its resume check expects full-sized parquets. For a "
                   "subsampled smoke, invoke run_downstream_embed.py directly.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.audit.exists():
        sys.exit(f"audit not found at {args.audit}")
    audit = json.loads(args.audit.read_text()).get("models", [])

    candidates = [m["hf_id"] for m in audit if m.get("branch") not in SUPERVISED_BRANCH]
    if args.branch is not None:
        # Map convenience aliases to canonical audit branch labels.
        branch_alias = {
            "ar": "ar_or_generative",
            "mlm": "mlm_or_encoder",
            "ar_or_generative": "ar_or_generative",
            "mlm_or_encoder": "mlm_or_encoder",
        }
        if args.branch not in branch_alias:
            sys.exit(f"--branch {args.branch!r} not recognised; expected one of "
                     f"{sorted(branch_alias.keys())}")
        wanted = branch_alias[args.branch]
        candidates = [
            m["hf_id"] for m in audit
            if m.get("branch") == wanted
            and m["hf_id"] in candidates  # keep prior supervised filter
        ]
    if args.only:
        patterns = [p.strip() for p in args.only.split(",") if p.strip()]
        candidates = [h for h in candidates if any(p in h for p in patterns)]
    if args.hf_ids:
        exact = {h.strip() for h in args.hf_ids.split(",") if h.strip()}
        unknown = exact - set(candidates)
        if unknown:
            sys.exit(f"--hf-ids contains models not in audit (or filtered out by "
                     f"--branch / --only): {sorted(unknown)}")
        candidates = [h for h in candidates if h in exact]

    pool_gpu_ids, gpu_source = resolve_pool_gpu_ids(args.gpu_ids, args.n_gpus)

    # Load panel_ids up front in scoring mode: needed both by the resume
    # check (when !--force) and by the post-exit parquet integrity gate in
    # run_sweep() (always).
    panel_ids: set[str] | None = None
    parquet_covers_panel = None
    if args.mode == "scoring":
        panel_path = (
            Path(args.panel) if args.panel
            else REPO_ROOT / "out_panel" / "main_panel.parquet"
        )
        try:
            import pandas as _pd
            panel_ids = set(_pd.read_parquet(
                panel_path, columns=["probe_id"]
            )["probe_id"])
        except Exception as exc:
            sys.exit(
                f"[sweep] mode=scoring needs to read {panel_path} for the "
                f"panel integrity check; got: {exc}"
            )
        # Lazy import the shared check
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.run_phase1_scoring import parquet_covers_panel

    # Same pattern for embed mode: pre-compute expected row counts per
    # (task, split) from the dna_foundation_benchmark CSVs, and import
    # parquet_complete from the child script for the integrity check.
    embed_expected_n: dict[tuple[str, str], int] | None = None
    parquet_complete_fn = None
    if args.mode == "embed":
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.run_downstream_embed import (
            TASKS as _EMBED_TASKS,
            parquet_complete as parquet_complete_fn,
            load_task_split,
        )
        bench_dir = args.benchmark_dir
        embed_expected_n = {}

        # Try reading expected row counts from the benchmark manifest first
        # (avoids requiring raw CSVs just for dry-run / resume checks).
        manifest_path = REPO_ROOT / "data" / "benchmark_manifests" / "downstream_tasks.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for task in _EMBED_TASKS:
                task_info = manifest.get("tasks", {}).get(task.rel_path)
                if task_info:
                    embed_expected_n[(task.name, "train")] = task_info["n_train"]
                    embed_expected_n[(task.name, "test")] = task_info["n_test"]
            if len(embed_expected_n) == len(_EMBED_TASKS) * 2:
                print(f"[sweep] embed expected row counts loaded from {manifest_path.name}",
                      flush=True)
            else:
                embed_expected_n = {}  # incomplete manifest, fall through

        # Fallback: read from actual CSVs if manifest didn't cover all tasks.
        if not embed_expected_n:
            try:
                for task in _EMBED_TASKS:
                    for split in ("train", "test"):
                        df = load_task_split(bench_dir, task, split)
                        embed_expected_n[(task.name, split)] = len(df)
            except Exception as exc:
                sys.exit(
                    f"[sweep] mode=embed needs task row counts. Either:\n"
                    f"  1. Ensure data/benchmark_manifests/downstream_tasks.json exists, or\n"
                    f"  2. Download task CSVs to {bench_dir}/data_processed/\n"
                    f"     (see data/benchmark_manifests/README.md)\n"
                    f"Error: {exc}"
                )

    # Resume: skip work that's already done. The "done" criterion depends on
    # which sweep we're driving:
    #   stability mode -> out_phase1/stability/<slug>.json with passes_gate
    #   scoring mode   -> out_phase1/scores/<slug>/probes.parquet covers
    #                     the full panel with finite sum_log_p in every row.
    #                     Set-equality alone would accept all-NaN parquets
    #                     produced by per-probe failures, so we delegate to
    #                     scripts.run_phase1_scoring.parquet_covers_panel.
    #   embed mode     -> out_phase2/embeddings/<slug>/<task>/{train,test}
    #                     .parquet present for all 6 tasks × 2 splits AND
    #                     each passes parquet_complete(): row count matches
    #                     expected_n AND ≥95% of rows have ALL embed dims
    #                     finite AND embed_* columns form the dense
    #                     [0..D-1] schema (validate_embed_columns). Mirrors
    #                     the contract enforced by load_embed_split at
    #                     classify time so a parquet accepted as "done"
    #                     here is guaranteed to load there.
    skipped: list[str] = []
    if not args.force:
        kept: list[str] = []
        # Scoring resume check honors --out override so an ablation sweep
        # with --out out_phase1/MLM_k1ablation_1000_scores does not get
        # confused by the canonical k=6 parquets in out_phase1/scores/.
        scores_dir = (
            Path(args.out) if args.out is not None
            else REPO_ROOT / "out_phase1" / "scores"
        )
        embeddings_dir = REPO_ROOT / "out_phase2" / "embeddings"
        for hf_id in candidates:
            slug = hf_id.replace("/", "__")
            if args.mode == "stability":
                stab_path = args.stability_dir / f"{slug}.json"
                if stab_path.exists():
                    try:
                        d = json.loads(stab_path.read_text())
                        if d.get("passes_gate"):
                            skipped.append(hf_id)
                            continue
                    except json.JSONDecodeError:
                        pass
            elif args.mode == "scoring":
                score_path = scores_dir / slug / "probes.parquet"
                ok, _reason = parquet_covers_panel(
                    score_path, panel_ids, n_panel=len(panel_ids)
                )
                if ok:
                    skipped.append(hf_id)
                    continue
                # else: parquet missing / incomplete / has NaN rows — re-run
            elif args.mode == "embed":
                # Done iff all 12 expected parquets are intact: existence +
                # row count match + ≥95% of rows with ALL embed dims
                # finite + dense embed_* schema (via the shared
                # validate_embed_columns helper). Mirrors the same check
                # the child uses for its own per-(task,split) resume so a
                # corrupted parquet from a killed sub-task doesn't cause
                # the parent to skip and the model never gets fixed.
                assert embed_expected_n is not None
                assert parquet_complete_fn is not None
                model_dir = embeddings_dir / slug
                all_ok = True
                for (task_name, split), n_exp in embed_expected_n.items():
                    path = model_dir / task_name / f"{split}.parquet"
                    if not parquet_complete_fn(path, n_exp):
                        all_ok = False
                        break
                if all_ok:
                    skipped.append(hf_id)
                    continue
                # else: at least one parquet missing or corrupt — re-run
            kept.append(hf_id)
        candidates = kept
        print(f"[sweep mode={args.mode}] skipping {len(skipped)} already-done; "
              f"will run {len(candidates)}")

    all_tasks = [
        (hf_id, route_model(hf_id, evo2_40b_gpus=args.evo2_40b_gpus))
        for hf_id in candidates
    ]
    skipped_by_gpu_need: list[tuple[str, int]] = []
    if args.max_gpus_per_model is not None:
        if args.max_gpus_per_model < 1:
            sys.exit("--max-gpus-per-model must be >= 1")
        tasks = []
        for hf_id, route in all_tasks:
            if route.gpus_needed > args.max_gpus_per_model:
                skipped_by_gpu_need.append((hf_id, route.gpus_needed))
            else:
                tasks.append((hf_id, route))
        if skipped_by_gpu_need:
            print(
                f"[sweep] --max-gpus-per-model={args.max_gpus_per_model}: "
                f"skipping {len(skipped_by_gpu_need)} model(s) needing more GPUs",
                flush=True,
            )
            for hf_id, need in skipped_by_gpu_need[:10]:
                print(f"  - needs {need} GPUs: {hf_id}", flush=True)
            if len(skipped_by_gpu_need) > 10:
                print(f"  ... {len(skipped_by_gpu_need) - 10} more", flush=True)
    else:
        tasks = all_tasks

    # Guard: any task whose gpus_needed exceeds the pool can never be
    # acquired, and run_sweep()'s "not progress and not running -> break"
    # would silently drop it. Fail fast with a clear message instead.
    n_pool_gpus = len(pool_gpu_ids)
    unschedulable = [(h, r.gpus_needed) for h, r in tasks if r.gpus_needed > n_pool_gpus]
    if unschedulable:
        msg_lines = [
            f"[sweep] {len(unschedulable)} task(s) need more GPUs than the pool "
            f"size ({n_pool_gpus}; ids={pool_gpu_ids}, source={gpu_source}):",
        ]
        for h, k in unschedulable:
            msg_lines.append(f"    needs {k} GPUs:  {h}")
        msg_lines.append(
            "Re-run with --n-gpus >= the largest gpus_needed (8 covers all "
            "current models), or pass --only <substring> to run a smaller "
            "subset that does not include those models."
        )
        sys.exit("\n".join(msg_lines))

    if args.dry_run:
        env_count = Counter(t[1].env for t in tasks)
        gpu_count = Counter(t[1].gpus_needed for t in tasks)
        print(f"[dry-run] {len(tasks)} tasks total")
        if skipped_by_gpu_need:
            print(f"  skipped by --max-gpus-per-model: {len(skipped_by_gpu_need)}")
        print(f"  gpu pool: {pool_gpu_ids}  (source={gpu_source})")
        print("  by env:")
        for env, n in env_count.most_common():
            print(f"    {env:10s}  {n:3d}")
        print(f"  by gpus_needed: {dict(gpu_count)}")
        # Estimated wall time at rough rates: 5s/probe small, 30s/probe big.
        # Just print the list of >1-GPU tasks since they dominate scheduling.
        big = [(h, r) for h, r in tasks if r.gpus_needed > 1]
        if big:
            print("  multi-GPU tasks:")
            for h, r in big:
                print(f"    gpus={r.gpus_needed}  {h}")
        return

    if not tasks:
        print("[sweep] nothing to do")
        return

    pool = GPUPool(pool_gpu_ids)
    t0 = time.time()
    results = run_sweep(
        tasks=tasks,
        pool=pool,
        n_probes=args.n_probes,
        panel=args.panel,
        log_dir=args.log_dir,
        order=args.order,
        mode=args.mode,
        force=args.force,
        stride=args.stride,
        out_dir=args.out,
        panel_ids=panel_ids,
        embed_expected_n=embed_expected_n,
        parquet_complete_fn=parquet_complete_fn,
        audit_path=str(args.audit),
        benchmark_dir=str(args.benchmark_dir),
    )
    total = time.time() - t0

    status_count = Counter(r[1] for r in results)
    print()
    print("=== sweep summary ===")
    print(f"wall time: {total:.0f}s  ({total/60:.1f} min)")
    print(f"models run: {len(results)}  (skipped already-passing: {len(skipped)})")
    if skipped_by_gpu_need:
        print(f"models skipped by --max-gpus-per-model: {len(skipped_by_gpu_need)}")
    for status, n in status_count.most_common():
        print(f"  {status:5s}: {n}")
    # Success label is mode-dependent: stability mode emits PASS, scoring
    # and embed modes emit DONE. Anything else (FAIL / ERR / CRASH / ? /
    # PARTIAL) is a failure worth listing in the summary.
    ok_label = "DONE" if args.mode in ("scoring", "embed") else "PASS"
    fails = [r for r in results if r[1] != ok_label]
    if fails:
        print()
        print(f"Non-{ok_label} models (check logs in", args.log_dir, "):")
        for hf_id, status, rc, elapsed in fails:
            print(f"  {status:5s}  {hf_id:60s}  rc={rc}  elapsed={elapsed:.0f}s")


if __name__ == "__main__":
    main()
