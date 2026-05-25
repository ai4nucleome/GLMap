# Env routing for phase 1 scoring

This file records which micromamba env to use for each model family in
the phase 1 rerun-stability / matrix-scoring pipeline. Built up
empirically during Stage 1 to reach 122/122 PASS on the rerun gate;
keep it next to the loader/runner code so future scoring sweeps don't
re-derive these mappings.

Single source of truth for runtime selection — when adding a new model,
update both this file and `scripts/run_rerun_stability.py:_audit_entry_to_spec`
if the model family needs a non-default env or loader.

## Why multiple envs

Most DNA foundation models pin their own CUDA-extension stack
(`mamba_ssm`, `causal_conv1d`, `flash_attn`, `triton`, plus a specific
`transformers` version that matches their custom `modeling_*.py`).
The combinations are mutually incompatible — `mamba_ssm 1.2` and
`mamba_ssm 2.x` can't coexist; `flash_attn` versions track
`torch`/CUDA; etc. Forcing every model into one env would mean
downgrading the rest. We instead run each family in its own env and
let the runner dispatch.

## Routing table

| Env | mamba_ssm | causal_conv1d | flash_attn | transformers | torch | Used for |
|---|---|---|---|---|---|---|
| `base` | — | — | — | latest | latest | NT v1/v2 standard, DNABERT 3-6, Botanic0, AIDO.DNA, GenSLM, megaDNA, PlasmidGPT, Mistral-DNA, ModernBERT-DNA, GROVER, ModernGENA, GENA-LM, Genos, AgroNT, NTv3 `_pre*`, **HuggingFaceBio/Carbon-500M/3B/8B** (Llama + HybridDNATokenizer, bf16, prepends `<dna>` and A-pads to 6 — `src/loaders/carbon.py`; tokenizer fetches `Qwen/Qwen3-4B-Base` tokenizer files at first load), …  (~80 models) |
| `caduceus` | 1.2.0 | 1.2.0 | 2.5.6 | 4.38 | 2.2 (py 3.8) | original `kuleshov-group/caduceus-*` 6 models |
| `PlantCAD` | 2.2.4 | 1.5.0.post8 | — | 4.49 | 2.5 | PlantCAD2 × 3, HybriDNA × 3, PlantBiMoE, **Jamba-DNA** (mamba_ssm 2.x + causal_conv1d are the only stack with both Mamba 2 and a recent transformers) |
| `dnabert2` | — | — | — | 4.29.2 | 1.13 | DNABERT-2-117M, DNABERT-S. **Intentionally has neither `triton` nor `flash_attn`** so `bert_layers.py`'s `from .flash_attn_triton import flash_attn_qkvpacked_func` raises at import time → its own try/except sets the symbol to `None` → PyTorch fallback path is taken with no monkey-patching from our side. |
| `gf` | — | — | 2.7.2 | 4.49 (upgraded from 4.29) | 2.1 | GenomeOcean × 4, NT-v2-50m-3mer. Needed transformers >= 4.36 for `cache_utils`; `pip install pyarrow` and `pip install 'transformers>=4.40,<4.50'` were the only setup steps. |
| `evo` | — | — | 2.7.4.post1 (cu12torch2.6, cxx11abiFALSE) | 5.8 | 2.6.0 + cu124 | Evo 1.x via `evo.Evo(model_name, device)` from the `evo-model` package (the old HF AutoModel path stopped working in transformers 5.x — dynamic-module loader can't resolve `PreTrainedModel` out of the cached `modeling_hyena.py`). See `src/loaders/evo1_loader.py`. Routes 5 models: `togethercomputer/evo-1-{8k,131k}-base`, `LongSafari/evo-1-8k-{crispr,transposon}`, `evo-design/evo-1.5-8k-base`. **Setup**: `pip install --force-reinstall --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl`. |
| `evo2` | — | — | 2.7.4.post1 (cu12torch2.6, cxx11abiFALSE) | 5.8.1 | 2.6.0 + cu124 | Evo 2.x via `evo2.Evo2(model_name)` (Vortex / StripedHyena 2). Routes 8 models: 7B series + 1B + 20B + 40B/40B_base + microviridae. **Setup**: `pip install --force-reinstall --no-deps https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`. **Two quirks at runtime**: (1) LD_LIBRARY_PATH must include torch/lib (not just env/lib) so flash_attn_2_cuda's c10 symbols resolve — see "Extended LD_LIBRARY_PATH for evo2" below. (2) The shipped `configs/evo2-*.yml` have `use_fp8_input_projections: True`; on this hardware FP8 needs a `transformer_engine` build with matching torch ABI which we don't have, so all configs must be patched to `False` (see "evo2 FP8 patch" below). 40B / 40B_base also need `CUDA_VISIBLE_DEVICES=0,1,...,7` and the StripedHyena shard merger has to complete (~80GB; do not kill mid-merge — see "evo2 40B merge" below). |
| `hyena-dna` | — | — | 1.0.7 | 4.57 | 2.7 | LongSafari/hyenadna-* × 7. Drives the standalone `HyenaDNAModel` from `models/modelsHFNoInfo/hyena-dna/standalone_hyenadna.py`; transformers is only used for the `PreTrainedTokenizer` base class. |
| `plantbimoe` | 1.2.0 | — | — | 4.38 | 2.2 (py 3.8) | An installed env supplied for PlantBiMoE; in practice we route PlantBiMoE through `PlantCAD` env because its loader needs only `mamba_ssm` (no causal_conv1d). Leaving the entry here for completeness; the routing table doesn't currently dispatch to it. |
| `gf_dnabert2` | 2.2.2 | — | 2.7.2 | 4.49 (upgraded) | 2.1 | Reserved for any future model that wants mamba_ssm 2.x without the full PlantCAD stack. **Not currently dispatched to** — Jamba moved to PlantCAD env after we found `causal_conv1d` wheel build fails here. Kept installed because Jamba's first working route was found here before. |

Absolute python paths are under `/nvme-data3/yusen/micomamba/envs/<env>/bin/python`.

## Runtime knobs

These environment / CLI flags are not env-specific but they're required
for certain families to actually work; bake them into whatever wrapper
launches the scoring runner.

### `CUDA_VISIBLE_DEVICES=<N>` + `--device cuda:0`

**Required for `mamba_ssm 2.x` Triton kernels (PlantCAD env).** The
kernels hard-code `cuda:0`; running the model on `cuda:N` for N != 0
fails with `ValueError: Pointer argument (at 0) cannot be accessed
from Triton (cpu tensor?)`. Workaround: remap GPU N to cuda:0 via
`CUDA_VISIBLE_DEVICES=N`. Affects PlantCAD2, HybriDNA, PlantBiMoE,
Jamba — none of which work with `--device cuda:N` directly.
`scripts/run_sweep.py --gpu-ids ...` performs this remap per subprocess
while still passing `--device cuda:0` to the worker.

### `LD_LIBRARY_PATH=/nvme-data3/yusen/micomamba/envs/PlantCAD/lib`

**Required for PlantCAD env.** Micromamba activation isn't a real shell
activation, so when we invoke the env's python directly, the dynamic
linker still finds the **system** `libstdc++.so.6` which is older than
GLIBCXX_3.4.29. `Pillow`'s `_imaging` extension (eagerly imported by
`transformers >= 4.30` via `chat_template_utils`) loads `libLerc.so.4`
which requires GLIBCXX_3.4.29+. Setting `LD_LIBRARY_PATH` to the env's
`lib/` makes it find the env's own `libstdc++.so.6.0.34` first.

### Extended LD_LIBRARY_PATH for evo2

```bash
TORCH_LIB=/nvme-data3/yusen/micomamba/envs/evo2/lib/python3.12/site-packages/torch/lib
LD_LIBRARY_PATH=/nvme-data3/yusen/micomamba/envs/evo2/lib:$TORCH_LIB
```

flash_attn_2_cuda.so was built referencing c10 symbols (`_ZN3c105ErrorC2...`) defined in torch's `libc10.so` and `libtorch_cpu.so`. Those .so files live under `<env>/lib/python3.12/site-packages/torch/lib/`, **not** `<env>/lib/`. Without the torch lib dir on LD_LIBRARY_PATH the dynamic linker can't resolve them and you get `ImportError: ... undefined symbol: _ZN3c105ErrorC2...`. `import torch` from Python doesn't fix this — by the time vortex.ops loads flash_attn_2_cuda the resolution path is already fixed.

The runner now hard-codes this prefix when launching the evo2 env; if you launch by hand, prepend it.

### evo2 FP8 patch (no Transformer Engine available)

The evo2 package's stock `configs/evo2-*.yml` set `use_fp8_input_projections: True` for every variant. On L20 + `transformer_engine 2.3.0`, the TE wheel was built against cxx11abi=TRUE torch but the installed `torch==2.6.0+cu124` is cxx11abi=FALSE — `import transformer_engine.pytorch` fails with `undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKNSt7__cxx11...`. Until that's resolved, fall back to bf16 by patching all seven configs:

```bash
for cfg in /nvme-data3/yusen/micomamba/envs/evo2/lib/python3.12/site-packages/evo2/configs/evo2-*.yml; do
    sed -i 's/use_fp8_input_projections: True/use_fp8_input_projections: False/' "$cfg"
done
```

This is a checkpoint-side patch (lives in the env's site-packages, not the repo). If the env is rebuilt or evo2 reinstalled, redo. Confirmed-working with `arcinstitute/evo2_{1b_base, 7b, 7b_base, 7b_262k, 20b}` and `evo-design/evo-2-7b-8k-microviridae` (`evo2_7b` 7B-class strictly enforces this fallback even at the vortex layer; the patch propagates through both checks).

### evo2 40B merge

`arcinstitute/evo2_40b*` ship weights as `.pt.part0` + `.pt.part1` (two ~40GB shards) plus a fix-up script that streams them into one `<HF_HOME>/evo2_40b.pt` file on first load. The merge takes ~5 min per shard on local disk. **Do not kill the loader mid-merge** — the partial file looks complete to subsequent runs but fails with `PytorchStreamReader failed reading zip archive: failed finding central directory`. Recovery: delete the half-merged file at `/data/yusen/software/.cache/huggingface/evo2_40b{,_base}.pt` and rerun (parts are preserved by default).

### `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

**Required for Evo 1.x.** The shell has `HF_ENDPOINT=https://hf-mirror.com`
set, and the mirror is missing some `trust_remote_code` files for the
evo-design / LongSafari Evo1 derivatives even when the model weights
are fully cached locally. Forcing offline mode makes transformers
resolve everything from the local cache.

Caveat: several Evo1 repos (LongSafari/evo-1-8k-crispr, evo-design/evo-1.5-8k-base,
evo-design/evo-1-7b-131k-microviridae, togethercomputer/evo-1-8k-base)
genuinely don't ship `tokenizer.py` + helper .py files. A one-off
copy from `togethercomputer/evo-1-131k-base`'s cache snapshot (which
is the canonical source — the auto_map even references it by name
via `togethercomputer/evo-1-131k-base--tokenizer.ByteTokenizer`) is
already done on disk. If the HF cache is wiped, this would need to be
re-done; see git history for the exact files copied.

### `device_map="auto"` (HFCausalLMLoader fallback)

**Triggered for models too large for a single GPU.** Genos-10B-v2
OOMs on a 46GB A100 in bfloat16. `HFCausalLMLoader.load` catches
`torch.cuda.OutOfMemoryError` and retries with `device_map="auto"` +
`torch_dtype=bfloat16` so accelerate shards layers across all visible
CUDA devices. Use `--gpu-ids 0,5,6,7` or an outer
`CUDA_VISIBLE_DEVICES=0,5,6,7` when launching to give it more memory to
spread across.

## How dispatch works

`scripts/run_rerun_stability.py:_audit_entry_to_spec` maps each
audit-listed hf_id to a `ModelSpec` with a `loader_kind`. Dispatch
chain in `scripts/run_phase1_scoring.py:_score_model` is:

```
hf_id                                       -> loader_kind   -> loader class                        in env
lingxusb/megaDNA                            -> megadna       -> src.loaders.megadna.MegaDNALoader   base
lingxusb/PlasmidGPT                         -> plasmidgpt    -> ...PlasmidGPTLoader                 base
GenSLM-*                                    -> genslm        -> ...GenSLMLoader                     base
living-models/Botanic0-*                    -> botanic       -> custom_mlm.BotanicLoader            base
plant-llms/PlantBiMoE                       -> plantbimoe    -> custom_mlm.PlantBiMoELoader         PlantCAD
LongSafari/hyenadna-*                       -> hyenadna      -> hyenadna.HyenaDNALoader             hyena-dna
genbio-ai/AIDO.DNA*                         -> aido          -> aido.AIDOLoader                     base
arcinstitute/evo2_*  +  evo-design/evo-2-7b-8k-microviridae
                                            -> evo2          -> evo2_loader.Evo2Loader              evo2
GenerTeam/GENERator-*  (NOT GENERanno)      -> generator     -> generator.GENERatorLoader           base
InstaDeepAI/NTv3_*                          -> ntv3          -> ntv3.NTv3MaskedLMLoader             base
HuggingFaceBio/Carbon-*                     -> carbon        -> carbon.CarbonCausalLMLoader         base
* (anything else, branch=ar_or_generative)  -> hf            -> HFCausalLMLoader                    see env table above
* (anything else, branch=mlm_or_encoder)    -> hf            -> HFMaskedLMLoader                    see env table above
```

The env column is selected by **`scripts/run_sweep.py:route_model(hf_id)`**, which is the source-of-truth Python expression of the table above. The sweep script orchestrates 122 models across the 8 envs onto an N-GPU pool, dispatches each as a subprocess with the right `python` / `CUDA_VISIBLE_DEVICES` / `LD_LIBRARY_PATH` / `HF_HUB_OFFLINE`, and is resume-safe (stability JSON for `--mode stability`; full-panel `probes.parquet` for `--mode scoring`). See "Operations: full-roster sweep" below for invocation patterns; the routing table above stays in sync with `route_model` by hand (when you add a model family that needs a non-default env, edit both).

## Adding a new model

1. Drop it into `models/download_models_list.txt`.
2. Rerun `scripts/audits/models.py` to refresh `data/audits/models.json`.
3. Try the standard path first: `base` env, `loader_kind="hf"` (no
   change to `_audit_entry_to_spec` needed).
4. If load fails on a missing extension / version mismatch / Triton
   compile error, look at this table — the family probably already has
   an env. Add an alias if needed.
5. If forward needs special args (species_ids, condition_ids, …), add
   a custom loader under `src/loaders/` and a dispatch rule in
   `_audit_entry_to_spec`. Mirror the pattern from
   `custom_mlm.BotanicLoader` (minimal wrapper using AutoModel +
   stride_pll_forward) or `evo1_loader._load_microviridae` (build a
   PreTrainedModel-style class by hand around a non-AutoModel
   checkpoint) depending on what's actually missing.
6. **Verify by running the rerun-stability runner with `--n-probes 5`
   on the new model and confirming `r=1.0 max_diff=0`** before adding
   it to any production sweep. Triton/CUDA kernels are silent failures
   waiting to happen — the gate is what catches "loaded but emits
   NaN/error on every probe" cases.

## Operations: full-roster sweep

`scripts/run_sweep.py` is the orchestrator. It reads the audit, routes
each model to its env via `route_model`, schedules onto a GPU pool
respecting per-task `gpus_needed`, launches `scripts/run_rerun_stability.py`
as a subprocess per model, aggregates results, and is resume-safe.

```bash
# Daily quick check that the current state still passes (skip already-PASS)
python scripts/run_sweep.py --n-probes 3

# Force-rerun a single family (or any substring match — comma-separated)
python scripts/run_sweep.py --force --only "PlantCAD2,HybriDNA,Jamba"

# See what would be launched without launching anything
python scripts/run_sweep.py --force --dry-run

# Stability sweep (Stage 1; lightweight, ~3 probes per model)
python scripts/run_sweep.py                       # default --mode stability
python scripts/run_sweep.py --n-probes 10

# Full 10,000-probe scoring sweep on the frozen Stage 2 panel (Stage 4 entry).
# Each subprocess writes its own probes.parquet under out_phase1/scores/;
# the final aggregate step (no --skip-aggregate) builds the matrices on cpu.
# --strict-aggregate makes the aggregate fail-fast on any missing/partial model.
python scripts/run_sweep.py --mode scoring
python scripts/run_phase1_scoring.py --from-audit --strict-aggregate  # final L/Q/D aggregate

# Cap the GPU pool (e.g. when sharing the box). Note: the full roster
# contains two 8-GPU models (evo2_40b, evo2_40b_base), so --n-gpus < 8
# will fail-fast unless you also use --only to drop them.
python scripts/run_sweep.py --n-gpus 8                                 # full roster
python scripts/run_sweep.py --n-gpus 4 --only "evo,nucleotide,DNABERT" # subset that fits

# Use a non-contiguous physical GPU set (equivalent to setting the outer
# CUDA_VISIBLE_DEVICES=0,5,6,7). The scheduler allocates from this list and
# still passes --device cuda:0 inside each masked subprocess.
python scripts/run_sweep.py --mode scoring --gpu-ids 0,5,6,7 --only "DNABERT"
CUDA_VISIBLE_DEVICES=0,5,6,7 python scripts/run_sweep.py --mode scoring --only "DNABERT"

# Run by resource class instead of model-name substring.
python scripts/run_sweep.py --mode scoring --gpu-ids 0 --max-gpus-per-model 1
python scripts/run_sweep.py --mode scoring --gpu-ids 0,5,6,7 --max-gpus-per-model 4
```

Each subprocess's stdout+stderr lands in `--log-dir` (default
`/tmp/sweep_logs/<slug>.log`). The stability JSONs land in
`out_phase1/stability/` per the worker's own contract; resume is
keyed on `passes_gate == True` in those JSONs.

The scheduler defaults to **single-GPU tasks first** (`--order
small-first`): all 118 1-GPU models go into the pool 8-way parallel
as fast as possible, then the 4-GPU tasks (evo2_20b + Genos-10B-v2)
pair up, then evo2_40b / 40b_base run serially with the full pool.
Reasoning: PASS reports stream in from the start (better UX, faster
failure detection), and a single 80GB evo2 .pt load with hot cache
is ~40 s — no real advantage to running it first. Pass
`--order big-first` if you want the inverse (multi-GPU tasks first,
small ones backfilling their idle slots).

## Rebuild checklist

If an env, the HF cache, or the repo's `models/modelsHFNoInfo/` tree is
wiped and you want to re-reach the Stage 1 `122/122 PASS` state, the
steps below need to be redone. Grouped by what they live in so each
section can be redone independently.

### A. Per-env `pip` install steps

```bash
# evo2 env (py3.12) — base packages for the evo2 inference path
/nvme-data3/yusen/micomamba/envs/evo2/bin/pip install transformers pyarrow sentencepiece

# evo env (py3.11) — extras the new evo-model package needs
/nvme-data3/yusen/micomamba/envs/evo/bin/pip install pyarrow protobuf sentencepiece

# gf env — upgrade transformers to one with cache_utils (>= 4.36) and add pyarrow.
# Needed for GenomeOcean × 4 + NT-v2-50m-3mer.
/nvme-data3/yusen/micomamba/envs/gf/bin/pip install pyarrow 'transformers>=4.40,<4.50'

# gf_dnabert2 env — same upgrade; reserved env, not currently dispatched to,
# but if you reach for it for a new mamba2 model the upgrade is the entry fee.
/nvme-data3/yusen/micomamba/envs/gf_dnabert2/bin/pip install pyarrow 'transformers>=4.40,<4.50'
```

### B. `flash_attn` wheels for evo / evo2 envs

PyPI mirrors do not ship binary flash_attn wheels; use the GitHub
releases page. Pick the wheel matching `cu12torch<TORCH_VERSION>` +
`cxx11abi=FALSE` (PyTorch's default linux wheels are
`manylinux2014` / `cxx11abi=FALSE`) + the env's cpython tag.

For the current torch 2.6.0+cu124 install in both evo and evo2:

```bash
# evo env (cp311)
/nvme-data3/yusen/micomamba/envs/evo/bin/pip install --force-reinstall --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

# evo2 env (cp312)
/nvme-data3/yusen/micomamba/envs/evo2/bin/pip install --force-reinstall --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

If a future torch upgrade changes the `torch2.6` part of the wheel
name, swap to the matching `torch<X.Y>` from the same GitHub release.
Verify by running `python -c "from flash_attn.modules.mha import MHA"`
in the env — silence is success.

### C. Env site-packages patches

#### C.1 evo2 FP8 → bf16 fallback (all 7 configs)

`transformer_engine 2.3.0` installed in the env is built against
`cxx11abi=TRUE` torch but our torch 2.6.0+cu124 is `cxx11abi=FALSE`,
so `import transformer_engine.pytorch` raises undefined-symbol. Until
that ABI mismatch is fixed (either by reinstalling TE built against
this torch ABI, or by swapping torch to a manylinux_2_28 wheel), the
evo2 inference must take the bf16 fallback path, which means every
`evo2-*.yml` config has to flip `use_fp8_input_projections` to False:

```bash
for cfg in /nvme-data3/yusen/micomamba/envs/evo2/lib/python3.12/site-packages/evo2/configs/evo2-*.yml; do
    sed -i 's/use_fp8_input_projections: True/use_fp8_input_projections: False/' "$cfg"
done
```

See `phase_1.md` § "Evo2 推理精度: bf16 fallback" for the methodology disclosure this patch entails.

### D. HF cache patches (env-independent)

These live under `/data/yusen/software/.cache/huggingface/hub/` and
are needed because the model authors uploaded incomplete repos
(missing tokenizer files, missing shard index variants, etc.).

#### D.1 Evo1 derivatives missing tokenizer + helper .py files

`evo-design/*` and `LongSafari/evo-1-*` ship their model weights but
their `tokenizer_config.json` references `tokenizer.ByteTokenizer`
from a local `tokenizer.py` that the authors didn't upload. The
canonical source is `togethercomputer/evo-1-131k-base`. Copy all .py
files from that snapshot into the dependent snapshots:

```bash
SRC=/data/yusen/software/.cache/huggingface/hub/models--togethercomputer--evo-1-131k-base/snapshots/78c715ab81852e02ec3b1c7e795dc7250d8c7625
for repo in \
    models--evo-design--evo-1.5-8k-base \
    models--evo-design--evo-1-7b-131k-microviridae \
    models--LongSafari--evo-1-8k-crispr \
    models--LongSafari--evo-1-8k-transposon \
    models--togethercomputer--evo-1-8k-base \
; do
    for snap in /data/yusen/software/.cache/huggingface/hub/${repo}/snapshots/*/; do
        for py in "$SRC"/*.py; do
            name=$(basename "$py")
            [ -e "${snap}${name}" ] || cp "$py" "${snap}${name}"
        done
    done
done
```

For `evo-design/evo-1-7b-131k-microviridae` specifically also copy
the tokenizer config + special tokens map + generation config (the
weights are pytorch_model.bin, no safetensors; we only need these
side-car files):

```bash
DST=$(echo /data/yusen/software/.cache/huggingface/hub/models--evo-design--evo-1-7b-131k-microviridae/snapshots/*/)
for f in tokenizer_config.json special_tokens_map.json generation_config.json; do
    [ -e "${DST}${f}" ] || cp "${SRC}/${f}" "${DST}${f}"
done
```

(The actual loader for microviridae — `src/loaders/evo1_loader.py:_load_microviridae` — no longer uses any of the cached .py files; it builds StripedHyena from the evo package directly. The copies above are still needed for the togethercomputer / LongSafari Evo1 models that DO go through evo.Evo()'s loader.)

#### D.2 togethercomputer/evo-1-8k-base — symlink shards from main → 1.1_fix

`evo.Evo("evo-1-8k-base")` requests revision `1.1_fix`, but that
snapshot directory only contains shard 3 of 3 + index files; shards 1
and 2 must be symlinked from the `main` revision (the blob content is
identical between revisions, only the snapshot dir is partially
populated):

```bash
MAIN=/data/yusen/software/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base/snapshots/a9be7b66485080893399ade87c7d34f81ad3e249
FIX=/data/yusen/software/.cache/huggingface/hub/models--togethercomputer--evo-1-8k-base/snapshots/6d7baa4482172f2c451ca4b36c87d50c8359a134
for shard in model-00001-of-00003.safetensors model-00002-of-00003.safetensors; do
    if [ -L "$MAIN/$shard" ] && [ ! -e "$FIX/$shard" ]; then
        ln -s "$(readlink "$MAIN/$shard")" "$FIX/$shard"
    fi
done
```

(If the entire `evo-1-8k-base` repo is re-downloaded fresh, `main` may
not exist yet — pull main first then redo this. Snapshot hashes
above are pinned because that's what `evo.Evo` checks.)

#### D.3 PlantBiMoE — copy .py files to snapshot root

`plant-llms/PlantBiMoE` ships its modeling code inside a
`plantbimoe/` subdirectory of the snapshot, but its `config.json`
`auto_map` references the files at snapshot root (e.g.
`configuration_plantbimoe.PlantbimoeConfig`, no `plantbimoe.`
prefix). Flatten:

```bash
SNAP=$(echo /data/yusen/software/.cache/huggingface/hub/models--plant-llms--PlantBiMoE/snapshots/*/)
for f in configuration_plantbimoe.py modeling_plantbimoe.py tokenization_plantbimoe.py; do
    [ -e "${SNAP}${f}" ] || cp "${SNAP}plantbimoe/${f}" "${SNAP}${f}"
done
```

#### D.4 AIDO.DNA — copy vocab.txt + rebuild safetensors index

`genbio-ai/AIDO.DNA-300M` and `AIDO.DNA-7B` ship a `model_type=rnabert`
config that transformers doesn't have natively registered. We import
`RNABertForMaskedLM` from
`models/modelsHFNoInfo/ModelGenerator/modelgenerator/huggingface_models/rnabert/`
in our loader, which means the same package's `vocab.txt` needs to be
present in each AIDO.DNA snapshot so `RNABertTokenizer(vocab_file=...)`
can find it:

```bash
SRC=/nvme-data3/yusen/worksapce/glm_mapping/genome_model_population_genetics/models/modelsHFNoInfo/ModelGenerator/modelgenerator/huggingface_models/rnabert/vocab.txt
for repo in models--genbio-ai--AIDO.DNA-300M models--genbio-ai--AIDO.DNA-7B; do
    for snap in /data/yusen/software/.cache/huggingface/hub/${repo}/snapshots/*/; do
        [ -e "${snap}vocab.txt" ] || cp "$SRC" "${snap}vocab.txt"
    done
done
```

`AIDO.DNA-7B` additionally ships sharded weights as `.safetensors` but
its index file is named `pytorch_model.bin.index.json` and references
`.bin` filenames. Build a matching `model.safetensors.index.json` that
points at the actual `.safetensors` shards, and drop the two tied keys
(`cls.predictions.decoder.{weight,bias}`) which aren't in the
safetensors files because they're tied to the embeddings at runtime:

```bash
SNAP=$(echo /data/yusen/software/.cache/huggingface/hub/models--genbio-ai--AIDO.DNA-7B/snapshots/*/)
/nvme-data3/yusen/micomamba/bin/python <<PY
import json
from safetensors.torch import safe_open
SNAP = "$SNAP"
d = json.loads(open(SNAP + "pytorch_model.bin.index.json").read())
new_map = {k: v.replace(".bin", ".safetensors") for k, v in d["weight_map"].items()}
# Drop keys that aren't actually in the safetensors shards (tied weights)
shards = sorted(set(new_map.values()))
actual = set()
for sh in shards:
    with safe_open(SNAP + sh, framework="pt") as f:
        actual.update(f.keys())
new_map = {k: v for k, v in new_map.items() if k in actual}
out = {"metadata": d["metadata"], "weight_map": new_map}
with open(SNAP + "model.safetensors.index.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote model.safetensors.index.json with", len(new_map), "keys (dropped", len(d['weight_map']) - len(new_map), ")")
PY
```

### E. Verify

After redoing whichever sections above apply, smoke-test by running
`scripts/run_rerun_stability.py --hf-ids "<canonical_id>" --device cuda:0 --n-probes 3`
through the appropriate env's python, and confirm `r=1.0 max_diff=0`.
One canary per family is enough to catch most regressions:

```bash
# canaries (run via the env each one belongs to per the routing table)
caduceus env:   kuleshov-group/caduceus-ph_4M
PlantCAD env:   kuleshov-group/PlantCAD2-Small-l24-d0768   # bring CUDA_VISIBLE_DEVICES + LD_LIBRARY_PATH
                Mishamq/HybriDNA-300M
                plant-llms/PlantBiMoE
                RaphaelMourad/Jamba-DNA-v1-114M-hg38
dnabert2 env:   zhihan1996/DNABERT-2-117M
gf env:         DOEJGI/GenomeOcean-100M
evo env:        togethercomputer/evo-1-8k-base
                evo-design/evo-1-7b-131k-microviridae
evo2 env:       arcinstitute/evo2_7b
hyena-dna env:  LongSafari/hyenadna-tiny-1k-seqlen
base env:       genbio-ai/AIDO.DNA-300M
                living-models/Botanic0-S
```
