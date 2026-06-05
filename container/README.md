# GLMap scoring containers (Apptainer / Singularity)

Containers for **running the 123 genomic language models** to recompute their
likelihood responses. 

For analysis only (using precomputed results, reproducing figures/tables) no
container is needed: `pip install -e .` gives the torch-free analysis stack
(see the [top-level README](../README.md)).

## Why per-group images

The scoring environments are mutually incompatible, different Python,
torch and CUDA, so they can never share one interpreter and are kept as
isolated envs.

The exact package manifest of each env is captured under
[`../models/env-specs/`](../models/env-specs/) (`pip freeze` per env) as the build reference.

## 🪞 Containers!

A single shared base, then one image per CUDA-compatibility group. Each group
image holds its env(s) as isolated micromamba envs and dispatches per family
at run time.

```
Base.def ──► base-cu128.sif         CUDA 12.8 devel + micromamba + build tools
                  │  (localimage bootstrap)
                  ├─► bio-default.sif   envs: base, dnabert2, megadna
                  ├─► bio-cu118.sif     envs: caduceus, gf, hyena-dna
                  ├─► bio-cu121.sif     envs: PlantCAD
                  └─► bio-evo.sif       envs: evo, evo2
```

## Download the prebuilt images (Hugging Face)

The four group images are published as a HuggingFace **dataset**,
[`Tim419/GLMap-containers`](https://huggingface.co/datasets/Tim419/GLMap-containers).
Download only the one(s) for the model families you want to score — each is
**self-contained** (the shared base is already inside; `base-cu128.sif` is
build-only):

```bash
hf download Tim419/GLMap-containers bio-default.sif --repo-type dataset --local-dir .   # 16 GB
hf download Tim419/GLMap-containers bio-cu118.sif   --repo-type dataset --local-dir .   # 19 GB
hf download Tim419/GLMap-containers bio-cu121.sif   --repo-type dataset --local-dir .   # 15 GB
hf download Tim419/GLMap-containers bio-evo.sif     --repo-type dataset --local-dir .   # 23 GB
```

Then run with `apptainer run --nv` (or `singularity run --nv` on nodes without
user namespaces) — see **Run** below.

| image | env(s) | model families |
|---|---|---|
| `bio-default.sif` | base / dnabert2 / megadna | NT, GENA-LM, ModernBERT, GROVER, Mistral-DNA, NTv3, … (most); DNABERT-2 / DNABERT-S; megaDNA |
| `bio-cu118.sif`   | caduceus / gf / hyena-dna | Caduceus; GenomeOcean; HyenaDNA |
| `bio-cu121.sif`   | PlantCAD                  | PlantCAD2 |
| `bio-evo.sif`     | evo / evo2                | Evo-1 / Evo-1.5; Evo-2 (7B) |

See [`../../models/env_routing.md`](../../models/env_routing.md) for the full
model → env routing.

## Build from source (maintainers)

On a host with Apptainer:

```bash
cd container          # %files paths + packed/ are relative to here

# 1. shared base (once)
apptainer build base-cu128.sif Base.def

# 2. each group image on top of the base
apptainer build bio-default.sif bio-default.def
apptainer build bio-cu118.sif   bio-cu118.def
apptainer build bio-cu121.sif   bio-cu121.def
apptainer build --fakeroot bio-evo.sif      bio-evo.def
```

`.sif` images are huge (10–20 GB each) and are **gitignored** — distribute via
a registry or Zenodo, not git.

## Run

Each group image dispatches by the `GLMAP_ENV` variable (which micromamba env
to use), then runs `python` inside it:

```bash
# score with the caduceus env from the cu118 group image
GLMAP_ENV=caduceus apptainer run --nv \
    --bind "$PWD":/work --pwd /work bio-cu118.sif \
    scripts/score/scoring_worker.py --from-audit --hf-ids kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16
```

`--nv` exposes the host GPU; `--bind $PWD:/work` mounts the GLMap checkout
(data, results, model weights). Map each model family → env via
[`../../models/env_routing.md`](../../models/env_routing.md).

## Full 123-model sweep — two backends

The scoring sweep runs the same way whether you built your own envs or use
these images; pick a backend:

```bash
# (A) your own micromamba envs (edit env_paths.yaml for your machine)
python scripts/score/run_scoring_sweep.py

# (B) the prebuilt images — no env setup, just point at the .sif directory
python scripts/score/run_scoring_sweep.py \
    --backend container --image-dir container \
    --hf-cache "$HF_HOME"
```

Backend (B) routes every model to its group image automatically (the
`ENV_IMAGE` map in `scripts/score/sweep_engine.py`), binds the checkout at
`/work` and the HF cache, and runs each via `apptainer run --nv` (use
`--container-runtime singularity` on compute nodes without user namespaces).
Both backends write the same `results/scores/AR_MLM_scores/<slug>/probes.parquet`
and accept the same flags (`--only`, `--hf-ids`, `--gpu-ids`, `--force`, …).
