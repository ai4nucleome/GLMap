# GLMap scoring containers (Apptainer / Singularity)

Containers for **running the 123 genomic language models** to recompute their
likelihood responses — they carry the GPU model-runtime environments. For
analysis only (using the precomputed results, reproducing figures/tables) no
container is needed: `pip install -e .` gives the torch-free analysis stack
(see the top-level README).

## Why per-group images (not one mega-image)

The 9 scoring environments are mutually incompatible — different Python,
torch and CUDA — so they can never share one interpreter and are kept as
isolated micromamba envs:

| micromamba env | Python | torch | CUDA | family signature |
|---|---|---|---|---|
| base       | 3.11.9  | 2.8.0        | 12.x  | transformers (NT / GENA-LM / ModernBERT / …) |
| dnabert2   | 3.8.20  | 1.13.1       | 11.7  | transformers 4.29 |
| megadna    | 3.10.0  | 2.9.0        | 12.x  | torch.load .pt |
| caduceus   | 3.8.0   | 2.2.0        | 11.8  | mamba-ssm 1.2, flash-attn 2.5.6 |
| gf         | 3.10.0  | 2.1.1        | 11.8  | evo-model, flash-attn 2.7.2 |
| hyena-dna  | 3.9.0   | 2.7.1        | 11.8  | flash-attn 1.0.7 |
| PlantCAD   | 3.11.14 | 2.5.1        | 12.1  | mamba-ssm 2.2.4 |
| evo        | 3.11.0  | 2.6.0        | 12.4  | evo-model, flash-attn 2.7.4 |
| evo2       | 3.12.0  | 2.6.0        | 12.4  | evo2 |

The exact package manifest of each env is captured under
[`../models/env-specs/`](../models/env-specs/) (`pip freeze` per env) as the build reference.

## Layered design

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

> Every group ships its envs **pre-built** as `packed/*.tar.gz` (conda-pack of
> the clean host envs, or a plain directory tarball for the conda/pip-polluted
> ones); the build only unpacks them, so no pip / network is needed at build
> time (a couple of envs add one small wheel — pyarrow, torchvision — from the
> Tsinghua mirror). All four group images are built and validated end-to-end.

## Build (on a host with Apptainer; e.g. the HPC)

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
