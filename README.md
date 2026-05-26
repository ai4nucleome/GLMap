# GLMap

**Profiling genomic language models as individuals in a population.**

<p align="center">
  <img src="assets/Fig1.png" alt="GLMap overview" width="80%"/>
</p>

GLMap is a training-free, architecture-agnostic framework for representing
and comparing genomic language models (GLMs) by their likelihood responses
over a fixed panel of DNA sequences. Applied to **123 publicly available
GLMs** scored on a panel of **10,000 DNA probes**, GLMap places autoregressive
(AR) and masked-language (MLM) models in a common space, yields model
distances that are stable to the choice of probes, and reflects known
relationships among models.

---

## Installation

**From source** (recommended for development and full reproducibility):

```bash
git clone https://github.com/ai4nucleome/GLMap.git
cd GLMap
pip install -e .            # core: analysis, matrix loading, figures
pip install -e .[scoring]   # adds torch + transformers for model scoring
pip install -e .[dev]       # adds pytest, build, twine
```

**After PyPI release** (forthcoming with the paper):

```bash
pip install ai4nucleome-glmap            # core
pip install ai4nucleome-glmap[scoring]   # + torch/transformers
```

> **Note**: `import glmap` does not trigger `import torch` or
> `import transformers`. Heavy dependencies are loaded on demand inside
> `get_loader()`, so the core install is usable for analysis and figures
> even without GPU packages installed.

---

## Quickstart: use precomputed GLMap artefacts

All precomputed artefacts for the paper's 123 models are included in
the source repository. No GPU, no model download, no scoring required.

```python
import glmap

# Load the 10,000-probe panel
panel = glmap.load_panel()       # (10000, 11) DataFrame

# Load precomputed matrices
V_AR  = glmap.load_matrix("V_AR")    # (64, 10000) raw AR responses
Vd_AR = glmap.load_matrix("Vd_AR")   # (64, 10000) double-centered

# Recompute the matrix pipeline from raw scores
info = glmap.fit_matrix(V_AR, clip_q=0.02)
# info["Vd"], info["D"], info["clip_threshold"], ...

# Project a new model into the existing Vd space
Vd_new = glmap.project(new_model_scores, info)

# Load the 123-model audit metadata
audit = glmap.load_audit()       # list of 123 dicts
specs = glmap.specs_from_audit() # list of 123 ModelSpec objects
```

---

## Re-run scoring and downstream evaluation

Reproducing the full pipeline from scratch requires GPUs, model weights,
and benchmark data. See the sections below for setup.

**Quick example** (single model, single GPU):

```bash
python scripts/run_phase1_scoring.py --from-audit \
    --hf-ids zhihan1996/DNABERT-2-117M --device cuda:0
```

**Full 123-model reproduction** (requires multiple environments + GPUs;
see [docs/env_routing.md](docs/env_routing.md)):

```bash
# 1. Parallel scoring across 123 models (workers use --skip-aggregate)
python scripts/run_sweep.py --mode scoring --audit data/audits/models.json

# 2. Build V/Vd/D matrices (CPU, after all scoring workers finish)
python scripts/run_phase1_scoring.py --from-audit --strict-aggregate

# 3. (Optional) Downstream diagnostics / PCA / GC-axis reports
python scripts/run_phase1_analysis.py

# 4. Parallel downstream embedding extraction (requires benchmark CSVs)
python scripts/run_sweep.py --mode embed --audit data/audits/models.json

# 5. Train linear probes and compute AUCs
python scripts/run_downstream_classify.py

# 6. Generate paper figures
python scripts/figures/fig2c_split_half_consistency.py --seed 123
python scripts/figures/fig3a_model_map_family.py
# ... (see scripts/figures/ for all figure scripts)
```

See [docs/env_routing.md](docs/env_routing.md) for per-family environment
setup required by the parallel sweep.

---

## Repository layout

```
GLMap/
├── src/glmap/              Python package (pip install -e .)
│   ├── loaders/            12 loader families (HF, evo, genslm, ...)
│   ├── scoring/            AR log-likelihood + MLM stride PLL
│   ├── panel/              Probe panel construction
│   ├── matrices/           clip + double-center + pairwise distances
│   ├── analysis/           PCA, GC-axis, heterozygosity
│   └── io/                 Parquet schema helpers
├── scripts/                CLI entry points for paper reproduction
│   ├── figures/            One script per paper figure
│   ├── tables/             One script per paper table
│   ├── audits/             Model + benchmark audit scripts
│   └── download_models/    HF model download helper
├── tests/                  217 pytest tests
├── data/
│   ├── audits/             123-model audit (models.json)
│   ├── panel_sources.yaml  Panel construction spec
│   └── benchmark_manifests/ Downstream task metadata
├── out_panel/              Prebuilt probe panel parquets
├── out_phase1/
│   ├── matrices/           V/Vd/D for AR and MLM branches
│   └── scores/             Per-model likelihood responses (slimmed)
├── out_phase2/
│   ├── downstream/         Per-model per-task AUC results
│   ├── phenotype_prediction/ RidgeCV prediction outputs
│   └── model_map/          t-SNE embeddings for Fig 3
├── figures/                Paper figure PDFs
├── tables/                 Paper table LaTeX sources
├── models/                 Model download manifest + setup scripts
└── docs/                   Methods, env routing, model catalog
```

---

## Pre-built artefacts vs user-downloaded data

| Included in this repository | User must download separately |
|---|---|
| Probe panel (10,000 probes, 8 MB) | HF model weights (~119 models via `hf download`) |
| V/Vd/D matrices for AR + MLM (20 MB) | 9 external model repos (`setup_external_models.sh`) |
| Per-model scores, slimmed (48 MB) | GenSLM pretrained weights (manual) |
| Downstream AUC results (6 MB) | Benchmark task CSVs from [DNA Foundation Benchmark](https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset) |
| Phenotype prediction outputs (2 MB) | |
| t-SNE model map embeddings | |
| Paper figures (23 PDFs) and tables (12 .tex) | |

---

## Model setup

**HuggingFace models** (119 of 123):

```bash
bash scripts/download_models/download_models_from_list.sh
```

**External models** (9 repos with custom loaders):

```bash
bash models/setup_external_models.sh
```

See [models/README.md](models/README.md) for details on megaDNA, GenSLM,
PlasmidGPT, and other special cases. Model weights follow their own
upstream licenses.

---

## Downstream benchmark setup

The 6 downstream classification tasks are from the
[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)
(Feng et al., 2025). Raw task CSVs are **not** bundled in this repository.

```bash
huggingface-cli download hfeng3/dna_foundation_benchmark_dataset \
    --repo-type dataset --local-dir data/dna_foundation_benchmark
```

See [data/benchmark_manifests/README.md](data/benchmark_manifests/README.md)
for expected directory layout and task details.

---

## The GLMap representation

<p align="center">
  <img src="assets/Fig2.png" alt="GLMap representation" width="90%"/>
</p>

The GLMap representation matrix *V_d* exhibits coherent block structure by
model family, and the split-half distance geometry is stable across
element-disjoint probe partitions (Pearson *r* = 0.835 over model-pair
distances).

<p align="center">
  <img src="assets/Fig3.png" alt="GLMap model map and prediction" width="90%"/>
</p>

The *V_d* representation predicts downstream task performance (mean AUC
Spearman ρ = 0.705 under random *K*-fold cross-validation).

---

## Acknowledgements

GLMap builds on the ideas and infrastructure of several outstanding
open-source projects:

- **[ModelMap](https://github.com/shimo-lab/modelmap)** (Oyama et al.,
  ACL 2025) — the clip + double-center pipeline applied to
  log-likelihood vectors originates from ModelMap's profiling of 1,000+
  natural-language LMs.
- **[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)**
  (Feng et al., 2025) — provides the curated suite of binary
  classification tasks used in our downstream evaluation.

We also thank the authors and maintainers of the 123 genomic language
models audited in this work for releasing their weights and code publicly.

---

## Citation

```bibtex
@article{hou2026glmap,
  title   = {Profiling genomic language models as individuals in a population},
  author  = {Hou, Yusen and Long, Weicai and Su, Houcheng and Feng, Junning and Zhang, Yanlin},
  journal = {In submission},
  year    = {2026}
}
```

---

## License

This repository is licensed under [Apache-2.0](LICENSE). Individual model
weights follow their own upstream licenses (see
[models/README.md](models/README.md)).
