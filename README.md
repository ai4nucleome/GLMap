# 🧬 🗺️ GLMap: Profiling genomic language models as individuals in a population

> 🌐 Language: [中文](README.zh.md) · **English**

> 📖 **Project page: [ai4nucleome.github.io/GLMap](https://ai4nucleome.github.io/GLMap/)**

<p align="center">
  <img src="assets/Fig1.png" alt="GLMap overview" width="80%"/>
</p>

GLMap is a training-free, architecture-agnostic framework for representing and comparing genomic language models (GLMs) by their likelihood responses over a fixed panel of DNA sequences. Applied to **123 publicly available GLMs** scored on a panel of **10,000 DNA probes**, GLMap places autoregressive (AR) and masked-language (MLM) models in a common space, yields model distances that are stable to the choice of probes, and reflects known relationships among models.

---

## Installation

### Reproduce analysis in paper via precomputed results

To use our precomputed 123-model PLL / log-likelihood responses over the
10,000-probe panel — plus the prebuilt panel, the V/Vd/D matrices and audit
metadata — and reproduce every figure/table, the install below is all you
need. **No GPU, no model weights, no scoring**; it makes `glmap` importable
with only lightweight, torch-free dependencies.

```bash
git clone https://github.com/ai4nucleome/GLMap.git
cd GLMap
pip install -e .
```

We recommend **Python 3.11.9**; the exact analysis-stack versions are pinned
in [`pyproject.toml`](pyproject.toml).

### Recomputing the 123-model scores

The 123 models span many **mutually incompatible runtime environments** —
different Python / PyTorch / CUDA versions per model family. We are packaging
these environments into container images so that recomputing the likelihood
responses for any model will be straightforward. **Coming soon.**

---

## Quickstart: use precomputed GLMap artefacts

All precomputed artefacts for the paper's 123 models are included in
the source repository. No GPU, no model download, no scoring required.

```python
import glmap

# Load the 10,000-probe panel (on disk: data/panels/main_panel.parquet).
# - From a repo checkout / $GLMAP_DATA_DIR: read locally.
# - From a pip install (no checkout): auto-downloaded from the GLMap
#   HuggingFace Dataset (Tim419/GLMap-panels) and cached.
panel = glmap.load_panel()       # (10000, 11) DataFrame

# Or load your own custom panel built with scripts/panel_build/
# panel = glmap.load_panel(path="my_panel.parquet")

# Load precomputed matrices by name. They resolve from the repo checkout
# (or $GLMAP_DATA_DIR), on disk at results/scores/matrices/<file>.npy:
#   V_AR->V_AR.npy   Vd_AR->V_d_AR.npy   D_AR->D_AR.npy   (+ the _MLM trio)
V_AR  = glmap.load_matrix("V_AR")    # (64, 10000)  raw AR responses   (MLM: 59 models)
Vd_AR = glmap.load_matrix("Vd_AR")   # (64, 10000)  double-centered
D_AR  = glmap.load_matrix("D_AR")    # (64, 64)     pairwise model distances

# Recompute the matrix pipeline from raw scores
info = glmap.fit_matrix(V_AR, clip_q=0.02)
# info["Vd"], info["D"], info["clip_threshold"], ...

# Project a new model into the existing Vd space
Vd_new = glmap.project(new_model_scores, info)

# Load the 123-model audit metadata
audit = glmap.load_audit()       # list of 123 dicts
specs = glmap.specs_from_audit() # list of 123 ModelSpec objects
```

> The panel is published as a HuggingFace Dataset at
> [`Tim419/GLMap-panels`](https://huggingface.co/datasets/Tim419/GLMap-panels)
> (CC-BY-NC-SA-4.0). `load_matrix` and `load_audit` read from the
> repository checkout or `$GLMAP_DATA_DIR` (not auto-downloaded).

---

## Repository layout

```
GLMap/
├── glmap/                  Python package (importable; `pip install -e .`)
│   ├── loaders/            Per-family model loaders (HF, evo, genslm, ...) + dispatch
│   ├── scoring/            AR log-likelihood + MLM stride PLL
│   ├── matrices/           clip + double-center + pairwise distances
│   └── formats_check/      Embedding-parquet schema validation
├── scripts/                CLI entry points for paper reproduction
│   ├── panel_build/        Panel construction + panel_sources.yaml spec
│   ├── figures/            One script per paper figure
│   ├── tables/             One script per paper table
│   ├── audits/             Model audit script + context overrides
│   └── 0_*.sh … 7_*.sh     Numbered pipeline drivers (audit → … → model map)
├── tests/                  pytest test suite
├── data/
│   ├── audits/             123-model audit (models.json)
│   ├── downstream_tasks/   Downstream task metadata
│   └── panels/             Prebuilt probe panel parquets
├── results/
│   ├── scores/             Scoring outputs
│   │   ├── matrices/       V/V_d/D for AR and MLM branches
│   │   └── AR_MLM_scores/  Per-model likelihood responses (slimmed)
│   ├── analysis/           Downstream + secondary analysis outputs
│   │   ├── benchmark_perform_prediction/
│   │   │   ├── per_model_AUC_result_6tasks/  Per-model per-task AUC results
│   │   │   ├── all_model_AUC_6tasks/         Aggregated (123×6) AUC matrix
│   │   │   └── phenotype_prediction/         Predict downstream AUC from GLMap signatures
│   │   ├── model_map/      t-SNE / MDS embeddings for Fig 3
│   │   └── MLM_stride-PLL_vs_true-PLL_1000samples/  k=1 vs k=6 PLL ablation (Fig S3)
│   ├── figures/            Paper figure PDFs
│   └── tables/             Paper table LaTeX sources
└── models/                 Model download manifest, setup scripts,
                            and per-family environment routing (env_routing.md)
```

---

## Pre-built artefacts included in this repository

Everything needed to reproduce the paper's analysis from precomputed results
ships with the repo — no model weights, no scoring required:

| Artefact | Size |
|---|---|
| Probe panel (10,000 probes) | 8 MB |
| V/Vd/D matrices for AR + MLM | 20 MB |
| Per-model likelihood responses, slimmed | 48 MB |
| Downstream AUC results | 6 MB |
| Phenotype prediction outputs | 2 MB |
| t-SNE model map embeddings | — |
| Paper figures (23 PDFs) and tables (12 .tex) | — |

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
  ACL *2025*) — the clip + double-center pipeline applied to
  log-likelihood vectors originates from ModelMap's profiling of 1,000+
  natural-language LMs.
- **[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)**
  (Feng et al., Nat. Comm. *2025*) — provides the curated suite of binary
  classification tasks used in our downstream evaluation.

We also thank the authors and maintainers of the **123 genomic language
models** audited in this work for releasing their weights and code publicly.

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

This repository uses **two licenses**:

- **Source code** (everything under `glmap/`, `scripts/`, `tests/`,
  etc.): [Apache-2.0](LICENSE).
- **Data artefacts** (`data/panels/`, `results/scores/matrices/`,
  `results/scores/AR_MLM_scores/`, `results/analysis/`): [CC-BY-NC-SA-4.0](LICENSE-DATA).
  These artefacts inherit the upstream Plant Genomic Benchmark license
  (1,600 probes drawn from PGB; CC-BY-NC-SA-4.0 via ShareAlike). They
  are usable for non-commercial research with attribution; commercial
  use requires obtaining the panel from a license-compatible source.

Individual model weights also follow their own upstream licenses (see
[models/README.md](models/README.md)).
