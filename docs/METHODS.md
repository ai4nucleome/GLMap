# Methods

This document describes the computational methods behind GLMap. It is a
standalone reference for the code in this repository; the paper provides
the full scientific context, motivation, and results.

---

## 1. Model collection

We audited **123 publicly available genomic language models** from the
Hugging Face Hub and GitHub. Models were included if they could process DNA
inputs of at least 1,024 bp (matching the maximum probe length in our panel).

The collection spans three axes of diversity:

- **Architecture**: transformer encoders, transformer decoders, state-space
  models (Mamba, StripedHyena), and hybrids.
- **Training paradigm**: autoregressive (AR, 64 models) and masked language
  modeling (MLM, 59 models).
- **Tokenization**: single-nucleotide, non-overlapping *k*-mer (k = 3, 6),
  and byte-pair encoding (BPE).

Parameter counts range from ~471K (Caduceus-118d) to 40B (Evo-2 40B).

The full model list is in `data/audits/models.json`. Loader dispatch is
handled by `glmap.loaders.dispatch`, which maps each audit entry to one
of 14 loader kinds. See `docs/env_routing.md` for the per-family
environment routing table.

---

## 2. DNA probe panel

The probe panel consists of **10,000 sequences** drawn from three published
genomic benchmark suites: GUE (DNABERT / DNABERT-2), the Plant Genomic
Benchmark (PGB), and the Nucleotide Transformer benchmark.

The panel covers **14 functional elements** across 4 biological categories:

| Category | Elements | Probes |
|---|---|---|
| Human (4,000) | promoter, enhancer, splice donor, splice acceptor | 1,400 + 1,400 + 600 + 600 |
| Plant (1,600) | chromatin accessibility, polyA, lncRNA, nascent RNA, splicing (donor + acceptor) | 450 + 350 + 300 + 200 + 300 |
| Fungi (2,700) | fungi genome (20 species), yeast genome | 1,500 + 1,200 |
| Virus (1,700) | virus species (25 species), virus variants (9 COVID lineages) | 1,100 + 600 |

Probe lengths range from 156 to 1,024 bp. Sampling is stratified by
functional element with a fixed random seed (42) for reproducibility.

The panel construction spec is in `data/panel_sources.yaml`. The prebuilt
panel is at `out_panel/main_panel.parquet` (deterministic, regenerable via
`python scripts/build_panel.py`).

---

## 3. Likelihood response computation

For each model and each probe sequence, we compute a single scalar score.

**AR models**: the sequence log-likelihood, computed in a single forward pass:

```
ℓ_AR(x) = Σ_{i=2}^{T} log p(t_i | t_{<i})
```

The sum runs over predictable positions only (i = 2 … T); the first
token has no left context and is excluded.

**MLM models**: the stride pseudo-log-likelihood (PLL), following
Salazar et al. (2020). Tokens are partitioned into *k* equally spaced
subsets; each subset is masked and scored together, covering all tokens
in *k* forward passes:

```
PLL(x) ≈ Σ_{i=1}^{T} log p(t_i | t_{\setminus i})    [estimated with stride k]
```

Each token t_i is scored conditioned on the full sequence with position
i masked.

The primary stride is **k = 6**. We validated this choice against the
exact per-token PLL (k = 1) on 51 of the 59 MLM models scored over a
1,000-probe subset (per-model Pearson *r*: median 0.999, range
[0.985, 1.000]; pooled *r* = 0.995 across 51,000 model–probe pairs).
Results are in `out_phase1/figS3_per_model_r.json`.

Precision and environment are loader-specific. Most models were scored in
FP32; larger models (Carbon, Evo-2 20B/40B, Genos-10B) used FP16 or
BF16 due to memory constraints. Per-family precision and GPU requirements
are documented in `docs/env_routing.md`.

---

## 4. Matrix construction: V → V_d → D

The likelihood responses of all models on all probes form the raw response
matrix **V** ∈ ℝ^{n × m} (n models, m probes).

### Lower-tail clipping

```python
V_clipped, threshold = glmap.clip_lower(V, q=0.02)
```

Cells below the 2nd percentile of all finite entries are floored to that
threshold, preventing catastrophic outliers from dominating the centering.

### Double-centering

```python
Vd, row_mean, col_mean, grand_mean = glmap.double_center(V_clipped)
```

Row centering removes each model's overall score level; column centering
(applied to the row-centered matrix) removes each probe's intrinsic
difficulty. The resulting **V_d** captures residual model–probe
interactions — the functional preferences that distinguish models from
one another.

The column mean used in centering is computed on the *row-centered*
matrix (not the original V). `grand_mean` is reported for diagnostics
only; it is not used in the centering formula.

### Pairwise distances

```python
D = glmap.pairwise_distances(Vd)    # D[i,j] = ||Vd[i] - Vd[j]||²
```

### Projecting new models

A new model can be projected into an existing V_d space without
refitting:

```python
info = glmap.fit_matrix(V_existing, clip_q=0.02)
Vd_new = glmap.project(V_new_row, info)
```

The new row is clipped at the fitted threshold, its own row mean is
subtracted, and the fitted column mean is subtracted. Existing rows
remain unchanged.

---

## 5. Downstream embedding evaluation

To test whether V_d carries information about model capability, we
predicted downstream task performance from GLMap signatures.

### Task setup

We used 6 binary classification tasks from the
[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)
(Feng et al., 2025): Yeast H4, enhancer, 5mC, promoter TATA (300 bp),
mouse TFBS 3, and *Arabidopsis* promoter TATA. Total: 48,439 samples.

Task metadata is in `data/benchmark_manifests/downstream_tasks.json`.

### Embedding extraction

For each model and task, we extracted sequence representations as the
mean pooling of the last hidden state over content tokens, with
loader-specific adaptations for non-standard architectures.

### Linear probing

A linear probe (L2-regularized logistic regression with StandardScaler
preprocessing) was trained on the frozen representations. We report the
test AUC for each model–task pair.

### Phenotype prediction

For each task, we used each model's V_d signature as input features and
its downstream AUC as the target, fitting a ridge regression (RidgeCV)
with 5-fold cross-validation over 5 random seeds.

Two cross-validation schemes:

- **Random K-fold**: mean AUC Pearson *r* = 0.681, Spearman ρ = 0.705.
- **Family-grouped K-fold** (no family shared between folds): mean AUC
  Pearson *r* = 0.501, Spearman ρ = 0.565.

Results are in `out_phase2/phenotype_prediction/`.

---

## 6. Robustness checks

### Split-half probe stability

We partitioned the 10,000-probe panel into two element-disjoint halves
(so the two halves contained entirely different functional elements),
computed V_d and the pairwise distance matrix D independently on each
half, and correlated the resulting model-pair distances (Pearson *r*
over the C(123, 2) = 7,503 distance pairs). This split-half correlation
of pairwise model distances was **0.835** (seed = 123), and was
consistent across multiple random partitions.

Precomputed figure: `figures/Fig2c-split-half-mantel_element-disjoint_seed123_with-pipeline.pdf`.

### Stride PLL approximation

The stride PLL (k = 6) closely matched the exact per-token PLL (k = 1)
across 51 of the 59 MLM models on a stratified 1,000-probe subset:

- Per-model Pearson *r*: median 0.999, mean 0.998, range [0.985, 1.000].
- Pooled across 51,000 model–probe pairs: Pearson *r* = 0.995.

This analysis validates the stride approximation and confirms that it
supports the stride approximation for the downstream GLMap analyses
reported in the paper. The 1,000-probe subset was sampled
to match the per-element composition of the full panel (seed = 42).

Results: `out_phase1/figS3_per_model_r.json`.
Figure: `figures/FigS3-stride_pll_per_model_r.pdf`.

### AR/MLM joint analysis

The AR/MLM branch label explains only **1.9%** of variance in V_d,
compared to **53.8%** for model family and **35.5%** for organization.
This was equally small in the raw scores (1.8%), indicating the
distinction is not an artefact of the centering transform. Per-branch
distance geometries are preserved after joint centering (minimum within-
branch Spearman ρ = 0.963).

---

## 7. Reproducibility notes

- All random operations use explicit seeds (panel: 42; phenotype
  prediction: 5 seeds per fold; split-half: configurable via `--seed`).
- The probe panel is deterministic given `data/panel_sources.yaml` and
  seed 42; `python scripts/build_panel.py` regenerates identical
  parquets.
- Per-model scores depend on model weights, CUDA version, and floating-
  point precision; small numerical differences (< 1e-4) are expected
  across hardware.
- The matrix pipeline (clip + double-center) is deterministic given the
  same input scores.
