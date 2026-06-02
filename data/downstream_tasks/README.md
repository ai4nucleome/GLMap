# Benchmark manifests

This directory contains machine-readable manifests for the benchmark tasks
used in GLMap, **without vendoring** the actual benchmark code or data.

## Downstream tasks (`downstream_tasks.json`)

Six binary classification tasks from the
[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)
(Feng et al., 2025). Total: 48,439 samples across 6 tasks.

### Obtaining the raw task CSVs

The raw `train.csv` and `test.csv` files for each task should be downloaded
from the upstream HuggingFace dataset:

```bash
# Option 1: download the full dataset
hf download hfeng3/dna_foundation_benchmark_dataset --repo-type dataset --local-dir data/dna_foundation_benchmark

# Option 2: clone the benchmark repo and follow its setup
git clone https://github.com/ChongWuLab/dna_foundation_benchmark
```

After downloading, the expected directory layout (relative to the repo root) is:

```
data/dna_foundation_benchmark/data_processed/
├── EMP/Yeast_H4/{train,test}.csv
├── enhancers/enhancer/{train,test}.csv
├── iDNA_ABF/5mC/{train,test}.csv
├── iPro-WAEL/Promoter_Arabidopsis_TATA/{train,test}.csv
├── mouse/mouse_TFBS_3/{train,test}.csv
└── prom/promoter_tata_300bps/{train,test}.csv
```

Each CSV has columns `Sequence` (DNA string) and `Label` (0 or 1).