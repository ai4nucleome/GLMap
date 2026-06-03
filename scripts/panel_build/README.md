# scripts/panel_build/

Panel construction code. Reads raw benchmark CSV/FASTA files and produces
the canonical 10,000-probe panel (`data/panels/main_panel.parquet`).

The prebuilt panel is already committed to this repository, so you only
need this directory if you want to:

1. **Regenerate** the canonical panel from scratch (verify reproducibility), or
2. **Build a custom panel** by editing `data/panel_sources.yaml`.

For loading the prebuilt panel, just use:

```python
import glmap
panel = glmap.load_panel()
```

---

## Layout

```
scripts/panel_build/
├── main_panel.py    Orchestrator: config parsing + sampling + ProbeRow assembly
├── readers.py       Per-format readers (CSV + 3 PGB FASTA variants)
└── README.md        (this file)
```

The 16/64-D `dinuc_vec` / `trinuc_vec` / GC helpers used during build live
in the library at `glmap.panel.composition` so figures and other callers
can reuse them.

---

## Build pipeline

1. **Download the three upstream benchmarks** (see "Upstream data" below).
2. **Edit `data/panel_sources.yaml`** if you want a custom panel (optional).
3. **Run the build**:

   ```bash
   bash scripts/0_build_10000_probes_dataset.sh
   ```

   Produces:
   - `data/panels/main_panel.parquet` (10,000 probes × 11 columns)
   - `data/panels/panel_manifest.json` (per-element / per-dataset counts + sub-seeds)
   - `data/panels/panel_summary.{md,tsv}` (human-readable summaries)

Build is deterministic given `seed: 42` in `panel_sources.yaml`.

---

## Upstream data

The build code reads from `data/{GUE, PGB, dna_foundation_benchmark}/`.
These directories are **not** included in this repository — download them
from their upstream sources:

### GUE (Genome Understanding Evaluation, DNABERT-2)

- **Download**: <https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T->
- **License**: Apache-2.0 (DNABERT-2 repository)
- **Citation**: Zhou, Z. et al. *DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome.* ICLR 2024.
- **Used for**: promoter, splice_donor, splice_acceptor, fungi_genome, yeast_genome, virus_species, virus_variants (≈ 5,600 probes)

Extract to `data/GUE/` so the layout looks like:
```
data/GUE/
├── prom/prom_300_{all,notata,tata}/{train,dev,test}.csv
├── splice/reconstructed/{train,dev,test}.csv
├── fungi/species_20/{train,dev,test}.csv
├── EMP/H3/{train,dev,test}.csv
└── virus/{species_40,covid}/{train,dev,test}.csv
```

### PGB (Plant Genomic Benchmark, AgroNT)

- **Download**: <https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark>
- **License**: **CC-BY-NC-SA-4.0** ⚠️ (non-commercial, share-alike)
- **Citation**: Mendoza-Revilla, J. et al. *A Foundational Large Language Model for Edible Plant Genomes.* Communications Biology 7, 835 (2024).
- **Used for**: chromatin_access, polyA, lncRNA, nascent_RNA, splicing_plant_donor, splicing_plant_acceptor (≈ 1,600 probes)

Extract to `data/PGB/` so the layout looks like:
```
data/PGB/
├── chromatin_access/{species}_{train,test}.fa
├── poly_a/{species}_{train,test}.fa
├── lncrna/{species}_{train,test}.fa
├── pro_seq/m_esculenta_{train,test}.fa
└── splicing/arabidopsis_thaliana_{donor,acceptor}_{train,test}.fa
```

### DNA Foundation Benchmark (DFB)

- **Download**: <https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset>
- **License**: Apache-2.0
- **Citation**: Feng, J. et al. *Benchmarking DNA Foundation Models for Genomic Sequence Classification.* (2025).
- **Used for**: enhancer (1,400 probes)

Extract to `data/dna_foundation_benchmark/` so the layout looks like:
```
data/dna_foundation_benchmark/data_processed/
└── enhancers/enhancer/{train,test}.csv
```

The same directory is also used by `scripts/run_downstream_embed.py` for
the 6 downstream classification tasks (see
`data/benchmark_manifests/downstream_tasks.json`).

---

## License attribution for the built panel

Because `data/panels/main_panel.parquet` contains 1,600 probes drawn from
PGB (CC-BY-NC-SA-4.0), the panel parquet itself is licensed under
**CC-BY-NC-SA-4.0** (the most restrictive upstream license, via
ShareAlike). The GLMap **code** in this repository remains under
Apache-2.0; only the data artefacts in `data/panels/` and downstream
matrices in `results/scores/matrices/` are CC-BY-NC-SA-4.0. See
[`LICENSE-DATA`](../../LICENSE-DATA) at the repo root.

When citing the panel or any derived matrix, please cite all three
upstream benchmarks (above) plus the GLMap paper.
