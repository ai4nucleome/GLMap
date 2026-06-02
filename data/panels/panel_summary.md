# Stage 2 probe panel — composition summary

Total probes: **10,000** across **14 functional elements** × **4 species groups** (Human / Plant / Fungi / Virus).

**Source benchmarks**:

- **GUE** — DNABERT-2, Zhou et al. 2024
- **PGB** — Plant Genomic Benchmark, Mendoza-Revilla et al. 2024
- **NT-DFB** — Nucleotide Transformer, Dalla-Torre et al. 2024

Probes are uniformly the **positive** examples from each source classification task (i.e. functional element instances, not background sequences); the gLM-scoring pipeline ignores the original task labels and uses the sequences only.

## Table 1 — per-element summary

| Element | Species group | Species | Source benchmark | Original task path | n | Length (bp) | Length median (IQR) | GC median (IQR) | % of panel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| promoter | Human | Homo sapiens | GUE | prom/prom_300_all, prom/prom_300_notata, prom/prom_300_tata | 1400 | 300 | 300 (300–300) | 0.60 (0.49–0.69) | 14.0 |
| enhancer | Human | Homo sapiens | NT-DFB | enhancers/enhancer | 1400 | 199–200 | 199 (199–200) | 0.48 (0.45–0.52) | 14.0 |
| splice_donor | Human | Homo sapiens | GUE | splice/reconstructed | 600 | 400 | 400 (400–400) | 0.45 (0.37–0.55) | 6.0 |
| splice_acceptor | Human | Homo sapiens | GUE | splice/reconstructed | 600 | 400 | 400 (400–400) | 0.48 (0.37–0.60) | 6.0 |
| chromatin_access | Plant | multi-species (n=7) | PGB | chromatin_access | 450 | 1000 | 1000 (1000–1000) | 0.49 (0.42–0.56) | 4.5 |
| polyA | Plant | multi-species (n=6) | PGB | poly_a | 350 | 400 | 400 (400–400) | 0.38 (0.34–0.45) | 3.5 |
| lncRNA | Plant | multi-species (n=6) | PGB | lncrna | 300 | 156–1001 | 459 (296–704) | 0.43 (0.38–0.52) | 3.0 |
| nascent_RNA | Plant | Manihot esculenta | PGB | pro_seq | 200 | 1000 | 1000 (1000–1000) | 0.28 (0.24–0.34) | 2.0 |
| splicing_plant_donor | Plant | Arabidopsis thaliana | PGB | splicing | 150 | 398 | 398 (398–398) | 0.39 (0.36–0.42) | 1.5 |
| splicing_plant_acceptor | Plant | Arabidopsis thaliana | PGB | splicing | 150 | 398 | 398 (398–398) | 0.39 (0.36–0.42) | 1.5 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | EMP/H3, EMP/H3K14ac, EMP/H3K36me3 | 1200 | 500 | 500 (500–500) | 0.38 (0.35–0.41) | 12.0 |
| fungi_genome | Fungi | multi-species (n=20) | GUE | fungi/species_20 | 1500 | 1024 | 1024 (1024–1024) | 0.48 (0.41–0.52) | 15.0 |
| virus_variants | Virus | multi-species (n=9) | GUE | virus/covid | 600 | 999 | 999 (999–999) | 0.38 (0.36–0.39) | 6.0 |
| virus_species | Virus | multi-species (n=25) | GUE | virus/species_40 | 1100 | 1024 | 1024 (1024–1024) | 0.28 (0.25–0.43) | 11.0 |

Notes:
- `Length (bp)` is the observed range; `Length median (IQR)` shows the central tendency. For most elements the panel build fixes the probe length, so range collapses to a single value.
- `GC median (IQR)` is computed on the panel's final selected probes after any length truncation / filtering.
- `% of panel` rounds to one decimal; columns may not sum to exactly 100 due to rounding.

## Supplementary Table S1 — per-source file detail

Within each functional element, the panel draws from one or more source files. This table shows the breakdown (41 rows). Useful when reviewers ask which species / which underlying task contributed a given subset of probes.

| Element | Species group | Species | Source benchmark | Source file (relative to data/) | n | Length median | GC median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| promoter | Human | Homo sapiens | GUE | GUE/prom/prom_300_all/train.csv | 467 | 300 | 0.623 |
| promoter | Human | Homo sapiens | GUE | GUE/prom/prom_300_notata/train.csv | 467 | 300 | 0.633 |
| promoter | Human | Homo sapiens | GUE | GUE/prom/prom_300_tata/train.csv | 466 | 300 | 0.533 |
| enhancer | Human | Homo sapiens | NT-DFB | dna_foundation_benchmark/enhancers/enhancer/train.csv | 1400 | 199 | 0.477 |
| splice_donor | Human | Homo sapiens | GUE | GUE/splice/reconstructed/train.csv | 600 | 400 | 0.449 |
| splice_acceptor | Human | Homo sapiens | GUE | GUE/splice/reconstructed/train.csv | 600 | 400 | 0.484 |
| chromatin_access | Plant | Sorghum bicolor | PGB | PGB/chromatin_access/sorghum_bicolor_train.fa | 65 | 1000 | 0.539 |
| chromatin_access | Plant | Zea mays | PGB | PGB/chromatin_access/zea_mays_train.fa | 65 | 1000 | 0.505 |
| chromatin_access | Plant | Arabidopsis thaliana | PGB | PGB/chromatin_access/arabidopis_thaliana_train.fa | 64 | 1000 | 0.350 |
| chromatin_access | Plant | Brachypodium distachyon | PGB | PGB/chromatin_access/brachypodium_distachyon_train.fa | 64 | 1000 | 0.522 |
| chromatin_access | Plant | Oryza sativa MH63 | PGB | PGB/chromatin_access/oryza_sativa_MH63_RS2_train.fa | 64 | 1000 | 0.461 |
| chromatin_access | Plant | Oryza sativa ZS97 | PGB | PGB/chromatin_access/oryza_sativa_ZS97_RS2_train.fa | 64 | 1000 | 0.470 |
| chromatin_access | Plant | Setaria italica | PGB | PGB/chromatin_access/setaria_italica_train.fa | 64 | 1000 | 0.535 |
| polyA | Plant | Oryza sativa japonica | PGB | PGB/poly_a/oryza_sativa_japonica_group_train.fa | 59 | 400 | 0.398 |
| polyA | Plant | Trifolium pratense | PGB | PGB/poly_a/trifolium_pratense_train.fa | 59 | 400 | 0.335 |
| polyA | Plant | Arabidopsis thaliana | PGB | PGB/poly_a/arabidopsis_thaliana_train.fa | 58 | 400 | 0.364 |
| polyA | Plant | Chlamydomonas reinhardtii | PGB | PGB/poly_a/chlamydomonas_reinhardtii_train.fa | 58 | 400 | 0.583 |
| polyA | Plant | Medicago truncatula | PGB | PGB/poly_a/medicago_truncatula_train.fa | 58 | 400 | 0.330 |
| polyA | Plant | Oryza sativa indica | PGB | PGB/poly_a/oryza_sativa_indica_group_train.fa | 58 | 400 | 0.409 |
| lncRNA | Plant | Glycine max | PGB | PGB/lncrna/g_max_train.fa | 50 | 589 | 0.373 |
| lncRNA | Plant | Manihot esculenta | PGB | PGB/lncrna/m_esculenta_train.fa | 50 | 493 | 0.384 |
| lncRNA | Plant | Sorghum bicolor | PGB | PGB/lncrna/s_bicolor_train.fa | 50 | 610 | 0.478 |
| lncRNA | Plant | Solanum lycopersicum | PGB | PGB/lncrna/s_lycopersicum_train.fa | 50 | 327 | 0.387 |
| lncRNA | Plant | Triticum aestivum | PGB | PGB/lncrna/t_aestivum_train.fa | 50 | 295 | 0.558 |
| lncRNA | Plant | Zea mays | PGB | PGB/lncrna/z_mays_train.fa | 50 | 498 | 0.541 |
| nascent_RNA | Plant | Manihot esculenta | PGB | PGB/pro_seq/m_esculenta_train.fa | 200 | 1000 | 0.282 |
| splicing_plant_donor | Plant | Arabidopsis thaliana | PGB | PGB/splicing/arabidopsis_thaliana_donor_train.fa | 150 | 398 | 0.391 |
| splicing_plant_acceptor | Plant | Arabidopsis thaliana | PGB | PGB/splicing/arabidopsis_thaliana_acceptor_train.fa | 150 | 398 | 0.389 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3/train.csv | 120 | 500 | 0.378 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K14ac/train.csv | 120 | 500 | 0.372 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K36me3/train.csv | 120 | 500 | 0.386 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K4me1/train.csv | 120 | 500 | 0.374 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K4me2/train.csv | 120 | 500 | 0.373 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K4me3/train.csv | 120 | 500 | 0.378 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K79me3/train.csv | 120 | 500 | 0.381 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H3K9ac/train.csv | 120 | 500 | 0.379 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H4/train.csv | 120 | 500 | 0.388 |
| yeast_genome | Fungi | Saccharomyces cerevisiae | GUE | GUE/EMP/H4ac/train.csv | 120 | 500 | 0.382 |
| fungi_genome | Fungi | multi-species (n=20) | GUE | GUE/fungi/species_20/train.csv | 1500 | 1024 | 0.479 |
| virus_variants | Virus | multi-species (n=9) | GUE | GUE/virus/covid/train.csv | 600 | 999 | 0.378 |
| virus_species | Virus | multi-species (n=25) | GUE | GUE/virus/species_40/train.csv | 1100 | 1024 | 0.284 |
