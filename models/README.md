# Model setup

GLMap scores 123 genomic language models. This directory holds the model
catalog and the setup scripts needed to obtain their weights and (for a
few families) their custom loading code.

## 1. Clone custom loading code

8 families cannot be loaded via standard HuggingFace `transformers` — they
need their own model code. Clone all of them:

```bash
bash models/setup_external_models.sh
```

This places each repo under `models/modelsHFNoInfo/<name>/`

| Family | Repo | Loader kind |
|---|---|---|
| Evo 1.x | [evo-design/evo](https://github.com/evo-design/evo) | `evo1` |
| Evo 2 | [ArcInstitute/evo2](https://github.com/ArcInstitute/evo2) | `evo2` |
| GenSLM | [ramanathanlab/genslm](https://github.com/ramanathanlab/genslm) | `genslm` |
| HyenaDNA | [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna) | `hyenadna` |
| megaDNA | [lingxusb/megaDNA](https://github.com/lingxusb/megaDNA) | `megadna` |
| AIDO.DNA | [genbio-ai/ModelGenerator](https://github.com/genbio-ai/ModelGenerator) | `aido` |
| PlantBiMoE | [HUST-Keep-Lin/PlantBiMoE](https://github.com/HUST-Keep-Lin/PlantBiMoE) | `plantbimoe` |
| PlantCAD2 | [kuleshov-group/PlantCaduceus](https://github.com/kuleshov-group/PlantCaduceus) | `plantcad2` |

## 2. Download model weights

Most models are hosted on the [Hugging Face Hub](https://huggingface.co/).
Download all weights listed in `download_models_list.txt`:

```bash
bash scripts/download_models/download_models_from_list.sh
```

Set `HF_HOME` to control the download cache location.

Two special cases are handled by the script:

- **GenSLM** (3 entries): these are local weight names, not HF repos, so
  the script skips them. Download the 3 checkpoints manually (see below).
- **megaDNA**: the audit keeps `lingxusb/megaDNA` as the canonical id but the weight (`megaDNA_phage_145M.pt`) is fetched from [`lingxusb/megaDNA_updated`](https://huggingface.co/lingxusb/megaDNA_updated) instead.

### GenSLM weights (manual)

After cloning the `genslm` repo (section 1), download the 3 pretrained
checkpoints and place them under `models/modelsHFNoInfo/genslm/weights/`:

```
models/modelsHFNoInfo/genslm/weights/
├── patric_25m_epoch01-val_loss_0.57_bias_removed.pt
├── patric_250m_epoch00_val_loss_0.48_attention_removed.pt
└── patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
```

See the [GenSLM README](https://github.com/ramanathanlab/genslm) for
download links.

## Files in this directory

- `download_models_list.txt` — the full 123-model scoring catalog.
- `evo-family-relationship.csv` — ground-truth Evo lineage labels as
  `(anchor, partner, label)` pairs (`1` = partner is a direct descendant /
  fine-tune of the anchor, `0` = unrelated); used by Table 3 / Fig 4a.
- `setup_external_models.sh` — clones the 8 upstream repos at pinned SHAs.

## Upstream licenses

The GLMap **code** is Apache-2.0. Individual model **weights** follow
their own upstream licenses (e.g. megaDNA weights are CC-BY-NC-4.0;
Evo-2 weights are Apache-2.0; etc.). Consult each model's HuggingFace
or GitHub page for licensing terms before redistribution or commercial use.
