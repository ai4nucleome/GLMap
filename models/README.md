# Model setup

GLMap scores 123 genomic language models. Most are loaded directly from the
[Hugging Face Hub](https://huggingface.co/); 9 require upstream GitHub repos
that are not standard HF checkpoints.

## HuggingFace models (114 of 123)

Download all HF-hosted models listed in `download_models_list.txt`:

```bash
bash scripts/download_models/download_models_from_list.sh
```

The script calls `huggingface-cli download` for each model. Set
`HF_HUB_CACHE` or `HF_HOME` to control the download location.

## External models (9 of 123)

These models ship as upstream GitHub repos with custom loading code
(torch.load `.pt`, non-HF architectures, or separate packages):

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
| PlasmidGPT | [lingxusb/PlasmidGPT](https://github.com/lingxusb/PlasmidGPT) | `plasmidgpt` |

Clone them all at the exact commits used in the paper:

```bash
bash models/setup_external_models.sh
```

This places each repo under `models/modelsHFNoInfo/<name>/` (gitignored).

## Environment routing

Different model families require different Python environments (e.g. Evo-2
needs a specific CUDA toolkit, GenSLM needs the `genslm` package). See
[docs/env_routing.md](../docs/env_routing.md) for the per-family routing
table.

## Files in this directory

- `download_models_list.txt` — the HF model download manifest (123 active
  entries + 2 commented-out sparse-bigbird).
- `fig4a-svm.csv` — Evo lineage model subset used in Table 1 / Fig 4a.
- `setup_external_models.sh` — clones the 9 upstream repos at pinned SHAs.
