# Model setup

GLMap scores 123 genomic language models. Most are loaded directly from the
[Hugging Face Hub](https://huggingface.co/); 8 require upstream GitHub repos
that are not standard HF checkpoints.

## HuggingFace models (119 of 123)

Download all HF-hosted models listed in `download_models_list.txt`:

```bash
bash scripts/download_models/download_models_from_list.sh
```

The script calls `hf download` for each model. The 3 GenSLM entries are
local weight names (not HF repos) and are skipped — see the GenSLM note
below. Set `HF_HOME` to control the download location.

## External models (8 of 123)

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

Clone them all at the exact commits used in the paper:

```bash
bash models/setup_external_models.sh
```

This places each repo under `models/modelsHFNoInfo/<name>/` (gitignored).

### Weight-source notes

- **megaDNA**: The GitHub clone provides the code package needed to
  `torch.load` the weight. The weight itself (`megaDNA_phage_145M.pt`,
  582 MB, CC-BY-NC-4.0) is fetched automatically by the download script
  from [`lingxusb/megaDNA_updated`](https://huggingface.co/lingxusb/megaDNA_updated)
  (the original `lingxusb/megaDNA` repo is no longer public). The audit
  keeps `lingxusb/megaDNA` as the canonical id for consistency with the
  scoring outputs.
- **GenSLM**: Requires a separate manual step. After cloning the
  `genslm` repo above, download the 3 pretrained checkpoints and place
  them under `models/modelsHFNoInfo/genslm/weights/`:

```
models/modelsHFNoInfo/genslm/weights/
├── patric_25m_epoch01-val_loss_0.57_bias_removed.pt
├── patric_250m_epoch00_val_loss_0.48_attention_removed.pt
└── patric_2.5b_epoch00_val_los_0.29_bias_removed.pt
```

See the [GenSLM README](https://github.com/ramanathanlab/genslm) for
download links.

## Environment routing

Different model families require different Python environments (e.g. Evo-2
needs a specific CUDA toolkit, GenSLM needs the `genslm` package). See
[docs/env_routing.md](../docs/env_routing.md) for the per-family routing
table.

## Files in this directory

- `download_models_list.txt` — the full 123-model scoring catalog (119 HF
  repos + 3 GenSLM local names + 1 megaDNA special-cased). Two additional
  bigbird-sparse models are commented out (excluded due to minimum seq_len
  incompatibility).
- `evo-family-relationship.csv` — Evo lineage model subset used in Table 1 / Fig 4a.
- `setup_external_models.sh` — clones the 8 upstream repos at pinned SHAs.

## Upstream licenses

The GLMap **code** is Apache-2.0. Individual model **weights** follow
their own upstream licenses (e.g. megaDNA weights are CC-BY-NC-4.0;
Evo-2 weights are Apache-2.0; etc.). Consult each model's HuggingFace
or GitHub page for licensing terms before redistribution or commercial use.
