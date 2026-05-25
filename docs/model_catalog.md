# 附录 A：基因组基座模型目录

最后更新：2026-05-06。范围：主要输入是 genomic DNA sequence，且能够直接编码、打分或生成 DNA 序列的模型。统一生物序列模型只有在明确支持 DNA sequence modeling 时才纳入。

## 纳入原则

- 不同参数量的变体分开列为不同条目；不同 tokenizer 变体也分开列出。
- 模型名、checkpoint、架构术语保留英文原文，便于检索与复现实验。
- "未报告" / "verify" 表示来源未明确给出该值，大规模实验前需要从 config 或 checkpoint 再确认。
- **第一阶段实验范围以 `phase1_model_selection.md` 为准**（NT、DNABERT、HyenaDNA、GenSLM 四家族）。本目录中标注 "推荐第一轮扩展集合" 的更宽列表是第一阶段验证 pipeline 后的扩展面板，不是第一阶段范围本身。
- 推荐第一轮扩展集合：DNABERT/DNABERT-2、Nucleotide Transformer/Codon-NT/AgroNT/NTv3、GENA-LM/ModernGENA、HyenaDNA、Caduceus/PlantCaduceus、GROVER、GenSLM、Evo/Evo 2、MegaDNA、GENERATOR、PlasmidGPT、gLM2。
- alignment/tree-based 模型与普通 single-sequence GLM 分开分析；GPN-family 放在 Follow-Up Candidates，待该分支打分协议设计完成后再纳入。
- 当前排除：RNA-only 或主要面向 RNA sequence-structure 的模型（如 OmniGenome）；protein-only LM；barcode-only 模型；supervised sequence-to-function / annotation 模型（Enformer、Borzoi、AlphaGenome、SegmentNT）。

## DNA 序列语言模型

| 家族 | 变体 / checkpoint | 架构类别 | Tokenizer / 分辨率 | 参数量 | 上下文 / 输入 | 训练范围 / 备注 | 来源 |
|---|---|---|---|---:|---|---|---|
| DNABERT | DNABERT-3 | BERT encoder | overlapping 3-mer | ~86M | 512 tokens | human GRCh38/hg38 | DNABERT paper; NVIDIA BioNeMo |
| DNABERT | DNABERT-4 | BERT encoder | overlapping 4-mer | ~86M | 512 tokens | human GRCh38/hg38 | DNABERT paper; NVIDIA BioNeMo |
| DNABERT | DNABERT-5 | BERT encoder | overlapping 5-mer | ~86M | 512 tokens | human GRCh38/hg38 | DNABERT paper; NVIDIA BioNeMo |
| DNABERT | DNABERT-6 | BERT encoder | overlapping 6-mer | ~86M | 512 tokens | human GRCh38/hg38; commonly used DNABERT baseline | DNABERT paper; NVIDIA BioNeMo |
| DNABERT-2 | DNABERT-2-117M | MosaicBERT-style encoder | BPE, vocab 4096 | 117M | 512 tokens default; often up to ~2 kb sequence after BPE | 135 species from GenBank | HF model card |
| Nucleotide Transformer | 500m-human-ref | Transformer encoder MLM | 6-mer-like nucleotide tokens | 500M | ~6 kb | human reference genome | Nature Methods; HF collection |
| Nucleotide Transformer | 500m-1000g | Transformer encoder MLM | 6-mer-like nucleotide tokens | 500M | ~6 kb | 3,202 human genomes | Nature Methods; HF collection |
| Nucleotide Transformer | 2.5b-1000g | Transformer encoder MLM | 6-mer-like nucleotide tokens | 2.5B | ~6 kb | 3,202 human genomes | Nature Methods; HF model card |
| Nucleotide Transformer | 2.5b-multi-species | Transformer encoder MLM | 6-mer-like nucleotide tokens | 2.5B | ~6 kb | 850 species | Nature Methods; HF model card |
| Nucleotide Transformer v2 | v2-50m-multi-species | efficient Transformer encoder MLM | standard NT-v2 tokenizer | 55.9M | 1,000 tokens / ~6 kb；HF max length 2,048 | 850 species | Nature Methods; HF collection |
| Codon-NT / Nucleotide Transformer v2 | nucleotide-transformer-v2-50m-3mer-multi-species | efficient Transformer encoder MLM | 3-mer / codon tokenizer | 50M | 1,000 tokens on HF card; verify bp-equivalent | 850 genomes; tokenizer-control variant for codon/protein-task analysis | Bioinformatics 2024; HF card; InstaDeep docs |
| Nucleotide Transformer v2 | v2-100m-multi-species | efficient Transformer encoder MLM | standard NT-v2 tokenizer | 97.9M | as above | 850 species | HF collection |
| Nucleotide Transformer v2 | v2-250m-multi-species | efficient Transformer encoder MLM | standard NT-v2 tokenizer | 250M | as above | best reported NT-v2 benchmark point in paper | Nature Methods; HF model card |
| Nucleotide Transformer v2 | v2-500m-multi-species | efficient Transformer encoder MLM | standard NT-v2 tokenizer | 500M | as above | 850 species | HF collection |
| AgroNT | agro-nucleotide-transformer-1b | Transformer encoder MLM | non-overlapping 6-mer tokenizer; vocab 4104 | 1B | 1,024 tokens, ~6,144 bp | edible/crop-focused plant genomes from 48 plant species; 472.5B training tokens | Communications Biology; HF card; InstaDeep docs |
| Nucleotide Transformer v3 | NTv3_8M_pre | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 7.69M | up to 1 Mb family claim; input multiple of 128 | pre-trained on OpenGenome2-scale DNA; exploration-scale checkpoint | InstaDeep GitHub; HF collection/card |
| Nucleotide Transformer v3 | NTv3_100M_pre | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 0.1B | up to 1 Mb family claim; input multiple of 128 | pre-trained representation model | InstaDeep GitHub; HF card |
| Nucleotide Transformer v3 | NTv3_650M_pre | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 0.7B | up to 1 Mb family claim; input multiple of 128 | pre-trained representation model | InstaDeep GitHub; HF collection |
| Nucleotide Transformer v3 | NTv3_8M_pre_8kb | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 7.69M | 8 kb context | 8-kb exploration checkpoint, not recommended for main results | HF collection/card |
| Nucleotide Transformer v3 | NTv3_100M_pre_8kb | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 0.1B | 8 kb context | 8-kb exploration checkpoint | HF collection |
| Nucleotide Transformer v3 | NTv3_650M_pre_8kb | U-Net-style conv -> Transformer -> deconv MLM | single-base character tokenizer; vocab 11 | 0.7B | 8 kb context | 8-kb exploration checkpoint | HF collection |
| Nucleotide Transformer v3 | NTv3_100M_post | conditioned U-Net-style model with MLM + supervised heads | single-base character tokenizer; vocab 11; species conditioning | 0.1B | up to 1 Mb family claim; input multiple of 128 | post-trained on ~16k functional tracks/annotations across 24 species; use representation/MLM outputs for GLM comparison | InstaDeep GitHub; HF collection |
| Nucleotide Transformer v3 | NTv3_650M_post | conditioned U-Net-style model with MLM + supervised heads | single-base character tokenizer; vocab 11; species conditioning | 0.7B | up to 1 Mb family claim; input multiple of 128 | post-trained on functional tracks; outputs MLM logits, embeddings, BigWig/BED heads | InstaDeep GitHub; HF card |
| Nucleotide Transformer v3 | NTv3_100M_post_131kb | conditioned U-Net-style post-trained model | single-base character tokenizer; vocab 11; species conditioning | 0.1B | 131 kb context | context-specific post-trained checkpoint | HF collection |
| Nucleotide Transformer v3 | NTv3_650M_post_131kb | conditioned U-Net-style post-trained model | single-base character tokenizer; vocab 11; species conditioning | 0.7B | 131 kb context | context-specific post-trained checkpoint | HF collection |
| Nucleotide Transformer v3 | NTv3_5downsample_pre | U-Net-style ablation MLM | single-base character tokenizer; vocab 11 | 0.6B | verify from config | 5-downsample architecture ablation; not recommended for main results | HF card |
| Nucleotide Transformer v3 | NTv3_5downsample_post | conditioned U-Net-style ablation with supervised heads | single-base character tokenizer; vocab 11; species conditioning | 0.6B | verify from config | post-trained 5-downsample ablation | HF collection |
| Nucleotide Transformer v3 | NTv3_5downsample_pre_8kb | U-Net-style ablation MLM | single-base character tokenizer; vocab 11 | 0.6B | 8 kb context | 5-downsample 8-kb ablation | HF collection |
| Nucleotide Transformer v3 | NTv3_5downsample_post_131kb | conditioned U-Net-style ablation with supervised heads | single-base character tokenizer; vocab 11; species conditioning | 0.6B | 131 kb context | 5-downsample post-trained 131-kb ablation | HF collection |
| GENA-LM | gena-lm-bert-base | BERT encoder MLM | 32k BPE, T2T split v1 tokenizer | 110M | 512 tokens, ~4.5 kb | human T2T v2 | NAR paper; HF card |
| GENA-LM | gena-lm-bert-base-t2t | BERT encoder MLM | 32k BPE, T2T+1KG+multispecies tokenizer | 110M | ~4.5 kb | T2T + 1000G SNP augmentation | NAR paper; HF collection |
| GENA-LM | gena-lm-bert-base-lastln-t2t | BERT encoder MLM | same 32k BPE | 110M | ~4.5 kb | last-layer norm variant | NAR paper |
| GENA-LM | gena-lm-bert-base-t2t-multi | BERT encoder MLM | same 32k BPE | 110M | ~4.5 kb | human + multispecies | NAR paper; HF collection |
| GENA-LM | gena-lm-bert-base-yeast | BERT encoder MLM | same 32k BPE | 110M | ~4.5 kb | yeast-specific | NAR paper; HF collection |
| GENA-LM | gena-lm-bert-base-fly | BERT encoder MLM | same 32k BPE | 110M | ~4.5 kb | Drosophila-specific | NAR paper; HF collection |
| GENA-LM | gena-lm-bert-base-athaliana | BERT encoder MLM | same 32k BPE | 110M | ~4.5 kb | Arabidopsis-specific | NAR paper; HF collection |
| GENA-LM | gena-lm-bert-large-t2t | BERT encoder MLM | same 32k BPE | 336M | ~4.5 kb | T2T + 1000G | NAR paper; HF collection |
| GENA-LM | gena-lm-bigbird-base-sparse | BigBird / sparse-attention encoder | 32k BPE, T2T split v1 | 110M | 4096 tokens, ~36 kb | human T2T v2 | NAR paper |
| GENA-LM | gena-lm-bigbird-base-sparse-t2t | BigBird / DeepSpeed sparse encoder | 32k BPE, T2T+1KG+multispecies tokenizer | 110M | ~36 kb | T2T + 1000G | NAR paper; HF collection |
| GENA-LM | gena-lm-bigbird-base-t2t | BigBird / HF sparse encoder | 32k BPE, T2T+1KG+multispecies tokenizer | 110M | ~36 kb | T2T + 1000G | NAR paper; HF config |
| ModernGENA | moderngena-base | ModernBERT encoder MLM | GENA-LM 32k BPE | 377M reported | long-context encoder | 443 vertebrate assemblies; TSS-upsampled windows | HF model card; OpenReview |
| ModernGENA | moderngena-large | ModernBERT encoder MLM | GENA-LM 32k BPE | 377M reported on current card; verify | long-context encoder | same as base | HF model card; OpenReview |
| GROVER | GROVER / BPE-600 | BERT encoder MLM | 通过 next-k-mer prediction 选择的 BPE；601-token vocab | 未报告；BERT-12L | up to 510 tokens | human hg19；透明 tokenizer/vocabulary 研究 | Nature Machine Intelligence; HF card |
| HyenaDNA | tiny-1k-seqlen | Hyena implicit-convolution decoder | single nucleotide | ~451k | 1,024 tokens | human reference genome | HyenaDNA paper; HF models |
| HyenaDNA | tiny-1k-seqlen-d256 | Hyena decoder | single nucleotide | ~1.66M | 1,024 tokens | human reference genome | HF models |
| HyenaDNA | tiny-16k-seqlen-d128 | Hyena decoder | single nucleotide | ~635k | 16,384 tokens | human reference genome | HF models |
| HyenaDNA | small-32k-seqlen | Hyena decoder | single nucleotide | ~4.07M | 32,768 tokens | human reference genome | HyenaDNA paper; HF card |
| HyenaDNA | medium-160k-seqlen | Hyena decoder | single nucleotide | ~14.2M | 160k tokens | human reference genome | HF models |
| HyenaDNA | medium-450k-seqlen | Hyena decoder | single nucleotide | ~28.2M | 450k tokens | human reference genome | HF models |
| HyenaDNA | large-1m-seqlen | Hyena decoder | single nucleotide | ~54.6M | 1M tokens | human reference genome | HyenaDNA paper; HF models |
| Caduceus | caduceus-ph_seqlen-131k_d_model-256_n_layer-16 | bidirectional Mamba / BiMamba | nucleotide tokens；RC augmentation | 未报告 | 131k tokens | 使用 reverse-complement augmentation，但不严格 equivariant | ICML paper; GitHub |
| Caduceus | caduceus-ps_seqlen-131k_d_model-256_n_layer-16 | RC-equivariant MambaDNA | nucleotide tokens；RC-equivariant | 未报告 | 131k tokens | reverse-complement parameter sharing / equivariance | ICML paper; GitHub |
| PlantCaduceus | PlantCaduceus_l20 | Caduceus/Mamba encoder MLM | nucleotide tokens | 20M | 未报告 | 16 angiosperm genomes | HF model card |
| PlantCaduceus | PlantCaduceus_l24 | Caduceus/Mamba encoder MLM | nucleotide tokens | 40M | 未报告 | 16 angiosperm genomes | HF model card |
| PlantCaduceus | PlantCaduceus_l28 | Caduceus/Mamba encoder MLM | nucleotide tokens | 112M | 未报告 | 16 angiosperm genomes | HF model card |
| PlantCaduceus | PlantCaduceus_l32 | Caduceus/Mamba encoder MLM | nucleotide tokens | 225M | 未报告 | 16 angiosperm genomes | HF model card |
| GenSLM | 25M Foundation | autoregressive Transformer | codon-level, 64 codons | 25M | 2,048 codon tokens | >110M prokaryotic gene sequences | GenSLM paper |
| GenSLM | 250M Foundation | autoregressive Transformer | codon-level | 250M | 2,048 codon tokens | prokaryotic foundation model | GenSLM paper |
| GenSLM | 2.5B Foundation | autoregressive Transformer | codon-level | 2.5B | 2,048 codon tokens | prokaryotic foundation model | GenSLM paper |
| GenSLM | 25B Foundation | autoregressive Transformer | codon-level | 25B | 2,048 codon tokens | prokaryotic foundation model | GenSLM paper |
| GenSLM | 25M SARS-CoV-2 | autoregressive Transformer | codon-level | 25M | 10,240 codon tokens | SARS-CoV-2 fine-tuned/evolution model | GenSLM paper |
| GenSLM | 250M SARS-CoV-2 | autoregressive Transformer | codon-level | 250M | 10,240 codon tokens | SARS-CoV-2 fine-tuned/evolution model | GenSLM paper |
| GenSLM | 123M CS-2 run | autoregressive Transformer | codon-level | 123M | 10,240 codon tokens | Cerebras scaling experiment | GenSLM paper |
| GenSLM | 1.3B CS-2 run | autoregressive Transformer | codon-level | 1.3B | 10,240 codon tokens | Cerebras scaling experiment | GenSLM paper |
| Evo | evo-1-8k-base | StripedHyena hybrid autoregressive | single-nucleotide / byte-level | 7B | 8,192 tokens | prokaryotic + phage OpenGenome | Arc/Together HF card; Science |
| Evo | evo-1-131k-base | StripedHyena hybrid autoregressive | single-nucleotide / byte-level | 7B | 131,072 tokens | long-context genome-scale model | Arc/Together HF card; Science |
| Evo | evo-1-8k-crispr | StripedHyena hybrid autoregressive | single-nucleotide / byte-level | 7B | 8,192 tokens | CRISPR-Cas fine-tune | Arc/Together HF card |
| Evo 2 | evo2_1b_base | StripedHyena 2 autoregressive | single nucleotide | 1B | 8,192 tokens | OpenGenome2; base checkpoint | Arc HF / GitHub |
| Evo 2 | evo2_7b_base | StripedHyena 2 autoregressive | single nucleotide | 7B | 8,192 tokens | OpenGenome2; base checkpoint | Arc HF / GitHub |
| Evo 2 | evo2_7b_262k | StripedHyena 2 autoregressive | single nucleotide | 7B | 262,144 tokens | intermediate long-context checkpoint | Arc HF / GitHub |
| Evo 2 | evo2_7b | StripedHyena 2 autoregressive | single nucleotide | 7B | 1M tokens | all domains of life | Arc HF / GitHub |
| Evo 2 | evo2_20b | StripedHyena 2 autoregressive | single nucleotide | 20B | 1M tokens | all domains of life; 40B-level performance claim | Arc HF card |
| Evo 2 | evo2_40b_base | StripedHyena 2 autoregressive | single nucleotide | 40B | 8,192 tokens | base checkpoint | Arc HF / NVIDIA |
| Evo 2 | evo2_40b | StripedHyena 2 autoregressive | single nucleotide | 40B | 1M tokens | all domains of life; trained on ~8.8-9T tokens | Arc HF / GitHub / NVIDIA |
| MegaDNA | megaDNA_78M | MEGABYTE / multiscale Transformer decoder CLM | single-nucleotide / byte-level vocabulary | 78M | up to 96 kb | bacteriophage genomes; smaller size variant | Nature Communications; GitHub |
| MegaDNA | megaDNA_145M / megaDNA_phage_145M | MEGABYTE / multiscale Transformer decoder CLM | single-nucleotide / byte-level vocabulary | 145M | up to 96 kb | original phage model; public checkpoint | Nature Communications; GitHub; HF |
| MegaDNA | megaDNA_277M | MEGABYTE / multiscale Transformer decoder CLM | single-nucleotide / byte-level vocabulary | 277M | up to 96 kb | larger phage model; gated HF checkpoint noted | GitHub; HF; DNA dialect review |
| MegaDNA | megaDNA_ecoli | MEGABYTE / multiscale Transformer decoder CLM | single-nucleotide / byte-level vocabulary | likely 145M; verify | up to 96 kb | E. coli phage fine-tuned derivative | GitHub |
| GENERator | GENERator-eukaryote-1.2b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 1.2B | 98 kb native; repo notes extension experiments to 1 Mb | eukaryotic DNA, 386B bp | arXiv; GitHub; HF |
| GENERator | GENERator-eukaryote-3b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 3B | 98 kb native; verify exact model max length | eukaryotic DNA, 386B bp | GitHub; HF |
| GENERator | GENERator-v2-eukaryote-1.2b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 1.2B | long-context, verify from config | eukaryotic DNA, 422B bp | GitHub; bioRxiv v2 |
| GENERator | GENERator-v2-eukaryote-3b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 3B | long-context, verify from config | eukaryotic DNA, 422B bp | GitHub; bioRxiv v2 |
| GENERator | GENERator-v2-prokaryote-1.2b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 1.2B | long-context, verify from config | prokaryotic DNA, 515B bp | GitHub; bioRxiv v2 |
| GENERator | GENERator-v2-prokaryote-3b-base | Llama-style Transformer decoder CLM | non-overlapping 6-mer tokenizer | 3B | long-context, verify from config | prokaryotic DNA, 515B bp | GitHub; bioRxiv v2 |
| PlasmidGPT | PlasmidGPT / Addgene GPT-2 | GPT-2 decoder CLM | BPE tokenizer, vocab 30,002 | 110M | 2,048 tokens | 153k engineered Addgene plasmids; domain-specific DNA generator | bioRxiv; GitHub; HF compatibility card |
| gLM2 | tattabio/gLM2_150M | Transformer encoder MLM | mixed scaffold tokenizer: amino-acid tokens for CDS, nucleotide tokens for intergenic DNA | 150M | genomic contig/scaffold; verify max length | OMG dataset; mixed DNA/protein genomic scaffold model | HF model card; GitHub |
| gLM2 | tattabio/gLM2_650M | Transformer encoder MLM | same mixed CDS/IGS tokenizer | 650M | genomic contig/scaffold; verify max length | larger gLM2 checkpoint | HF model card; GitHub |
| LucaOne | LucaOne mixed nucleic-acid/protein model | Transformer encoder MLM + semi-supervised tasks | unified 39-token nucleotide/amino-acid vocabulary; token-type embeddings | 1.8B | 1,280 tokens | trained on DNA/RNA/protein sequences from 169,861 species; DNA-capable mixed-modal model | Nature Machine Intelligence |
| CD-GPT | CD-GPT-1b | GPT-like decoder-only CLM + central-dogma pretraining | shared BPE multi-molecule vocabulary, 64k tokens | 1B | README/论文片段未报告；需从 config/checkpoint 确认 | DNA/RNA/protein generative model；Stage 1 mono-sequence + Stage 2 central-dogma pretraining | bioRxiv; GitHub |
| CD-GPT | CD-GPT-1b-s | GPT-like decoder-only CLM + central-dogma + protein-structure pretraining | shared BPE multi-molecule vocabulary, 64k tokens | 1B | 未报告；需确认 | 增加 Stage 3 protein-structure pretraining；仅在测试 mixed-modal transfer effects 时纳入 | GitHub |
| CD-GPT | CD-GPT-1b-reverse-translation | GPT-like decoder-only fine-tuned model | shared BPE multi-molecule vocabulary, 64k tokens | 1B | 未报告；需确认 | 面向 protein-to-codon reverse translation 的 fine-tuned 下游衍生模型，不是 base checkpoint | GitHub |

## 边缘 / 后续候选模型

这些模型与课题相关，但进入主实验面板前需要进一步核实来源、输入协议或分析分支。

| 家族 | 已知公开变体 | 需要核实的原因 |
|---|---|---|
| D3LM-from-NT | `Hengchang-Liu/D3LM-from-nt` (~55.9M) | 可能来自 NT；需要判断应作为 base model 还是 downstream derivative。 |
| ChatNT | NT + Vicuna-style instruction model | DNA-capable text-to-text model，但表示方法需要 prompt protocol，而不是原始 likelihood。 |
| GPN | `songlab/gpn-brassicales` (65.9M) | Arabidopsis + Brassicales 的 single-sequence ConvNet masked DNA LM；偏 variant-effect，不是普通 broad GLM。 |
| GPN-MSA | `songlab/gpn-msa-sapiens` (85.7M) | 基于 MSA columns 的 alignment-based DNA LM；需要和 single-sequence GLM 分开设计表示协议。 |
| PhyloGPN | `songlab/PhyloGPN` (83.2M) | 训练时使用 Zoonomia alignment + F81 substitution likelihood，**推理只需 single sequence**（sliding ConvNet, 481 bp 感受野）。归 Follow-Up 的真正原因是输出为 F81 substitution rate matrix 参数，不是 token probability，需要单独打分协议。 |
| GPN-Star | human/vertebrate、human/mammal、human/primate 200M；mouse/chicken/fly 86.1M；C. elegans/Arabidopsis 25.9M | 更强的 phylogeny-aware GPN 后续模型；在决定是否分离 single-sequence 与 alignment-required 模型后再纳入。 |

## 来源链接

- DNABERT paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11025658/
- NVIDIA DNABERT BioNeMo card: https://docs.nvidia.com/bionemo-framework/1.10/models/dnabert.html
- DNABERT-2 HF card: https://huggingface.co/multimolecule/dnabert2
- Nucleotide Transformer paper: https://www.nature.com/articles/s41592-024-02523-z
- Nucleotide Transformer HF collection: https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer
- Nucleotide Transformer GitHub docs: https://github.com/instadeepai/nucleotide-transformer/blob/main/docs/nucleotide_transformer.md
- Codon-NT paper: https://academic.oup.com/bioinformatics/article/40/9/btae529/7745814
- Codon-NT HF card: https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species
- AgroNT paper: https://www.nature.com/articles/s42003-024-06465-2
- AgroNT HF card: https://huggingface.co/InstaDeepAI/agro-nucleotide-transformer-1b
- NTv3 InstaDeep paper page: https://instadeep.com/research/paper/a-foundational-model-for-joint-sequence-function-multi-species-modeling-at-scale-for-long-range-genomic-prediction/
- NTv3 GitHub overview: https://github.com/instadeepai/nucleotide-transformer
- NTv3 HF collection: https://huggingface.co/collections/InstaDeepAI/nucleotide-transformer-v3
- NTv3 8M pre HF card: https://huggingface.co/InstaDeepAI/NTv3_8M_pre
- NTv3 100M pre HF card: https://huggingface.co/InstaDeepAI/NTv3_100M_pre
- NTv3 650M post HF card: https://huggingface.co/InstaDeepAI/NTv3_650M_post
- NTv3 5-downsample HF card: https://huggingface.co/InstaDeepAI/NTv3_5downsample_pre
- GENA-LM NAR paper: https://academic.oup.com/nar/article/53/2/gkae1310/7954523
- GENA-LM GitHub / ModernGENA README: https://github.com/AIRI-Institute/GENA_LM
- ModernGENA base: https://huggingface.co/AIRI-Institute/moderngena-base
- ModernGENA large: https://huggingface.co/AIRI-Institute/moderngena-large
- GROVER paper: https://www.nature.com/articles/s42256-024-00872-0
- GROVER HF card: https://huggingface.co/PoetschLab/GROVER
- HyenaDNA paper: https://pubmed.ncbi.nlm.nih.gov/37426456/
- HyenaDNA HF card: https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen
- Caduceus ICML paper: https://proceedings.mlr.press/v235/schiff24a.html
- Caduceus GitHub: https://github.com/kuleshov-group/caduceus
- PlantCaduceus HF cards: https://huggingface.co/kuleshov-group/PlantCaduceus_l20
- GenSLM paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9709791/
- Evo-1 HF card: https://huggingface.co/togethercomputer/evo-1-131k-base
- Evo / Science DOI listed on HF card: https://www.science.org/doi/abs/10.1126/science.ado9336
- Evo 2 GitHub: https://github.com/arcinstitute/evo2
- Evo 2 40B HF card: https://huggingface.co/arcinstitute/evo2_40b
- Evo 2 20B HF card: https://huggingface.co/arcinstitute/evo2_20b
- NVIDIA Evo 2 model card: https://build.nvidia.com/arc/evo2-40b/modelcard
- MegaDNA paper: https://www.nature.com/articles/s41467-024-53759-4
- MegaDNA GitHub: https://github.com/lingxusb/megaDNA
- MegaDNA 145M HF checkpoint: https://huggingface.co/lingxusb/megaDNA_updated
- MegaDNA variants HF checkpoint: https://huggingface.co/lingxusb/megaDNA_variants
- GENERator GitHub: https://github.com/GenerTeam/GENERator
- GENERator paper page: https://huggingface.co/papers/2502.07272
- PlasmidGPT GitHub: https://github.com/lingxusb/PlasmidGPT
- PlasmidGPT HF compatibility card: https://huggingface.co/UCL-CSSB/PlasmidGPT
- gLM2 150M HF card: https://huggingface.co/tattabio/gLM2_150M
- gLM2 650M HF card: https://huggingface.co/tattabio/gLM2_650M
- GPN paper: https://pubmed.ncbi.nlm.nih.gov/37883436/
- GPN HF card: https://huggingface.co/songlab/gpn-brassicales
- GPN-MSA paper: https://www.nature.com/articles/s41587-024-02511-w
- GPN-MSA HF collection: https://huggingface.co/collections/songlab/gpn-msa
- PhyloGPN paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11908359/
- PhyloGPN HF card: https://huggingface.co/songlab/PhyloGPN
- GPN-Star HF collection: https://huggingface.co/collections/songlab/gpn-star
- OmniGenome HF card, excluded as primarily RNA: https://huggingface.co/yangheng/OmniGenome-186M
- LucaOne paper: https://www.nature.com/articles/s42256-025-01044-4
- CD-GPT bioRxiv / DOI: https://doi.org/10.1101/2024.06.24.600337
- CD-GPT GitHub: https://github.com/TencentAI4S/CD-GPT
- Genomic Language Models review table: https://pmc.ncbi.nlm.nih.gov/articles/PMC11275703/
- DNA dialect review: https://link.springer.com/article/10.1038/s44320-025-00184-4
