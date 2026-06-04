# 🧬 🗺️ GLMap: Profiling genomic language models as individuals in a population

> 🌐 Language: **中文** · [English](README.md)

<p align="center">
  <img src="assets/Fig1.png" alt="GLMap overview" width="80%"/>
</p>

GLMap 是一个**免训练、与架构无关**的框架,通过基因组语言模型(genomic language models, GLMs)在一组固定 DNA 序列面板上的**似然响应**来表征并比较这些模型。我们将 GLMap 应用于 **123 个公开可用的 GLM**、在 **10,000 条 DNA 探针**面板上打分,把自回归(AR)模型和掩码语言(MLM)模型放进同一个空间,得到的模型间距离对探针的选择稳定,并能反映模型之间已知的关系。

---

## 安装

### 用预计算结果复现论文中的分析

如果你只想**使用我们预计算好的 123 个模型在 10,000 条探针面板上的
PLL / log-likelihood 响应**,以及预构建的探针面板、V/Vd/D 矩阵和审计元数据,
并复现全部图/表——下面这套安装就够了。**无需 GPU、无需模型权重、无需打分**;
它会让 `glmap` 可被 import,只安装轻量、不含 torch 的依赖。

```bash
git clone https://github.com/ai4nucleome/GLMap.git
cd GLMap
pip install -e .
```

我们推荐 **Python 3.11.9**。分析栈在以下固定版本下开发并测试通过:

| 包 | 版本 | | 包 | 版本 |
|---|---|---|---|---|
| numpy | 2.2.6 | | seaborn | 0.13.2 |
| pandas | 2.3.3 | | umap-learn | 0.5.9.post2 |
| pyarrow | 23.0.1 | | scikit-learn | 1.8.0 |
| scipy | 1.17.1 | | PyYAML | 6.0.3 |
| matplotlib | 3.10.8 | | huggingface_hub | 0.36.2 |

> **注意**:`import glmap` 不会触发 `import torch` 或 `import transformers`。
> 重依赖在 `get_loader()` 内部按需加载,因此即使没有装 GPU 相关包,核心
> 安装也足以用于分析和画图。如需自己重新打分或开发加载器,见
> [models/README.md](models/README.md)。

---

## 快速开始:使用预计算的 GLMap 产物

论文中 123 个模型的全部预计算产物都已包含在源码仓库里。无需 GPU、无需下载模型、无需打分。

```python
import glmap

# 加载 10,000 条探针面板。
# - 从仓库 checkout:本地读取。
# - 从 pip 安装(无 checkout):自动从 GLMap HuggingFace
#   Dataset(Tim419/GLMap-panels)下载并缓存。
panel = glmap.load_panel()       # (10000, 11) DataFrame

# 或加载你用 scripts/panel_build/ 自建的面板
# panel = glmap.load_panel(path="my_panel.parquet")

# 加载预计算矩阵(它们位于仓库 / $GLMAP_DATA_DIR)
V_AR  = glmap.load_matrix("V_AR")    # (64, 10000) 原始 AR 响应
Vd_AR = glmap.load_matrix("Vd_AR")   # (64, 10000) 双中心后

# 从原始分数重新运行矩阵流水线
info = glmap.fit_matrix(V_AR, clip_q=0.02)
# info["Vd"], info["D"], info["clip_threshold"], ...

# 将一个新模型投影到已有的 Vd 空间
Vd_new = glmap.project(new_model_scores, info)

# 加载 123 模型的审计元数据
audit = glmap.load_audit()       # 123 个 dict 的列表
specs = glmap.specs_from_audit() # 123 个 ModelSpec 对象的列表
```

> 面板以 HuggingFace Dataset 形式发布于
> [`Tim419/GLMap-panels`](https://huggingface.co/datasets/Tim419/GLMap-panels)
> (CC-BY-NC-SA-4.0)。`load_matrix` 和 `load_audit` 从仓库 checkout 或
> `$GLMAP_DATA_DIR` 读取(不会自动下载)。

---

## 重新运行打分与下游评测

从零复现完整流水线需要 GPU、模型权重和基准数据。环境配置见下文各节。

**快速示例**(单模型、单 GPU):

```bash
python scripts/score/scoring_worker.py --from-audit \
    --hf-ids zhihan1996/DNABERT-2-117M --device cuda:0
```

**完整 123 模型复现**(需多个环境 + 多 GPU,见
[models/env_routing.md](models/env_routing.md)):

```bash
# 1. 跨 123 模型并行打分(worker 使用 --skip-aggregate)
python scripts/score/run_scoring_sweep.py --audit data/audits/models.json

# 2. 构建 V/Vd/D 矩阵(CPU,在所有打分 worker 完成后)
python scripts/score/scoring_worker.py --from-audit --strict-aggregate

# 3. 并行提取下游 embedding(需要基准 CSV)
python scripts/downstream_tasks/run_embed_sweep.py --audit data/audits/models.json

# 4. 训练线性探针并计算 AUC
python scripts/downstream_tasks/run_downstream_classify.py

# 5. 生成论文图
python scripts/figures/fig2c_split_half_consistency.py --seed 123
python scripts/figures/fig3a_model_map_family.py
# ...(全部图脚本见 scripts/figures/)
```

并行 sweep 所需的各家族环境配置见
[models/env_routing.md](models/env_routing.md)。

> 💡 顶层另有一组编号驱动脚本(`scripts/0_*.sh … 7_*.sh`),把上述步骤串成可顺序执行的流水线(模型审计 → 下载权重 → 建面板 → 打分 → MLM 稳定性/消融 → 下游 embedding/AUC → 表型预测 → 模型图)。

---

## 仓库结构

```
GLMap/
├── glmap/                  Python 包(可 import;`pip install -e .`)
│   ├── loaders/            各家族模型加载器(HF, evo, genslm, ...)+ 分发
│   ├── scoring/            AR 对数似然 + MLM stride PLL
│   ├── matrices/           clip + 双中心 + 成对距离
│   └── formats_check/      Embedding-parquet schema 校验
├── scripts/                论文复现的 CLI 入口
│   ├── panel_build/        面板构建 + panel_sources.yaml 规范
│   ├── figures/            每个论文图一个脚本
│   ├── tables/             每个论文表一个脚本
│   ├── audits/             模型审计脚本 + context overrides
│   └── 0_*.sh … 7_*.sh     编号流水线驱动(审计 → … → 模型图)
├── tests/                  pytest 测试套件
├── data/
│   ├── audits/             123 模型审计(models.json)
│   ├── downstream_tasks/   下游任务元数据
│   └── panels/             预构建探针面板 parquet
├── results/
│   ├── scores/             打分输出
│   │   ├── matrices/       AR 与 MLM 分支的 V/V_d/D
│   │   └── AR_MLM_scores/  每模型似然响应(精简)
│   ├── analysis/           下游 + 二级分析输出
│   │   ├── benchmark_perform_prediction/
│   │   │   ├── per_model_AUC_result_6tasks/  每模型每任务 AUC 结果
│   │   │   ├── all_model_AUC_6tasks/         聚合 (123×6) AUC 矩阵
│   │   │   └── phenotype_prediction/         用 GLMap 指纹预测下游 AUC
│   │   ├── model_map/      Fig3 的 t-SNE / MDS 嵌入
│   │   └── MLM_stride-PLL_vs_true-PLL_1000samples/  k=1 vs k=6 PLL 消融(Fig S3)
│   ├── figures/            论文图 PDF
│   └── tables/             论文表 LaTeX 源
└── models/                 模型下载清单、配置脚本、
                            各家族环境路由(env_routing.md)
```

---

## 仓库自带产物 vs 用户自行下载的数据

| 本仓库已包含 | 用户需另行下载 |
|---|---|
| 探针面板(10,000 探针,8 MB) | HF 模型权重(约 119 个,经 `hf download`) |
| AR + MLM 的 V/Vd/D 矩阵(20 MB) | 8 个外部模型仓库(`setup_external_models.sh`) |
| 每模型分数,精简版(48 MB) | GenSLM 预训练权重(手动) |
| 下游 AUC 结果(6 MB) | [DNA Foundation Benchmark](https://huggingface.co/datasets/hfeng3/dna_foundation_benchmark_dataset) 的基准任务 CSV |
| 表型预测输出(2 MB) | |
| t-SNE 模型图嵌入 | |
| 论文图(23 个 PDF)与表(12 个 .tex) | |

---

## 模型配置

**HuggingFace 模型**(123 中的 119 个):

```bash
bash scripts/0_download_models_from_list.sh
```

**外部模型**(8 个带自定义加载器的仓库):

```bash
bash models/setup_external_models.sh
```

megaDNA、GenSLM 等特殊情况的细节见
[models/README.md](models/README.md)。模型权重遵循其各自上游许可证。

---

## 下游基准配置

6 个下游分类任务来自
[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)
(Feng et al., 2025)。原始任务 CSV **不**随本仓库一起分发。

```bash
huggingface-cli download hfeng3/dna_foundation_benchmark_dataset \
    --repo-type dataset --local-dir data/dna_foundation_benchmark
```

期望的目录布局与任务细节见
[data/downstream_tasks/README.md](data/downstream_tasks/README.md)。

---

## GLMap 表示

<p align="center">
  <img src="assets/Fig2.png" alt="GLMap representation" width="90%"/>
</p>

GLMap 表示矩阵 *V_d* 呈现出按模型家族聚成的连贯块状结构,且其 split-half
距离几何在按功能元件互斥划分探针时保持稳定(模型对距离的 Pearson *r* = 0.835)。

<p align="center">
  <img src="assets/Fig3.png" alt="GLMap model map and prediction" width="90%"/>
</p>

*V_d* 表示能预测下游任务表现(随机 *K* 折交叉验证下,平均 AUC 的
Spearman ρ = 0.705)。

---

## 致谢

GLMap 建立在若干优秀开源项目的思想与基础设施之上:

- **[ModelMap](https://github.com/shimo-lab/modelmap)**(Oyama et al.,
  ACL *2025*)——将 clip + 双中心流水线应用于对数似然向量的做法,源自
  ModelMap 对 1,000+ 自然语言 LM 的画像。
- **[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)**
  (Feng et al., Nat. Comm. *2025*)——提供了我们下游评测所用的、经过整理的
  二分类任务套件。

我们也感谢本工作所审计的 **123 个基因组语言模型**的作者与维护者公开发布其权重与代码。

---

## 引用

```bibtex
@article{hou2026glmap,
  title   = {Profiling genomic language models as individuals in a population},
  author  = {Hou, Yusen and Long, Weicai and Su, Houcheng and Feng, Junning and Zhang, Yanlin},
  journal = {In submission},
  year    = {2026}
}
```

---

## 许可证

本仓库采用**双许可证**:

- **源代码**(`glmap/`、`scripts/`、`tests/`、`scripts/panel_build/` 等下的所有内容):
  [Apache-2.0](LICENSE)。
- **数据产物**(`data/panels/`、`results/scores/matrices/`、
  `results/scores/AR_MLM_scores/`、`results/analysis/`):[CC-BY-NC-SA-4.0](LICENSE-DATA)。
  这些产物继承上游 Plant Genomic Benchmark 的许可证(面板中 1,600 条探针取自
  PGB;经 ShareAlike 条款为 CC-BY-NC-SA-4.0)。可在署名前提下用于非商业研究;
  商业用途需从与许可证兼容的来源获取面板。

各模型权重亦遵循其各自的上游许可证(见
[models/README.md](models/README.md))。
