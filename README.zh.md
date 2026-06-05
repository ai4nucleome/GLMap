# 🧬 🗺️ GLMap: Profiling genomic language models as individuals in a population

> 🌐 Language: **中文** · [English](README.md)
> 
> 📖 **项目网站: [ai4nucleome.github.io/GLMap](https://ai4nucleome.github.io/GLMap/)**

<p align="center">
  <img src="assets/Fig1.png" alt="GLMap overview" width="80%"/>
</p>

GLMap 是一个**免训练、与架构无关**的框架,通过基因组语言模型(genomic language models, GLMs)在一组固定 DNA 序列面板上的**似然响应**来表征并比较这些模型。我们将 GLMap 应用于 **123 个公开可用的 GLM**、在 **10,000 条 DNA 探针**面板上打分,把自回归(AR)模型和掩码语言(MLM)模型放进同一个空间,得到的模型间距离对探针的选择稳定,并能反映模型之间已知的关系。

---

## 安装

### 用预计算结果复现论文中的分析

如果只想**复现论文中的全部图/表**而不是从头计算 123 个模型的 GLMap 表示，请使用下方的安装命令。**无需 GPU、无需模型权重、无需打分**;
我们推荐 **Python 3.11.9**;分析环境所需的 pip包 的精确版本已固定在[`pyproject.toml`](pyproject.toml) 中。

```bash
git clone https://github.com/ai4nucleome/GLMap.git
cd GLMap
pip install -e .
```

> **注意**: 这套命令不会安装 `torch`  或 `transformers`，即没有装 GPU 相关包,这些安装足以用于分析和画图。

### 重新计算 123 个模型的分数

这 123 个模型分属**多套互不兼容的运行环境**(不同模型家族的 Python /
PyTorch / CUDA 版本各不相同)。重新计算似然响应有**两种方式**:

- **自己配置环境** —— 按家族建好各 micromamba 环境
  (见 [`models/env_routing.md`](models/env_routing.md)),然后跑
  `python scripts/score/run_scoring_sweep.py`。
- **用我们预构建的容器镜像** —— 4 个 Apptainer/Singularity 镜像覆盖全部
  123 个模型的环境,以 HuggingFace dataset 形式发布于
  [`Tim419/GLMap-containers`](https://huggingface.co/datasets/Tim419/GLMap-containers)。
  同一条 sweep 加 `--backend container` 即可,**无需配置环境**。

镜像下载 + 模型→镜像对照见 [`container/README.md`](container/README.md);
模型权重和外部加载代码见 [`models/README.md`](models/README.md)。

---

## 快速开始:使用预计算的 GLMap 产物

论文中 123 个模型的全部预计算产物都已包含在源码仓库里。无需 GPU、无需下载模型、无需打分。

```python
import glmap

# 两种加载 10,000 条探针面板的方式(磁盘上:data/panels/main_panel.parquet)。
# - 本地读取。
# - 或自动从 HuggingFace (Tim419/GLMap-panels) 下载。
panel = glmap.load_panel()       # (10000, 11) DataFrame

# 或你可以自己构建你想要的数据面板
# panel = glmap.load_panel(path="my_panel.parquet")

# 按名字("V_AR")或路径加载预计算矩阵。
V_AR  = glmap.load_matrix("results/scores/matrices/V_AR.npy")    # (64, 10000)  原始 AR 响应   (MLM: 59 个模型)
Vd_AR = glmap.load_matrix("results/scores/matrices/V_d_AR.npy")   # (64, 10000)  双中心后
D_AR  = glmap.load_matrix("results/scores/matrices/D_AR.npy")    # (64, 64)     模型两两距离

# 从原始分数重新运行矩阵流水线
info = glmap.fit_matrix(V_AR, clip_q=0.02)

# 将一个新模型投影到已有的 Vd 空间
Vd_new = glmap.project(new_model_scores, info)

# 加载 123 模型的审计元数据
audit = glmap.load_audit()       # 123 个 dict 的列表
specs = glmap.specs_from_audit() # 123 个 ModelSpec 对象的列表
```

> 面板以 HuggingFace Dataset 形式发布于
> [`Tim419/GLMap-panels`](https://huggingface.co/datasets/Tim419/GLMap-panels)
> (CC-BY-NC-SA-4.0)。

---

## 仓库结构

```
GLMap/
├── glmap/                  Python 包
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
│   │   └── MLM_stride-PLL_vs_true-PLL_1000samples/  true PLL vs Stride PLL 消融(k=6, Fig S3)
│   ├── figures/            论文图 PDF
│   └── tables/             论文表 LaTeX 源
└── models/                 模型下载清单、配置脚本、
```

---

## 仓库自带产物

用预计算结果复现论文分析所需的一切都已随仓库提供——无需模型权重、无需打分:

| 产物 | 
|---|
| 探针面板(10,000 探针) |
| AR + MLM 的 V/Vd/D 矩阵  |
| 每模型似然响应,精简版  |
| 下游 AUC 结果 |
| 表型预测输出 |
| t-SNE 模型图嵌入 |
| 论文图(23 个 PDF)与表(12 个 .tex) |

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
  ACL *2025*)
- **[DNA Foundation Benchmark](https://github.com/ChongWuLab/dna_foundation_benchmark)**
  (Feng et al., Nat. Comm. *2025*)

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
