# GLMap 公开仓库整理计划

> 目标：把 `/nvme-data3/yusen/worksapce/glm_mapping/genome_model_population_genetics`
> 整理成一份可以作为论文附带代码发布的公开仓库
> （目的地：`/nvme-data3/yusen/worksapce/glm_mapping/GLMap-code-public`）。
>
> 全程**只**在 `GLMap-code-public/` 下工作，不修改原仓库任何文件。
> 原仓库作为 read-only source 提供素材。

---

## 0. 已经拍板的决策

| 项目 | 选择 |
|---|---|
| License | **Apache-2.0** |
| `out_phase1/scores/` 投放方式 | **先瘦身；若任一文件 >95 MB，则外置到 HuggingFace Dataset** |
| `METHODS.md` 语言 | **英文重写** |
| 是否包含 `dna_foundation_benchmark/` 代码 | **暂不 vendor**（避免第三方 license 风险；公开 task manifest / 下载说明） |
| PyPI distribution name | **`ai4nucleome-glmap`** |
| Python import name | **`glmap`** |

---

## 1. PyPI / import 名称口径

公开仓库统一采用：

- PyPI distribution name: `ai4nucleome-glmap`
- Python import name: `glmap`

也就是说，`pyproject.toml` 里写 `name = "ai4nucleome-glmap"`，但源码目录仍是
`src/glmap/`，用户使用 `import glmap`。

`tools/pypi_placeholder/` 暂时不进入 GitHub 公开代码整理；PyPI reserve / release
之后单独处理。

---

## 2. 顶层目录设计

```
GLMap-code-public/
├── pyproject.toml              # name = "ai4nucleome-glmap", import glmap, Apache-2.0
├── LICENSE                     # Apache-2.0 全文
├── README.md                   # 公开版（quickstart / install / 目录解释 / cite）
├── CITATION.cff                # 论文引用
├── CHANGELOG.md
├── .gitignore                  # 本阶段先不公开，后续 release 前再纳入
│
├── src/
│   └── glmap/                  # Python 包（src-layout）
│       ├── __init__.py         # public API（见 §4）
│       ├── loaders/            # 12 种 loader（HF / megaDNA / evo / evo2 / genslm / ...）
│       ├── scoring/            # AR sum_log_p + MLM stride PLL
│       ├── panel/              # panel 构建
│       ├── matrices/           # clip + double-center + pairwise distances
│       ├── analysis/           # downstream 分析工具
│       ├── figures/            # 共用绘图工具
│       └── io/                 # parquet 读写
│
├── scripts/                    # CLI 调用入口（保留 shell 调用方式，import 改 glmap.*）
│   ├── build_panel.py
│   ├── build_control_panel.py
│   ├── build_k-stride-PPL_ablation_subset.py
│   ├── run_phase1_scoring.py
│   ├── run_phase1_analysis.py
│   ├── run_downstream_embed.py
│   ├── run_downstream_classify.py
│   ├── run_rerun_stability.py
│   ├── run_sweep.py
│   ├── *.sh                    # 并行 orchestration（保留 + 加路径警告）
│   ├── audits/                 # models.py + benchmarks.py + common.py
│   ├── figures/                # paper 用的 14 个 figure script
│   ├── tables/                 # paper 用的 4 个 table script
│   ├── analysis/               # ar_mlm_merge_diagnostic.py + run_fig3_model_map_embedding.py
│   └── download_models/        # HF 模型下载脚本
│
├── tests/                      # pytest 测试套件（迁移后以实际测试数为准，要求全过）
│
├── tools/                      # 本阶段先不公开；后续单独整理
│
├── data/
│   ├── audits/                 # models.json + benchmarks.json + *.md + context_overrides.yaml
│   ├── benchmark_manifests/     # task/source manifests; no vendored third-party benchmark code
│   └── panel_sources.yaml
│
├── out_panel/                  # 预构建 panel parquet（确定性 seed=42）
│   ├── main_panel.parquet
│   ├── control_panel.parquet
│   ├── MLM_k1ablation_1000_main_panel.parquet
│   ├── *_manifest.json
│   ├── panel_summary.md
│   └── panel_summary_*.tsv
│
├── out_phase1/
│   ├── matrices/               # V/Vd/D_AR + V/Vd/D_MLM + metadata
│   ├── scores/                 # 瘦身后（每模型 ≈ 2 MB × 123 = ≈ 250 MB）
│   ├── MLM_k1ablation_1000_scores/  # 瘦身后（≈ 10 MB）
│   ├── models/
│   ├── probes/
│   ├── reports/
│   ├── stability/
│   └── figS3_per_model_r.json
│
├── out_phase2/
│   ├── downstream/             # 每模型每任务 AUC（≈ 6 MB）
│   ├── matrices/
│   ├── model_map/
│   └── phenotype_prediction/   # ≈ 2 MB
│
├── figures/                    # 所有 paper figure PDF（≈ 6 MB）
├── tables/                     # 所有 paper LaTeX 表（≈ 568 KB）
│
├── models/
│   ├── download_models_list.txt
│   ├── fig4a-svm.csv
│   └── setup_external_models.sh    # 新写：clone 9 个上游 repo
│
└── docs/
    ├── METHODS.md              # 英文重写（distilled from phase_1/2.md）
    ├── env_routing.md          # 9 个 micromamba env 的 routing 表
    └── model_catalog.md        # 描述性 catalog（从根目录搬过来）
```

---

## 3. 不会进入公开仓库的内容

| 路径 | 体积 | 原因 |
|---|---|---|
| `data/{GUE,PGB,dna_foundation_benchmark,dnabert-s_eval,genomic-benchmarks}` | 71 GB | 原始 benchmark 数据，readers 从上游下载 |
| `data/panel_sources/` | 1.1 GB | 中间产物，可重生成 |
| `out_phase2/embeddings/` | 36 GB | 下游模型 hidden states，可重生成 |
| `models/modelsHFNoInfo/` | 946 MB | vendored 上游 repo，用 `setup_external_models.sh` 让用户自取 |
| `dna_foundation_benchmark/` 代码与 `data_processed/` | 21 GB+ | 暂不 vendor；避免第三方 license 风险，改用 manifest + 上游链接 |
| `dna_foundation_benchmark/attention_weights/` | 7.7 MB | 大体积中间产物 |
| `scripts/logs/` | 8 MB | 工作运行日志 |
| `.trash/` | 112 KB | 已废弃的代码 |
| `CLAUDE.md` | 13 KB | Claude Code 内部指令 |
| `GOAL.md` | 24 KB | 内部 roadmap |
| `phase_0.md / phase_1.md / phase_2.md / phase_3.md` | 100 KB | 内部 protocol 工作文档（提炼后进 `docs/METHODS.md`） |
| `paper.md` | 21 KB | 论文草稿 |
| `docs/ar_mlm_merge_diagnostic.md` | -- | 关键数字已进 paper supplementary |
| `out_phase1/scores/*/probes.parquet` 中的 `token_log_probs` 列 | 2 GB | 每个 token 的 log probs，矩阵已经聚合掉，不需要原始数据 |
| `activations_debug.log`、`checkpoints/`、各种 `__pycache__/`、`.pytest_cache/`、`.claude/`、`.codex/`、`.agents/` | 杂项 | 临时/IDE/工具产物 |

---

## 4. Python 包 Public API（`glmap/__init__.py`）

```python
"""
GLMap: Profiling genomic language models as individuals in a population.
"""

# ---- 加载预构建制品 ----
load_panel(name="main")              # -> pd.DataFrame (10000 probes)
load_control_panel()                  # -> pd.DataFrame
load_matrix(name)                     # name ∈ {V_AR,Vd_AR,D_AR,V_MLM,Vd_MLM,D_MLM}
load_audit()                          # -> list[dict] (123 模型 metadata)

# ---- 矩阵流水线 ----
clip_lower(V, q=0.02)                 # -> (V_clipped, threshold)
double_center(V_clipped)              # -> (Vd, row_mean, col_mean, grand_mean)
fit_matrix(V, clip_q=0.02)            # -> dict 含 V_clipped / Vd / D / clip_threshold / ...
pairwise_distances(Vd)                # 平方欧氏距离
project(scores_row, fit_info)         # 把新模型的 raw V 投影进现有 Vd 空间

# ---- 模型评分 ----
get_loader(hf_id, **kwargs)           # 自动 dispatch 到 12 种 loader_kind 之一
score_panel(loader, panel, stride=6)  # -> pd.DataFrame, 包含 sum_log_p / ell_per_base / ...
score_sequence(loader, seq, stride=6) # 单条 fast path

# ---- 元数据 / 路径 ----
data_dir()                            # 包内 data/ 的绝对路径
panel_dir()                           # 同 out_panel/
matrices_dir()                        # 同 out_phase1/matrices/
__version__                           # "1.0.0"
```

CLI 不在第一版注册（即不通过 `pyproject.toml [project.scripts]` 暴露），
保持 `python scripts/build_panel.py ...` 的调用方式。

---

## 5. 实施阶段（10 个 stage，建议每个 stage 一个 commit）

### Stage 1 · 仓库骨架与元数据
- 使用现有 git repo（当前 remote 指向 `git@github.com:ai4nucleome/GLMap.git`），不重新 `git init`
- 暂不公开 `.gitignore`；先保留本地 ignored 状态，release 前再单独纳入
- 写 `LICENSE`（Apache-2.0 全文）
- 写 `pyproject.toml`：
  - `name = "ai4nucleome-glmap"`、`version = "1.0.0"`、`license = "Apache-2.0"`
  - `requires-python = ">=3.10"`
  - **核心依赖**（必装）：`torch`, `transformers`, `pandas`, `numpy`,
    `scikit-learn`, `matplotlib`, `pyarrow`, `umap-learn`, `scipy`, `PyYAML`,
    `seaborn`
  - **核心依赖只保证 analysis / figures / matrix loading 可用**
  - 大模型 scoring 依赖不承诺通过 pip extras 一步装好；按 `docs/env_routing.md` 分 family 说明环境
  - `dev = ["pytest", "build", "twine"]`
- 写 `CITATION.cff` 指向论文（DOI 占位等出版后补）
- 写 `CHANGELOG.md` 标记 v1.0.0 entry

### Stage 2 · 迁移 `src/` → `src/glmap/`
- 把原仓库 `src/{analysis,io,loaders,matrices,panel,scoring,figures}/`
  整体拷贝到新仓库 `src/glmap/`（不复制 `__pycache__/`）
- 写脚本一次性把 `src.X` 全部改成 `glmap.X`
  （`grep -rln "from src\." | xargs sed -i 's/from src\./from glmap./g'`，
   类似 import 句也改）
- 修 `src/glmap/__init__.py`：暴露 §4 列出的 public API

### Stage 3 · 迁移 `scripts/`
- 拷贝 `scripts/` 全部（排除 `logs/`、`__pycache__/`）
- 同样把 import 从 `src.X` 改成 `glmap.X`
- 在并行 shell 脚本顶部加注释 warning：硬编码的 micromamba env 路径需用户修改

### Stage 4 · 迁移 `tests/`
- 拷贝 `tests/` 整体
- 改 import
- 在新仓库根目录 `pip install -e .[dev]`（或 `python -m pip install -e .`）
- `pytest -q`，目标：当前测试套件全过（以迁移时实际测试数为准）

### Stage 5 · 迁移小体积数据制品
- `data/audits/` 全量 → `data/audits/`
- `data/panel_sources.yaml` → `data/panel_sources.yaml`
- `out_panel/` 全量（panel parquet + manifest + summary，≈ 8 MB） → `out_panel/`
- `out_phase1/matrices/` 全量（≈ 20 MB） → `out_phase1/matrices/`
- `out_phase1/{models,probes,reports,stability}` + `figS3_per_model_r.json` → 对应位置
- `out_phase2/{downstream,matrices,model_map,phenotype_prediction}` → 对应位置
- `figures/` 全量（≈ 6 MB） → `figures/`
- `tables/` 全量（≈ 568 KB） → `tables/`
- `model_catalog.md` → `docs/model_catalog.md`
- `docs/env_routing.md` → `docs/env_routing.md`

### Stage 6 · 瘦身 `out_phase1/scores/` 或外置到 HuggingFace Dataset
写 `tools/strip_token_logprobs.py`：

```python
# 伪代码
for model_dir in source.glob("out_phase1/scores/*/"):
    df = pd.read_parquet(model_dir / "probes.parquet")
    df = df.drop(columns=["token_log_probs"])  # 占大头的 list 列
    out = target / model_dir.name
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "probes.parquet", index=False)
```

- 应用到 `out_phase1/scores/`（123 模型，目标 ≈ 250 MB）
- 应用到 `out_phase1/MLM_k1ablation_1000_scores/`（以当前实际模型数为准；当前不是 56 全量）
- 瘦身后检查单文件大小：若任一文件 >95 MB，则 `scores/` 不进 git，改放 HuggingFace Dataset，git 内只保留 manifest / download script / checksum
- 在 `out_phase1/scores/README.md` 加一行说明：
  > "`token_log_probs` column removed to fit git limits.
  > Re-run `scripts/run_phase1_scoring.py` to regenerate per-token vectors."

### Stage 7 · `models/` 与外部依赖
- `models/download_models_list.txt` + `models/fig4a-svm.csv` → 直接拷贝
- `scripts/download_models/` → 整体拷贝
- 新写 `models/setup_external_models.sh`：clone 9 个上游 repo
  （evo / evo2 / genslm / hyena-dna / megaDNA / ModelGenerator / PlantBiMoE /
   PlantCaduceus / PlasmidGPT）
- `models/README.md` 解释这一切是怎么协同工作的

### Stage 8 · Benchmark manifests / external data notes
- 暂不 vendor `dna_foundation_benchmark/` 代码，避免第三方 license 风险
- 新建 `data/benchmark_manifests/`：记录任务名、来源、上游链接、样本数、label 信息
- 在 `docs/METHODS.md` 和 README 中说明 raw task CSVs 从上游数据源获取
- 如后续确实要 vendor 第三方代码，先补 `THIRD_PARTY_NOTICES.md` 和来源 commit/hash

### Stage 9 · 文档
- 写新的 `README.md`（公开版）：
  ```
  # GLMap
  一行简介 + 论文 cite

  ## Quickstart       # 5–10 行最小示例：score 一个新模型
  ## Installation
  ## Repository layout
  ## Reproducing the paper
  ## Citation
  ## License
  ```
- 写 `docs/METHODS.md`（英文重写）：
  - § Scoring protocol：AR sum_log_p + MLM stride PLL（含 k=6 选择理由）
  - § Panel construction：14 个 functional element 的来源 + 取样策略
  - § Matrix pipeline：clip(q=0.02) → double-center → distances
  - § Downstream evaluation：mean-pooling + L2 logistic regression + 6 tasks
  - § Phenotype prediction：RidgeCV + random K-fold + family GroupKFold
  - § Robustness：split-half Mantel + k=1 vs k=6 stride PLL
  - 不收录"Stage X gate"、"claim ladder"、F_ST 框架等已退役的内部叙事

### Stage 10 · 验证 + 初始 commit
- `cd GLMap-code-public && pip install -e .[dev] && pytest -q`，确认迁移后的测试套件全过
- 试跑一次 `python -c "import glmap; panel = glmap.load_panel(); print(panel.shape)"` 烟雾测试
- 试跑 figure 脚本一个：`python scripts/figures/fig2c_split_half_consistency.py --seed 123`
- `du -sh .`，确认总体积 < 500 MB
- 分阶段 commit（建议 8–10 个 semantic commits）：
  ```
  feat: scaffold pyproject + LICENSE + README skeleton
  feat: add glmap package skeleton with public API
  feat: add CLI scripts for scoring, matrices, figures
  feat: add test suite
  data: prebuilt panel and matrices artefacts
  data: per-model scores (slimmed, token_log_probs dropped)
  data: downstream AUCs and phenotype prediction outputs
  feat: external model setup script
  data: benchmark manifests and external data notes
  docs: README, METHODS.md, env routing
  ```
- 最终打 v1.0.0 tag

---

## 6. 风险与备选方案

| 风险 | 应对 |
|---|---|
| Stage 4 测试失败：部分 fixture 依赖 `out_phase1/scores/` 全量原始数据 | 在 `tests/conftest.py` 加 marker `@pytest.mark.requires_full_scores`；CI 默认 skip，把"完整测试"和"快速测试"分两组 |
| Stage 6 瘦身后任一文件 >95 MB 或总体积仍超出 GitHub 友好范围（>500 MB） | 把 `out_phase1/scores/` 外置到 HuggingFace Dataset（`ai4nucleome/GLMap-scores`），git 只保留 `out_phase1/matrices/`、manifest、checksum 和下载说明 |
| Stage 7 上游 repo 在某些时候被作者删库 | `setup_external_models.sh` 里同时提供 commit SHA fallback；用户在 setup 时锁定具体版本 |
| Loader 依赖（evo2 / genslm 等）安装难度高 | core install 只保证 analysis / matrix loading / figures；scoring 依赖按 family 写入 `docs/env_routing.md`，不承诺 pip extras 一步解决 |
| 论文还没接受，但代码先放出来 | `CHANGELOG.md` 第一行写明 "v1.0.0 — preprint companion (论文 in submission)"，正式接受后切 v1.0.1 加 DOI |

---

## 7. 需要用户配合的事项

1. **确认 PyPI distribution 使用 `ai4nucleome-glmap`，Python import 使用 `glmap`**（PyPI release 之后单独处理）
2. **真实 ORCID**（用于 `CITATION.cff`）
3. **目标 GitHub repo URL**（论文里写的 `https://github.com/ai4nucleome/GLMap`——这个 repo 是否已创建？是否给我推权限？）
4. **接受/拒绝 §6 的退路方案**（特别是 scores 外置到 HF Dataset 那条）

---

## 8. 时间预估

| Stage | 预估耗时 |
|---|---|
| 1 · 骨架 | 15 min |
| 2 · src/ 迁移 + import 重写 | 30 min |
| 3 · scripts/ 迁移 | 20 min |
| 4 · tests 迁移 + 跑通 | 30 min |
| 5 · 小制品拷贝 | 10 min |
| 6 · 瘦身 scores | 15 min（含跑脚本时间） |
| 7 · models 外部依赖 | 20 min |
| 8 · DNA foundation benchmark | 10 min |
| 9 · README + METHODS.md（英文重写） | 60 min |
| 10 · 验证 + commit | 30 min |
| **合计** | **≈ 4 小时**（其中 ≈ 60 min 是 LLM token，其余是脚本运行 / IO 等待） |

---

## 9. 我建议的下一步

1. 你审一遍这份计划，标注要改的地方
2. 我把 §7 提到的 4 项你的回答收齐
3. 我开始按 Stage 1 → 10 推进，每个 stage 完成后短报告 + 等你 OK 再进下一个
4. 全部完成后切 v1.0.0 tag，等你 PyPI 上传完发布

是否按这个流程走？
