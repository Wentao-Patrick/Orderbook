# EA_recherche

该仓库汇总了围绕高频金融数据（订单簿、成交、事件、因果分析、Hawkes 建模、VIX 复现等）的研究代码、实验笔记与中间结果。

## 1. 仓库定位

- 目标：集中管理课程/研究中的实验脚本、Notebook 与结果可视化。
- 研究对象：以 Sanofi（FR0000120578）相关数据为核心，包含订单簿、成交与事件序列分析。
- 主要语言：Python（Jupyter Notebook + 脚本）。

## 2. 顶层结构总览

- `causal_zovko/`：Zovko 思路复现与因果发现流程（含独立 README 与 docs）
- `Hawkes/`：Hawkes 过程建模、重标定检验与相关图表
- `log_volume/`：订单簿体量（log volume）特征构建与分布分析
- `orderbook_construction/`：订单簿重建、市场状态处理、可视化验证
- `Trade/`：成交曲线、订单流与跳变分析
- `VIX_replicate/`：VIX 相关实验复现
- `euronextparis/`：原始或半原始市场数据目录（体量较大）
- `msc_decoded/`：解码/对齐后的事件与区间数据
- `Lectures/`：文献与课程阅读材料（PDF）
- `_tmp_fitrs/`、`._tmp_pdftext/`：临时处理产物（建议不纳入版本管理）
- 根目录零散 `*.ipynb/*.csv/*.parquet/*.png/*.mp4`：跨模块实验与导出结果

更详细目录说明见：`docs/PROJECT_STRUCTURE.md`。

## 3. 快速开始

### 3.1 建议环境

- Python 3.10+
- 常用包：numpy、pandas、matplotlib、scipy、networkx、scikit-learn（按子项目需要增补）

### 3.2 推荐流程

1. 先阅读各子目录说明（尤其 `causal_zovko/README.md`）。
2. 在对应子目录运行脚本/Notebook，避免在根目录混跑导致路径混乱。
3. 大规模数据处理优先脚本化，Notebook 以分析和可视化为主。

## 4. 数据与发布建议（上传 GitHub 前）

- 本仓库包含大量数据与多媒体文件（CSV/Parquet/PNG/MP4/PDF/ZIP）。
- 若公开发布，建议区分：
  - 必须保留：代码、核心说明、少量可复现样例数据。
  - 建议移除或改用 Git LFS：大体量原始数据、视频、批量中间结果。
- 已在根目录提供 `.gitignore`，用于排除缓存、临时文件与系统文件。

## 5. 你可以如何继续整理（可选）

- 将根目录零散实验文件按主题归档至 `notebooks/`、`data/processed/`、`figures/`。
- 给每个子项目补充最小 README（输入、输出、运行命令、依赖）。
- 新增 `requirements.txt` 或 `environment.yml` 固化环境。

## 6. 相关文档

- 子项目说明：`causal_zovko/README.md`
- 仓库结构地图：`docs/PROJECT_STRUCTURE.md`
- 上传前检查清单：`docs/PRE_GITHUB_CHECKLIST.md`
