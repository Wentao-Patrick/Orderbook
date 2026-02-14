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
- `msc_decoded/`：解码/集合竞价 连续竞价时间计算
- `Lectures/`：文献与课程阅读材料（PDF）
- `_tmp_fitrs/`、`._tmp_pdftext/`：临时处理产物
- 根目录零散 `*.ipynb/*.csv/*.parquet/*.png/*.mp4`：跨模块实验与导出结果

更详细目录说明见：`docs/PROJECT_STRUCTURE.md`。

