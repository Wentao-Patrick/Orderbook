# EA_recherche - 项目结构梳理

## 总体说明

本仓库目前是“多主题研究合集”结构：每个子目录对应一个研究方向，根目录保留部分跨主题实验文件。

## 目录说明

### 1) 数据与原始输入

- `euronextparis/`
  - Euronext 原始/半原始数据目录（层级深、体量大）。
- `msc_decoded/`
  - 解码后的事件与区间文件（如 `decoded_events.csv`、`merged_intervals.csv`）。
- 根目录部分数据文件
  - 例如 `decoded_full_trade_information.csv`、`sanofi_book_events.parquet`、`sanofi_book_snapshots_1s.parquet`。

### 2) 研究子项目

- `causal_zovko/`
  - 因果分析主流程（特征构建、XCF、PC/CIT）。
  - 结构完整：`scripts/`、`data/`、`results/`、`figures/`、`docs/`。
- `Hawkes/`
  - Hawkes 建模与检验脚本、Notebook 与图表。
- `log_volume/`
  - 订单簿 volume 相关特征与统计分析。
- `orderbook_construction/`
  - 订单簿构建、市场状态处理与可视化。
- `Trade/`
  - 成交与订单流分析、变点/跳变相关实验。
- `VIX_replicate/`
  - VIX 复现实验与图表。

### 3) 文献与临时目录

- `Lectures/`
  - PDF 文献资料。
- `_tmp_fitrs/`、`._tmp_pdftext/`
  - 临时处理目录（建议不进 Git）。

## 推荐的 GitHub 展示方式

1. 根 README 作为总入口。
2. 子项目 README 作为二级入口（先从 `causal_zovko/` 开始，后续可补齐其他目录）。
3. 对大文件采用以下策略之一：
   - 不上传（仅保留样例）；
   - 使用 Git LFS；
   - 在 README 提供外部数据获取说明。

## 可选后续重构（不影响当前上传）

- 新增 `notebooks/` 汇总根目录零散 Notebook。
- 新增 `data/raw`、`data/processed` 统一数据层。
- 新增 `figures/` 汇总论文/报告最终图。
