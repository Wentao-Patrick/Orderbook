# Gamma Distribution Analysis for LOB Log-Volumes

## 项目结构

分析Sanofi订单簿的日志体积分布，并拟合Gamma分布参数的项目。

### 第一步：样本构造 (LOB，按侧分开)

**文件**: `lob_sample_construction.py`

#### 数学定义

对于每个snapshot的第i档（i ≤ K），定义：
- $V_i^{bid}(t_j)$: bid侧第i档的队列量
- $V_i^{ask}(t_j)$: ask侧第i档的队列量

构造的样本集：
$$X_{bid} = \{\log V_i^{bid}(t_j) : V_i^{bid}(t_j) > 0, i \leq K\}$$
$$X_{ask} = \{\log V_i^{ask}(t_j) : V_i^{ask}(t_j) > 0, i \leq K\}$$

其中 $K = 10$（默认前10档）。

#### 使用方法

```python
from lob_sample_construction import LOBSampleConstructor
import numpy as np

# 初始化构造器
constructor = LOBSampleConstructor("path/to/orderbook.csv", max_depth=10)

# 构造样本（对一系列时间戳）
timestamps = np.arange(0, 60_000_000_000, 5_000_000_000)  # 纳秒
constructor.construct_samples(timestamps)

# 获取样本和统计信息
log_vols_bid, log_vols_ask = constructor.get_samples()
stats = constructor.get_summary_stats()
```

#### 实现细节

1. **深度提取** (`extract_depth_volumes`):
   - 从订单簿重放器获取指定时间戳的快照
   - 按价格聚合各档的订单（同一档可能有多个订单）
   - 提取bid侧最高的K档和ask侧最低的K档

2. **样本构造** (`construct_samples`):
   - 遍历所有时间戳
   - 对每个快照提取体积
   - 取正体积的对数，分别存储到bid和ask列表

3. **统计汇总** (`get_summary_stats`):
   - 计算样本量、均值、标准差、最小值、最大值
   - 按侧分别报告

---

## 第二步：经验分布可视化

**文件**: `visualization.py`, `step2_visualization.py`

### 可视化类型

#### 1. 经验直方图 (Empirical Histograms)
```python
LOBVisualization.plot_empirical_histograms(
    log_volumes_bid, 
    log_volumes_ask,
    bins=100,
    save_path="histograms.png"
)
```
并排显示bid和ask的密度直方图

#### 2. 分离直方图 (Separate Full-Page Histograms)
```python
LOBVisualization.plot_separate_histograms(
    log_volumes_bid,
    log_volumes_ask,
    bins=100,
    save_dir="."
)
```
为每一侧生成单独的全页直方图

#### 3. 经验CDF
```python
LOBVisualization.plot_empirical_cdf(
    log_volumes_bid,
    log_volumes_ask,
    save_path="cdf.png"
)
```

#### 4. QQ图 (预备检验)
```python
LOBVisualization.plot_qq_against_normal(
    log_volumes_bid,
    log_volumes_ask,
    save_path="qq_normal.png"
)
```

### 统计汇总
```python
stats = LOBVisualization.get_comparison_stats(
    log_volumes_bid,
    log_volumes_ask
)
# 返回: n, mean, std, skewness, kurtosis, min, max, q1, median, q3
```

### 使用示例

```bash
python step2_visualization.py
```

或在交互式环境中：
```python
from lob_sample_construction import LOBSampleConstructor
from visualization import LOBVisualization
import numpy as np

# 构造样本
constructor = LOBSampleConstructor("path/to/orderbook.csv", max_depth=10)
timestamps = np.arange(0, 3_600_000_000_000, 5_000_000_000)
constructor.construct_samples(timestamps)

# 获取样本
bid_log, ask_log = constructor.get_samples()

# 可视化
LOBVisualization.plot_empirical_histograms(bid_log, ask_log)
LOBVisualization.plot_empirical_cdf(bid_log, ask_log)
```

---

## 第三至五步：Gamma 分布拟合与可视化

**文件**: `gamma_fitting.py`, `step3_5_complete_pipeline.py`

### 第3步：参数估计 (MLE)

```python
from gamma_fitting import GammaDistributionFitter

fitter = GammaDistributionFitter(log_volumes_bid, log_volumes_ask)
bid_result, ask_result = fitter.fit()
```

**自动决策策略**：
- 检测负log-volume的比例
- 若 > 10%（默认阈值）：改为拟合原始体积 $V$（而非 $\log V$）
- 否则：拟合正log-volume样本

**参数**：
- $k$：形状参数 (shape)
- $\theta$：尺度参数 (scale)
- $\text{loc}$：位置参数（固定为0）

### 第4步：叠加理论密度

```python
fitter.plot_fitted_histograms(bins=100, save_path="gamma_fit.png")
```

输出：
- 经验直方图（半透明蓝/红）
- Gamma PDF 曲线（红/深红，线宽加粗）
- 参数显示在标题中
- KS检验 p-value 显示在图表

### 第5步：结果输出

```python
print(fitter.get_summary_report())
```

输出包含：
- 原始样本量 vs 实际拟合样本量
- Gamma参数：$k$, $\theta$
- 数据统计：均值、标准差
- 拟合优度：KS统计量、p-value
- 假设检验结论

### 完整管道示例

```bash
python step3_5_complete_pipeline.py
```

执行流程：
1. 构造LOB样本
2. 绘制经验分布
3. 拟合Gamma分布（自动决策fit_type）
4. 绘制拟合曲线
5. 输出完整报告和参数对比

### 关键特性

**自适应拟合策略**：
- 检查 $\log V$ 的负值比例
- 若过多，自动切换到拟合 $V$（而非 $\log V$）
- 避免截断数据导致的偏差

**拟合优度检验**：
- Kolmogorov-Smirnov (KS) 检验
- p-value > 0.05：数据与Gamma分布相容
- p-value < 0.05：拒绝Gamma分布假设

**参数对比**：
- BID vs ASK 的 $k$ 比值：反映两侧流动性均衡
- BID vs ASK 的 $\theta$ 比值：反映体积水平差异

---

## 文件结构总结

```
gamma_distribution_analysis/
├── lob_sample_construction.py    # Step 1: 样本构造
├── visualization.py              # Step 2: 经验分布可视化
├── gamma_fitting.py              # Step 3-5: 拟合 & 结果
├── step2_visualization.py        # Step 2 示例脚本
├── step3_5_complete_pipeline.py  # Step 3-5 完整管道
└── README.md                     # 本文档
```

## 使用流程

### 最快开始

```python
from step3_5_complete_pipeline import complete_analysis_pipeline

results = complete_analysis_pipeline(
    orderbook_file="path/to/orderbook.csv",
    max_depth=10,
    snapshot_interval_ns=5_000_000_000,      # 5秒
    analysis_duration_ns=3_600_000_000_000,  # 1小时
)
```

### 详细控制

```python
from lob_sample_construction import LOBSampleConstructor
from gamma_fitting import GammaDistributionFitter
import numpy as np

# Step 1: 构造
constructor = LOBSampleConstructor("orderbook.csv", max_depth=10)
timestamps = np.arange(0, 3_600_000_000_000, 5_000_000_000)
constructor.construct_samples(timestamps)

# Step 3: 拟合
bid_log, ask_log = constructor.get_samples()
fitter = GammaDistributionFitter(bid_log, ask_log)
fitter.fit()

# Step 4: 绘图
fitter.plot_fitted_histograms()
fitter.plot_separate_fitted_histograms(save_dir="results/")

# Step 5: 报告
print(fitter.get_summary_report())
```

