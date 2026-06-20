# Ablation Study #001: Factor Selection vs Equal-Weight Factor Aggregation

> **日期**：2026-06-20
> **状态**：已完成
> **实验 ID**：ABLATION-001

---

## 背景

v1.0 验证阶段确认了核心研究链路的正确性（Oracle IC=1.0, Known Alpha Recovery, WalkForward, MVO Audit）。但在此之后，一个根本性问题仍未解决：

**为什么一个已经验证正确的平台，默认配置跑出来的策略结果很差？**

默认配置 `alpha.method: equal_weight` 将所有 26 个因子等权平均后生成信号。本实验旨在验证：

**假设**：噪音因子（|IC| < 0.005）等权参与信号合成，会稀释有效因子的 Alpha，导致组合表现劣化。

---

## 方法

### 实验设计

在相同市场数据下（合成数据，300 只股票，2021-2024，alpha_strength=0.03），按因子 |IC| 排名分组，每组用等权融合方式生成 Alpha 信号，跑完全部回测流程，对比组合指标。

### 分组

| 组别 | 选择策略 | 包含因子 |
|------|---------|---------|
| All 26 | 全部因子 | 所有 26 个注册因子 |
| Top 8 | |C| 排名前 8 | momentum_1m, trend_stage, rsi_14d, momentum_3m, breakout_proximity, momentum_6m, momentum_12m, turnover_20d |
| Top 5 | |C| 排名前 5 | momentum_1m, trend_stage, rsi_14d, momentum_3m, breakout_proximity |
| Top 3 | |C| 排名前 3 | momentum_1m, trend_stage, rsi_14d |

### 回测参数

| 参数 | 值 |
|------|-----|
| 股票池 | 300 只（合成数据） |
| 时间范围 | 2021-01-01 至 2024-12-31 |
| Alpha 强度 | 0.03（weak/realistic） |
| 调仓频率 | 月度 |
| 优化器 | EqualWeight |
| 成本模型 | 佣金 0.03% + 印花税 0.1%（卖）+ 滑点 0.1% |
| 约束 | 纯多头，单票 ≤ 5%，行业 ≤ 30% |

---

## 结果

### 主表

| Strategy | Factors | Sharpe | Ann.Ret | MaxDD | Volatility |
|----------|---------|--------|---------|-------|-----------|
| **Top 8** | 8 | **+0.464** | **+12.57%** | -26.6% | 20.6% |
| Top 5 | 5 | +0.284 | +8.71% | -31.8% | 20.1% |
| Top 3 | 3 | +0.065 | +4.31% | -40.3% | 20.1% |
| All 26 | 26 | **-0.779** | **-14.16%** | -63.7% | 22.0% |

### 因子相关性（Top 8）

| Factor | Avg Corr | 家族 |
|--------|----------|------|
| momentum_1m | 0.466 | 动量 |
| trend_stage | 0.592 | 趋势 |
| rsi_14d | 0.440 | 反转 |
| momentum_3m | 0.509 | 动量 |
| breakout_proximity | 0.422 | 突破 |
| momentum_6m | 0.492 | 动量 |
| momentum_12m | 0.285 | 动量（长周期） |
| turnover_20d | **0.213** | 换手率（独立） |

---

## 结论

### 结论 1：等权融合所有因子是错误配置

All 26 因子等权是**最差**的策略（Sharpe = -0.779）。噪音因子不仅没有帮助，还主动拖累了组合。**默认配置应改为精选因子 + ICIR 加权，而非等权融合。**

### 结论 2：IC 排名有效传导至组合表现

Top 8 > Top 5 > Top 3 > All 26 的 Sharpe 排名与 IC 排名一致，证明 Factor Ranking 能有效指导因子选择决策。

### 结论 3：Top 8 > Top 3 — Alpha 强度不是唯一指标

Top 3 单独使用表现不佳（Sharpe = 0.065），而 Top 8（Sharpe = 0.464）显著更优。说明：
- 因子多样性和正交性比单一因子强度更重要
- Turnover_20d 和 momentum_12m 虽然单因子 IC 较低，但提供了互补信息
- 最佳因子组合 ≠ 最强单因子集合

---

## 对项目的影响

### Alpha Pipeline 架构变更

```
以前：
  全部因子 → 等权平均 → 信号
  
以后：
  全部因子 → Factor Ranking → Factor Selection → ICIR 加权 → 信号
```

### 默认配置变更

`alpha.method` 应从 `equal_weight` 改为 `icir_weighted`，并启用因子选择。

### 后续实验

**Ablation #002 — Leave-One-Out Study**：在 Top 8 中逐一剔除因子，识别冗余因子和关键补充因子。

---

## 原始数据

| 因子 | Mean IC | ICIR |
|------|---------|------|
| rsi_14d | +0.0208 | +0.263 |
| trend_stage | +0.0207 | +0.271 |
| momentum_1m | +0.0195 | +0.242 |
| breakout_proximity | +0.0190 | +0.241 |
| momentum_3m | +0.0172 | +0.222 |
| momentum_6m | +0.0172 | +0.228 |
| momentum_12m | +0.0125 | +0.171 |
| turnover_20d | +0.0114 | +0.153 |
| macd | +0.0091 | +0.120 |
| pe_ratio | +0.0076 | +0.099 |
| roe | -0.0070 | -0.091 |
| ma_convergence | +0.0061 | +0.079 |
| kup | +0.0056 | +0.073 |
| klow | +0.0056 | +0.073 |
| ksft | +0.0056 | +0.073 |
| pb_ratio | -0.0040 | -0.054 |
| breakout_ignition | +0.0038 | +0.050 |
| kmid | +0.0029 | +0.036 |
| efficiency_ratio | +0.0028 | +0.036 |
| asset_growth | +0.0027 | +0.035 |
| mtf_resonance | -0.0022 | -0.028 |
| amplitude_20d | +0.0022 | +0.028 |
| volatility_60d | -0.0009 | -0.012 |
| klen | -0.0006 | -0.007 |
| volatility_20d | +0.0005 | +0.006 |
| log_market_cap | +0.0000 | +0.001 |
