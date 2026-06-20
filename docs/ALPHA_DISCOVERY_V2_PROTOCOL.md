# Alpha Discovery v2 实验协议

> **冻结日期**：2026-06-20
> **状态**：已冻结 — 协议锁定期内不得增加/修改实验项目
> **关联文档**：[NO_LOOKAHEAD_CONTRACT.md](NO_LOOKAHEAD_CONTRACT.md) — 本协议所有实验均遵守该契约

---

## 1. 研究背景与动机

v1.0 验证阶段完成三项核心发现：

| # | 发现 | 来源 |
|---|------|------|
| 1 | Top 8 因子 vs All 26 因子 **Sharpe 差 1.24** | Ablation #001 |
| 2 | 8 个有效因子压缩为 **4 个独立 Alpha 簇** | Ablation #002 |
| 3 | 4 个代表因子解释 Top 8 **82% 收益** | Ablation #003 |
| 4 | 真实 A 股中合成数据信号方向全部相反 | Real A-Share Validation |

**核心问题**：Alpha 从哪来？哪些方向错了？什么时候有效？

本协议冻结以下三个研究问题的实验设计，禁止在结果产出前修改实验参数。

---

## 2. 数据集定义

### 2.1 数据源

| 参数 | 值 | 说明 |
|------|-----|------|
| Provider | `SyntheticDataProvider` | 可复现，IC 水平已知 |
| Asset count | 500 | 默认股票池 |
| Date range | 2021-01-01 ~ 2025-12-31 | 5 年 |
| Alpha strength | `0.06` | 默认中等强度 |
| Benchmark | 等权组合 | — |
| Rebalance | Monthly | 每月最后一个交易日 |

### 2.2 真实 A 股验证（Phase 2 后可选扩展）

| 参数 | 值 |
|------|-----|
| Provider | `BaostockDataProvider` |
| Asset count | 200（从 A 股全市场 5000+ 中采样） |
| Date range | 2018-01-01 ~ 2025-12-31 |
| 说明 | 仅用作交叉验证，不参与主实验结论 |

---

## 3. 因子池

### 3.1 完整因子列表

本协议使用以下 8 个因子（即 Ablation #001 确认的 Top 8）：

```
momentum_1m      短期动量（21 日）
momentum_3m      中期动量（63 日）
momentum_6m      中期动量（126 日）
momentum_12m     长期动量（252 日，skip 21）
rsi_14d          相对强弱
turnover_20d     换手率
trend_stage      趋势阶段
breakout_proximity  突破接近度
```

### 3.2 Alpha 簇映射

按 Ablation #002 的聚类结果：

| Alpha 簇 | 因子 | 标签 |
|----------|------|------|
| **Cluster A: Short Reversal** | `rsi_14d`, `momentum_1m`, `breakout_proximity` | 短期/反转信号 |
| **Cluster B: Medium Trend** | `trend_stage`, `momentum_3m`, `momentum_6m` | 中期趋势 |
| **Cluster C: Long Trend** | `momentum_12m` | 长期趋势 |
| **Cluster D: Liquidity** | `turnover_20d` | 流动性 |

---

## 4. 研究问题与实验设计

---

### 研究问题 1（RQ1）：Alpha 来源于哪些 Alpha 簇？

**假设**：
- **H₀**：四个簇对组合 Alpha 贡献相近（IC/Sharpe 无显著差异）
- **H₁**：少数簇贡献大部分 Alpha（分布不均）

#### 实验 1.1 — 单簇独立评估

运行四个簇各自的信号，不与其他簇混合：

| 实验 | 信号成分 | 预期用途 |
|------|----------|---------|
| A | Cluster A 等权合成 | 评估 Short Reversal 独立表现 |
| B | Cluster B 等权合成 | 评估 Medium Trend 独立表现 |
| C | Cluster C 等权合成 | 评估 Long Trend 独立表现 |
| D | Cluster D 等权合成 | 评估 Liquidity 独立表现 |

#### 实验 1.2 — 组合归因

| 实验 | 信号成分 | 说明 |
|------|----------|------|
| A+B | Cluster A + B | 短期 + 中期趋势 |
| A+C | Cluster A + C | 短期 + 长期趋势 |
| A+D | Cluster A + D | 短期 + 流动性 |
| B+C | Cluster B + C | 中期 + 长期趋势 |
| B+D | Cluster B + D | 中期 + 流动性 |
| C+D | Cluster C + D | 长期 + 流动性 |
| A+B+C | Cluster A + B + C | 所有趋势类 |
| A+B+D | Cluster A + B + D | 排除长期趋势 |
| A+C+D | Cluster A + C + D | 排除中期趋势 |
| B+C+D | Cluster B + C + D | 排除短期反转 |
| ALL | A + B + C + D | 四个簇全量 |
| TOP8 | 全部 8 个原始因子等权 | Baseline（来自 v1） |

#### 实验 1.3 — 相关性矩阵

计算四个簇信号两两之间的：
- Pearson 相关系数
- Spearman 秩相关系数
- 时间序列对齐（日频）

---

### 研究问题 2（RQ2）：哪些 Alpha 方向错了？

**假设**：
- **H₀**：所有簇的信号方向与合成数据预设方向一致
- **H₁**：部分簇在真实 A 股中信号方向相反

#### 实验 2 — Cluster Sign Flip

| 实验 | Short Rev | Mid Trend | Long Trend | Liquidity |
|------|-----------|-----------|------------|-----------|
| Baseline | + | + | + | + |
| Flip-A | **−** | + | + | + |
| Flip-B | + | **−** | + | + |
| Flip-C | + | + | **−** | + |
| Flip-D | + | + | + | **−** |

每个实验运行完整回测，以 Baseline 为参照，关注 **ΔIC** 和 **ΔSharpe**。

#### 判定规则

如果翻转后同时满足：
1. IC 绝对值提升（\|IC_flipped\| > \|IC_original\|）
2. Sharpe 提升（Sharpe_flipped > Sharpe_original + 0.1）
3. 方向翻转（sign(IC_flipped) = −sign(IC_original)）

则判定该簇**方向错误**。

---

### 研究问题 3（RQ3）：Alpha 稳定性与市场状态依赖性

**假设**：
- **H₀**：各簇 Alpha 在不同时间窗口和市場状态下表现一致
- **H₁**：Alpha 存在显著的时变性或 Regime Dependency

#### 实验 3.1 — 逐年稳定性

固定时间窗口（不可调整）：

| 窗口 | 年份 | 市场背景（事后描述用，非实验参数） |
|------|------|-------------------------------|
| W1 | 2021 | — |
| W2 | 2022 | — |
| W3 | 2023 | — |
| W4 | 2024 | — |
| W5 | 2025 | — |

每个窗口计算各簇及 ALL 组合的：
- IC / ICIR
- Sharpe Ratio
- Max Drawdown
- Turnover

#### 实验 3.2 — 市场状态分解

数据集的日期范围被切分为三种市场状态（定义标准冻结如下）：

| 状态 | 定义（固定规则，不得事后修改） |
|------|------------------------------|
| **Bull** | benchmark 月收益 > +2% |
| **Bear** | benchmark 月收益 < −2% |
| **Sideways** | benchmark 月收益 ∈ [−2%, +2%] |

计算各簇在不同市场状态下的 Sharpe 与 IC。

---

## 5. 评价指标

所有实验统一使用以下指标，**禁止临时增加**：

| 指标 | 缩写 | 计算方式 | 阈值/关注点 |
|------|------|---------|-------------|
| Rank IC | IC | Spearman 秩相关系数（因子 vs 下期收益） | 绝对值 > 0.02 为有信号 |
| ICIR | ICIR | IC 均值 / IC 标准差 | > 0.5 为稳定 |
| Sharpe Ratio | Sharpe | 年化收益 / 年化波动 | > 0.5 为可用 |
| 最大回撤 | MDD | 峰值到谷值最大跌幅 | < −30% 需标注 |
| 单边换手率 | Turnover | 平均每期权重变化 / 2 | > 50% 需标注 |

**禁止的指标扩展**：Calmar Ratio、Sortino Ratio、Information Ratio、Win Rate、盈亏比、分位数收益、因子归因瀑布图——这些不在本协议范围内。

---

## 6. 回测参数

所有实验使用相同回测参数：

| 参数 | 值 |
|------|-----|
| Rebalance frequency | Monthly |
| Rebalance date | 每月最后一个交易日 |
| Execution | 次日收盘价 |
| Commission | 0.03% 双边 |
| Stamp tax | 0.1% 仅卖出 |
| Slippage | 0.05% 固定 |
| Lot size | 100 股 |
| Position limit | 单票 ≤5% |
| Sector limit | 行业 ≤30% |
| Long only | 是 |
| Optimizer | EqualWeight |

优化器使用等权而非 MVO/RiskParity——目的是隔离信号质量与优化器的影响。

---

## 7. 判定标准与报告模板

### 7.1 显著性判定

| 层级 | 标准 | 行动 |
|------|------|------|
| 🔴 强信号 | IC > 0.05 且 Sharpe > 1.0 | 独立候选 Alpha |
| 🟡 可用信号 | IC > 0.02 且 Sharpe > 0.5 | 可纳入组合 |
| ⚪ 弱信号 | IC > 0.01 或 Sharpe > 0.2 | 需进一步验证 |
| ⚫ 噪声 | IC ≤ 0.01 且 Sharpe ≤ 0.2 | 排除 |

### 7.2 报告模板

每个研究问题产出一个独立报告，格式固定：

```
## RQ{N}: {问题简述}

### Hypothesis
H0: ...
H1: ...

### Result Summary
| Experiment | IC | ICIR | Sharpe | MDD | Turnover |
|------------|----|------|--------|-----|----------|

### Key Findings
1. ...
2. ...

### Conclusion
接受/拒绝 H0。理由：...
```

---

## 8. 协议冻结条款

1. **冻结日期**：2026 年 6 月 20 日。在此日期前可修改协议。
2. **锁定范围**：RQ 列表、实验矩阵、评价指标、数据集、回测参数。
3. **解冻条件**：以下情况可解冻并修订协议：
   - 本协议中所有已冻结实验全部执行完毕
   - 或有明确统计证据表明某个设计偏差导致结论不可信（需书面记录理由）
4. **Protocol Amendment**：解冻后的变更需以 Amendment 形式追加，不修改原始条款。

---

## 9. 执行顺序

```
Step 1  [当前]   冻结协议 ✓
Step 2           RQ1 — Cluster Attribution（实验 1.1 + 1.2 + 1.3）
Step 3           RQ1 结果分析与结论
Step 4           RQ2 — Cluster Sign Flip
Step 5           RQ2 结果分析与结论
Step 6           RQ3 — Stability + Regime
Step 7           RQ3 结果分析与结论
Step 8           Discovery v2 综合报告
Step 9           [可选] 解冻协议 / 进入 Discovery v3
```

每个步骤产出结果后再进入下一步，不跳过、不并行修改协议。

---

## 10. 风险控制

| 风险 | 应对 |
|------|------|
| 发现意外结果后想加实验 | 记录为 Protocol Amendment，当前轮次不动 |
| 某个实验报错/无法运行 | 修复代码 bug 后重新运行该实验，保持参数不变 |
| 合成数据与真实 A 股结论冲突 | 以合成数据结论为主，真实 A 股标注为"交叉验证参考" |
| 结果不显著 | 记录 H₀ 未被拒绝，这是有效结论，不是失败 |
