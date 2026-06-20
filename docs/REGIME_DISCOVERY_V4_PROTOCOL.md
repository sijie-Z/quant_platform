# Regime Discovery v4 — 实验协议

> **冻结日期**：2026-06-21
> **前序**：Market Structure Map v1 确认 A 股存在反转结构带（S=40/H=80）
> **但**：滚动稳定性检验显示该结构高度不稳定——52% 窗口正收益，Sharpe 范围 [-1.46, 0.95]
> **核心问题**：α 不是静态的。α = f(市场状态)。市场状态是什么？

---

## 1. 研究问题

### RQ6: Regime Discovery — 市场状态定义

**问题**：A 股的反转结构依赖于哪些市场状态？

**假设**：
- **H₀**：市场状态与 Sharpe(S,H) 无关 → 结构是稳定的（已被反驳）
- **H₁**：不同市场状态下，Sharpe(S,H) 响应面显著不同

### RQ7: Regime-Conditioned Heatmap

完整的条件热力图：Sharpe(Signal_H, Hold_H | Regime)

### RQ8: Regime Predictability

**问题**：市场状态本身是否可预测？如果能预测状态，就可以切换策略。

---

## 2. 数据集

| 参数 | 值 |
|------|-----|
| Provider | Baostock（真实 A 股） |
| Universe | ~101 只 |
| Date range | 2018-01-01 ~ 2025-12-31 |
| Benchmark | 等权组合 |
| Signal/Hold horizons | S={5,10,20,40,60,80,120,200} × H={5,10,20,40,60,80,120,200} |

---

## 3. 实验设计

### 实验 6.1 — 基于基准收益的状态分类（协议固定，不可修改）

使用与 Discovery v2 协议相同的定义：

| 状态 | 定义 |
|------|------|
| **Bull** | 基准月收益 > +2% |
| **Bear** | 基准月收益 < −2% |
| **Sideways** | 基准月收益 ∈ [−2%, +2%] |

### 实验 6.2 — 基于波动率的状态分类

| 状态 | 定义 |
|------|------|
| **Low Vol** | 月波动率（日收益 std）处于历史后 1/3 |
| **High Vol** | 月波动率处于历史前 1/3 |
| **Normal Vol** | 中间 1/3 |

### 实验 7 — Regime-Conditioned Heatmap

对每种状态划分，重复 Market Structure Map v1 的完整流程：

1. 将 2018-2025 按状态分割为时间片段
2. 在每个状态的时间片段上独立运行全 8×8 = 64 单元 Sharpe 矩阵
3. 输出每个状态的条件热力图

**判定规则**：
- 如果同一 (S,H) 单元在不同状态下的 Sharpe 差值 > 0.5：**状态对该单元有显著影响**
- 如果最优单元在不同状态下位置不同（如 Sideways 下 S=40/H=80 最优，Bear 下 S=5/H=20 最优）：**市场状态改变反转结构**

### 实验 8 — 状态持续性分析

1. 计算 Bull/Bear/Sideways 的持续期分布
2. 计算状态转换概率矩阵
3. 评估"基于当前状态调整策略"的可行性

---

## 4. 评价指标（同 v3 协议）

Sharpe, 年化收益, MDD, n_periods

---

## 5. 执行顺序

```
Step 1  冻结协议 ✅
Step 2  实验 6.1 — Bull/Bear/Sideways 分类
Step 3  实验 7.1 — Regime-conditioned heatmap（基准收益状态）
Step 4  实验 6.2 — Volatility 分类  
Step 5  实验 7.2 — Regime-conditioned heatmap（波动率状态）
Step 6  实验 8 — 状态持续性分析
Step 7  Regime Discovery v4 综合报告
```

---

## 6. 风险控制

- **小样本风险**：状态分解后每个状态的时间片段变短，Sharpe 估计值方差增大。结果为探索性，非确证性。
- **多重比较**：64 单元 × 3 状态 = 192 个 Sharpe 估计值。关注宏观模式，不追求单个单元的统计显著性。
- **过拟合**：不以"找到某个状态下的最高 Sharpe"为目标，以"理解结构如何随状态变化"为目标。
