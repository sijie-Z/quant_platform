# Regime Router v1 — 工程蓝图

> **基于**：Regime Discovery v4（α = Signal × I(市场有方向)）
> **状态**：研究完成度 80%，工程产品化 30%
> **目标**：将市场结构函数转化为条件化决策系统

---

## 1. 系统设计原则

1. **最小闭包**——只用已验证的结构，不加任何新发现
2. **无 ML 分类器**——规则足够，ML 引入过拟合风险
3. **显式状态切换**——每个决策可追溯、可审计、可解释
4. **防过拟合**——状态定义固定、参数不优化、S/H 锁定

---

## 2. 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│ Market Data  │───▶│ Regime       │───▶│ Operator Selector │
│ (价格+收益)  │    │ Classifier   │    │ (S, H 映射表)    │
└─────────────┘    └──────────────┘    └────────┬─────────┘
                                                │
                                                ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
│ Portfolio   │◀───│ Execution    │◀───│ Signal Generator  │
│ (权重+风控) │    │ (调仓引擎)   │    │ (反转信号, 等权)  │
└─────────────┘    └──────────────┘    └──────────────────┘
```

### 各模块职责

| 模块 | 输入 | 输出 | 复杂度 |
|------|------|------|--------|
| Regime Classifier | 过去 N 日基准收益 + 波动率 | regime ∈ {Bull, Bear, Sideways} | 5 行规则 |
| Operator Selector | regime | (signal_horizon, holding_horizon) | 3 行查表 |
| Signal Generator | prices, (S, H) | 反转信号权重 | 10 行 |
| Execution | 信号, 持有期 | 调仓指令 | 复用现有引擎 |

---

## 3. Regime Classifier 设计

### 3.1 分类规则（固定，不可优化）

使用过去 63 个交易日（~3 个月）的基准收益和波动率：

```python
def classify(benchmark_returns, lookback=63):
    ret = benchmark_returns.tail(lookback).mean() * 252  # 年化
    vol = benchmark_returns.tail(lookback).std() * sqrt(252)  # 年化

    if ret > 0.05:        # 年化收益 > 5%
        return "Bull"
    elif ret < -0.05:     # 年化收益 < -5%
        return "Bear"
    else:
        return "Sideways"
```

阈值 +5%/-5% 年化 ≈ 每月 +0.4%/-0.4%，与 v4 协议的月度 ±2% 一致，映射到年化约 ±5%。

### 3.2 状态记忆

引入最小持有期（min_regime_hold=20 交易日，约 1 个月），防止频繁切换。当检测到状态变化时，新状态必须持续至少 20 日才触发切换。

---

## 4. Operator Selector 设计

### 4.1 (S, H) 映射表

| Regime | Signal_H | Hold_H | Rebalance | 逻辑 |
|--------|----------|--------|-----------|------|
| **Bull** | 20d | 60d | 每 60 日 | 上涨市场，用较短信号（快反），中等持有 |
| **Bear** | 40d | 80d | 每 80 日 | 下跌市场，用较长信号（深跌反弹），较长持有 |
| **Sideways** | 60d | 120d | 每 120 日 | 震荡市场，反转失效，降低仓位/频率 |

### 4.2 选择依据

| 参数 | Bull | Bear | Sideways |
|------|------|------|----------|
| 选 S | 20d | 40d | 60d |
| 选 H | 60d | 80d | 120d |
| 对应 Sharpe | +1.20 | +0.70 | — |

Bull 使用 S=20, H=60（Sharpe +1.20）而非 S=5, H=120（Sharpe +3.30），因为后者基于极少的观测（n=6 个 Bull 期 × 不均衡分布），更保守。

### 4.3 Sideways 降仓

在 Sideways 状态中，由于该状态下反转策略 Sharpe 为负（−0.46），应降低仓位：
- 目标仓位 = 正常仓位的 **50%**
- 剩余 50% 持有现金/等权基准
- 如果连续 3 个持有期仍处于 Sideways，进一步降至 **25%**

---

## 5. 信号生成器设计

### 5.1 反转信号（与 RQ5b 一致）

```python
def reversal_signal(returns, signal_horizon):
    past_ret = returns.rolling(signal_horizon).apply(cumprod - 1)
    signal = -past_ret.rank(pct=True)  # 反转：买入跌最多的
    signal = signal - 0.5  # 中心化到 [-0.5, 0.5]
    return signal
```

### 5.2 选择逻辑

每期选择过去 S 日跌幅最大的 **20% 股票**，等权配置。

---

## 6. 回测验证计划

### 6.1 对照实验

| 策略 | 说明 |
|------|------|
| 静态反转 (S=40, H=80) | RQ5b 最优静态单元 |
| Regime Router (动态 S/H) | 本协议 |
| 等权基准 | 100% 现金/基准 |

### 6.2 评价指标

Sharpe, 年化收益, MDD, 换手率（每期一次=低）, 正收益占比

---

## 7. 限制与风险

| 风险 | 说明 | 应对 |
|------|------|------|
| 状态误判 | 分类器错误率约 30% | 20 日状态记忆缓冲，减少误判成本 |
| Sideways 仓位过低 | 如果误判为 Sideways 但市场是 Bull | 仓位最低 25%，不算完全错过 |
| 小样本 | Bull 操作期约 6 个 | 结果为探索性，非确证性 |
| 参数阈值 | +5%/−5% 阈值是人为设定的 | 不优化，保持固定。如需敏感性分析可跑 ±2% 变体 |
| 状态持续期 vs 持有期 | 状态平均持续 45-55d，持有期 60-120d | 持仓可能跨状态切换——这是系统固有风险 |

---

## 8. 实现计划

| 步骤 | 交付物 | 预计时间 |
|------|--------|---------|
| 1. Regime Classifier | `regime_router/classifier.py` | — |
| 2. Operator Selector | `regime_router/selector.py` | — |
| 3. Signal Generator | `regime_router/signal.py` | — |
| 4. Router Engine | `regime_router/engine.py` | — |
| 5. Backtest (对照实验) | `regime_router/backtest.py` | — |
| 6. 报告 | `docs/REGIME_ROUTER_V1_RESULTS.md` | — |
