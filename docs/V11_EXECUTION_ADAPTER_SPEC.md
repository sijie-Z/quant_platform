# v1.1 Execution Adapter Layer — 设计规范

> **核心问题**：研究空间（S=40, H=80）与执行空间（月频 ~20d）之间存在持有期不匹配
> **核心原则**：不改 BacktestEngine。改造略适配执行约束，而非反之。

---

## 1. 问题定义

### 1.1 不匹配的具体形式

| 维度 | 研究最优 (RQ5b) | 执行约束 (Engine) |
|------|----------------|-------------------|
| 信号窗口 | 40 日 | 任意，OK |
| 持有期 | **80 日** | 20 日（月频） |
| 调仓节奏 | 每 80 日一次 | 每月一次 |
| 持仓 | 单期单向 | 单期单向 |

差距 = 80d / 20d = **4x**。引擎每月换仓，但信号的最优持有期是 4 个月。

### 1.2 为什么不能改引擎

```
直接改引擎风险:
  - 80d 持有期意味着每年 ~3 次交易 → 样本太少
  - 改变 rebalance 逻辑影响所有已有回测
  - 修撮合/持仓生命周期 → 工程膨胀
  - 没有新增认知价值
```

### 1.3 正确解法

不是改引擎的 rebalance 频率，而是**改变策略的表达形式**——将 80d 持有期策略投影到 20d 调仓框架中。

---

## 2. Execution Adapter 架构

```
Strategy (S=40, H=80)
  │ signal = f(past_40d_return)
  ▼
Execution Adapter
  │ projection: 80d → 4 × 20d overlapping tranches
  ▼
BacktestEngine (monthly rebalance, unmodified)
  │ 每月接收一个 tranche 的信号
  ▼
Position Manager
  │ accumulate 4 tranches, each held 80d
  ▼
Portfolio (aggregated)
```

### 2.1 Position Overlap 机制

核心思想：**不集中持有 80 天，而是每月开一个新仓位，每个仓位运行 80 天后关闭。**

```
时间线 (每月 = 20 交易日):

Month 0:  开仓 Tranche_0 (持有到 Month 3)
Month 1:  开仓 Tranche_1 (持有到 Month 4)  ← 此时持仓: T0 + T1
Month 2:  开仓 Tranche_2 (持有到 Month 5)  ← 此时持仓: T0 + T1 + T2
Month 3:  开仓 Tranche_3 (持有到 Month 6)  ← 此时持仓: T0+T1+T2+T3 (满仓)
          平仓 Tranche_0 (持有期满 80d)
Month 4:  开仓 Tranche_4 (持有到 Month 7)  ← 此时持仓: T1+T2+T3+T4 (满仓)
          平仓 Tranche_1
...
```

**效果**：
- 每月调仓（与引擎兼容）
- 每个仓位持有 80 天（与研究结论一致）
- 稳态下始终持有 4 个重叠仓位
- 每个仓位 1/4 权重，总仓位 100%

### 2.2 信号生成

每月（每个调仓日）：
1. 计算当日所有股票过去 40 日累计收益
2. 选择跌幅最大的 20% 作为该 tranche 的持仓
3. 记录 tranche 的入仓时间戳和入仓价格

### 2.3 平仓逻辑

每个 tranche 在其入仓后的第 80 个交易日平仓：
- 所有仓位一次性卖出
- 计算 realized PnL = 卖出价 / 买入价 - 1

---

## 3. 实现方案

### 3.1 Adapter 接口

```python
class ExecutionAdapter:
    """将 research strategy 投影到 execution constraint space."""

    def __init__(self, signal_h=40, hold_h=80, engine_freq_days=20):
        self.signal_h = signal_h
        self.hold_h = hold_h
        self.freq_days = engine_freq_days
        self.n_tranches = hold_h // freq_days  # 80/20 = 4
        self.tranches = []  # 活跃仓位

    def on_rebalance(self, date, prices, returns):
        """每月调仓时调用."""

        # 1. 生成新 tranche 信号
        new_weights = self._generate_tranche(returns, date)

        # 2. 检查是否有到期的 tranche
        matured = [t for t in self.tranches if t.is_matured(date)]
        for t in matured:
            self._close_tranche(t, prices, date)

        # 3. 开新仓
        self.tranches.append(Tranche(date, new_weights))

        # 4. 返回当月净权重（新开 - 平仓）
        return self._net_weights(prices, date)
```

### 3.2 与现有引擎的集成

不修改 BacktestEngine，而是在其**上层**包装：

```python
class RegimeRouterWithAdapter(RegimeRouterStub):
    """带 Execution Adapter 的 Router."""

    def __init__(self):
        super().__init__()
        self.adapter = ExecutionAdapter(signal_h=40, hold_h=80)

    def run_backtest(self, returns, prices, benchmark):
        # 使用 adapter 投影后的信号
        adapted_signal = self._build_adapted_signal(returns, prices)
        # 传入引擎（月频）
        return super().run_backtest(
            returns, prices, benchmark,
            signal_override=adapted_signal
        )
```

---

## 4. 预期效果

### 4.1 最乐观估计

| 指标 | 研究 (RQ5b) | 引擎 (当前) | Adapter (预期) |
|------|------------|-----------|---------------|
| Sharpe | 0.45 | -0.27 | **0.30-0.40** |
| 年化收益 | ~6% | -3.24% | ~4-5% |
| 最大回撤 | -34% | -45% | -35~40% |

Sharp loss = 0.45 → 0.35 ≈ **22% alpha decay from execution constraints**.

### 4.2 alpha decay 来源分析

| 来源 | 幅度 | 说明 |
|------|------|------|
| 持有期分散 (4 tranches) | -0.05 | 重叠持仓降低集中度 |
| 月频 vs 80d 信号对齐 | -0.03 | 信号可能在月内衰减 |
| 交易成本 | -0.02 | 每月调仓 vs 每季调仓成本 3x |
| **合计 loss** | **~-0.10** | **0.45 → 0.35** |

### 4.3 验证方法

1. 在 RQ5b 自定义回测上增加 adapter layer
2. 对比 adapter 回测 vs 原始 80d 回测
3. 计算 alpha decay 是否在预期范围内

---

## 5. 不做的内容（v1.1 边界）

| 不做 | 原因 |
|------|------|
| 改 BacktestEngine | 无认知增量 |
| 加 regime classifier | v1.0 stub 已完成 |
| 自定义持有期支持 | 用 adapter 替代 |
| 优化器替换 | 等权已验证 |
| 因子修改 | 研究链已关 |
| 信号改进 | S=40/H=80 已确认 |

---

## 6. 实现优先级

```
P0: ExecutionAdapter (signal projection + position overlap)
P1: RegimeRouterWithAdapter (集成到现有架构)
P2: alpha decay verification (对比 RQ5b 基线)
P3: adapter-based regime router v1.1 完整验证

不做: BacktestEngine 修改, 新 classifier, 新因子
```
