# Production Architecture v1: 80d Reversal + Vol Filter

> 从 statistical edge → operational edge
> 目标不是更高的 Sharpe，是在真实市场中活下来

---

## 1. 系统架构

```
                      ┌─────────────────────┐
                      │   Data Layer         │
                      │   Baostock / 实时行情 │
                      └─────────┬───────────┘
                                │ prices
                                ▼
┌─────────────────────────────────────────────────────────┐
│                   Strategy Layer                         │
│                                                         │
│  1. Market State       2. Signal           3. Execution │
│     vol calc →         past_40d_ret →      rebalance    │
│     filter decision    rank bottom 20%     schedule     │
│                        equal weight        (every 80d)  │
└─────────────────────────────────────────────────────────┘
         │                      │                 │
         ▼                      ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                   Risk Overlay                           │
│                                                         │
│  • Max drawdown limit (-30%)     • Position limit (5%)  │
│  • Volatility kill-switch        • Sector cap (30%)     │
│  • Capital at risk per trade     • Cash threshold       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Execution Engine                       │
│                                                         │
│  • Order scheduling (single batch per rebalance)        │
│  • Slippage estimation (vol-adjusted)                   │
│  • Partial fill handling                                │
│  • Trade logging                                        │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   Monitoring                             │
│                                                         │
│  • Sharpe tracker (rolling 12m)                         │
│  • Drawdown alert                                       │
│  • Trade P&L vs expected P&L (parity check)             │
│  • Regime drift detection                               │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 策略参数（已锁定，不修改）

| 参数 | 值 |
|------|-----|
| Signal window | 40d |
| Holding period | 80d |
| Selection | bottom 20% by past return |
| Weighting | equal weight |
| Rebalance | every 80 trading days |
| Vol filter threshold | 70th percentile of 20d rolling vol |

---

## 3. Execution Layer

### 3.1 Rebalance Schedule

- 每 80 个交易日执行一次调仓
- 不提前, 不推迟
- 所有交易在同一日完成 (batch execution)

### 3.2 Order Type

| 方向 | 订单类型 | 说明 |
|------|---------|------|
| 买入 | 市价单/限价单 | 新选中的股票 |
| 卖出 | 市价单/限价单 | 持仓满 80d 的股票 |

### 3.3 Slippage Estimation

```
slippage_est = 0.0005 + 0.5 * spread_ratio
where:
  spread_ratio = (ask - bid) / mid_price
```

### 3.4 Cash Management

- 每次调仓满仓 (除 vol filter 触发时为 100% 现金)
- 不保留现金缓冲

---

## 4. Risk Overlay

### 4.1 Hard Limits（不可违反）

| 规则 | 阈值 | 触发动作 |
|------|------|---------|
| 最大回撤 | -30% (从峰值) | 停止交易, 全部平仓 |
| 单票上限 | 5% | 建仓时强制限制 |
| 行业集中度 | 30% | 建仓时检查 |

### 4.2 Soft Limits（触发警告）

| 规则 | 阈值 | 动作 |
|------|------|------|
| Rolling Sharpe (12m) | < 0 | 打印警告, 标记审查 |
| Vol filter 连续触发 | 连续 2 次 | 打印警告 |
| 执行成本偏离 | > 预算 20% | 打印警告 |

### 4.3 Kill-Switch

手动触发。触发后：平所有仓位，转入现金。恢复需人工确认。

---

## 5. Monitoring

### 5.1 Core Metrics（每日检查）

| 指标 | 频率 | 用途 |
|------|------|------|
| 当前持仓 | 每日 | 确认持仓与预期一致 |
| 未实现 P&L | 每日 | 及时发现异常 |
| Market vol | 每日 | vol filter 决策 |
| 距离下次调仓 | 每日 | 跟踪 schedule |

### 5.2 Trade-Level Logging

每次调仓记录：

```
timestamp
action (buy/sell/skip)
asset code
quantity
price
cost (commission + slippage)
expected exit date
```

### 5.3 Performance Tracking

- Rolling 12-month Sharpe
- 累计 P&L vs 理论 P&L（backtest 预期）
- 执行偏差分析

---

## 6. 已知局限（不修复, 但记录）

| 局限 | 影响 | 原因 |
|------|------|------|
| 101 只样本 | 代表性风险 | 未扩展至全市场 |
| 月频 Barra 数据 | 信号精度 | 日频数据不可得 |
| 无市场冲击模型 | 大资金时偏离 | 当前资金量级下忽略 |
| 无日内交易 | 无法捕捉日内反转 | 策略设计如此 |

---

## 7. 启动检查清单

- [ ] 数据源配置完成 (Baostock / 替代)
- [ ] 策略参数配置完成
- [ ] Vol filter 阈值确认
- [ ] Risk overlay 参数配置
- [ ] Kill-switch 测试完成
- [ ] 日志系统配置
- [ ] 监控 Dashboard 配置
- [ ] 首次调仓模拟运行
