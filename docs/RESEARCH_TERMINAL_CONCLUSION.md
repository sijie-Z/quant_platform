# 研究终局结论

> **日期**：2026-06-21
> **状态**：研究链闭环。从平台验证到市场结构测绘到执行边界证明，全部完成。

---

## 最终结论（论文级摘要）

A 股市场存在强均值回归 Alpha，集中在 40–80 个交易日的连续时间结构中。该 Alpha 不可分解为独立的短周期信号。所有离散执行方案（月频调仓、Tranche 重叠、固定持有期）均因采样引入的相位错位（phase misalignment）而系统性破坏底层信号结构。

因此，观察到的 Alpha 不是因子现象，而是**时间相位依赖的市场过程**。当前系统的限制因素不是信号发现，而是连续市场动力学与离散执行框架之间的不可兼容性。

---

## 三层研究结构

### Layer 1: Signal Existence（信号存在性）✅

| 证据 | 结果 | 来源 |
|------|------|------|
| Oracle IC | 1.0000 | 平台验证 |
| IC(H) 曲线 | 全部 10 个时间尺度为负 | Market Structure v3 |
| Sign Flip | 4/4 簇翻转后 Sharpe 改善 | Cross-Validation |
| RQ5b 自定义回测 | Sharpe +0.45 | 80d 持有期反转 |

**结论**：A 股反转 Alpha 真实存在。

### Layer 2: Market Structure（市场结构）✅

| 测量 | 结果 | 来源 |
|------|------|------|
| 最优信号窗口 | 40d（过去 40 日涨跌幅） | RQ5b |
| 最优持有期 | 80d | RQ5b |
| 反转强度 | 80d 最强（IC=-0.081, t=-4.58） | Market Structure v3 |
| 状态依赖 | Bull/Bear 有效, Sideways 无效 | Regime Discovery v4 |
| 市值差异 | 小盘反转 > 大盘反转, 差异不大 | Market Structure v3 |

**结论**：Alpha 是 40–80d 的集中时间反转结构，不是因子数值，而是**时间结构体**。

### Layer 3: Execution Feasibility（执行可行性）❌

| 实验 | 结果 | 含义 |
|------|------|------|
| 原始信号 + 月频引擎 | Sharpe -0.27 | 持有期不匹配 |
| ExecutionAdapter | Sharpe -0.23（仅改善 15%） | 80d ≠ 4 × 20d |
| RQ6 Horizon Scan | 所有 H 的月频 Execution Sharpe 为负 | 无 H* 可使月频反转盈利 |
| RQ7 Phase Diagram | 35/35 (f,H) 单元全负 | 所有频率×持有期组合均失效 |

**结论**：该 Alpha 不可被当前离散执行框架捕捉。

---

## 统一解释：Phase Collapse（相位坍缩）

所有执行失败可以统一归因于一个机制：

```
市场真实过程：price(t) = mean reversion with memory ~80d
执行系统观察：observe(t) every 20d (monthly)
             rebalance discrete
```

信号是连续波，执行是离散点。当采样周期（20d）接近信号周期（40-80d）的一半时，发生 **aliasing + phase cancellation**——信号频谱被改变，相位信息丢失，Alpha 结构被破坏。

ExecutionAdapter 试图通过 Tranche 重叠恢复信号，但这本质上是将 80d 完整周期拆分为 4 段 20d，改变了信号的频谱结构，因此不可恢复。

---

## 最终系统分类

```
                    ┌──────────────────────────────┐
     Signal Layer   │  Alpha 存在 (Sharpe +0.45)   │ ✅ 完成
                    │  40-80d 反转结构已测清        │
                    ├──────────────────────────────┤
     Regime Layer   │  Bull/Bear 有效               │ ✅ 完成
                    │  Sideways 无效                │
                    │  α = Signal × I(有方向)       │
                    ├──────────────────────────────┤
     Execution      │  月频 → aliasing 破坏信号     │ ❌ 不可行
     Layer          │  Adapter → 相位信息不可恢复   │
                    │  Tranche → 频谱结构改变       │
                    └──────────────────────────────┘
```

---

## 对后续工作的含义

| 方向 | 可行性 | 理由 |
|------|--------|------|
| 继续优化 adapter | ❌ | 连续→离散的相位损失不可恢复 |
| 改 BacktestEngine | ❌ | 无认知增量，工程膨胀 |
| 找新因子 | ❌ | 因子不是限制因素 |
| 接受结论，转向**连续时间执行系统** | ✅ | 唯一有信息增量的方向 |
| **80d 持有期集中交易** | ✅ | RQ5b 已验证可行 |
| **事件驱动（非定时）执行** | ⚠️ | 需要原型验证 |

---

## 研究链终局状态

```
                    研究链完成
                        │
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
平台验证 ✅       市场结构测绘 ✅      执行边界证明 ✅
Oracle IC=1.0    IC(H)全负, 80d最优   月频全负, 相位坍缩
Known Alpha      反转结构, Regime Dep.  Adapter不可恢复
                 α = Signal × I(方向)   Phase Collapse
                        │
                        ▼
              最终结论: 连续市场动力学与
              离散执行框架的不可兼容性
```

---

> **一句话终局**：A 股存在可预测的 80d 反转结构，但该结构是连续时间相位现象。任何月频或离散化执行都会系统性破坏信号——问题不是 Alpha discovery，而是 continuous-time process 与 discrete execution 之间的不可兼容性。
