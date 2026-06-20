# 研究终局结论

> **日期**：2026-06-21
> **状态**：研究链闭环。从平台验证到执行边界证明，全部完成。
> **修正版本 v2**：NO-GO ZONE 限定在已测试控制策略族内，不排除高阶控制空间。

---

## 最终结论（精确版本）

A 股市场存在强均值回归 Alpha，集中在 40–80 个交易日的连续时间结构中。
该 Alpha 在固定频率执行（月频）下因采样 aliasing 和相位错位系统性失效。
在简单的相位感知代理（MA sign、自相关）下同样无法恢复。

**但此结论仅限于已测试的低维控制策略空间。**
不排除高阶潜在状态模型（状态空间、小波域、非线性控制）能恢复信号。

---

## 三层研究结构

### Layer 1: Signal Existence ✅

| 证据 | 结果 | 来源 |
|------|------|------|
| Oracle IC | 1.0000 | 平台验证 |
| IC(H) 曲线 | 全部负 | Market Structure v3 |
| Sign Flip | 全部有效 | Cross-Validation |
| RQ5b 回测 | Sharpe +0.45 | 80d 持有期 |

### Layer 2: Execution Feasibility（测试范围内） ✅

| 执行方式 | 结果 | 来源 |
|---------|------|------|
| 固定 80d 持有期 | Sharpe +0.45 | RQ5b（真实交易需此方案） |
| 月频 + BacktestEngine | Sharpe -0.27 | RQ6 |
| 4×20d Tranche overlap | Sharpe -0.23 | ExecutionAdapter |
| Phase-conditioned MA | Sharpe -0.26 | Probe |
| Phase-conditioned AC | Sharpe -1.03 | Probe |

### Layer 3: Advanced Latent Control ❓

| 方法 | 状态 | 说明 |
|------|------|------|
| 小波域相位估计 | ❌ 未测试 | 可能恢复相位结构 |
| 状态空间模型/HMM | ❌ 未测试 | 潜在相位变量推断 |
| 非线性控制策略 | ❌ 未测试 | 连续时间控制 |

---

## NO-GO ZONE 正确定义

> 80-day reversal alpha is unrecoverable within the tested low-dimensional control policy space (fixed-frequency + simple phase proxies), without ruling out higher-order latent-state control formulations.

✔ 可写在 README 中的版本：
- 月频执行 + 反转信号 = 结构性不兼容
- 简单相位感知代理未改善

⚠️ 不可写的版本：
- "Alpha 不可交易"（过度泛化）
- "离散执行系统性破坏信号"（控制空间未展开）

---

## 研究链终局状态

```
Layer 1: Signal existence      ✅ 完成 (Sharpe +0.45, IC结构稳定)
Layer 2: Execution feasibility ✅ 完成 (控制策略族内的NO-GO ZONE已测绘)
Layer 3: Latent state control  ❓ 未探索 (状态空间/小波/非线性控制)

最终判定: 研究链在"已测试假设空间"内闭环。
          不排除扩展假设空间后结论改变。
```
