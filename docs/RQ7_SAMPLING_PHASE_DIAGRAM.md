# RQ7: Sampling–Holding Phase Diagram

> **冻结日期**：2026-06-21
> **前序**：RQ6 证明月频执行使所有 H 的 Sharp 为负——采样频率破坏信号
> **核心假设**：月频（20d）低于反转信号（40-80d）的 Nyquist 频率，导致 aliasing
> **核心问题**：临界采样频率 f_critical 是多少？执行流形（execution manifold）在哪里？

---

## 1. 理论基础

### Nyquist 采样定理
- 反转信号的自然周期 T ≈ 40-80d（IC 曲线确认）
- 月频采样间隔 Δt = 20d
- 当 Δt > T/2 时，信号发生 aliasing（频率混叠）
- T/2 = 20-40d → 月频采样（20d）**刚好处于 aliasing 边界**

### 实验假设
- **H₀**：采样频率不影响 Sharpe（信号是 frequency-independent）
- **H₁**：存在临界采样频率 f*，低于此频率信号被破坏

---

## 2. 实验空间

2D 网格：

```
Sampling Interval (f) ∈ [1d, 5d, 10d, 20d, 40d]
Holding Period (H)    ∈ [5, 10, 20, 40, 80, 120, 160]

信号窗口 S = max(5, H // 2)
```

总计 5 × 7 = 35 个单元。每个单元运行一次完整的 position overlap 回测。

---

## 3. 实验层

### Layer 1: 理想连续基准（f=1d）
每日调仓，无约束。定义"理论信号空间"的上限。

### Layer 2: 离散采样曲面（全网格）
对每个 (f, H) 运行 position overlap 回测：
- 每 f 天开一个新 tranche
- 每个 tranche 持有 H 天
- 聚合所有活跃 tranche → 每日组合收益

### Layer 3: Aliasing 诊断
Signal Reconstruction Loss = Sharpe_ideal(H) − Sharpe_exec(f, H)

---

## 4. 评价指标

| 指标 | 意义 |
|------|------|
| Sharpe(f, H) | 核心——每个单元的年化 Sharpe |
| Reconstruction Loss | 理想 vs 执行的 Sharpe 差距 |
| Critical f* | 使 Sharpe 从正转负的临界采样间隔 |
| Turnover | 每单元的年化换手率 |

---

## 5. 判定规则

- 如果 f* < 20d：月频采样确实低于 Nyquist 频率，需要更高频采样
- 如果 f* > 40d：信号对采样频率不敏感，失败另有原因
- 如果 Sharpe(f, H) 在 f ≤ 10d 且 H ≈ 80 处为正值且显著：**确认采样频率是核心瓶颈**

---

## 6. 输出

1. **Phase Diagram Heatmap**：Sharpe(f, H) 二维矩阵
2. **Optimal Ridge**：每个 f 的最佳 H
3. **f_critical**：Sharpe 首次变为负的临界采样间隔
4. **结论**：A 股反转 Alpha 的执行流形

---

## 7. 执行计划

```
Step 1  冻结协议 ✅
Step 2  实现 phase diagram sweep 脚本
Step 3  运行 35 个单元
Step 4  输出 heatmap + 关键指标
Step 5  报告
```
