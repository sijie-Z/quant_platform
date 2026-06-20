# RQ6: Execution-Aware Horizon Re-optimization

> **冻结日期**：2026-06-21
> **前序**：Adapter 实证 80d alpha ≠ 4 × 20d alpha，时间分解非线性
> **核心问题**：在真实 execution constraint 下，最优反转窗口 H* 是多少？

---

## 1. 研究背景与动机

### 1.1 已建立的发现

| 发现 | 证据 |
|------|------|
| A 股全区间反转 | IC(H) 在全部 10 个时间尺度为负 |
| 研究最优 H=80 | RQ5b: Sharpe=0.45 (自定义回测, 80d 持有) |
| 执行破坏 alpha | BacktestEngine 月频: Sharpe=-0.27 (gap=0.72) |
| Adapter 仅部分修复 | Position overlap: 从 -0.27 到 -0.23 (修复 ~15%) |
| 80d ≠ 4×20d | 时间分解非线性, 信息损失不可逆 |

### 1.2 核心矛盾

```
Research optimum:  H*=80,  Sharpe=0.45
Execution optimum: H=20 (月频), Sharpe=-0.27

两个最优不在同一个 horizon 上。
问题: 哪个 H 在 execution constraint 下仍保持正 Sharpe？
```

### 1.3 假设

- **H₀**：研究最优和执行最优在同一 horizon（H=80 同时是 execution-fixed point）
- **H₁**：执行约束将最优 horizon 拉向更短周期（H* < 80）
- **H₂**：存在一个 H* 使得 Research Sharpe(H) 和 Execution Sharpe(H) 的差距最小

---

## 2. 实验设计

### 2.1 扫描参数

| 参数 | 值 |
|------|-----|
| H (持有期) | [10, 20, 30, 40, 60, 80, 120] |
| S (信号窗口) | H/2 (信号窗口保持为持有期的一半, 基于 RQ5b 发现) |
| Rebalance | 月频 (与 BacktestEngine 一致) |

### 2.2 每个 H 运行两种回测

**回测 A — Research（理想执行）**：
- 与 RQ5b 一致的自定义回测
- 持有期 = H，调仓周期 = H
- 买入 bottom 20% → 持有 H 天 → 卖出
- 目的：测量信号在理想执行下的 Sharpe（不受约束干扰）

**回测 B — Execution（真实约束）**：
- 通过 BacktestEngine 月频调仓
- 信号窗口 = H/2
- 每月买卖一次
- 目的：测量同一信号在真实执行下的 Sharpe

### 2.3 评价指标

| 指标 | 公式 | 意义 |
|------|------|------|
| Research Sharpe | Sharpe(return_series_A) | 信号质量, 不受执行干扰 |
| Execution Sharpe | Sharpe(return_series_B) | 信号在真实约束下的可实现质量 |
| Alpha Decay | Research − Execution | 执行带来的损失 |
| Decay Ratio | (Research − Execution) / \|Research\| | 标准化损失率 |
| Turnover | 平均单边换手率 | 执行成本的代理变量 |

### 2.4 判定规则

执行最优 H* 满足：
1. Execution Sharpe > 0 (正收益)
2. Decay Ratio < 50% (不超过一半的信号被执行破坏)
3. Turnover < 50% (换手率合理)

如果 H* 显著偏离 80（如 H*=40），则 **拒绝 H₀，接受 H₁**——执行约束确实将最优 horizon 拉向更短周期。

---

## 3. 分析方法

### 3.1 双曲线分析

```
Sharpe
  │
  │    Research Sharpe(H)    ← 随 H 递增 (信号积累)
  │      ↗
  │     ↗    Execution Sharpe(H)  ← 先增后减 (执行约束)
  │    ↗      ↙
  │   ↗    ↙
  │  ↗  ↙
  │ ↗ ↙
  │╳
  └───────────────────── H
     H* ← 两条曲线的交叉点是 system optimum
```

### 3.2 预期结果（假设 H₁ 成立）

| H | Research Sharpe | Execution Sharpe | Decay | Decay Ratio |
|---|:-:|:-:|:-:|:-:|
| 10 | 低 | 低 | 小 | — |
| 20 | 中 | 中 | 中 | ~60% |
| 40 | 高 | **~0.30** | 小 | **~30%** ← 可能是 H* |
| 60 | 高 | ~0.15 | 中 | ~50% |
| 80 | 最高 | -0.27 | 大 | ~160% |
| 120 | 中 | 低 | 大 | — |

---

## 4. 执行计划

```
Step 1  冻结协议 ✅
Step 2  实现 horizon 扫描脚本
Step 3  运行 7 × 2 = 14 个回测
Step 4  生成 Research vs Execution 双曲线
Step 5  识别 H*
Step 6  报告: Execution-Aware Horizon Discovery
```

## 5. 不做的内容

❌ 改 BacktestEngine
❌ 加新因子/信号
❌ regime 分类器
❌ Adapter 优化
❌ position overlap

## 6. 风险控制

| 风险 | 应对 |
|------|------|
| H 样本太少 (7 个) | 分辨率足够的, 不需要更细致扫描 |
| Research vs Execution 回测不可比 | 信号源相同, 仅执行方式不同, 可比性成立 |
| 找到 H* 后想直接上线 | 这是 exploratory finding, 不是 trading signal |
