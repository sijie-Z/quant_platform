# Ablation Study #003: Cluster Representative Study

> **日期**：2026-06-20
> **状态**：已完成
> **实验 ID**：ABLATION-003
> **前置**：ABLATION-001, ABLATION-002

---

## 背景

Ablation #002 证明 Top 8 因子可聚类为 4 个独立 Alpha 信号簇。本实验进一步验证：**用每个簇的一个代表因子构建组合，能恢复 Top 8 的多少收益？**

**研究问题**：Alpha 的本质是因子数量还是独立信号数量？

---

## 方法

### 实验设计

| 策略 | 因子 | 预期 |
|------|------|------|
| All 26（基准最差） | 全部 26 因子等权 | Sharpe 最低 |
| Top 8（基准最优） | |C| 排名前 8 等权 | 基线 Sharpe = +0.464 |
| **Cluster-4** | rsi_14d + trend_stage + momentum_12m + turnover_20d | 能否接近 Top 8？ |
| Cluster-4 + momentum_1m | 4 代表 + 短期簇第二因子 | 簇内是否有互补信息？ |
| Cluster-4 + breakout_proximity | 4 代表 + 短期簇第三因子 | 同上 |

### 回测参数
- 合成数据，300 只股票，2021-2024，alpha_strength=0.03
- 等权融合因子 → Rank 归一化 → EqualWeight 优化器
- 月频调仓，佣金 0.03% + 印花税 0.1%（卖）+ 滑点 0.1%

---

## 结果

| 策略 | 因子数 | Sharpe | 总收益 | 最大回撤 | vs Top 8 |
|------|--------|--------|--------|---------|---------|
| **Top 8** | 8 | **+0.464** | +63.27% | -26.6% | 100% |
| Cluster-4 + breakout | 5 | +0.416 | +57.22% | -26.5% | **89.7%** |
| Cluster-4 + mom_1m | 5 | +0.415 | +57.40% | -24.6% | 89.4% |
| **Cluster-4** | **4** | **+0.381** | **+53.87%** | **-25.1%** | **82.1%** |
| All 26 | 26 | -0.779 | -46.85% | -63.7% | -168% |

---

## 结论

### 结论 1：4 个独立信号解释了 Top 8 的 82% 收益

用一半的因子数恢复了 82.1% 的 Sharpe（0.381 vs 0.464），最大回撤几乎一致（-25.1% vs -26.6%）。这验证了核心假设：**Alpha 的本质是独立信号数量，不是因子数量。**

### 结论 2：同簇因子并非完全冗余

Cluster-4 + momentum_1m（短期簇第二因子）将 Sharpe 从 0.381 提升至 0.415（+8.9%），说明 rsi_14d 和 momentum_1m 在短期信号簇内有互补信息，并非完全冗余。

### 结论 3：26 → 8 → 4 的逐步收敛成立

```
All 26 (Sharpe -0.779)   → 噪音因子稀释 Alpha
    ↓
Top 8 (Sharpe +0.464)    → Factor Ranking 有效
    ↓
Cluster-4 (Sharpe +0.381) → Alpha ≈ 独立信号数量
```

---

## Alpha Discovery v1 最终结论

本实验是三轮 Ablation 研究的收尾。Alpha Discovery v1 的完整发现链如下：

### 发现链

```
26 原始因子
    ↓ Factor Ranking (ABLATION-001)
12 个因子 |IC| < 0.005（噪音）
8 个因子有实际预测力
    ↓ Correlation Clustering (ABLATION-002)
8 因子 = 4 个独立 Alpha 簇
    ↓ Representative Selection (ABLATION-003)
4 个代表因子解释 Top8 82% 收益
```

### Alpha 分类体系（Alpha Taxonomy）

| 信号簇 | 代表因子 | 信号含义 | 独立程度 |
|--------|---------|---------|---------|
| 短期反转/超买超卖 | rsi_14d | RSI 均值回归 | 中等 |
| 中期趋势 | trend_stage | 120 日价格位置 | 高（与短期信号互补） |
| 长期趋势 | momentum_12m | 12 月累计收益 | 高（时间尺度独特） |
| 流动性 | turnover_20d | 换手率/关注度 | **最高（最独立）** |

### 对项目架构的影响

**Alpha Pipeline 应升级为：**

```
以前：
  26 Factors → Equal Weight → Portfolio

现在：
  26 Factors → Ranking → Clustering → Representative Selection → Weighted → Portfolio
```

---

## 后续研究

### v1.2 候选方向

1. **真实 A 股验证**：在沪深 300 或全 A 数据上重复 Ablation #001-#003
2. **Cluster-4 + 1 优化**：找到最佳的 5 因子组合（4 独立信号 + 1 互补因子）
3. **ICIR 加权替代等权**：在信号簇层面使用 ICIR 加权而非簇内等权

### 已知限制

- 全部实验基于合成数据（alpha_strength=0.03），结果在真实市场可能不一致
- Candle 因子（kup/klow/kmid/klen/ksft）在合成数据中因 OHLC 生成简单而表现弱
