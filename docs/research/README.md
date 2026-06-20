# Research — Quant Platform v1.1

> 本目录记录平台的系统性量化研究实验。
> 每份研究报告包含：假设、方法、结果、结论、对项目的影响。

---

## 研究链：Alpha Discovery v1

从 26 个原始因子到 4 个独立 Alpha 信号的完整发现过程：

```
26 原始因子
    ↓ Factor Ranking (ABLATION-001)
12 个噪音因子被识别
8 个因子有实际预测力
    ↓ Correlation Clustering (ABLATION-002)
8 因子 = 4 个独立 Alpha 簇
    ↓ Representative Selection (ABLATION-003)
4 个代表因子解释 Top8 82% 收益
```

### 研究报告

| 编号 | 名称 | 核心发现 | 状态 |
|------|------|---------|------|
| #001 | [Factor Selection](ABLATION_001_FACTOR_SELECTION.md) | 等权融合所有因子会显著稀释 Alpha。Top 8 vs All 26 差距 **1.24 Sharpe** | ✅ |
| #002 | [Correlation Clustering](ABLATION_002_CORRELATION_CLUSTERING.md) | 8 个因子 = **4 个独立 Alpha 信号簇**：短期反转、中期趋势、长期趋势、流动性 | ✅ |
| #003 | [Cluster Representatives](ABLATION_003_CLUSTER_REPRESENTATIVES.md) | 4 个代表因子解释 Top 8 的 **82% 收益**（Sharpe 0.381 vs 0.464） | ✅ |

### Alpha 分类体系（Alpha Taxonomy）

| 信号簇 | 代表因子 | IC | 含义 |
|--------|---------|-----|------|
| 短期反转/超买超卖 | rsi_14d | 0.021 | RSI 均值回归 |
| 中期趋势 | trend_stage | 0.021 | 价格在 120 日历史区间位置 |
| 长期趋势 | momentum_12m | 0.013 | 12 月累计收益（跳过 1 月） |
| 流动性 | turnover_20d | 0.011 | 换手率/关注度 |

---

## 研究原则

1. **可复现**：每份报告记录完整实验参数和数据来源
2. **单变量**：每次只改变一个变量
3. **有对照组**：每份报告都有清晰的基准（Benchmark）
4. **结论导向**：每份报告回答一个具体的研究问题
