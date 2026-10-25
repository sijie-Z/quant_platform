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
| #001 | [Factor Selection](ABLATION_001_FACTOR_SELECTION.md) | 等权融合所有因子会显著稀释 Alpha。Top 8 vs All 26 差距 **1.24 Sharpe** | ⚠️ **数字作废（#60）** — 产出于空信号缺陷修复前 |
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

---

## 真实数据研究（2026-10）

| 文档 | 内容 |
|---|---|
| [REAL_DATA_EVIDENCE_2026-10-24.md](REAL_DATA_EVIDENCE_2026-10-24.md) | 101 池与宽池的真实数据证据、2026 holdout、口径审计（§20）、成本压力测试（§21） |
| [DEPLOYMENT_DECISION_2026-10-25.md](DEPLOYMENT_DECISION_2026-10-25.md) | 部署决策备忘、复核门槛、90 天执行计划 |
| [FREEZE_MANIFEST_2026-10-25.md](FREEZE_MANIFEST_2026-10-25.md) | 冻结策略清单（sha256/tag）、变更与终止规则 |
| [DATA_DEFECT_LEDGER_2026-10-25.md](DATA_DEFECT_LEDGER_2026-10-25.md) | 15 项历史数据缺陷台账与处置状态 |
| [PREREG_ETF_ROTATION_2026-10-25.md](PREREG_ETF_ROTATION_2026-10-25.md) | ETF 动量轮动预注册（数据、参数、验收标准） |
| [ETF_ROTATION_RESULTS_2026-10-25.md](ETF_ROTATION_RESULTS_2026-10-25.md) | ETF 试点结果：OOS 达标、2026 holdout 失败 → 冻结 |
| [PREREG_CB_DOUBLE_LOW_2026-10-25.md](PREREG_CB_DOUBLE_LOW_2026-10-25.md) | 可转债双低轮动预注册（下一个收益来源；尚未跑回测） |
| [EXTERNAL_REVIEW_PACKAGE_2026-10-25.md](EXTERNAL_REVIEW_PACKAGE_2026-10-25.md) | 给外部评审的自包含材料与处置记录（§13） |
