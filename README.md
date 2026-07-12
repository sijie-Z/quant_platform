# A-Share Multi-Factor Quant Research Platform

<p align="center">
  <img src="https://img.shields.io/badge/Research%20OS-v0.1-brightgreen" alt="Research OS">
  <img src="https://img.shields.io/badge/Milestone-M4-blue" alt="Milestone">
  <img src="https://img.shields.io/badge/Registry-27%20runs-success">
</p>

**不是回测框架，是 AI 原生量化研究操作系统。**

Trust → Knowledge → Alpha → Production

---

## 项目已完成

### M1 — Research OS MVP（v0.1-research-os, 锁定）

3 个诚实因子运行，Trust Metadata 全程记录。

| Factor | IC | ICIR | IC>0% | PIT | Adjust |
|--------|-----|------|-------|------|--------|
| volatility_20d | +0.0334 | +0.1181 | 54% | false | hfq |
| volatility_60d | +0.0349 | +0.1144 | 53% | false | hfq |
| momentum_12m | +0.0114 | +0.0515 | 54% | false | hfq |
| reversal | +0.0208 | +0.0979 | 53% | false | hfq |

**能力证明**：
- ✅ Registry：25+ 条机器可读研究记录（全部 pit=false, DSR=insufficient_trials）
- ✅ Trust Metadata：数据源、复权方式、偏差警告持久化
- ✅ Auto Report：WARNING 从 Registry 字段自动生成，非人工编写
- ✅ Cross-Run Comparison：一句 SQL 回答跨因子知识问题
- ✅ Path Reuse：新增因子零框架改动
- ✅ Zero Legacy Modification：4 万行旧代码一个字节未动

### M2 — Factor Diagnostics（完成）

对 Low Vol vs Momentum 做逐年 IC 诊断。

**结论**：Low Vol 好于 Momentum 因为 **年际稳定性**——6 年从未转负，Momentum 波动大且 2026 年转负。

### M3 — Factor Zoo（完成）

10 个因子横向比较。

**核心发现**：**A 股是反转市场。** 所有动量因子（1m/3m/6m）IC 均为负，低波动率因子和反转因子最强。

### M3.5 — Candidate Validation（完成）

volatility_20d 得分 87/100，reversal 65/100。vol_20d 在牛市和熊市均有效。

### M4 — Strategy Validation（进行中）

**目标**：IC 能否变成真实收益？

**M4.1 组合回测（✗ 完成）**：volatility_20d Top20 Long/Bottom20 Short，月调仓，包含佣金 3bp + 印花税 10bp + 滑点 5bp。结果：

| 指标 | 值 |
|------|-----|
| CAGR | -18.05% |
| Sharpe | -0.96 |
| Max Drawdown | -69.01% |
| Excess vs CSI300 EW | -30.20% |

**结论**：IC=+0.0334 的信号 **不能直接** 转化为真实组合收益。这正是 Research OS 的价值——不粉饰数据。信号稳定性（年际正）不保证策略可投资性（组合层成本/换手/波动吃掉 alpha）。

---

## 下一步

- [ ] **M4.2**：Long-Only variant（Top 10% equal-weight）
- [ ] **M4.3**：Walk-Forward OOS（2021-2023 train, 2024-2026 test）
- [ ] **M4.4**：Factor combination（vol_20d + reversal + quality）
- [ ] **M5**：Production candidate pipeline

---

## 架构

```
Trust（可信数据层：akshare hfq + CSI300）
  ↓
Knowledge（Registry: 27 records, SQL-queryable）
  ↓
Research（3 Factor Diagnostics + 10 Factor Zoo + 2 Candidate Validation）
  ↓
Strategy Validation（M4 Portfolio Backtest）
  ↓
Alpha（未来）
  ↓
Production（未来）
```

```
Capability Layer (framework/contracts/)  →  开源能力层
Knowledge Layer (lab/)                   →  私有研究资产
Production Layer (prod/)                 →  极简生产系统
```

---

## 数据源

| 名称 | 类型 | 用途 |
|------|------|------|
| AkShare (TX) | 免费 A 股 | v0.1 全部 Run 的数据源 |
| hfq 后复权 | 复权方式 | 截面排序安全，按 IPO 基准 |
| CSI300 | 成分股 | 全部 Run 的 universe |

---

## 运行

```bash
# Setup
python -m venv .venv
.venv\Scripts\python -m pip install akshare pandas numpy pyarrow --no-user

# Run a factor
.venv\Scripts\python -c "import sys; sys.path.insert(0,'D:/Desktop'); from quant_platform.lab.runs.first_honest_research_run import run; run()"

# Query Registry
.venv\Scripts\python -c "import sqlite3, json; c=sqlite3.connect('data/trading.db'); [print(json.loads(r[0]).get('icir')) for r in c.execute(\"SELECT evaluation FROM research_runs WHERE status='success'\")]"
```

## Registry 查询

```sql
-- 按 ICIR 排名所有因子
SELECT factor, json_extract(evaluation,'$.ic_mean') as ic, json_extract(evaluation,'$.icir') as icir
FROM research_runs WHERE status='success'
ORDER BY CAST(json_extract(evaluation,'$.icir') AS REAL) DESC;

-- 跨因子比较（含信任元数据）
SELECT factor, json_extract(evaluation,'$.icir'), json_extract(universe_meta,'$.pit'), json_extract(data_meta,'$.adjust')
FROM research_runs;

-- 查看失败 Run
SELECT run_id, status, reason FROM research_runs WHERE status='failed';
```

---

## 治理

| 文件 | 内容 |
|------|------|
| `CONSTITUTION.md` | 7 条原则 + Flywheel + Anti-Goals |
| `ARCHITECTURE.md` | 四包 monorepo + 依赖方向 |
| `ROADMAP.md` | Milestone 线 + 版本准入准则 |
| `docs/ADR/` | 4 条长期决策记录 |
| `docs/MILESTONE_1.md` | M1 总结 |
| `docs/MILESTONE_1_RETROSPECTIVE.md` | M1 复盘（4 个关键问题 + 证伪假设） |

---

## 许可

MIT License — 仅供教学和研究使用，不构成投资建议。
