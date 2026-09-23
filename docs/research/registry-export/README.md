# Registry 导出（2026-09-23）

这些文件是 `data/trading.db` 与 `data/factor_research.db` 里**表格的文本导出**。

## 为什么要导出

那两个 `.db` 被 `.gitignore` 挡着（`data/*.db`），**所以它们从没进过 git**。这台机器被清掉的话，这个仓库唯一的机器可读研究记录就没了。

导出成 JSON/CSV 是为了让记录能进版本库：数据库本身不适合提交，但它的内容值得留。

## 里面是什么

### `research_runs.json` —— Trust Registry

`lab/registry/__init__.py` 写的那张表。**43 行，2026-07-10 → 2026-09-23。**

| 维度 | 分布 |
|---|---|
| 数据源 | **akshare 34 行（真实数据）**、synthetic 9 行 |
| 状态 | success 36、diagnostic 7 |
| 有非空 `ic_mean` | 35 / 43 |
| **`validity`** | **全部 43 行都是 `valid`** |
| **`affected_by` 非空** | **0 行** |

每一行带 `run_id` / `timestamp` / `slice` / `factor` / `factor_params` / `evaluation` / `input_hash` / `report_path` / `warnings`。

**注意最后两行统计**：这张表建了 `validity` 和 `affected_by` 两列、配了 `mark_validity()` 和 `list_affected()` 两个方法 —— 本仓库的 Principle 1（"所有不可信必须机器可识别"）就落在这里。**而这个机制一次都没被调用过。** 见 issue #71。

`input_hash` 只覆盖**声明的输入**，不含代码版本 —— 没有任何一行带 commit。所以一行记录可以被定位到时间，但不能被定位到某个代码版本。

### `factor_evaluation_history.csv` —— 114 行

`factor_name` / `signal_date` / `rank_ic` / `pearson_ic` / `icir` / `coverage` / `n_assets`。

**用之前先核一下**：抽样的行里 `pearson_ic = 0.0`、`coverage = 0.0`、`n_assets = 0` —— 这三个字段看起来是退化的（未计算而非真为零）。`rank_ic` 和 `icir` 有值。

### 纸面交易的运行状态

`trading_sessions.csv`（620）、`trading_pnl_history.csv`（2829）、`trading_orders.csv` / `trading_trades.csv` / `trading_signals.csv`（各 60）、`trading_events.csv`（3）。

这是 `live_runner` 循环跑出来的模拟状态。**不是研究结论**，留档备用。

## 两个使用提醒

1. **`research_runs.json` 的最后一行是验证运行产生的。** `run_1790145888_91bd43cb82120133`（2026-09-23T06:44:48，`synthetic_momentum_1m`）来自一次外部可用性测试执行的 `main.py factor` 命令，不是一份研究结论。它对应的报告文件没有保留。

2. **这 43 行里哪些结论还成立，取决于它跑在什么时候。** 2026-09 的清算修掉了空 alpha 信号、成交时点、ST 全样本过滤等问题（见 `docs/HANDOVER.md`）—— **在那之前的回测性能数字全部作废。** 分档见 issue #72。

## 重新生成

```bash
# 需要 .venv312
python - <<'PY'
import sqlite3, json, csv
# 见本目录文件的来源；trading.db 的表：research_runs / sessions / orders
# / trades / signals / pnl_history / events
# factor_research.db 的表：factor_evaluation_history
PY
```

原始数据库仍在 `data/trading.db` 与 `data/factor_research.db`（本机）。
