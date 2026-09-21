# No-Lookahead Contract（前视偏差零容忍契约）

> 这是本平台所有量化研究代码必须遵守的铁律。任何违反此契约的代码都是 bug。
> 回测的高收益只有在零前视偏差的前提下才有意义。

> ⚠️ **当前有两条未满足**（2026-09-21 复核）：**第八条**（ST 用公告日期）是真实的未来函数，
> 且方向上是抬高回测的；**第七条**（行业用生效日期）未满足，但方向上只是陈旧、不构成未来函数。
> 两条的现状、证据与修复方向都在各自小节里。
>
> 本次只逐条追踪了第二、第六条（alpha 权重与 IC shift 链条，未发现违反）。
> 第一、三、四、五条**本次没有重新复核**，不要按"未列出即无问题"理解。

---

## 第一条：价格因子 — 只能用 signal_date 及之前的数据

**规则**：计算因子值时，只使用 signal_date 当日及之前已产生的价格数据。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `factors/technical.py` 所有 compute() 方法 | 基于 prices DataFrame 当前行及之前行的滚动/窗口计算 |
| `factors/processing.py` 横截面处理 | 每日期独立处理，不跨日期泄漏 |

**不允许的写法**：
```python
# ❌ 使用了未来数据
factor = prices.shift(-1).rolling(21).mean()

# ❌ 全量数据计算后再切片
all_factors = some_function(all_prices)
signal = all_factors.loc[signal_date]  
# 如果 some_function 内部用了未来数据，这行切片不管用
```

**正确的写法**：
```python
# ✅ 点-in-time：每个日期只用该日期及之前的数据
factor = prices.rolling(21).mean()
# rolling 默认只用到当前行，天然因果
```

---

## 第二条：Alpha 信号权重 — 只能用历史上的 IC

**规则**：IC/ICIR 加权计算某期信号时，只使用该期之前的 IC 历史数据。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `alpha/combination.py:75-81` | `ic_hist = ic_s[ic_s.index < date]` — 严格小于 signal_date |
| `alpha/combination.py:153-159` | 同上，ICIR 加权路径 |

**不允许的写法**：
```python
# ❌ 用了包含 signal_date 及未来的 IC 数据
full_ic = rank_ic(factor, forward_returns)
weights = full_ic.rolling(252).mean()  # rolling 泄漏了当日 IC
```

**正确的写法**：
```python
# ✅ 只用 signal_date 之前的 IC
ic_hist = ic_series[ic_series.index < signal_date]
mean_ic = ic_hist.tail(252).mean()
```

---

## 第三条：基本面因子 — 只能用 report_date，不能用 fiscal_period_end

**规则**：基本面数据的使用受限于**报告发布日期（report_date / publish_date）**，而不是财务周期结束日（fiscal_period_end）。

A 股实际情况：季报结束后 1-2 个月才发布。用 fiscal_period_end 会使用实际尚未公开的数据。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `data/providers/synthetic.py:467-468` | publish_date = qdate + 40-50 天（模拟 A 股披露延迟） |
| `data/providers/synthetic.py:494-497` | ffill 后保留 publish_date 列 |

**不允许的写法**：
```python
# ❌ 用 fiscal_period_end 作为可用日期
financials.loc['2024-03-31']  # 2024Q1 的财务数据在当年 4月底才公布
```

**正确的用法**：
```python
# ✅ 用 publish_date 过滤
mask = financials['publish_date'] <= signal_date
```

---

## 第四条：Walk-Forward 验证 — 每个 fold 内用 train-only 数据重算信号

**规则**：Walk-Forward 的每个 fold 中，test 期的信号必须用 train 期的数据重新计算，不能使用全量预计算信号。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `backtest/walkforward.py:47-71` | 每个 fold 内传入 `factors`/`alpha_kwargs` 参数，基于 train 数据重新计算信号 |

**允许的写法**：
```python
for train_idx, test_idx in folds:
    train_data = data.iloc[train_idx]
    # 用 train_data 重新计算因子和信号
    train_factors = compute_factors(train_data)
    train_signal = generate_alpha(train_factors)
    # 在 test 期评估
    evaluate(train_signal, test_data)
```

**不允许的写法**：
```python
# ❌ 用全量预计算信号做 Walk-Forward
signal = compute_signal(data)  # 全量数据计算
for train_idx, test_idx in folds:
    test_signal = signal.iloc[test_idx]  # 但 signal 制作时用了 test 数据
```

---

## 第五条：合成数据嵌入式 Alpha — 仅用于演示

**规则**：`embedded_alpha=True` 只在面试演示和集成测试中使用。研究因子质量时必须关闭（`embedded_alpha=False`），此时收益是纯噪声。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `data/providers/synthetic.py:231-280` | `if self.embedded_alpha:` 条件分支 |
| `data/providers/synthetic.py:13-15` | 文档警告：`Never use embedded_alpha=True to validate strategy performance` |
| `config/default.yaml:16` | `embedded_alpha: true`（默认开，面试友好） |

---

## 第六条：IC 计算 — shift 链条正确

**规则**：IC = 因子值(t) 与 未来收益(t→t+1) 的相关性。因子值用截至 t 的价格计算，收益用 t+1 减去 t 的价格。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `factors/evaluation.py:21-65` | factor 在 t，returns 是 t→t+1 的收益 |

**已验证无 shift 链条错误**：pipeline 中 returns 已做 shift(-1)（t→t+1 收益率），IC 计算不再重复 shift。

---

## 第七条：行业分类 — 用生效日期

**规则**：行业分类用 effective_date，不是静态标签。当股票在回测期间发生行业变更时，使用变更生效后的分类。

**状态**：⚠️ **当前不成立**（2026-09-21 复核）。

`neutralize` 接受的是 `sector_map: Series`（asset → sector）并在每个日期上复用同一份映射
（`factors/processing.py:99`、`:126-129`），而 `main.py` 传进去的是 `metadata["sector"]`——
provider 的**样本起始分类**。实测（synthetic，120 只，2023-01-01 ~ 2025-12-31，seed 17）：
38 只（32%）在样本内发生行业变更，它们**全部日期的中性化都用的是起始行业**。

**方向上不是未来函数**：用变更生效**之前**的分类是**陈旧**，不是偷看未来——它永远不会读到 t
之后的分类。所以这一条既不会抬高回测业绩，也不违反"无未来函数"；它只是让中性化对那 32% 的
股票在变更后的区间内失效。

**已在代码中实现的位置（部分）**：

| 位置 | 实现 |
|------|------|
| `data/providers/synthetic.py:605-641` | 行业分类附带 effective_date，~5%/半年变更率 |
| `data/pipeline.py:188-215` | `get_industry_map(as_of_date)` 按 effective_date 返回当期分类 |
| ⚠️ 缺口 | `get_industry_map` **没有任何生产调用方**（grep 仅命中自身定义与测试）；`neutralize` 也不接受按日期的映射。要真正做到这一条，需要把按日期的行业映射从 provider 一路传到 `process_factor` |

---

## 第八条：ST 状态 — 用公告日期

**规则**：ST 标记用 announce_date，不是 trigger_date。交易所发布 ST 公告之前，市场不知道。

**状态**：⚠️ **当前不成立，且这一条是真实的未来函数**（2026-09-21 复核）。

`_filter_universe`（`data/pipeline.py:97`）用的是 `metadata["is_st"]`，而 provider 对该字段的定义是
"这只股票是否**在样本内某时点**变成 ST"（`data/providers/synthetic.py:555`，`trigger_idx` 随机落在
`[60, len-60]`）。于是样本一开始就排除了约 3% 的股票——而这 3% 恰好是后来出事的股票，属于
**逆向选择**，方向上会抬高所有基于该股票池的回测。

`get_st_status(as_of_date)` 按 announce_date 判断，但**没有任何生产调用方**。而且
`_load_point_in_time_data()` 在 `_filter_universe()` **之后**才执行（`data/pipeline.py:61` 与 `:74`），
所以即使去调用它，过滤时 `_st_timeseries` 也还是 `None`。

**修复方向**（未实施，会改变研究数字）：`valid_assets` 是"整个回测共用一个集合"，无法表达
"某只股票从某个日期起不可交易"。要做到第八条，需要把它改成按日期的可交易掩码，并让回测消费它。

---

## 补充：如发现违反此契约

1. 这是一个 bug，提交 issue 或 PR
2. 修复后在该文件中更新"已在代码中实现的位置"
3. 如果该 bug 影响了之前的回测结果，标注 affected 版本

---

*最后更新：2026-09-21*
