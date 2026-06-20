# No-Lookahead Contract（前视偏差零容忍契约）

> 这是本平台所有量化研究代码必须遵守的铁律。任何违反此契约的代码都是 bug。
> 回测的高收益只有在零前视偏差的前提下才有意义。

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

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `data/providers/synthetic.py:583-601` | 行业分类附带 effective_date，~5%/半年变更率 |
| `factors/processing.py:119-148` | neutralization 支持 point-in-time sector map |

---

## 第八条：ST 状态 — 用公告日期

**规则**：ST 标记用 announce_date，不是 trigger_date。交易所发布 ST 公告之前，市场不知道。

**已在代码中实现的位置**：

| 位置 | 实现 |
|------|------|
| `data/providers/synthetic.py:504-548` | ST 有 trigger_date / announce_date 两个字段，中间差 1-3 个交易日 |
| `data/pipeline.py` | 使用 announce_date 做 ST 过滤 |

---

## 补充：如发现违反此契约

1. 这是一个 bug，提交 issue 或 PR
2. 修复后在该文件中更新"已在代码中实现的位置"
3. 如果该 bug 影响了之前的回测结果，标注 affected 版本

---

*最后更新：2026-06-18*
