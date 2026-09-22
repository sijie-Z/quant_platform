# 决策请求 — 回测引擎的执行时点约定（BUG-28）

> **要拿去外部评审的话，用 [`DECISIONS_FOR_REVIEW.md`](DECISIONS_FOR_REVIEW.md)。**
> 那一份是把本文加上另外两个待定问题（BUG-26 股票池、BUG-13 限价卖单）重新整理成的
> 自包含版本，可以直接粘贴，不需要仓库上下文。本文保留为 BUG-28 的详细单篇记录。
>
> **用途**：一个我**故意没有单方面决定**的问题。它不改变代码正确性，但会**第二次改变所有研究数字**，
> 且"改代码"和"改文档"两个方向都站得住。请给意见。
>
> **本文件自包含**：§1 重建上下文，§2 是事实与证据（含我刚验证的部分），§3 是两个方案及其代价，
> §4 是我倾向的判断与理由，§5 是需要你回答的问题。
>
> 日期：2026-09-20　分支：`develop` @ `d19d226`

---

## §1 背景

quant_platform 是一个单人维护的 A 股多因子研究平台。它的核心卖点是 **No-Lookahead（无未来函数）**，
仓库里有一份正式的 `docs/NO_LOOKAHEAD_CONTRACT.md`（7 条），并且 A 股十大实盘陷阱文档里
第 6 条就是 **T+1**。

本周的修复已经**两次**改动了研究数字：

1. **BUG-03**（成本模型被系统性减半）—— 修复后所有历史回测收益数字失效；
2. **BUG-27**（协方差窗口包含了自己即将赚到的那一天）—— 本次刚修，
   `backtest/engine.py` 的窗口从 253 根 bar（含 1 根未来）改为 252 根已实现 bar。

BUG-28 是同一族的第三个，但**性质不同**：前两个是无可争议的 bug，这个是**约定选择**。

---

## §2 事实与证据

### 2.1 这套代码里 `returns` 的含义

`data/pipeline.py:279`：

```python
self.returns = close.pct_change(fill_method=None).shift(-1)
# shift(-1): return from today's close to tomorrow's close
```

**注意**：`returns.loc[t]` 是 **close(t) → close(t+1)** 的**前向**收益，不是trailing收益。
这个约定在整个仓库里是一致的（IC 计算、alpha pipeline、`benchmark` 都按它写）。

### 2.2 引擎实际怎么做

`backtest/engine.py` 的 `_get_rebalance_dates()` 返回**每月最后一个交易日**（如 1/31）。
`run()` 在该日取 `signal.loc[rdate]` 算出目标权重；`_simulate_pnl()` 里：

```python
if next_rdate is not None and date >= next_rdate:
    ...
    current_weights = target_weights.copy()
...
daily_ret_assets = returns.loc[date].reindex(current_weights.index, fill_value=0.0)
portfolio_return = (current_weights * daily_ret_assets).sum()
```

在 `date == rdate` 当天，新权重就生效并开始赚 `returns.loc[rdate]`，即
**close(rdate) → close(rdate+1)**。

**翻译成人话**：在 rdate 收盘价算出信号 → **用同一个收盘价成交** → 赚下一个收盘的收益。
这是业界常见的 "trade at the close" 约定。

### 2.3 信号在 close(rdate) 时是否真的可知？（我专门验证过）

这是判断"是不是未来函数"的关键，我让一个独立 agent 逐条追了消费链，**结论是可知的**：

| 位置 | 代码 | 结论 |
|---|---|---|
| `alpha/combination.py:80`（`combine_ic_weighted`） | `ic_hist = ic_s[ic_s.index < date]` | 严格 `<`，**不含当期 IC** |
| `alpha/combination.py:154`（`combine_icir_weighted`） | 同上 | 严格 `<` |
| `alpha/regime.py:163`、`:241` | 同上 | 严格 `<` |
| `alpha/ml_signal.py:602-609` | `dates = index[start_idx:end_idx]`，行区间 `[train_start, i)` | 训练集不含当期标签 |
| `factors/evaluation.py:42-48` | `period>1` 时 `rolling(period)...shift(-(period-1))` | 正确 |

**所以：信号在 close(rdate) 时完全可知，引擎没有偷看未来。**
它只是一个**成交价假设的乐观化**（假设你能在自己刚观测到的那个收盘价上成交）。

### 2.4 但文档说是次日成交

| 文档 | 原文 |
|---|---|
| `backtest/engine.py:10`（模块 docstring） | "weights computed on the last trading day of each month, **executed at next day's close**" |
| `data/ASHARE_PITFALLS.md:82` | "the engine uses `returns.shift(-1)` — **today's signal executes at tomorrow's close**, so you're never buying and selling the same stock on the same day" |
| `CLAUDE.md:976` / `AGENTS.md:976` | "T+1 ... 日频用 `shift(-1)` **次日执行**" |
| `data/ASHARE_PITFALLS.md:84`（面试话术） | "Our signal construction uses `shift(-1)` for **next-day execution**" |

**四处声明，代码一处都没有兑现。** 按次日收盘成交，应该赚 `returns.loc[rdate+1]`
（= close(rdate+1) → close(rdate+2)），而不是 `returns.loc[rdate]`。

### 2.5 为什么这条在**日频**下不只是文档问题

月频调仓天然规避 T+1（买卖间隔 ≥20 个交易日）。但 `rebalance_frequency: "daily"` 是
`config/default.yaml` 的合法选项，`CLI` 也支持。日频 + 同收盘成交 =
**当天买、当天卖（同一收盘价）**，这正是 A 股 T+1 禁止的行为，
而平台对外宣称"处理了 T+1"。

---

## §3 两个方案与代价

### 方案 A：改代码，兑现文档（次日收盘成交）

```python
# _simulate_pnl 中，新权重从 rdate 的下一根 bar 之后才生效
```

- ✅ 与四处文档、与面试话术、与"处理了 T+1"的宣称一致
- ✅ 消除"用自己刚观测到的收盘价成交"这个乐观假设
- ❌ **所有历史回测数字再变一次**（第三次），且方向上是**变差**（收益下降、Sharpe 下降）
- ❌ 需要重新生成所有已有报告；已有 Registry 里的研究结论与新的不可比

### 方案 B：改文档，描述代码（同收盘成交）

- ✅ 不动任何数字，历史结论保持可比
- ✅ 从"无未来函数"的定义看，代码本身是**自洽**的（信号确实在 close(t) 可知）
- ❌ 要改 4 处文档 + 面试话术，且必须**删除**或**改写** T+1 的宣称
  （日频下"当天买卖同一收盘价"确实是 T+1 违规）
- ❌ 把一个已被写进简历级材料的卖点降级

### 方案 C（我倾向的折中）

**代码改成次日成交（方案 A），但把日频单独处理**：

- 月频/周频：信号在 rdate 收盘生成，**次日收盘成交**；
- 日频：同样次日成交 —— 这同时**自动满足 T+1**，不需要额外特判。

即：A 已经涵盖了 T+1，不需要 C 的特殊分支。C 只是说明"为什么 A 也让 T+1 宣称成立"。

---

## §4 我倾向的判断

**倾向 A（改代码）**，理由：

1. **四处文档 + 一份契约 + 一段面试话术**都写了次日成交，代码一处都没做。在这种矛盾里，
   默认应该改代码去兑现文档，而不是改文档去迁就代码——除非代码的行为是**有意的**且文档是错的。
   我查过这一点：`_simulate_pnl` 里的 `date >= next_rdate` 判据来自**初始提交**
   （`git log -S "date >= next_rdate" -- backtest/engine.py` 只有 `aec0bbb` 一条），
   此后 5 次改动都没碰过它。**它不是被谁改坏的，是一开始就与自己的 docstring 不一致。**
   这是一个从第 0 天起就没兑现过的承诺，而不是一个有意的取舍。
2. 平台**对外出售的核心能力就是诚实性**。一份说"A"做"B"的文档，比一个乐观化假设更伤。
3. T+1 的宣称在日频下当前是**假的**，这是实质性错误，不只是措辞。

**但我不单方面执行**，因为：

- 它会让**第三次**改写全部历史数字；
- 修复方向会让所有已发表的结果**变差**，这类改动应该由项目所有者确认；
- 项目已有 `docs/RESEARCH_TERMINAL_CONCLUSION.md` 等基于当前数字写成的结论文档，
  是否接受"全部重跑"是**产品决策**而非技术决策。

---

## §5 需要你回答的问题

1. **A 还是 B？** 如果你的答案是 A，是否有必要先做一次"新旧数字对照"（用同一份配置跑两遍，
   把差异量化出来）再合并？还是直接合并、把对照作为 PR 描述的一部分？
2. **A 方案下，`turnover` 和成本应该在 rdate 扣还是次日扣？** 我倾向次日（真实现金流出时点），
   但这会让 `turnover_history` 的日期索引也发生变化，需要确认。
3. **已经写好的研究结论**（M1/M2/M3 报告、Registry 里的记录）要不要标注"基于旧执行约定"？
   还是直接重跑覆盖？平台自己的原则是"所有不可信的必须机器可识别"，我倾向**加标注**而不是静默覆盖。
4. 如果选 B：**T+1 的对外宣称要不要一起删掉**？还是保留但限定在"月频调仓天然规避"的语境里？

---

## §6 附：本次已做但与本决策无关的修复

| Bug | 内容 | 状态 |
|---|---|---|
| BUG-27 | 协方差窗口包含未来 1 根 bar（253 → 252 根已实现 bar） | 已修（PR #37，已合并） |
| BUG-45 | `/api/analysis/ic-decay` 与 `/correlation` 伪造曲线与相关矩阵 | 已修（PR #36，待合并） |
| — | 实盘信号生成器把 `np.random` 当 `forward_returns` 喂给 AlphaPipeline | 已修（PR #37，已合并） |
| — | `POST /api/run` 从未跑通：pandas 真值判断 + 运行存储存 Pydantic 对象 | 已修（PR #38，待合并） |

**注意 BUG-27 与 BUG-28 是同一根 bar 的两面**：BUG-27 是"优化器看到了未来收益"（已修，
无可争议），BUG-28 是"成交时点约定"（本文件要你决定）。
