# M4 — Strategy Validation（归档）

> **归档性质：研究证据，不是受维护的工具。**
> 这些脚本是 M4 结论的证据链，按"研究记录不应因不再被调用而删除"的原则保留。
> 它们**不在 CI 覆盖范围内**，也没有被改造成可复用的模块 —— 这是有意的。

---

## M4 回答的问题

M3 的 Factor Zoo 里，`volatility_20d` 是**最强因子**（IC = +0.0334，ICIR = +0.12）。
M4 的问题是：

> **这个 IC 能不能转成组合收益？**

**结论：不能。** 在月频固定调仓下，交易成本与换手吃掉了 alpha。
`README.md` 的研究里程碑一节记录的就是这条结论，本目录是它的证据。

---

## 脚本

| 脚本 | 内容 |
|---|---|
| `m4_backtest.py` | **核心回测**。`volatility_20d` Top20 Long / Bottom20 Short，月频调仓，完整 A 股成本模型（佣金 + 印花税 + 滑点）。M4 的主结果由它产出 |
| `m4_backtest_lo.py` | **Long-Only 变体**。假设：M4.1 的负 Sharpe 来自空头腿（空头侧成本 + 负 carry）。测试去掉空头后是否改善 |
| `m4_complete.py` | **完整分析**。沿用 `m4_backtest.py` 已验证的组合循环（不做新的会计处理），再把 IC → 超额收益的泄漏逐段拆开 |
| `m4_alpha_leakage.py` | **Alpha 泄漏审计**。沿真实的组合构建与执行路径，把 alpha 从 IC 到净超额逐步分解 |
| `m4_return_attribution.py` | **收益归因**。把 Long-Only 组合收益拆成市场 beta、因子暴露、成本等分量 |
| `m4_benchmark_attribution.py` | **基准归因**。回答：Long-Only 是提供了真正更好的风险收益结构，还是只是吃到了市场 beta？ |

---

## ⚠️ 使用这些脚本前必须知道的三件事

归档时逐一核实过，不是推测：

1. **硬编码了绝对路径。** 6 个脚本全部含 `sys.path.insert(0, "D:/Desktop")` ——
   它们只能在原作者的机器布局下运行。要复跑必须先改这一行。
2. **需要联网。** 通过 `akshare` 拉取 CSI300 成分股与真实行情（`ak.index_stock_cons(symbol="000300")`），
   不是合成数据。
3. **会写入真实 Registry。** 脚本直接连 `data/trading.db` 的 `research_runs` 表并落盘报告 ——
   也就是项目的**研究知识库**。复跑会在其中新增记录，这与
   `tests/test_lab/test_synthetic_factor_run.py` 的隔离要求是同一个关切。

另外：这些脚本**没有在 canonical Python 3.12 环境中复跑过**。它们的输出属于当时的环境，
不应假定与当前环境一致（参见 `docs/AUDIT_REVIEW_ROUND2.md` §2.7 关于依赖未固定的分析）。

---

## 目录约定

`docs/research/` 下的其它文件是**已定稿的研究报告**（ABLATION 系列、Alpha Discovery、
Market Structure、Regime Discovery 等），每份包含假设 / 方法 / 结果 / 结论。
本目录不同：存放的是**产生结论的脚本本体**，没有配套的独立报告 —— 相关叙述在
`README.md` 的研究里程碑表和 `docs/RESEARCH_TERMINAL_CONCLUSION.md` 里。
