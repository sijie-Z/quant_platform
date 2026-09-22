# Bug 巡检报告 — 2026-09-13

> **巡检方式**：4 个独立 agent 分区通读（资金账务 / 未来函数 / 状态生命周期 / API 契约），
> 每条发现要求给出 `file:line` + 代码原文 + 具体失效场景 + 可达性证据（grep 输出）。
>
> **本文件的诚实标注**：每条都标注了**由谁验证**。标「我复核」的是我逐条读过代码、
> 查过调用方后独立确认的；标「agent 报告」的是 agent 提供了执行输出但我尚未逐条复核的。
> **不要把「agent 报告」当作我已经背书。**
>
> 代码基线：`main` @ `4ba9683`（PR #16 未合并）

---

## 已确认：P0 / 高危

### BUG-01 · Kill Switch 接在死实例上，Monitor 面板的熔断按钮对引擎无效 【我复核】

**位置**：`api/monitor.py:104,124` · `api/routes.py:2307,2352` · `trading/live_runner.py:189`

全仓库有 **4 个互不相通的 `RiskMonitor` 实例**：

```python
api/monitor.py:124          _core_risk = RiskMonitor()   # MonitorDashboard 打这个
api/routes.py:2307          _core_risk = RiskMonitor()   # /api/core/risk 打这个
api/routes.py:2352          _core_risk = RiskMonitor()   # 每次 trading/start 再新建
trading/live_runner.py:189  self._risk = RiskMonitor()
```

`monitor._core_risk` 与 `routes._core_risk` 之间**没有任何赋值关联**（grep 证据见下），
而引擎持有的是 `routes._core_risk`（`routes.py:2360` 传入 `risk_monitor=`）。

```bash
$ grep -n "_core_risk" api/monitor.py api/routes.py
api/monitor.py:104:_core_risk = None
api/monitor.py:124:            _core_risk = RiskMonitor()
api/routes.py:2307:_core_risk = RiskMonitor()
api/routes.py:2352:    _core_risk = RiskMonitor()          # ← 每次 start 重新创建
api/routes.py:2360:        risk_monitor=_core_risk,
```

**失效场景**：在 MonitorDashboard 点 Kill Switch → `POST /api/monitor/kill-switch` →
面板显示 `active: true` 并弹出提示 → **引擎继续下单**，因为 `trading/engine.py:360` 检查的是
另一个实例的 `kill_switch_active`。同理 `POST /api/monitor/config` 改的风控限额引擎永远看不到。

**加重**：`routes.py:2352` 每次 `trading/start` 都重建 `_core_risk`，所以即使走
`/api/core/risk/kill-switch`，也只对**最后一个**引擎有效；此前启动的引擎全部失联。

**影响**：平台对外宣称的「实时风控熔断 + Kill Switch」在最主要的入口上是失效的。

---

### BUG-02 · `LiveRunner` 从不 mark-to-market，报告的 equity 实际只是现金 【我复核 + agent 复现】

**位置**：`trading/live_runner.py:517-519` · `trading/broker.py:100-107,415-416`

```python
# live_runner.py:517
def _get_equity(self) -> float:
    acct = self._broker.get_account()
    return acct.get("total_equity", self._initial_cash)
```

```python
# broker.py:415
market_value = sum(p.market_value for p in self._positions.values())
total_equity = self._cash + market_value
```

`Position.market_value` 全仓库**只有一处被写**：`broker.py:107` 的 `update_price()`，
它只由 `update_market_prices()` 调用。而 `update_market_prices` 的调用者**只有**：

```bash
$ grep -rn "update_market_prices" --include=*.py .
trading/broker.py:444:    def update_market_prices(prices)      # 定义
trading/engine.py:505:  ...update_market_prices(...)      # 唯一的调用方，事件驱动引擎
```

`live_runner.py` **从不调用它** → 每个持仓的 `market_value` 永远是 `0.0` →
`total_equity == cash`。

**失效场景（agent 执行，1,000,000 起始资金）**：
```
account: {'cash': 839063.73, 'total_equity': 839063.73, 'total_pnl': -160936.27}
```
用 16 万买了股票，报告出来的却是 **−16% 收益**（其实那是花掉的钱）。

**连带**：
- `SessionReport` 的 `final_value` / `total_return_pct` / `annualized_return_pct` /
  `sharpe_ratio` / `max_drawdown_pct` 全部失真
- 失真的 equity 又回流到仓位计算（`live_runner.py:285` `target = equity * 0.04`）→ 越买越少
- 失真值被写进 `store.save_pnl_snapshot`（`live_runner.py:452`），而 `NAVCalculator`
  正是读这个表

**同一文件的相关问题**：`_generate_signals`（`:288-298`）只产生 `"side": "buy"`，
`_execute_signal`（`:324-345`）从不与现有持仓比较 → **每天把整个篮子重买一遍，持仓永远无法退出**。

---

### BUG-03 · 回测交易成本被系统性减半 【我复核】⚠️ 会改变所有历史研究结论

**位置**：`backtest/engine.py:225` · `backtest/cost_model.py:96-103`（`backtest/capacity.py:273` 同）

```python
# engine.py:225 —— 传的是【单边】turnover
turnover = (target_weights - current_weights).abs().sum() / 2
rebalance_cost = self.cost_model.compute_costs(turnover)
```

```python
# cost_model.py:96-103
commission_cost = abs_turnover * commission_rate
if is_sell is not None:
    stamp_cost = abs_turnover * stamp_tax_rate * is_sell.astype(float)
else:
    stamp_cost = abs_turnover * stamp_tax_rate * 0.5      # ← 关键
```

那个 `* 0.5` **只有在输入是「双边成交额」时才成立**（假设一半是卖出）。
但调用方传的是单边——而且这是项目**自己声明的约定**：

```python
# portfolio/constraints.py:16
max_turnover: Maximum one-sided turnover per rebalance.
```

后果：佣金、印花税、滑点**每一项都少收一半**。单边 30% 换手的月频策略，
每年凭空少算约 130bp 成本。

**为什么这是最需要决策的一条**：它**不是**一个边缘 bug，而是**回测引擎的定价基准错了**——
意味着**README 里以及所有历史回测的 Sharpe / 收益数字都是在成本被低估一半的前提下得到的**。

---

## 已确认：中危

### BUG-04 · `POST /api/trading/start` 无重复启动保护，第二次调用孤儿化第一个引擎 【我复核】

`api/routes.py:2387-2388` 无条件 `engine.start(); _live_engine = engine`，只覆盖不检查。
`LiveTradingEngine.start()` 自己的 `if self._running: return` 是**实例级**的，挡不住新建实例。

**失效场景**：双击按钮 / 两个标签页 / 脚本连发 → 两个引擎各自起 `TradingScheduler` 线程、
各自持 broker、交易同一批标的，**同时往同一个 `data/trading.db` 写**。第一个引擎失去引用，
API 再也停不掉它，线程会持续下单直到进程退出。

### BUG-05 · EventBus → WebSocket 桥接静默丢弃所有来自工作线程的事件 【agent 报告，我未复核】

`api/routes.py:121-129` 的 `_on_bus_event` 里 `asyncio.get_event_loop()` 在工作线程中抛
`RuntimeError` 后直接 `return`。而引擎的所有事件都从工作线程发布
（`trading/engine.py:217` 起 `threading.Thread`；`:435` 发 `order.filled`）。

**后果**：UI 上「实时交易事件流」在引擎真正交易时是空的。

### BUG-06 · `MultiStrategyManager` 的 `current_value` 从不初始化 【agent 报告（含复现），我未复核】

`strategy/multi_strategy.py:47` 默认 `current_value = 0.0`，`add_strategy` 不初始化，
而 `update_strategy_pnl` 把它当权益基数累加 → **+1% 的收益被报成 −99%** 并触发 `loss_alert`。

单元测试里有一行「workaround」注释正好证明这个初始化是缺失的：
```python
# tests/test_strategy/test_multi_strategy.py:30
# Initialize current_value to capital_allocated
self.mgr.states[sid].current_value = self.mgr.states[sid].capital_allocated
```

### BUG-07 · `NAVCalculator` 跨日重启会重置高水位，凭空产生业绩报酬 【我复核】

`operations/nav.py:93` 调 `get_nav_history(days=1)`，而实现是：

```python
cutoff = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
rows = conn.execute("SELECT * FROM nav_history WHERE date > ? ...", (cutoff,))
```

严格 `>` → **昨天的记录不满足条件**，只有今天的能返回。任何跨日历日的进程重启
（`trading/engine.py:162` 每次启动都 new 一个 `NAVCalculator`）都会把 HWM 重置为 1.0、
份额重置为 10,000,000 → **在已经处于高水位时仍向投资人计提高额业绩报酬**。

（附带：`except (AttributeError, Exception): pass` 吞掉一切异常。）

### BUG-08 · 多策略的目标仓位被拿去对账共享的无归属持仓簿 【agent 报告，我未复核】

`strategy/portfolio_orchestrator.py:158-169` 用**同一个** `exec_engine.positions` 去核对
**每一个**策略的当前持仓。策略 A 持有 600519，策略 B 的目标里没有它 → B 调 `rebalance()`
会把 A 的持仓全卖掉。且 `:295-307` 把**整个组合**的 P&L 记到**每个**策略头上。

---

## 已确认：低危 / 需要判断

| 编号 | 问题 | 位置 |
|---|---|---|
| BUG-09 | `save_session` 用 `INSERT OR REPLACE`，三条停止路径都不带 `started_at` → 每次停止都把开始时间改写成停止时间、`total_trades` 归零 | `quant_core/store.py:417` |
| BUG-10 | 租户隔离是假的：`positions` 主键只有 `code`，且引擎从不把 `tenant_id` 传给 store | `quant_core/store.py:90` · `trading/engine.py:466` |
| BUG-11 | 全局 `AsyncEventBus` 从未 `start()`；且 `stop()` 不清 `_consumer_tasks` → **无法重启** | `core/events.py:203` · `event_bus_v2.py:733` |
| BUG-12 | 状态机的 HALTED / ERROR 在生产中**不可达**：kill switch 不碰状态机 | `core/state_machine.py:41` |
| BUG-13 | `SimulatedBroker` 限价**卖单在参考价永远无法成交**（best bid 比 best ask 低 1bp），被拒订单还留在簿里 | `trading/broker.py:201-228` |
| BUG-14 | `LiveTradingEngine` 的 P&L 快照在**同一周期内买入**的持仓按 0 计价（价格更新在订单循环之前） | `trading/engine.py:504` vs `:452` |
| BUG-15 | VWAP 切片会产出**负数量**（钳位对象写错）：200 股 / 10 片 → `[100,100,0,-100,…,-700]` | `execution/algorithms.py:181-199` |
| BUG-16 | OMS 声称支持 T+1 但 `_trade_date_offset` **只被赋值从未被读**，当日买入可当日卖出 | `execution/oms.py:57,78` |
| BUG-17 | `SimulatedBroker` 现金可为负：0.1% 缓冲覆盖不住 ¥5 最低佣金 | `trading/broker.py:291-298` |
| BUG-18 | `PaperBroker._simulate_partial_fill` 的 `scale` 在首笔后被重复递减 → 后续成交全塌到 1 股 | `execution/paper_broker.py:271-289` |
| BUG-19 | 配置写 `slippage_model: "impact"`，但引擎从不传 `daily_volume`/`volatility` → **实际跑的是固定滑点** | `config/default.yaml:92` · `backtest/engine.py:227` |
| BUG-20 | 风控下单前检查传 `{"code": ...}`，而 `RiskMonitor` 读 `order.get("ticker")` → **持仓限额永远从零开始检查** | `trading/live_runner.py:331` · `risk/circuit_breaker.py:193` |

---

## 建议的修复分批

| 批次 | 内容 | 特点 |
|---|---|---|
| **第 1 批** | BUG-01（Kill Switch 单例化）、BUG-04（重复启动保护） | 安全功能失效 + 孤儿进程，改动局部、**不改变研究结论** |
| **第 2 批** | BUG-02（LiveRunner mark-to-market）、BUG-07（NAV 高水位） | 报告/资金正确性，不改研究结论 |
| **第 3 批** | BUG-03（成本模型） | ⚠️ **需要单独决策**：修正后所有历史回测数字失效 |
| **第 4 批** | BUG-15/16/17/18/20 等 | 逐个独立，可并行 |
| **第 5 批** | BUG-05/06/08/09/10/11/12/13/14/19 | 需要设计判断，非机械修复 |

---

## 第三组发现：未来函数 / 时间对齐（agent 报告，我没有逐条复核）

> **这组的分量最重**：它指出平台**三个"验证级"宣称不成立**。所有发现都经过调用链追踪，
> 其中 4 条由 agent 执行代码验证（附重现命令）。

平台统一的收益约定（所有下游都依赖它）：

```python
# data/pipeline.py:276-280
self.returns = close.pct_change(fill_method=None).shift(-1)
# shift(-1): return from today's close to tomorrow's close
```

### BUG-21 · Walk-forward 的「OOS」指标实际覆盖训练期+测试期 【HIGH】

`backtest/walkforward.py:144-147,159-171,182`。走**推荐路径**（传入 `factors`）时，
`_compute_signal_in_sample` 返回覆盖 `train+test` 的信号，引擎在整个区间上模拟 P&L，
而该整段序列被当作 "OOS" 追加、拼接。

**agent 执行验证**（900 天，train=252，test=63）：
```
fold  reported_sharpe  test_only_sharpe  reported_n  test_n
  0        -1.098          -2.256            315        63
aggregate reported: -0.76（拼接后的"OOS"覆盖全部 900 天）
```
消费方：`main.py walkforward`（输出标题写着 "OOS"）、`strategy/research_validation.py`
（喂给 Strategy Gates）、前端 WalkForward 面板。

（旧路径 `factors=None` 是只跑测试期的，没问题。）

### BUG-22 · 基本面因子使用了 `publish_date` 之前的数据 【HIGH】

`data/providers/synthetic.py:517-520` 从**财报期末**（而非公告日）向前填充，且用 `bfill()`
把首份财报推回样本起点。`factors/fundamental.py` 与 `process_factor` 的中性化都直接用。

**agent 执行验证**：2021-06-30 的报告 `publish_date=2021-08-17`，但在 2021-07-01 就已可见
（提前 33 个交易日）；2021-05-17 才公告的报告从 2021-01-01 就被使用。

**契约落空**：`docs/NO_LOOKAHEAD_CONTRACT.md` 第三条声称已实现。真正做 PIT 的
`DataPipeline.get_financials_as_of()` / `get_st_status()` / `get_industry_map()` **全部存在**，
但 grep 显示**只被测试调用**，`DataPipeline.run()` / `main.py` / API 从不调用。

IC 影响（同一数据实测）：`pb_ratio` −0.0072（现状）vs −0.0051（PIT 修正后）。

### BUG-23 · ML 交叉验证按「拍平后的 (日期,股票) 行」切分，训练与测试共享日期 【HIGH】

`alpha/ml_signal.py:359-379`。`train_size=504` 被当作 **504 行**解释，而每行是一个
(日期, 股票) 对 → 504 行 ≈ 12 个交易日，而非 2 年；`gap=10` 也变成 10 行 ≈ 不足一个横截面。

**agent 执行验证**（50 股 / 600 天 / 24966 行 ≈ 43 行/天）：
```
fold 0: train 2020-01-29..2020-02-12, test 2020-02-12..2020-02-13, 同日重叠: 1 天
fold 2/3/4: 同日重叠各 1 天
```
→ 训练集与测试集共享真实日期，purge gap 没有 purge 掉任何东西。
`main.py ml train` 与 `POST /api/ml/train` 报告的 `test_ic` / `test_icir` 因此接近样本内。

### BUG-24 · `POST /api/walkforward` **直接从样本内净值曲线伪造折叠结果** 【HIGH，用户可见】

`api/routes.py:1582-1601`：注释写着 "Generate synthetic walk-forward results based on the run data"，
把全样本回测（用全样本信号跑的）的净值曲线切片、贴上 "oos_equity" 标签、
再从一个切片收益率反推一个 Sharpe。**不重算信号、不重训模型。** 前端 WalkForward 面板消费它。

### BUG-25 · `MLSignalGenerator.train()` 在全样本上拟合，`predict()` 再输出所有日期的信号 【HIGH（潜在）】

`alpha/ml_signal.py:403-407` + `:437-482`。因果版本 `generate()` 存在且正确，CLI 用的是它；
但 `train()`+`predict()` 这一对公开 API 目前无生产调用者 —— **是一把上了膛的枪**。

### 其余（中低）

| 编号 | 问题 | 位置 |
|---|---|---|
| BUG-26 | 股票池用**全样本** ST 状态过滤（PIT 路径是死代码） | `data/pipeline.py:95` |
| BUG-27 | 回测协方差窗口**包含了它即将赚到的那一天收益**（`lookback_end + 1`） | `backtest/engine.py:119-122` |
| BUG-28 | 引擎在与信号同一个收盘价成交，与其自身 docstring 矛盾（差一根 bar） | `backtest/engine.py:10` vs `:220-234` |
| BUG-29 | `PureVolatilityFactor` 的 AR 系数在**全样本**上拟合；且遭遇停牌股会抛 `ValueError`（被 `main.py:171` 吞成 warning，因子静默消失） | `factors/technical.py:637-662` |
| BUG-30 | 行业中性化不是 PIT（契约第七条落空） | `factors/processing.py:113-129` |

## 第三组**验证为真**的负面结果（同样有价值）

| 宣称 | 结论 |
|---|---|
| `factors/evaluation.py` 的 shift 链条（CLAUDE.md 特别强调过） | ✅ **正确**。实测 max\|diff\| = **0.00e+00**（N=1,2,3 对照手算） |
| 技术因子因果性（截断测试：用 `prices[:450]` 重算对比第 449 行） | ✅ 动量/波动/RSI/MACD/效率比/趋势阶段/网络中心性全部 max\|diff\| = 0 |
| 横截面处理逐日进行（非全样本归一化） | ✅ Numba 与 pandas 两条路径都是按行迭代 |
| `alpha/combination.py` 的 point-in-time IC 加权 | ✅ `ic_hist = ic_s[ic_s.index < date]`，严格取历史 |
| `factors/ic_monitor.py` 滚动 IC | ✅ `common_dates[i-window:i]` 排除当日（保守） |

**结论**：日常流水线的因子/IC 算术**确实是对的**，但平台**三个验证级能力**站不住 ——
walk-forward 的 "OOS" 含训练期、ML 交叉验证按行切分、基本面在公告前可见。

## 附带发现的非未来函数缺陷

- `factors/technical.py:492-520` `MAConvergenceFactor.compute` 返回 **Series 而非 DataFrame**
  → 在流水线里退化成单列；而 `rank_ic_numba` 按位置取 `.values`，把这一列和 418 只股票逐行
  相关，得到 1304 个「有值但无意义」的 IC（不是空 IC，纯 pandas 回退路径下才是空）。
  **【已修复 2026-09-22】**
- `api/routes.py:2659` 向 `generate()` 传它不接受的 `force_retrain=` → **`/api/ml/predict` 必然 500**
- `research/validation.py` Deflated Sharpe 里 `observed_sr` 年化后代入按日计算的公式
  → z 统计量被放大约 √252，该检验几乎恒显著

---

## 第四组发现：API / CLI 契约（agent 报告 + 我复核了 CRITICAL）

> agent 在巡检期间注意到有新提交落地，**全部发现都在 `0a93746` 上重新复现过**。
> 其中一条直接指出**我那次 kill-switch 修复不完整**（见 BUG-31）。

### BUG-31 · `POST /api/run` 永远无法完成 —— schema 拒绝流水线自己构造的数据 【CRITICAL · 我已复核】

```python
# api/schemas.py:163
monthly_returns: dict[str, list[float]] | None = None
```
```python
# api/routes.py:751-755 —— 实际构造的形状完全不同
monthly = {
    "years":  [str(y) for y in mdf.index.tolist()],   # list[str]
    "months": list(range(1, 13)),                     # list[int]
    "data":   mdf.values.tolist(),                    # list[list[float]]
}
```

**我亲自复现**：
```
ValidationError: 2 validation errors for ChartData
monthly_returns.data.0  Input should be a valid number [input_type=list]
```

**后果**：**主产品端点从来无法产出结果**。`_execute_pipeline` 捕获后把运行标记为 `failed`，
`GET /api/run/{id}/result` 返 404。`/api/demo` 逃过一劫是因为它没有 `response_model`。

### BUG-32 · `_run_store` 存的是 Pydantic 对象，所有消费方都当 dict 用 【HIGH】

生产者 `api/routes.py:377-407` 存 `{"performance": PerformanceMetrics(...), "chart_data": ChartData(...)}`，
而消费方全部用 `.get(...)`：

```
_run_walkforward  -> AttributeError: 'ChartData' object has no attribute 'get'
_run_monte_carlo  -> AttributeError: 'ChartData' object has no attribute 'get'
_decompose_risk   -> AttributeError: 'FactorICItem' object has no attribute 'get'
_detect_regime    -> AttributeError: 'ChartData' object has no attribute 'get'
```

**注**：这一条**修正了我文档里 BUG-24 的描述**——`/api/walkforward` 目前**根本走不到**
那段伪造折叠结果的代码，它在 `routes.py:1573` 就先崩了；而且 `/api/run` 从不成功，
`_run_store` 始终是空的。

### BUG-33 · **我上一次的 kill-switch 修复漏了第三个入口** 【HIGH · 已修复】

`api/routes.py:1990` 还有第三个构造点，被 `/api/risk/status`、`/api/risk/kill-switch`、
`/api/risk/check-order` 使用。前端有**三个** kill switch 调用点，我上次只覆盖了两个：

```
api/index.js:114  /risk/kill-switch        ← 漏掉的那个（Terminal 风控面板按的就是它）
api/index.js:270  /core/risk/kill-switch   ← 已修
api/index.js:295  /monitor/kill-switch     ← 已修
```

我在 commit message 里写了 "route **every** construction site through it" —— **那句话是错的**。
已在 `60d9d28` 补齐并验证。**这是本次巡检中由独立检查抓出的、我自己的失误。**

### BUG-34 · `POST /api/monitor/config` 静默忽略 `max_position_pct` 【HIGH · 已修复】

`api/monitor.py:319` 写 `risk.limits.max_position_pct`，而 dataclass 字段名是
`max_single_position_pct`。`@dataclass` 会默默接受这个野字段，**没有任何代码读它** →
接口返回 `{"updated": ["max_position_pct"]}` 报告成功，而**实际执行的限额一动没动**。

### 其余（agent 报告，我未逐条复核）

| 编号 | 问题 | 位置 |
|---|---|---|
| BUG-35 | `POST /api/report/html` 双重失败：先用错参数调 `_build_chart_data`，再把 Pydantic 对象当 dict 传给 `html_report` → 下载栏永远 500 | `routes.py:2123,2130` |
| BUG-36 | `POST /api/trading/start {"broker":"qmt"}` 必然 500：传 `qmt_path=`，而 `QMTBroker.__init__` 的参数是 `account=/server=/password=` | `routes.py:2344` |
| BUG-37 | `python main.py strategy run` 崩溃：`DashboardGenerator` 这个类**根本不存在**（只有函数 `generate_dashboard`） | `main.py:1459` |
| BUG-38 | `POST /api/data/quality` 必然 500：`numpy.bool_` 不可 JSON 序列化 | `routes.py:2289` |
| BUG-39 | `/api/fundamentals/stats` 被 `/api/fundamentals/{code}` 遮蔽（注册顺序问题）→ 返回一只叫 "stats" 的假股票 | `routes.py:3061` vs `:3113` |
| BUG-40 | `POST /api/portfolio/import` 正则写成了 `r"\\d{6}"`（转义过头）→ **静默导入 0 只股票**，还返回 `{"status":"ok"}` | `routes.py:1247` |
| BUG-41 | `main.py run` **破坏它自己刚存的配置快照**：`load_config` 会 `pop` 调用方的 dict，而它在 `vm.save()` 之前执行 → 快照缺 `constraints`/`covariance`/`var`，`config rollback` 会静默回退这些段 | `utils/config.py:72` + `main.py:266,272` |
| BUG-42 | 缓存命中时 `turnover_20d` **静默退化成价格 SMA 代理**（缓存只存 5 元组，不含 turnover）→ 同样的输入、不同的因子值 | `main.py:289,296` + `factors/technical.py:132` |
| BUG-43 | `/api/analysis/ic-decay` 与 `/api/analysis/correlation` 总是取**最旧**的那次运行（`_run_store` 里没有 `started_at`，`max()` 对空串退化为第一个键） | `routes.py:1783,1819` |
| BUG-44 | `_compute_attribution` 用了上一轮循环残留的 `factor_df` | `routes.py:687` |

**额外印证**：BUG-05（EventBus→WebSocket 桥接）被独立确认——从 ThreadPoolExecutor 工作线程调用
`_update_status` 时 `_broadcast_status` 调用次数为 **0**。

---

## 待补

- 无。四组巡检已全部完成。
