# quant_platform — Bug 巡检结果与优先级评审请求

> **用途**：把一次覆盖全仓库的 bug 巡检结果（44 条）提交外部评审，请给出**修复优先级排序**，
> 并对其中两条需要方向性判断的问题给出意见。
>
> **本文件自包含**：§0 重建全部必要上下文，§1 说明这些发现的**可信度分级**，
> §2 是可供排序的完整矩阵，§3–§5 是分类细节与我已修的部分，§6 是需要你回答的问题，
> §7 是必须遵守的约束条件。
>
> 日期：2026-09-13　代码基线：`main` @ `4ba9683`（另有三个 PR 待合并，见 §4）

---

## §0 背景

### 0.1 项目是什么

**quant_platform** —— 单人维护的 A 股多因子量化**研究 + 交易**平台，公开在 GitHub。
320 个 Python 文件 / 57,683 行；42 个 Vue/JS 文件；1,300+ 测试；97 个 REST 端点；19 个 CLI 命令。

架构分两层：

- **v1 平台层**：研究（数据→因子→Alpha→组合→回测）→ 风控 → 执行 → 实盘 → Web 仪表盘
- **v2 研究 OS 层**（2026-07 起）：转向"诚实研究"。核心原则 **Truth First / Point-in-time /
  Knowledge compounds**；每份研究记录都把数据源、复权方式、PIT 状态、偏差警告写进 SQLite Registry，
  报告里的告警由字段**机器生成**而非手写

### 0.2 项目对外宣称的三件"验证级"能力

这决定了哪些 bug 更严重：

1. **零未来函数** —— 有 `docs/NO_LOOKAHEAD_CONTRACT.md`（8 条契约），CLAUDE.md 专门强调
   "IC 计算不存在 shift 链条错误"
2. **Walk-forward OOS 验证** —— 宣传为"避免过拟合的黄金标准"
3. **实时风控熔断 + Kill Switch** —— 5 级风险等级 + 一键熔断

### 0.3 为什么做这次巡检

用户要求"去巡检有没有 bug，认真看看可以修"。此前刚完成一轮 CI 基建重建
（从红到绿、加了分支保护、建了 canonical Python 3.12 环境，1,313 个测试 0 失败）。

### 0.4 巡检方式

4 个独立 agent 分区通读，互不重叠：

| 组 | 范围 |
|---|---|
| A · 资金/账务正确性 | `trading/` `execution/` `operations/` `backtest/` |
| B · 未来函数/时间对齐 | `factors/` `alpha/` `backtest/` `data/pipeline.py` `research/` |
| C · 状态/持久化/生命周期 | `core/` `daemon/` `strategy/` `kernel/` `services/` |
| D · API/CLI 契约 | `api/` `main.py` `app.py` `compliance/` `reporting/` |

对每条的硬性要求：给出 `file:line` + 代码原文 + **具体失效场景** + **可达性证据**（grep 输出）。
对"从未被调用 / 从未被持久化"这类**否定性断言**，必须附上执行过的 grep 命令与输出。

---

## §1 这些发现的可信度分级（请按此权重采信）

**这一点很重要**——不要把全部 44 条当作同等可信：

| 标注 | 含义 | 数量 |
|---|---|---|
| **【我复核】** | 我逐条读过代码、查过调用方、部分亲自执行复现后独立确认 | ~12 条 |
| **【agent 报告】** | agent 提供了执行输出与 grep 证据，但我**没有**逐条复核 | ~32 条 |

巡检过程中已经出现过三类"看起来像 bug 但需要核实"的情况，所以这个分级不是形式主义：

- agent 报"4 个报告文件 IC 全是 None" → 实际只有 1 个是（差点据此删掉研究证据）
- 我曾把本机 126 个测试 error 全归因为"缺包" → 实际约 119 个是 `tmp_path` 权限问题
- 我曾声称某次修复"覆盖了所有构造点" → 独立复核发现漏了三分之一

**执行复现过的**：BUG-01/02/03/07/31/33/34 有实际运行输出。
**仅代码阅读的**：其余。

---

## §2 结果矩阵（供优先级排序）

> **先说清计数口径**（否则按 ID 反查会对不上）：
> **历史发现总数 = 44 条**（BUG-01 … BUG-44，编号唯一、无重复计数）
> **其中已修复 = 3 条**（BUG-01 / BUG-33 / BUG-34，见 §4）
> **当前开放 = 41 条**
>
> 下面矩阵里标「已修」的行仅作为历史记录保留，**不构成待办**。

图例：**SV** = 严重度；**研究** = 是否改变研究结论/失效已有结果；**量级** = 修复工作量估计

### 类别 ①：验证能力失效（"平台宣称的能力不成立"）

| ID | SV | 问题 | 影响面 | 研究 | 量级 |
|---|---|---|---|---|---|
| BUG-21 | **HIGH** | Walk-forward 的 "OOS" 指标实际**覆盖训练期+测试期** | `main.py walkforward`、Strategy Gates、前端面板全部展示错误 OOS | 是 | 中 |
| BUG-22 | **HIGH** | 基本面因子在 **`publish_date` 之前就可见**（实测提前 33 个交易日） | 所有用基本面因子的研究结论；`NO_LOOKAHEAD_CONTRACT.md` 第三条落空 | 是 | 中 |
| BUG-23 | **HIGH** | ML 交叉验证按**拍平的 (日期,股票) 行**切分 → 训练/测试**共享真实日期**，purge gap 无效 | `main.py ml train`、`POST /api/ml/train` 报告的 test_ic/test_icir 接近样本内 | 是 | 小 |
| BUG-24 | **HIGH** | `POST /api/walkforward` **从样本内净值曲线伪造折叠结果**（代码注释直白写着 "Generate synthetic walk-forward results"） | 前端 WalkForward 面板 | 是（展示层） | 中 |
| BUG-25 | HIGH(潜在) | `MLSignalGenerator.train()` 在全样本拟合，`predict()` 输出所有日期信号 | 目前**无生产调用者**——是"上了膛的枪" | 是 | 小 |
| BUG-26 | MED | 股票池用**全样本** ST 状态过滤（PIT 路径是死代码） | 幸存者偏差类问题 | 是 | 中 |
| BUG-27 | MED-LOW | 回测协方差窗口**包含了它即将赚到的那一天收益** | 组合优化权重 | 是 | 小 |
| BUG-28 | LOW-MED | 引擎在与信号**同一个收盘价**成交，与自身 docstring 矛盾（差一根 bar） | 回测执行假设 | 是 | 需判断 |
| BUG-29 | MED-LOW | AR 系数在**全样本**上拟合；且遇停牌股抛 `ValueError`（被吞成 warning，因子静默消失） | 该因子 | 是 | 小 |
| BUG-30 | LOW | 行业中性化不是 PIT | 行业分类变更的股票 | 是 | 中 |

> **重要补充**：同一组巡检也用**可执行证据证实了若干宣称是真的** —— IC shift 链条实测
> `max|diff| = 0.00e+00`；技术因子的截断测试全部为 0；横截面处理确实逐日进行；
> `alpha/combination.py` 的 PIT IC 加权确实严格只取历史。**因子/IC 的日常算术是对的。**

### 类别 ②：静默失效（接口报成功，实际什么都没做）

| ID | SV | 问题 | 影响面 | 研究 | 量级 |
|---|---|---|---|---|---|
| BUG-34 | **HIGH** | `POST /api/monitor/config` 字段名写错（`max_position_pct` vs `max_single_position_pct`）→ 报告 `updated` 但**限额从未改变** | 用户以为风险限额生效了 | 否 | 已修 |
| BUG-40 | **HIGH** | `POST /api/portfolio/import` 正则转义过头 `r"\\d{6}"` → **静默导入 0 只股票**，仍返回 `{"status":"ok"}` | 持仓导入功能完全失效且不报错 | 否 | 小 |
| BUG-41 | **HIGH** | `main.py run` **破坏它自己刚存的配置快照**：`load_config` 会 `pop` 调用方的 dict，而它在 `vm.save()` 之前执行 → 快照缺 `constraints`/`covariance`/`var`；`config rollback` 会静默回退这些段 | 配置版本管理功能被自己破坏 | 是（配置） | 小 |
| BUG-42 | MED | 缓存命中时 `turnover_20d` **静默退化成价格 SMA 代理**（缓存只存 5 元组不含 turnover）→ 同样输入、不同因子值 | 破坏"确定性缓存"契约 | 是 | 小 |
| BUG-05 | MED | EventBus→WebSocket 桥接在工作线程抛 `RuntimeError` 后静默 `return` → **UI 实时事件流永远是空的** | 宣称的"实时推送"失效 | 否 | 中 |
| BUG-06 | MED-HIGH | 多策略 `current_value` 从不初始化 → **+1% 收益被报成 −99%** 并触发亏损告警 | 多策略面板全错 | 否 | 小 |
| BUG-10 | MED | 租户隔离是假的：`positions` 主键只有 `code`，引擎也从不把 `tenant_id` 传给 store | 多租户 | 否 | 中 |

### 类别 ③：必然崩溃（可复现的 500 / ImportError）

| ID | SV | 问题 | 影响面 | 研究 | 量级 |
|---|---|---|---|---|---|
| **BUG-31** | **CRITICAL** | `POST /api/run` **永远无法完成** —— `ChartData.monthly_returns` 声明为 `dict[str, list[float]]`，而代码构造的是 `{"years":[str], "months":[int], "data":[[float]]}`，最后一个阶段必然 `ValidationError` | **主产品端点从不产出结果**；`GET /api/run/{id}/result` 恒 404 | 否 | 小 |
| BUG-32 | HIGH | `_run_store` 存 Pydantic 对象，所有消费方当 dict 用 → walkforward / montecarlo / decompose / regime **全部 500** | 4 个高级分析端点 | 否 | 中 |
| BUG-33 | HIGH | Kill-switch 有**第三个**构造点漏修（`/api/risk/*`，正是 Terminal 风控面板按的按钮） | 安全功能 | 否 | 已修 |
| BUG-35 | HIGH | `POST /api/report/html` 双重失败（先传错参数，再把 Pydantic 对象当 dict） | HTML 报告下载栏恒 500 | 否 | 小 |
| BUG-36 | MED | `POST /api/trading/start {"broker":"qmt"}` 传 `qmt_path=` 而构造函数要 `account=/server=` | QMT 实盘接入 | 否 | 小 |
| BUG-37 | MED | `python main.py strategy run` 崩溃：`DashboardGenerator` 这个**类根本不存在**（只有函数） | 一个 CLI 命令 | 否 | 小 |
| BUG-38 | MED | `POST /api/data/quality` 必然 500（`numpy.bool_` 不可 JSON 序列化） | 数据质量面板 | 否 | 小 |
| BUG-39 | LOW | `/api/fundamentals/stats` 被 `/api/fundamentals/{code}` 遮蔽 → 返回一只叫 "stats" 的假股票 | 一个端点 | 否 | 小 |
| BUG-43 | LOW | IC 衰减/相关矩阵端点总是取**最旧**那次运行 | 分析面板 | 否 | 小 |
| BUG-44 | LOW | `_compute_attribution` 用了上一轮循环残留的变量 | 归因数值（通常无害） | 是 | 小 |

### 类别 ④：资金与状态正确性

| ID | SV | 问题 | 影响面 | 研究 | 量级 |
|---|---|---|---|---|---|
| BUG-01 | **HIGH** | **Kill Switch 接在死实例上**（多个 `RiskMonitor` 互不相通）→ 面板显示"已熔断"而**引擎继续下单** | 核心安全功能 | 否 | 已修 |
| BUG-02 | **HIGH** | `LiveRunner` **从不 mark-to-market** → 报告的 equity 就是现金。100 万本金买了 16 万股票，报告 **−16% 收益**。且 `_generate_signals` 只产生 `buy`，**持仓永远无法退出** | 实盘试跑的全部报告指标 | 否 | 中 |
| **BUG-03** | **HIGH** | **回测成本被系统性减半**：调用方传单边 turnover（项目自己在 `constraints.py:16` 声明的约定），而 `cost_model` 的 `stamp × 0.5` **只在双边输入下成立** | **README 与所有历史回测的 Sharpe/收益数字** | **是（全部）** | 小 |
| BUG-04 | MED-HIGH | `POST /api/trading/start` 无重复启动保护 → 第二次调用**孤儿化第一个引擎**（再也停不掉，线程持续下单，两个引擎写同一个 DB） | 实盘安全 | 否 | 小 |
| BUG-07 | **HIGH** | `NAVCalculator` 跨日重启重置高水位（`days=1` + `date > cutoff` 严格不等）→ **凭空向投资人计提高额业绩报酬** | 基金运营 / 投资人资金 | 否 | 小 |
| BUG-08 | MED | 多策略的目标仓位被拿去对账**共享的无归属**持仓簿 → 策略 A 的持仓会被策略 B 卖掉 | 多策略 | 否 | 中 |
| BUG-09 | MED | `save_session` 用 `INSERT OR REPLACE` 且停止路径不带 `started_at` → 每次停止都把开始时间改写成停止时间 | 会话记录 | 否 | 小 |
| BUG-11 | LOW-MED | 全局 `AsyncEventBus` 从未 `start()`；且 `stop()` 不清 `_consumer_tasks` → **无法重启** | 潜在 | 否 | 中 |
| BUG-12 | LOW | 状态机的 HALTED/ERROR 在生产中**不可达**；kill switch 不碰状态机 | 状态展示 | 否 | 中 |
| BUG-13 | MED | `SimulatedBroker` 限价**卖单在参考价永远无法成交**（best bid 比 best ask 低 1bp）；被拒订单还留在簿里 | 模拟撮合 | 是（模拟） | 中 |
| BUG-14 | MED | `LiveTradingEngine` 的 P&L 快照把**同一周期内买入**的持仓按 0 计价（价格更新在订单循环之前） | 持久化的 P&L 历史 | 否 | 小 |
| BUG-15 | MED | VWAP 切片会产出**负数量**（钳位对象写错）：200 股/10 片 → `[100,100,0,-100,…,-700]` | 执行算法 | 否 | 小 |
| BUG-16 | MED | OMS 声称支持 T+1，但 `_trade_date_offset` **只被赋值从未被读** → 当日买入可当日卖出 | A 股规则合规 | 否 | 中 |
| BUG-17 | LOW-MED | `SimulatedBroker` 现金可为负（0.1% 缓冲 < ¥5 最低佣金） | 模拟账务 | 否 | 小 |
| BUG-18 | LOW-MED | `PaperBroker._simulate_partial_fill` 的 `scale` 被重复递减 → 首笔之后全塌到 1 股 | Paper 指标 | 否 | 小 |
| BUG-19 | MED | 配置写 `slippage_model: "impact"`，但引擎从不传 `daily_volume`/`volatility` → **实际跑的是固定滑点** | 回测成本 | 是 | 小 |
| BUG-20 | MED | 风控下单前检查传 `{"code":...}` 而 `RiskMonitor` 读 `order.get("ticker")` → **持仓限额永远从零开始检查** | 风控 | 否 | 小 |

---

## §3 三个必须放在一起看的主题

### 主题一：平台的三件"验证级"宣称，逐条核对后的精确状态

不是笼统的"有两件站不住"，而是**逐条定位到底哪一条成立、哪一条只完成一半、哪一条完全没接线**：

| 能力 | 状态 |
|---|---|
| 价格因子 PIT | ✅ **成立**（截断测试 max\|diff\| = 0） |
| IC 加权只用历史 | ✅ **成立**（`ic_s[ic_s.index < date]`） |
| Walk-forward **信号生成** | ✅ **成立**（每折确实只用 train-only 重算） |
| Walk-forward **OOS 指标** | ❌ **不成立** —— 指标窗口覆盖 train+test（BUG-21） |
| 基本面 **PIT** | ❌ **不成立** —— 只生成了 `publish_date`，从未按它过滤（BUG-22） |
| ML **时间隔离** | ❌ **不成立** —— CV 按 `(日期,股票)` 拍平行切分，训练/测试共享日期（BUG-23） |

**关键区分**：BUG-21 **不是**信号泄漏 —— 信号构造是对的，错的是**报告的指标窗口**。
把它笼统说成"walk-forward 有未来函数"会**夸大**它，也会掩盖真实缺陷的位置。

### 主题二：Web 层几乎不可用，而且互相遮蔽

BUG-31 让 `POST /api/run` 从不成功 → `_run_store` 永远为空 → 依赖它的 4 个端点必然 500（BUG-32）
→ 而它们本来也因 Pydantic/dict 混用会 500。

**修 BUG-31 是解开这一串的钥匙**：修好它才能让其他 API bug 暴露出来。

### 主题三：一堆"报成功但没做"的静默失效

BUG-34（限额没改）、BUG-40（导入 0 只股票还返回 ok）、BUG-41（自己破坏配置快照）、
BUG-42（缓存命中悄悄换因子）、BUG-05（实时推送恒空）。

这一类**比崩溃更危险**，因为没有错误信号。对一个宣称"偏差必须机器可识别"的项目，
静默失效是最直接的价值冲突。

---

## §4 我已经修好的（三个 PR 待合并）

| PR | 内容 | 验证 |
|---|---|---|
| #16 | Docker 非 root 用户 + 两个 `.dockerignore` 缺陷（构建上下文 412MB → 1.09MB） | 容器内实测：uid=10001、8 个目录可写、应用代码路径（Registry/报告/缓存）可写、代码本身只读 |
| #17 | Kill Switch 完整修复（BUG-01 + BUG-33 + BUG-34） | 三个引用同一对象、熔断同步、限额真的改变；267 tests pass |
| — | `docs/AUDIT_BUG_PATROL_2026-09-13.md` —— 44 条的完整证据文档 | 逐条标注验证来源 |

**我修错过一次**：第一次 Kill Switch 修复漏了第三个入口，commit message 里却写了
"route every construction site"。是独立复核抓出来的。**这件事说明这些发现值得再验一遍。**

---

## §5 需要你回答的问题

### Q1（主问题）请给出修复优先级排序

44 条，单人维护，有分支保护（所有改动必须走 PR 且过 4 个必需 CI 检查）。
请给出你认为的**批次划分**（哪些必须立刻做、哪些可以攒一批、哪些可以永久搁置）。

### Q2 BUG-03（成本模型减半）该不该现在修？

- 现状：回测成本每一项都少收一半 → **所有历史回测的 Sharpe/收益数字都是在成本被低估的
  前提下得到的**
- 修好后：数字会变差，**README 和所有已发布结果失效**
- 项目原则："研究结果必须可解释、可复现"

**这是"修正错误"与"保持历史可比"之间的冲突。** 我倾向修（一个算错成本的引擎，它的结论本来
就不该被信任），但这是项目所有者的决定。

### Q3 主题一（验证能力失效）该怎么处理？

有两个方向，我不确定哪个对：

- **(a) 改代码**：把 walk-forward 改成真正只跑测试期、接上 PIT accessor、修 CV 切分
- **(b) 同时改宣称**：在修好之前，先把文档/UI 里那些不成立的宣称撤掉或标注

考虑到这个项目**主动写了 `NO_LOOKAHEAD_CONTRACT.md`** 并把它当作卖点，(b) 可能更符合它的
价值观——但这会让简历/README 上的亮点减少。**你怎么看？**

### Q4 那些"agent 报告但我未复核"的 32 条，该在修之前复核到什么程度？

我的判断是：**修复任何一条之前必须自己复核那一条**（因为已经有 3 次过度断言的先例）。
但这会让 44 条的修复周期变长。有没有更高效的把关方式？

---

## §6 约束条件（排序时必须考虑）

1. **研究可复现性优先于修复速度** —— 任何改变数值输出的修复，都意味着已有结论失效
2. **单人维护** —— 没有第二个人 review，所以"修复引入新 bug"的风险由作者独自承担
3. **分支保护已生效** —— 每个 PR 必须过 `Lint & type check` / `Test (py3.11)` /
   `Test (py3.12)` / `Docker image` 四个必需检查
4. **依赖版本被冻结** —— `requirements.txt` 的 30 个顶层 pin 是研究环境的一部分，
   不能为了让某个修复变容易而升级（有实证：pandas 3.x 会破坏 4 处生产代码）
5. **CI 必须保持绿** —— 上一次 main 变红就是这轮工作的起因

---

## §7 附：项目自己声明但代码未实现的东西（供判断"宣称 vs 实现"的差距）

> `docs/NO_LOOKAHEAD_CONTRACT.md` 有 8 条契约，每条都带一张「**已在代码中实现的位置**」表。
> 逐条比对后：

| 契约条款 | 声明 | 实际情况 |
|---|---|---|
| **第一条** 价格因子只用 signal_date 及之前 | ✅ | ✅ 实测成立（截断测试 max\|diff\|=0） |
| **第二条** Alpha 权重只用历史 IC | ✅ | ✅ 成立（`ic_s[ic_s.index < date]`） |
| **第三条** 基本面只能用 `publish_date`，不能用 `fiscal_period_end` | ✅ | ⚠️ **只实现了一半**。契约列的"实现位置"只有 **publish_date 的生成与保留**（`synthetic.py:467,494`），没有**按它过滤**；而契约自己在正文里把 `mask = financials['publish_date'] <= signal_date` 写作"正确的用法"——**那个 mask 从未被应用**。实测：2021-06-30 的财报在 **2021-07-01 就可见**（`publish_date` 是 2021-08-17） |
| **第四条** Walk-Forward 每折用 train-only 重算信号 | ✅ | ✅ **成立**——信号确实只用 train 期 IC 计算（agent 与我都确认）。**BUG-21 不是这条的违规**：错的是**报告的指标窗口**（含训练期），不是信号泄漏。这个区分很重要 |
| **第五条** 合成数据嵌入式 Alpha 仅用于演示 | ✅ | 未核 |
| **第六条** IC 计算 shift 链条正确 | ✅ | ✅ **实测成立**（max\|diff\| = 0.00e+00） |
| **第七条** 行业分类用生效日期 | ✅ | ❌ `factors/processing.py:113-129` 只接受一张静态 map，没有日期维度 |
| **第八条** ST 状态用公告日期 | ✅ | ❌ `data/pipeline.py:95` 用全样本 `is_st` 过滤 |

**三个做 PIT 的 accessor（`get_financials_as_of` / `get_st_status` / `get_industry_map`）全部存在，
但 grep 显示只在测试里被调用**——规则写好了、工具造好了，就是没接上。

### 契约自己规定的修复程序（与你回答 Q3 直接相关）

`NO_LOOKAHEAD_CONTRACT.md` 末尾的「补充」写了发现违规时该怎么办：

```
1. 这是一个 bug，提交 issue 或 PR
2. 修复后在该文件中更新"已在代码中实现的位置"
3. 如果该 bug 影响了之前的回测结果，标注 affected 版本
```

**第 3 条正是 Q2/Q3 的答案所在**——项目自己已经规定了"受影响的回测结果要标注 affected 版本"。
这暗示它预期的处理方式是「修代码 + 标注影响范围」，而不是「因为怕影响结果所以不修」。
但它也要求标注 affected 版本，而目前没有任何版本标注机制。

### 其他宣称 vs 实现

| 声明 | 位置 | 实际情况 |
|---|---|---|
| "单一主实现" | `ARCHITECTURE_V2.md` ADR-0003 | ❌ 风控曾有 **4 个**实例（已修为 1 个） |
| 实时风控熔断 + Kill Switch | README / UI | ❌ 面板按钮曾对引擎完全无效（已修，且**修了两次**） |
| 租户隔离 | `core/context.py` | ❌ `positions` 表主键只有 `code`，`tenant_id` 只是个普通列 |
| A 股 T+1 | `execution/oms.py` docstring | ❌ `_trade_date_offset = 0  # For T+1 simulation` 全仓库只有这一行，从未被读 |
| 缓存确定性 | `utils/cache.py` | ❌ 缓存命中时 `turnover_20d` 静默退化成价格 SMA |
