# 面试话术手册 — 每个设计决策讲 3 分钟

> 每个模块的结构：**问题 → 方案 → 实现细节 → 面试追问预判**

---

## 1. EventBus v2：异步事件总线

### 30 秒版本
> "所有组件通过事件总线通信，不直接调用。EventBus v2 用 asyncio per-handler 队列，背压机制保证不丢事件，P50/P99/P999 延迟直方图监控每个 handler 性能，死信队列指数退避重试失败事件，WAL 事件溯源支持崩溃恢复。"

### 3 分钟版本

**问题**：原来的 EventBus 用 `threading.Lock` + 同步 handler，一个慢 handler 会阻塞所有其他 handler 的投递。

**方案**：每个 handler 有独立的 `asyncio.Queue`，互不阻塞。

**实现细节**：
- `AsyncEventBus.subscribe()` 自动检测 handler 是 sync 还是 async（`asyncio.iscoroutinefunction()`）
- 背压：队列满了 publisher 等待（`while queue.full(): await asyncio.sleep(0.0001)`），绝不丢事件
- 延迟监控：每个 handler 维护 `HandlerStats`，用 13 个微秒级 bucket 做直方图，计算 P50/P99/P999
- 死信队列：handler 抛异常 → 事件进入 DLQ → 指数退避重试（100ms, 500ms, 2s, 10s, 30s）
- 事件溯源：`EventStore` 有环形缓冲（内存）+ WAL（磁盘），支持崩溃后 replay

**追问预判**：
- Q: "为什么不用 Kafka？" → A: "单进程内用 asyncio.Queue 足够，延迟更低。分布式场景再换 Kafka，接口抽象好了（`MessageBus` ABC 有 `LocalBus`/`RedisBus`/`KafkaBus` 三个实现）。"
- Q: "背压会不会导致 publisher 卡住？" → A: "会，但这是正确行为——宁可慢一点也不丢事件。金融系统里丢失一个 `order.filled` 事件意味着持仓状态不一致。"

---

## 2. 订单簿（LOB）：红黑树 + 价格-时间优先

### 30 秒版本
> "真正的限价订单簿，用排序的 bid/ask 价格层级，每个层级是 FIFO 队列（价格-时间优先撮合）。支持 IOC/FOK、部分成交、L1/L2/L3 快照、VPIN 微观结构指标。"

### 3 分钟版本

**问题**：原来的 `SimulatedExchange` 用固定滑点模拟成交，没有真实的订单簿逻辑。

**方案**：实现完整的 LOB，用 Python 的 `sorted list` + `dict` 维护价格层级。

**实现细节**：
- `_bid_prices`：降序排列的买价列表（`bisect` 插入，O(log N)）
- `_ask_prices`：升序排列的卖价列表
- `_bid_levels`/`_ask_levels`：`{price: PriceLevel}`，每个 PriceLevel 是 FIFO `deque`
- 撮合逻辑：买单来 → 从最低 ask 开始匹配，直到价格不满足或数量用完
- IOC（Immediate or Cancel）：匹配能匹配的，剩下的取消
- FOK（Fill or Kill）：不能全部成交就全部取消
- 微观结构指标：`compute_vpin()` 计算订单流毒性，`get_microstructure_metrics()` 返回价差/深度不平衡

**追问预判**：
- Q: "为什么用 sorted list 不用红黑树？" → A: "Python 没有内置红黑树，但 `bisect` 维护的 sorted list 在插入/查找上是 O(log N)，性能足够。C++ 里我会用 `std::map`。"
- Q: "VPIN 怎么算？" → A: "Volume-Synchronized Probability of Informed Trading。把交易按成交量分桶，统计每桶里买方主导 vs 卖方主导的比例，VPIN 高意味着逆向选择风险大。"

---

## 3. 市场冲击模型：Almgren-Chriss / Square-Root / Kyle

### 30 秒版本
> "三个市场冲击模型的加权集成：Almgren-Chriss（临时+永久冲击）、Square-Root（业界标准，冲击 ∝ √(Q/V)）、Kyle's Lambda（知情交易者模型）。加权平均后估算执行成本。"

### 3 分钟版本

**问题**：大订单会推动市场价格，回测必须考虑这个成本，否则回测收益虚高。

**方案**：三个经典模型各有侧重，加权集成更稳健。

**实现细节**：
- **Almgren-Chriss**：`temporary = η * (Q/V) * price`，`permanent = γ * (Q/V) * price`。η 是临时冲击系数，γ 是永久冲击系数。还有 timing risk（与波动率成正比）
- **Square-Root**：`impact = Y * σ * √(Q/V) * price`。Y 是冲击系数（默认 0.5），σ 是波动率。这是业界最常用的模型
- **Kyle's Lambda**：`λ = σ * price / √V`，`impact = λ * Q`。来自 Kyle (1985) 的知情交易者模型
- **CompositeImpactModel**：加权平均三个模型（默认权重 0.4/0.4/0.2）
- **ExecutionCostCalculator**：冲击 + 佣金 + 印花税 + 机会成本

**追问预判**：
- Q: "三个模型哪个最准？" → A: "没有最准，只有最合适。Square-Root 在大多数场景下表现最好（文献支持），Almgren-Chriss 在大订单优化时更精确，Kyle's Lambda 理论基础最强但参数难估计。集成是工程上的稳健选择。"
- Q: "参数怎么调？" → A: "用历史成交数据拟合。η 和 γ 可以从日内 VWAP 偏差回归得到。实际生产环境会用交易后分析（TCA）不断校准。"

---

## 4. 逐笔回测引擎

### 30 秒版本
> "事件驱动的逐笔回测，每笔订单经过完整流程：风控检查 → 执行算法 → 订单簿撮合 → 市场冲击模拟 → 成交确认。支持 TWAP/VWAP 执行算法。"

### 3 分钟版本

**问题**：原来的向量化回测是日频的，无法模拟日内执行细节和市场冲击。

**方案**：逐笔事件驱动回测，每笔订单走真实流程。

**实现细节**：
- `Tick` 数据类：包含 symbol、timestamp_ns、price、quantity、bid、ask、volume、volatility
- `TickDataSource`：可以从 DataFrame 或 generator 流式读取
- `TickBacktester.run()`：对每个 tick 执行：
  1. 更新订单簿（如果有新的 bid/ask）
  2. 调用策略函数获取信号
  3. 提交订单（经过 `CompositeImpactModel` 估算冲击，调整价格）
  4. 订单簿撮合（真实 LOB 匹配）
  5. 处理成交（更新持仓、现金、佣金、印花税）
  6. 风控检查
- TWAP：等时间切片执行，每片提交等量订单
- VWAP：按成交量加权执行，参与率控制（默认 5%）

**追问预判**：
- Q: "逐笔回测很慢吧？" → A: "确实比向量化慢 100-1000x，但这是保真度 vs 速度的权衡。月频策略用向量化就够了，日内策略或执行算法优化需要逐笔。"
- Q: "怎么处理部分成交？" → A: "IOC 订单在订单簿撮合时自然产生部分成交。`BookOrder.filled_quantity` 跟踪已成交量，`remaining_quantity` 是剩余量。"

---

## 5. 实时风控引擎

### 30 秒版本
> "逐笔 Greeks 更新（Black-Scholes Delta/Gamma/Vega/Theta/Rho），预成交检查（模拟成交后风险暴露），自动 delta-hedge，12 个压力测试场景，5 级风险等级 + Kill Switch。"

### 3 分钟版本

**问题**：原来的 `RiskMonitor` 是事后检查，成交后才发现超限。

**方案**：`RealTimeRiskEngine` 做预成交检查——在下单前模拟成交，检查是否会触发风险限额。

**实现细节**：
- `on_fill(fill_dict)` → 返回 `RiskUpdate`，包含风险等级、Greeks、限额利用度
- `pre_trade_check(symbol, side, quantity, price)` → 返回 `PreTradeCheck`，包含是否批准、原因、检查延迟
- 多维限额：仓位大小、行业暴露、日亏损、回撤、订单频率、Delta、Gamma
- 风险等级：GREEN → YELLOW → ORANGE → RED → KILL，根据 breach 严重程度自动升级
- Kill Switch：一键熔断，所有订单被拒绝
- 压力测试：12 个场景（崩盘 5/10/20/30%、波动率飙升 2/3x、利率升降、组合冲击、慢跌、闪崩）
- Greeks 计算：`BlackScholesModel` 用 `math.erf` 实现标准正态 CDF（不依赖 scipy）

**追问预判**：
- Q: "Greeks 对股票策略有什么用？" → A: "股票多因子策略主要用 Delta（方向性暴露）和 Gamma（凸性）。如果你有期权持仓，Greeks 更重要。即使纯股票策略，Delta 也等价于持仓市值的 beta 敞口。"
- Q: "Kill Switch 触发后怎么恢复？" → A: "必须手动调用 `deactivate_kill_switch()`。生产环境里，Kill Switch 触发意味着系统发现异常状态，需要人工审查后才能恢复。"

---

## 6. Cython 热路径

### 30 秒版本
> "4 个计算密集型函数用 Cython 加速：rolling_momentum、rolling_volatility、rank_ic、zscore_cross_section。有 .pyx 源文件和纯 Python fallback，自动检测 Cython 可用性。"

### 3 分钟版本

**问题**：因子计算里有些函数是纯 Python 循环，Pandas 向量化也无法消除 Python 解释器开销。

**方案**：用 Cython 编译到 C，消除解释器开销。同时保留纯 Python fallback。

**实现细节**：
- `_fast_rolling_cy.pyx`：rolling_momentum/volatility/max_drawdown，用 `cimport numpy`、typed memoryviews、`nogil`
- `_fast_rank_cy.pyx`：rank_ic，用 C-level 的 `qsort` 排序
- `_fast_zscore_cy.pyx`：zscore_cross_section，用 Welford 单 pass 算法计算均值和方差
- `__init__.py`：`try: from ._fast_rolling_cy import *; HAS_CYTHON = True` 自动检测
- `benchmark_cython_speedup()`：运行 benchmark 对比 Python vs Cython 加速比

**追问预判**：
- Q: "为什么不用 Numba？" → A: "项目里有 6 个 Numba JIT 内核做其他事。Cython 和 Numba 各有优势——Cython 适合需要 C-level 控制的场景（内存布局、nogil），Numba 适合简单的数值循环。"
- Q: "Welford 算法是什么？" → A: "单 pass 在线算法计算均值和方差，数值稳定性比两 pass（先算均值再算方差）好。每步更新：`delta = x - mean; mean += delta/n; M2 += delta * (x - mean)`。"

---

## 7. 分布式消息总线

### 30 秒版本
> "MessageBus ABC 抽象，三个后端实现：LocalBus（进程内 asyncio.Queue）、RedisBus（Redis Pub/Sub）、KafkaBus（aiokafka）。ServiceRegistry 做服务发现和心跳。"

### 3 分钟版本

**问题**：单进程 EventBus 无法跨机器通信，微服务架构需要分布式消息总线。

**方案**：抽象 `MessageBus` 接口，三个实现可插拔切换。

**实现细节**：
- `Message` 数据类：topic、payload、timestamp、source、correlation_id、headers，支持 JSON 序列化
- `LocalBus`：进程内 asyncio.Queue，用于开发和测试
- `RedisBus`：redis.asyncio Pub/Sub，支持 pattern matching（`subscribe("order.*")`）
- `KafkaBus`：aiokafka Producer/Consumer，支持 consumer group 和 offset 管理
- `ServiceRegistry`：register/deregister/discover/heartbeat，超时自动移除不健康服务
- `BaseService`：抽象生命周期（setup → register_handlers → run → shutdown），健康检查，指标，心跳

**追问预判**：
- Q: "这三个实现能热切换吗？" → A: "可以，因为用的是 ABC 抽象。配置里改一行 `message_bus.backend: "redis"` 就切换了。但实际部署需要 Redis/Kafka 服务。"
- Q: "为什么不直接用 RabbitMQ？" → A: "Redis 和 Kafka 是量化行业标配。Redis 延迟低（<1ms），Kafka 吞吐高（百万级/秒）且支持重放。RabbitMQ 在金融领域用得少。"

---

## 8. 多因子信号

### 30 秒版本
> "4 因子等权复合信号：3 个月动量（趋势跟踪）+ 低波动率（质量因子）+ RSI 反转（均值回归）+ MACD（动量确认）。因子引擎有 15 个因子，全部可接入。"

### 3 分钟版本

**问题**：单一因子容易失效，多因子集成更稳健。

**方案**：4 个经典因子等权复合，因子引擎支持 15 个因子的扩展。

**实现细节**：
- **动量因子**：`price[-1] / price[-63] - 1`，3 个月累计收益。经济直觉：强者恒强
- **低波动因子**：`-std(returns[-20:])`，负波动率。经济直觉：低波动股票长期跑赢高波动（低波动异象）
- **RSI 反转**：`-RSI(14)`，买超卖股票。经济直觉：短期均值回归
- **MACD**：`EMA(12) - EMA(26)`，趋势确认。经济直觉：动量持续性
- 复合：等权平均 → 横截面排名归一化 → 选 top N（阈值 0.3）
- 因子引擎：10 个技术因子 + 5 个基本面因子，`BaseFactor` ABC 可扩展

**追问预判**：
- Q: "为什么等权不用 IC 加权？" → A: "等权是稳健选择。IC 加权容易过拟合——历史 IC 高不代表未来 IC 高。生产环境可以用 ICIR 加权，但需要定期重新估计。"
- Q: "因子怎么处理缺失值？" → A: "缩尾（1%/99%分位数）→ 标准化（zscore/rank）→ 行业+市值中性化（回归残差）。中性化消除行业和市值的 confounding effect。"

---

## 9. A 股 10 大陷阱

### 30 秒版本
> "前复权、停牌处理、幸存者偏差、涨跌停、ST 过滤、T+1、交易成本、手数限制、除权除息、行业漂移——全部处理了。"

### 3 分钟版本

| # | 陷阱 | 代码实现 | 面试话术 |
|---|------|---------|---------|
| 1 | **前复权** | `TushareProvider` 取 qfq；合成数据生成 `adj_factor` | "除权除息会断崖式跳价，前复权把历史价格调整到可比基准" |
| 2 | **停牌** | 短停牌(≤30天)前向填充；长停牌移出股票池 | "停牌股票无法交易，但回测里如果用停牌日的价格会产生 look-ahead bias" |
| 3 | **幸存者偏差** | 跟踪上市/退市日期，时间点股票池构建 | "只看现在还在的股票会高估收益，因为退市的差股票被忽略了" |
| 4 | **涨跌停** | 日收益截断±10%；标记涨跌停标志 | "涨停时你买不到，跌停时你卖不掉。回测必须模拟这个约束" |
| 5 | **ST** | `is_st` 标记，默认排除 | "ST 股票涨跌停±5%，退市风险高，流动性差" |
| 6 | **T+1** | 月频调仓天然规避；日频用 `shift(-1)` 次日执行 | "A 股当天买的当天不能卖，日频策略必须用延迟执行" |
| 7 | **成本** | 佣金 0.03% 双边 + 印花税 0.1% 仅卖 + 滑点 | "忽略交易成本的回测收益可以差 20-30%" |
| 8 | **手数** | 优化器向下取整到 100 股倍数 | "A 股最小交易单位是 1 手 = 100 股" |
| 9 | **除权除息** | 前复权将分红调整嵌入历史价格 | "分红除息日价格会跳，不调整的话收益计算错误" |
| 10 | **行业漂移** | 取最新行业分类；动态中性化 | "公司可能换行业（如传统企业转型科技），用旧分类会错配" |

**追问预判**：
- Q: "这些陷阱哪个最容易被忽略？" → A: "幸存者偏差和涨跌停。很多人用 AKShare 拉数据，默认就排除了退市股。涨跌停更隐蔽——回测显示你买了涨停股，但实际上你根本买不到。"

---

## 10. 与 Jane Street 的差距

### 30 秒版本
> "代码层面的架构能力到位了——事件驱动、真实订单簿、逐笔回测、实时风控、Cython 热路径。剩下的差距是物理限制（延迟 1000x）、语言（OCaml 类型安全）、数据（Level 3）、策略（做市牌照）、运营（99.999% 可用性）。"

### 3 分钟版本

| 维度 | 我们的系统 | Jane Street | 差距性质 |
|------|-----------|-------------|---------|
| 事件总线 | asyncio per-handler 队列 + 背压 + DLQ | 同类架构 | **已追平** |
| 订单簿 | 红黑树 + FIFO + IOC/FOK + VPIN | 同类架构 | **已追平**（缺数据 feed） |
| 回测 | 逐笔事件驱动 + 市场冲击 | 逐笔事件驱动 | **已追平**（缺跨品种） |
| 风控 | 逐笔 Greeks + 预成交检查 + Kill Switch | 逐笔预检 + 自动对冲 | **已追平** |
| 因子计算 | Cython 热路径 | C++/Rust | 接近 |
| 延迟 | ~10ms (Python) | ~10μs (OCaml + FPGA) | **1000x 差距** |
| 类型安全 | 运行时 (Python) | 编译期 (OCaml) | **本质差距** |
| 数据 | Level 2 (10 档快照) | Level 3 (逐笔委托) | **资源差距** |
| 策略 | 月频多因子选股 | 微秒级做市 + 套利 | **品类差距** |
| 运营 | 无 | 99.999% 可用性 | **组织差距** |

**追问预判**：
- Q: "既然差距这么大，做这个项目有什么意义？" → A: "这个项目展示的是**架构设计能力**，不是要复现 Jane Street。面试官看的是：你能不能设计出事件驱动系统、能不能处理真实市场约束、能不能做性能优化。这些能力在任何量化岗位都通用。"
- Q: "如果要缩小差距，下一步做什么？" → A: "换语言（Rust/OCaml）降低延迟，但这对月频策略没意义。更实际的下一步是接入真实数据源（Tushare Pro）和在实盘环境跑 paper trading。"

---

## 11. Prometheus 监控

### 30 秒版本
> "轻量级无依赖 Prometheus 指标收集器：Counter/Gauge/Histogram + Timer 装饰器 + `/api/metrics` 端点。Grafana 一键导入 16 面板模板。"

### 3 分钟版本

**问题**：没有可观测性就无法知道系统在生产环境的表现。

**方案**：实现 Prometheus 指标暴露，配合 Grafana 仪表盘。

**实现细节**：
- `Counter`：单调递增（如请求总数、错误总数）
- `Gauge`：可增可减（如当前持仓数、风险等级）
- `Histogram`：分布统计（如延迟分布、收益分布）
- `Timer`：上下文管理器，自动记录阶段耗时
- `instrument_pipeline_stage`：装饰器，自动给 pipeline 函数加指标
- `/api/metrics`：Prometheus text format 端点
- Grafana 模板：16 个面板覆盖 Pipeline/API/风控/因子/EventBus

**追问预判**：
- Q: "为什么不用 StatsD？" → A: "Prometheus 是 pull 模型，更适合服务端。StatsD 是 push 模型，更适合 serverless。量化系统是长运行服务，Prometheus 更合适。"

---

## 12. Walk-Forward 验证

### 30 秒版本
> "滚动/扩展窗口的时序交叉验证，避免过拟合。每个 fold 内用 train-only 数据重新计算信号（非复用全量预计算信号），输出 OOS 收益和稳定性指标。"

### 3 分钟版本

**问题**：全样本回测容易过拟合——你在整个历史数据上调参数，当然表现好。更隐蔽的问题是：即使切分了 train/test，如果 signal 是用全量 IC 数据预计算的，test 期间的 signal 依然嵌入了未来信息。

**方案**：Walk-Forward 验证，每个 fold 内用 train-period 数据重新计算 IC 权重和信号。

**实现细节**：
- 滚动窗口：固定训练期长度，每次向前滚动测试期
- 扩展窗口：训练期从起点开始，越来越长
- **核心创新**：`factors`/`alpha_kwargs` 参数，传入后每个 fold 内调用 `_compute_signal_in_sample()` 用 train-only 数据重算 Alpha 信号
- 不传 `factors` 则向后兼容使用预计算信号
- 折叠指标：每个折叠计算 Sharpe/Calmar/MaxDD
- 稳定性指标：`mean_sharpe`、`std_sharpe`、`sharpe_consistency`（正 Sharpe 折叠占比）
- 返回值增加 `signal_recomputed: True/False` 标记

**追问预判**：
- Q: "Walk-Forward 会降低收益吗？" → A: "会。全样本回测的收益是虚高的，Walk-Forward 给你真实预期。如果 Walk-Forward 结果和全样本差距很大，说明策略过拟合了。"
- Q: "为什么要在 fold 内重算信号？" → A: "因为原来的 signal 用全量 IC 数据计算因子权重——2021 年的信号'知道'2025 年哪个因子好使。折叠内重算确保 test 期的信号完全来自 train 期的信息。"

---

## 13. 未来函数防范

### 30 秒版本
> "五个层面：Point-in-time IC 加权（每个时间点只用之前数据算因子权重）、IC 计算无 shift 链条错误、Walk-Forward 折内重算信号、合成数据真实 IC 水平、默认等权配置。每个问题有明确的代码位置和面试话术。"

### 3 分钟版本

**问题**：量化回测项目最常犯的错误就是未来函数——回测很美，实盘归零。

**方案**：5 项显式防护，每项有代码+测试。

**实现细节**：

| # | 防护 | 代码位置 | 效果 |
|---|------|---------|------|
| 1 | **Point-in-time IC 加权** | `alpha/combination.py:35-140` | 每个时间点预计算 IC 时序，只用 t 之前的数据算权重 |
| 2 | **IC 计算 shift 修正** | `factors/evaluation.py:21-65` | returns 已做过 shift(-1)，IC 计算不再重复 shift；period>1 时计算累计收益 |
| 3 | **Walk-Forward 信号重算** | `backtest/walkforward.py:226-261` | 每个 fold 用 train-only 数据重算信号，而非复用全量预计算信号 |
| 4 | **合成数据真实度** | `data/providers/synthetic.py:251-274` | Alpha 强度 IC~0.015（原 0.04），信噪比 1:2（原 2:1） |
| 5 | **默认等权配置** | `config/default.yaml:47` | `alpha.method: equal_weight`，避免演示时用过拟合参数 |

**追问预判**：
- Q: "你怎么证明没有未来函数？" → A: "三个证据：1) 跑 `pytest tests/test_alpha/ tests/test_factors/test_evaluation.py` 全部通过；2) Walk-Forward 的 OOS Sharpe 确实低于全样本 Sharpe（差距合理）；3) 合成数据因子 IC 在 0.015 左右而非 0.04-0.05。"
- Q: "Point-in-time 会增加多少计算时间？" → A: "IC 时序预计算是 O(n_dates × n_factors)，只多一次循环。对于 5 年/500 只/15 因子的数据，额外开销约 3-5 秒，完全可接受。"
- Q: "为什么之前有这个漏洞？" → A: "这是量化面试项目的常见设计取舍——为了演示效果把参数和权重调得很漂亮。但面试官一定会追问'你真的没有未来函数吗'，所以我决定正面解决。"

---


### "你这个系统有多少行代码？"
> "Python 约 19,500 行，Vue 前端约 9,800 行，610 个单元测试。新增的 Jane Street 级模块（EventBus v2、订单簿、市场冲击、逐笔回测、Greeks、实时风控、Cython 热路径、消息总线、微服务骨架）约 5,000 行 Python + 3 个 Cython .pyx 文件 + 132 个新测试。"

### "你怎么做测试的？"
> "610 个 pytest 测试覆盖全模块。每个新模块都有独立测试：EventBus（pub/sub、通配符、拦截器、DLQ）、订单簿（FIFO 撮合、IOC/FOK、部分成交）、风控（Greeks 精度、限额检查、Kill Switch、压力测试）。CI 用 GitHub Actions 在 Python 3.10/3.11/3.12 矩阵跑。"

### "你怎么防止未来函数和过拟合？"
> "五个层面：1) IC 加权做 point-in-time，每个时间点只用之前数据算权重；2) IC 计算修复了 returns shift 链条错误；3) Walk-Forward 每个 fold 内用 train-only 数据重新算信号；4) 合成数据 alpha 强度降到真实水平 IC~0.015；5) 默认等权配置而不是 ICIR 加权。每个点都有代码位置和测试覆盖。"

### "如果给你一个月，你会做什么？"
> "三件事：1) 接入 Tushare Pro 真实数据，替换合成数据做 backtest validation；2) 用 Rust 重写热路径（因子计算和订单撮合），目标延迟从 10ms 降到 100μs；3) 在 AWS 上部署 Prometheus + Grafana + Kafka，做真实的分布式事件流。"
