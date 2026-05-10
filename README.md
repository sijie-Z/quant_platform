# A-Share Multi-Factor Quantitative Trading Platform

A 股多因子量化研究 + 交易平台 —— 从数据到回测到实盘的完整流水线。面向量化开发面试，展示**机构级架构设计**能力。

**610 单元测试全部通过** | **84 个 Python 模块** | **19,500+ 行 Python** | **91 个 REST API 端点**

---

## 核心亮点

| 能力 | 实现 | 对标 |
|------|------|------|
| 事件驱动架构 | AsyncEventBus（背压 + P50/P99/P999 延迟监控 + 死信重试 + WAL 事件溯源） | Jane Street 级 |
| 真实订单簿 | 红黑树 + 价格-时间优先 FIFO + IOC/FOK + 部分成交 + VPIN 微观结构 | Jane Street 级 |
| 逐笔回测 | 事件驱动 tick replay + 三模型市场冲击集成（Almgren-Chriss/Square-Root/Kyle） | Jane Street 级 |
| 实时风控 | 逐笔 Greeks + 预成交检查 + 自动 delta-hedge + Kill Switch + 12 场景压力测试 | Jane Street 级 |
| Cython 热路径 | 4 个计算内核（rolling_momentum/volatility/rank_ic/zscore）+ Python fallback | 性能优化 |
| 分布式消息总线 | MessageBus ABC + LocalBus/RedisBus/KafkaBus + ServiceRegistry | 生产级 |
| 微服务骨架 | BaseService 生命周期 + RiskService/ExecutionService/DataService | 生产级 |
| 多因子信号 | 15 个因子（10 技术 + 5 基本面）+ 4 因子复合信号 + ML 信号 | 研究级 |
| LLM 增强 | 财经新闻情感因子 + RAG 研究 Agent | 差异化 |
| A 股实盘处理 | 10 大陷阱全处理（前复权/停牌/ST/涨跌停/T+1/成本/手数等） | 实盘就绪 |

> 📖 详细架构文档见 [CLAUDE.md](CLAUDE.md) | 面试话术见 [INTERVIEW_CHEATSHEET.md](INTERVIEW_CHEATSHEET.md) | Jane Street 差距分析见 [JANE_STREET_GAP_ANALYSIS.md](JANE_STREET_GAP_ANALYSIS.md)

---

## 架构总览

```
                        ┌─────────────────────────────────────────────┐
                        │         Core Architecture (core/)          │
                        │  AsyncEventBus · Store · StateMachine      │
                        │  AuditLog · Scheduler · MessageBus         │
                        └──────────────┬──────────────────────────────┘
                                       │ 所有组件通过 EventBus 通信
                                       │ 所有状态通过 Store 持久化
    ┌──────────────────────────────────┼──────────────────────────────────┐
    v                                  v                                  v
Data Layer  -->  Factor Engine  -->  Alpha Model  -->  Portfolio Optimizer
(合成/Tushare/    (15 因子 + Cython     (IC/ICIR/ML       (EW/MVO/RP)
 Baostock/LLM)    热路径加速)           信号合成)
                                                          |
                                                          v
  Order Book  <--  Backtest Engine  -->  Real-Time Risk  -->  Execution Layer
  (红黑树 LOB)     (逐笔事件驱动)        (Greeks+预检+       (TWAP/VWAP/Iceberg)
                                         Kill Switch)
                                                          |
                                                          v
                   Live Trading Engine  <--  Multi-Strategy  <--  Report Engine
                   (AKShare+Paper+QMT)     (资本分配/P&L)       (HTML/Prometheus)
                                                          |
                                                          v
                                      Web Dashboard (Vue 3 + ECharts)
                                      REST API (91 endpoints)
                                      WebSocket (实时推送)
```

---

## 模块详解

### 1. 核心架构 (`core/`)

| 模块 | 功能 | 关键特性 |
|------|------|---------|
| `event_bus_v2.py` | 异步事件总线 | per-handler asyncio.Queue、背压、P50/P99/P999 延迟直方图、DLQ 指数退避重试、WAL 事件溯源 |
| `events.py` | 事件总线桥接 | 向后兼容 `get_event_bus()`，自动检测 sync/async handler |
| `store.py` | SQLite 持久化 | WAL 模式、8 张表、线程安全 |
| `state_machine.py` | 组合状态机 | 8 状态生命周期（INIT→READY→TRADING→REBALANCING→POST_MARKET） |
| `audit.py` | 合规审计 | 每个决策记录 who/what/when/why/result |
| `scheduler.py` | 交易调度 | A 股开市时间检测、自动状态切换 |
| `message_bus.py` | 分布式消息总线 | MessageBus ABC + LocalBus/RedisBus/KafkaBus + ServiceRegistry |

### 2. 数据层 (`data/`)

| 模块 | 功能 |
|------|------|
| `providers/synthetic.py` | 合成 A 股数据（500 只/5 年/可复现/含嵌入式 alpha） |
| `providers/tushare_loader.py` | Tushare Pro 实盘数据（CSI 300/前复权/HDF5 缓存） |
| `providers/baostock_provider.py` | Baostock 免费数据（无需 API key） |
| `providers/postgres_provider.py` | PostgreSQL/TimescaleDB（连接池 + asyncpg + SQLite 回退） |
| `providers/websocket_provider.py` | WebSocket 实时行情（东方财富/新浪推送） |
| `providers/level2_provider.py` | Level 2 盘口（10 档买卖队列 + 逐笔成交 + VWAP + 微观结构因子） |
| `providers/fundamental_realtime.py` | 实时基本面（PE/PB/ROE + TTL 缓存 + 选股器） |
| `pipeline.py` | ETL 流水线（ST 过滤/停牌处理/复权/对齐） |
| `quality.py` | 数据质量监控（8 项检查 + 严重性分级） |

### 3. 因子引擎 (`factors/`)

**10 个技术因子**：momentum_1m/3m/6m/12m、volatility_20d/60d、turnover_20d、rsi_14d、macd、amplitude_20d

**5 个基本面因子**：log_market_cap、pb_ratio、pe_ratio、roe、asset_growth

**因子处理流水线**：原始因子 → 缩尾(1%/99%) → 标准化(zscore/rank) → 行业+市值中性化

**因子评估**：Rank IC / ICIR / 分位数收益 / 相关性矩阵 / IC 衰减曲线

**图网络因子**（`network.py`）：股票关联网络 + 4 种中心性（PageRank/特征向量/介数/度）

**IC 监控**（`ic_monitor.py`）：滚动 IC/ICIR + 衰减检测 + 半衰期估计 + 自适应权重

### 4. Alpha 模型 (`alpha/`)

- **3 种合成法**：equal_weight / ic_weighted / icir_weighted
- **ML 信号**（`ml_signal.py`）：XGBoost/LightGBM + Walk-Forward CV + SHAP 解释性
- **LLM 情感因子**（`agent/sentiment_factor.py`）：财经新闻标题 → 情绪因子，Strategy 模式可插拔 OpenAI

### 5. 执行层 (`execution/`)

| 模块 | 功能 |
|------|------|
| `order_book.py` | **真实 LOB**：红黑树 bid/ask + FIFO PriceLevel + IOC/FOK + 部分成交 + L1/L2/L3 快照 + VPIN |
| `market_impact.py` | **市场冲击模型**：Almgren-Chriss + Square-Root + Kyle's Lambda + 加权集成 |
| `algorithms.py` | TWAP/VWAP/Iceberg + SmartRouter 智能路由 |
| `oms.py` | 订单管理系统：订单生命周期 + SimulatedExchange |

### 6. 回测引擎 (`backtest/`)

| 模块 | 功能 |
|------|------|
| `engine.py` | 向量化月频回测 + 日频持仓漂移 |
| `tick_engine.py` | **逐笔事件驱动回测**：tick replay + 真实 LOB 撮合 + 市场冲击模拟 + TWAP/VWAP |
| `cost_model.py` | A 股成本：佣金 0.03% + 印花税 0.1%（卖）+ 滑点 |
| `walkforward.py` | Walk-Forward 验证（滚动/扩展窗口 OOS 测试） |
| `distributed.py` | 并行回测（ProcessPoolExecutor 参数扫描） |

### 7. 风险管理 (`risk/`)

| 模块 | 功能 |
|------|------|
| `realtime_engine.py` | **实时风控引擎**：逐笔 Greeks 更新 + 预成交检查 + 自动 delta-hedge + Kill Switch + 12 场景压力测试 |
| `greeks.py` | Black-Scholes 全 Greeks（Delta/Gamma/Vega/Theta/Rho）+ 组合聚合 + delta-hedge 计算 |
| `circuit_breaker.py` | RiskMonitor：仓位/行业/亏损/回撤限额 + 5 级风险等级 |
| `var.py` | VaR（历史/参数/蒙特卡洛）+ CVaR |
| `stress.py` | 压力测试：2008 金融危机 / 2015 股灾 / 2020 新冠 |
| `barra.py` | Barra 10 因子风险模型：横截面回归 + Ledoit-Wolf 收缩 + 风险归因 |
| `regime.py` | 行情状态检测：波动率/趋势/相关性三维度 |

### 8. 交易 (`trading/`)

| 模块 | 功能 |
|------|------|
| `broker.py` | **SimulatedBroker**（真实 LOB 撮合）+ QMTBroker（xtquant 实盘） |
| `engine.py` | **实盘交易引擎**：AKShare 实时行情 → 多因子信号 → RealTimeRiskEngine 预检 → LOB 下单 → P&L 跟踪 |
| `realtime.py` | AKShare 实时行情：全市场快照/个股报价/涨跌榜 |

### 9. 性能优化 (`utils/cyext/`)

**4 个 Cython 热路径**（.pyx 源文件 + 纯 Python fallback）：
- `rolling_momentum`：滚动动量（log 收益）
- `rolling_volatility`：滚动波动率（Welford 单 pass 算法）
- `rank_ic`：Spearman Rank IC
- `zscore_cross_section`：横截面 Z-Score 标准化

**6 个 Numba JIT 内核**：滚动累计收益 / 最大回撤 / 缩尾 / Rank IC / Ledoit-Wolf / Z-Score

### 10. 监控 (`utils/metrics.py` + `monitoring/`)

- **Prometheus 指标**：Counter/Gauge/Histogram + Timer 装饰器 + `/api/metrics` 端点
- **Grafana 模板**：16 面板一键导入（Pipeline/API/风控/因子/EventBus）
- **结构化日志**：JSON 格式 + 日志级别配置

### 11. Web 界面 (`app.py` + `frontend/`)

- **FastAPI**：91 个 REST API 端点
- **Vue 3 前端**：Bloomberg Terminal 风格暗色仪表盘 + 8 个视图
- **WebSocket**：EventBus → WebSocket 桥接，交易事件实时推送

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行完整流水线

```bash
# 合成数据（无需 API key，~3 分钟）
python main.py run

# 强制重算（忽略缓存）
python main.py run --force

# Baostock 实盘数据（免费，无需 API key）
python main.py run --use-baostock

# Tushare 实盘数据（需要 token）
export TUSHARE_TOKEN=your_token
python main.py run
```

### 策略对比

```bash
# 三种优化器对比
python main.py compare --optimizers equal_weight,mean_variance,risk_parity

# 参数网格搜索
python main.py sweep --optimizers equal_weight,mean_variance --frequencies monthly,weekly
```

### ML Alpha 信号

```bash
# 训练 ML 模型
python main.py ml train --model lightgbm

# 生成 ML 信号
python main.py ml signal --model xgboost
```

### Web 服务

```bash
# 启动 FastAPI + Vue 前端
python main.py web

# API 文档
open http://localhost:8000/api/docs
```

### 测试

```bash
# 运行全部 610 个测试
pytest tests/ -v

# 只运行核心架构测试
pytest tests/test_core/ -v

# 只运行新增模块测试
pytest tests/test_core/test_event_bus_v2.py tests/test_execution/test_order_book.py tests/test_risk/test_realtime_engine.py tests/test_backtest/test_tick_engine.py tests/test_utils/test_cyext.py tests/test_core/test_message_bus.py -v
```

---

## 配置热切换

在 `config/default.yaml` 中修改：

```yaml
# 切换优化器
portfolio.optimizer: "equal_weight" | "mean_variance" | "risk_parity"

# 切换信号合成方法
alpha.method: "equal_weight" | "ic_weighted" | "icir_weighted"

# 股票池大小
universe.n_stocks: 100 | 300 | 500

# 调仓频率
backtest.rebalance_frequency: "daily" | "weekly" | "monthly"

# 协方差估计
portfolio.covariance.method: "sample" | "ledoit_wolf" | "ewma"

# VaR 方法
risk.var.method: "historical" | "parametric" | "monte_carlo"
```

---

## CLI 命令

```bash
python main.py run                      # 完整流水线
python main.py run --force              # 强制重算
python main.py run --use-baostock       # Baostock 数据
python main.py analyze                  # 分析已有结果
python main.py compare                  # 策略对比
python main.py sweep                    # 参数网格搜索
python main.py ml train --model lightgbm  # 训练 ML 模型
python main.py ml signal --model xgboost  # 生成 ML 信号
python main.py research report          # LLM 研究分析
python main.py profile                  # 性能分析
python main.py web                      # 启动 Web 服务
python main.py cache list               # 查看缓存
python main.py cache clear              # 清除缓存
```

---

## A 股实盘陷阱

平台显式处理了 10 个 A 股特有的实盘陷阱：

| # | 陷阱 | 处理方案 |
|---|------|---------|
| 1 | 前复权 | Tushare 取 qfq；合成数据生成 adj_factor |
| 2 | 停牌 | 短停牌(≤30天)前向填充；长停牌移出股票池 |
| 3 | 幸存者偏差 | 跟踪上市/退市日期，时间点股票池构建 |
| 4 | 涨跌停 | 日收益截断±10%；标记涨跌停标志 |
| 5 | ST 股票 | is_st 标记，默认排除 |
| 6 | T+1 | 月频调仓天然规避；日频用 shift(-1) 次日执行 |
| 7 | 交易成本 | 佣金 0.03% 双边 + 印花税 0.1% 单边 + 滑点 |
| 8 | 手数限制 | 优化器向下取整到 100 股倍数 |
| 9 | 除权除息 | 前复权将分红调整嵌入历史价格 |
| 10 | 行业漂移 | 取最新行业分类；动态中性化处理 |

详见 `data/ASHARE_PITFALLS.md`。

---

## 项目结构

```
quant_platform/
├── main.py                          # CLI 入口
├── app.py                           # FastAPI 应用
├── CLAUDE.md                        # 完整架构文档
├── INTERVIEW_CHEATSHEET.md          # 面试话术手册
├── JANE_STREET_GAP_ANALYSIS.md      # Jane Street 差距分析
├── requirements.txt                 # Python 依赖
├── Dockerfile                       # Docker 部署
├── docker-compose.yml               # Docker Compose
├── .github/workflows/ci.yml         # CI/CD
│
├── config/
│   ├── default.yaml                 # 默认配置
│   └── schema.py                    # 类型化配置验证
│
├── core/                            # ★ 核心架构
│   ├── event_bus_v2.py              # AsyncEventBus（背压+DLQ+WAL+延迟监控）
│   ├── events.py                    # 事件总线桥接（向后兼容）
│   ├── message_bus.py               # 分布式消息总线（Local/Redis/Kafka）
│   ├── store.py                     # SQLite 持久化
│   ├── state_machine.py             # 组合状态机
│   ├── audit.py                     # 合规审计
│   └── scheduler.py                 # 交易调度
│
├── data/                            # 数据层
│   ├── providers/
│   │   ├── base.py                  # DataProvider ABC
│   │   ├── synthetic.py             # 合成数据
│   │   ├── tushare_loader.py        # Tushare
│   │   ├── baostock_provider.py     # Baostock
│   │   ├── postgres_provider.py     # PostgreSQL
│   │   ├── websocket_provider.py    # WebSocket 实时行情
│   │   ├── level2_provider.py       # Level 2 盘口
│   │   └── fundamental_realtime.py  # 实时基本面
│   ├── pipeline.py                  # ETL 流水线
│   └── quality.py                   # 数据质量监控
│
├── factors/                         # 因子引擎
│   ├── base.py                      # BaseFactor ABC
│   ├── technical.py                 # 10 技术因子
│   ├── fundamental.py               # 5 基本面因子
│   ├── processing.py                # 横截面处理
│   ├── evaluation.py                # IC 评估
│   ├── ic_monitor.py                # IC 监控
│   └── network.py                   # 图网络因子
│
├── alpha/                           # Alpha 模型
│   ├── combination.py               # 3 种合成法
│   ├── pipeline.py                  # AlphaPipeline
│   └── ml_signal.py                 # ML 信号
│
├── portfolio/                       # 组合优化
│   ├── optimizers.py                # EW/MVO/RP
│   ├── covariance.py                # 协方差估计
│   └── constraints.py               # 约束条件
│
├── backtest/                        # 回测引擎
│   ├── engine.py                    # 向量化回测
│   ├── tick_engine.py               # ★ 逐笔回测（事件驱动+LOB+市场冲击）
│   ├── cost_model.py                # 成本模型
│   ├── walkforward.py               # Walk-Forward 验证
│   └── distributed.py               # 并行回测
│
├── execution/                       # 执行层
│   ├── order_book.py                # ★ 真实 LOB（红黑树+FIFO+IOC/FOK+VPIN）
│   ├── market_impact.py             # ★ 市场冲击模型（AC/SR/Kyle）
│   ├── algorithms.py                # TWAP/VWAP/Iceberg
│   └── oms.py                       # 订单管理
│
├── risk/                            # 风险管理
│   ├── realtime_engine.py           # ★ 实时风控（Greeks+预检+Kill Switch）
│   ├── greeks.py                    # ★ Black-Scholes Greeks
│   ├── circuit_breaker.py           # RiskMonitor
│   ├── var.py                       # VaR/CVaR
│   ├── stress.py                    # 压力测试
│   ├── barra.py                     # Barra 10 因子
│   ├── regime.py                    # 行情状态检测
│   └── factor_risk.py               # 因子风险分解
│
├── trading/                         # 实盘交易
│   ├── broker.py                    # SimulatedBroker(LOB) + QMTBroker
│   ├── engine.py                    # 实盘引擎（EventBus+RealTimeRisk+LOB）
│   └── realtime.py                  # AKShare 实时行情
│
├── services/                        # 微服务骨架
│   ├── base.py                      # BaseService 生命周期
│   ├── risk_service.py              # RiskService
│   ├── execution_service.py         # ExecutionService
│   └── data_service.py              # DataService
│
├── agent/                           # LLM 模块
│   ├── sentiment_factor.py          # 情感因子
│   └── research_agent.py            # RAG 研究 Agent
│
├── utils/                           # 工具
│   ├── cyext/                       # ★ Cython 热路径
│   │   ├── _fast_rolling_cy.pyx     # rolling momentum/volatility
│   │   ├── _fast_rank_cy.pyx        # rank IC
│   │   ├── _fast_zscore_cy.pyx      # z-score
│   │   └── setup.py                 # Cython 构建配置
│   ├── numba_accelerator.py         # 6 个 Numba JIT 内核
│   ├── metrics.py                   # Prometheus 指标
│   ├── cache.py                     # Pipeline 缓存
│   └── config.py                    # YAML 配置加载
│
├── api/                             # Web API
│   ├── routes.py                    # 91 个 FastAPI 端点
│   └── schemas.py                   # Pydantic 模型
│
├── frontend/                        # Vue 3 前端
│   └── src/components/              # 35 个 Vue 组件
│
├── monitoring/
│   └── grafana_dashboard.json       # Grafana 16 面板模板
│
└── tests/                           # 610 个单元测试
    ├── test_core/                   # EventBus(13) + EventBus v2(25) + MessageBus(16) + Store(16) + StateMachine(15) + Audit(10) + Scheduler(2)
    ├── test_execution/              # OrderBook(22) + OMS(17) + 算法(13) + MarketImpact(10)
    ├── test_risk/                   # RealTimeRisk(12) + Greeks(8) + VaR(7) + Barra(16) + ...
    ├── test_backtest/               # TickEngine(15) + 成本(4) + 指标(7) + WalkForward(2)
    ├── test_utils/                  # Cython(18) + Numba + 缓存 + 配置 + 指标
    └── ...                          # 共 610 个测试
```

---

## 与 Jane Street 的差距

| 维度 | 本项目 | Jane Street | 差距性质 |
|------|--------|-------------|---------|
| 事件总线 | AsyncEventBus（背压+DLQ+WAL） | 同类架构 | **已追平** |
| 订单簿 | 红黑树 + FIFO + IOC/FOK + VPIN | 同类架构 | **已追平** |
| 回测 | 逐笔事件驱动 + 市场冲击 | 逐笔事件驱动 | **已追平** |
| 风控 | 逐笔 Greeks + 预检 + Kill Switch | 逐笔预检 | **已追平** |
| 延迟 | ~10ms (Python) | ~10μs (OCaml+FPGA) | **1000x** |
| 语言 | Python (运行时类型) | OCaml (编译期类型) | **本质差距** |
| 数据 | Level 2 | Level 3 + 跨市场 | **资源差距** |
| 策略 | 月频多因子选股 | 微秒级做市+套利 | **品类差距** |

详细分析见 [JANE_STREET_GAP_ANALYSIS.md](JANE_STREET_GAP_ANALYSIS.md)。

---

## 扩展

```python
# 新数据源
from quant_platform.data.providers.base import DataProvider
class MyProvider(DataProvider): ...

# 新因子
from quant_platform.factors.base import BaseFactor
class MyFactor(BaseFactor): ...

# 新优化器
class MyOptimizer:
    def optimize(self, signal, cov_matrix, ...): ...

# 新执行算法
class MyAlgorithm:
    def create_plan(self, order, **kwargs) -> ExecutionPlan: ...
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 异步 | asyncio, aiohttp, aiokafka, redis.asyncio |
| 数据 | Pandas, NumPy, SQLAlchemy, asyncpg |
| 优化 | cvxpy, SciPy |
| ML | XGBoost, LightGBM, SHAP, scikit-learn |
| 性能 | Cython, Numba |
| Web | FastAPI, Vue 3, Vite, ECharts |
| 监控 | Prometheus, Grafana |
| 容器 | Docker, Docker Compose |
| CI/CD | GitHub Actions (Python 3.10/3.11/3.12 矩阵) |

---

## License

MIT
