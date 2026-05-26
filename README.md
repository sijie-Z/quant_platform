# A股多因子量化交易平台

> **从数据到回测到实盘的完整量化流水线** —— 事件驱动架构 · 实时风控熔断 · 多因子信号 · 实盘交易引擎
> 面向量化开发岗位面试，展示**机构级架构设计**与**A股实盘工程**能力

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/测试-1077%20通过-brightgreen?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/Python模块-96-orange" alt="Modules">
  <img src="https://img.shields.io/badge/Python代码-30K%2B-yellow" alt="Lines">
  <img src="https://img.shields.io/badge/Vue组件-37个-47d" alt="Vue Components">
  <img src="https://img.shields.io/badge/API端点-97个-red?logo=fastapi" alt="API">
  <img src="https://img.shields.io/badge/因子数-15个-purple" alt="Factors">
  <img src="https://img.shields.io/badge/许可证-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#%E6%9E%B6%E6%9E%84%E6%A6%82%E8%A7%88">架构概览</a> •
  <a href="#%E6%A0%B8%E5%BF%83%E6%A8%A1%E5%9D%97">核心模块</a> •
  <a href="#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B">快速开始</a> •
  <a href="#cli-%E5%91%BD%E4%BB%A4">CLI 命令</a> •
  <a href="#%E6%8A%80%E6%9C%AF%E6%A0%88">技术栈</a>
</p>

---

## 亮点

| 能力 | 实现 | 级别 |
|:-----|:-----|:------|
| **事件驱动架构** | EventBus (topic pub/sub + 通配符 + 拦截器 + 死信队列 + WAL) | 机构级 |
| **实时风控熔断** | RiskMonitor 下单前检查 + 5级风险等级 + Kill Switch 一键熔断 | 机构级 |
| **多因子信号引擎** | 15因子 (10技术 + 5基本面) + ICIR加权 + ML信号 (XGBoost/LightGBM) | 研究级 |
| **实盘交易引擎** | AKShare实时行情 → 多因子信号 → 风控预检 → Paper/QMT实盘 → 实时P&L | 生产级 |
| **A股陷阱全处理** | 10大A股实盘陷阱：复权/停牌/ST/T+1/幸存者偏差/涨跌停/成本/手数/除权/行业漂移 | 生产级 |
| **未来函数零容忍** | Point-in-time IC加权 + Walk-Forward折内重算 + 合成数据真实IC水平 | 生产级 |
| **Barra风险模型** | 10因子横截面回归 + Ledoit-Wolf收缩 + 风险归因分解 | 机构级 |
| **IC自动衰减监控** | 滚动IC/ICIR + 衰减检测 + 半衰期估计 + 自适应权重 + 三级告警 | 机构级 |
| **跨资产接口** | InstrumentType/Instrument/AssetUniverse 统一抽象，支持股票/ETF/期货/期权 | 机构级 |
| **LLM增强选股** | 财经新闻情感因子 (Strategy模式 + 可插拔OpenAI) + RAG研究Agent | 差异化 |
| **Web实时监控** | Bloomberg Terminal风格仪表盘 + WebSocket实时推送 + Grafana监控面板 | 生产级 |

> 完整架构文档: [CLAUDE.md](CLAUDE.md) | 面试指南: [INTERVIEW_CHEATSHEET.md](INTERVIEW_CHEATSHEET.md) | 新手入门: [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)

---

## 架构概览

```
                        ┌─────────────────────────────────────────────┐
                        │         核心架构层 (core/)                  │
                        │  EventBus · Store · StateMachine · Audit   │
                        │  Scheduler · RiskMonitor · CircuitBreaker  │
                        └──────────────┬──────────────────────────────┘
                                       │ 所有组件通过EventBus通信
                                       │ 所有状态通过Store持久化
    ┌──────────────────────────────────┼──────────────────────────────────┐
    v                                  v                                  v
  数据层  ───>  因子引擎  ───>  Alpha模型  ───>  组合优化器
 (合成数据/      (15个因子)     (IC/ICIR/ML    (EW/MVO/RP)
  Tushare/                      信号)
  Baostock/
  WebSocket/
  Level2)
                                                          |
                                                          v
    增强回测  <──  回测引擎  ──>  风险管理  ──>  执行层
    (WalkForward/  (向量化月频/   (VaR/压力/    (TWAP/VWAP/
    蒙特卡洛/      成本模型)      Barra/风控)     Iceberg)
    并行扫描)
                                                          |
                                                          v
                   实盘交易引擎  <──  多策略管理  <──  报告引擎
                   (AKShare+     (资本分配/      (HTML报告/
                    Paper+QMT)    聚合P&L)         Prometheus)
                                                          |
                                                          v
                              Web界面 (Vue 3 + ECharts)
                              REST API (97个端点)
                              WebSocket (实时推送)
```

### 架构设计原则

- **事件驱动**：EventBus 解耦所有组件，topic-based pub/sub，通配符匹配，死信队列
- **全状态持久化**：SQLite WAL 模式，8张表（订单/持仓/成交/PnL/信号/会话/事件/配置）
- **状态机管理**：8个生命周期状态（INIT→READY→TRADING→REBALANCING→POST_MARKET），合法转换强制校验
- **合规审计**：每个信号/下单/成交/状态变更都记录 who/what/when/why/result
- **ABC 抽象接口**：DataProvider / BaseFactor / PortfolioOptimizer 全部可插拔

---

## 核心模块

### 1. 核心架构层 `core/` — 平台的神经系统

| 模块 | 功能 | 关键特性 |
|------|------|----------|
| `events.py` | EventBus | topic pub/sub, 通配符, 拦截器, 死信队列, 环形缓冲历史 |
| `store.py` | SQLite持久化 | WAL模式, 8张表, 线程安全, 索引优化 |
| `state_machine.py` | 状态机 | 8状态, 合法转换校验, entry/exit hooks |
| `scheduler.py` | 交易调度器 | A股开市时间(9:30-11:30,13:00-15:00), 自动状态切换 |
| `audit.py` | 合规审计 | 三路输出: SQLite+EventBus+Logger |
| `context.py` | 多租户上下文 | contextvars 协程安全, tenant_id隔离 |
| `instrument.py` | 跨资产抽象 | InstrumentType/Instrument/AssetUniverse, 股票/ETF/期货/期权 |

### 2. 数据层 `data/` — 多源数据流水线

| 模块 | 功能 |
|------|------|
| `providers/synthetic.py` | 合成A股数据 (500只/5年/可复现/嵌入式Alpha) |
| `providers/tushare_loader.py` | Tushare Pro 实盘 (CSI300/前复权/HDF5缓存) |
| `providers/baostock_provider.py` | Baostock 免费数据 (无需API key, 日/周/月/实时) |
| `providers/postgres_provider.py` | PostgreSQL/TimescaleDB (连接池+asyncpg+SQLite回退) |
| `providers/websocket_provider.py` | WebSocket实时行情 (东方财富/新浪推送) |
| `providers/level2_provider.py` | Level 2盘口 (10档买卖队列+逐笔成交+VWAP+微观结构因子) |
| `providers/fundamental_realtime.py` | 实时基本面 (PE/PB/ROE/ROA/毛利率+缓存+选股器) |
| `providers/connection_pool.py` | 连接池 (多源路由+熔断器+健康检查) |
| `pipeline.py` | ETL流水线 (ST过滤/停牌处理/复权/对齐) |
| `quality.py` | 数据质量监控 (8项检查+严重性分级) |

### 3. 因子引擎 `factors/` — 15个因子 + 评估体系

**10个技术因子**：
| 因子 | 参数 | 说明 |
|------|------|------|
| momentum_1m | period=21 | 过去1个月累计收益 |
| momentum_3m | period=63 | 过去3个月累计收益 |
| momentum_6m | period=126 | 过去6个月累计收益 |
| momentum_12m | period=252, skip=21 | 过去12个月(跳过最近1个月避免反转) |
| volatility_20d | period=20 | 20日波动率 |
| volatility_60d | period=60 | 60日波动率 |
| turnover_20d | period=20 | 20日平均换手率 |
| rsi_14d | period=14 | 相对强弱指数 |
| macd | fast=12, slow=26, signal=9 | MACD 离差值 |
| amplitude_20d | period=20 | 20日平均振幅 |

**5个基本面因子**：log_market_cap / pb_ratio / pe_ratio / roe / asset_growth

**因子处理流水线**：原始值 → 缩尾(1%/99%) → 标准化(zscore/rank) → 行业+市值中性化(回归残差)

**评估体系**：Rank IC / Pearson IC / ICIR / 分位数收益 / 因子相关性矩阵 / IC衰减曲线

**增强模块**：
- **图网络因子** (`network.py`)：股票关联网络 + PageRank/特征向量/介数/度 中心性度量
- **因子正交化** (`orthogonalization.py`)：Gram-Schmidt / PCA / 对称正交
- **因子择时** (`factor_timing.py`)：RegimeBasedTimer + 指数平滑 + regime权重调整
- **IC实时监控** (`ic_monitor.py`)：滚动IC/ICIR + 衰减检测 + 半衰期估计 + 自适应权重 + 三级告警

### 4. Alpha模型 `alpha/` — 信号生成

- **3种合成方法**：等权 / IC加权 / ICIR加权
- **ML信号** (`ml_signal.py`)：XGBoost/LightGBM + Walk-Forward时序CV + SHAP可解释性 + 自动重训练
- **市场状态因子** (`regime.py`)：CompositeRegimeDetector 集成到Alpha流水线

### 5. 组合优化 `portfolio/`

| 优化器 | 求解器 | 说明 |
|--------|--------|------|
| EqualWeight | — | 1/N 等权基准 |
| MeanVariance | cvxpy | 最大化 `w·α - γ·w'Σw` |
| RiskParity | cvxpy | 等风险贡献 |

**约束条件**：纯多头 / 单票≤5% / 行业≤30% / 换手≤30% / 手数100股
**协方差估计**：样本协方差 / Ledoit-Wolf收缩 (Numba加速) / EWMA

### 6. 回测引擎 `backtest/` — 向量化 + 增强分析

- **向量化回测**：月频调仓 + 日频持仓漂移
- **A股成本模型**：佣金0.03%(双边) + 印花税0.1%(仅卖出) + 滑点
- **Walk-Forward验证**：滚动/扩展窗口OOS测试，折内重算信号防止泄漏
- **蒙特卡洛模拟**：Block Bootstrap + Student-t参数化模拟
- **并行回测**：ProcessPoolExecutor 多进程参数扫描
- **策略容量估算**：参与率限制 + 冲击成本 + AUM-收益曲线

### 7. 执行层 `execution/` — 机构级订单处理

| 模块 | 功能 |
|------|------|
| `oms.py` | 订单管理系统：订单生命周期 + SimulatedExchange模拟撮合 |
| `algorithms.py` | TWAP(等时间切片) / VWAP(成交量加权) / Iceberg(冰山隐藏) + SmartRouter |
| `tca.py` | TCA分析：Implementation Shortfall / Arrival Price / VWAP分解 |
| `paper_broker.py` | 增强Paper Trading：延迟模拟(零→五级) / 随机部分成交 / 撤单失败模拟 / L2回放 |

### 8. 风险管理 `risk/` — 多维度风控体系

| 模块 | 功能 |
|------|------|
| `circuit_breaker.py` | 实时风控：仓位/行业/亏损/回撤/订单频率限额 + 5级风险等级 + Kill Switch |
| `var.py` | VaR (历史/参数/蒙特卡洛) + CVaR |
| `stress.py` | 压力测试：2008金融危机 / 2015股灾 / 2020新冠 |
| `barra.py` | Barra 10因子风险模型：横截面回归 + Ledoit-Wolf收缩 + 风险归因 |
| `regime.py` | 行情状态检测：波动率(40%) + 趋势(35%) + 相关性(25%) |
| `healthcheck.py` | 开盘前自检：数据连接/资金/持仓/路由/风控限额 → 失败阻断发单 |

### 9. 实盘交易 `trading/` — 面对市场的核心模块

| 模块 | 功能 |
|------|------|
| `broker.py` | 券商接口：SimulatedBroker(Paper) + QMTBroker(xtquant) + XTPBroker + BrokerRegistry工厂 |
| `engine.py` | 交易引擎：后台线程 + 实时行情 → 多因子信号 → 风控预检 → 下单 → P&L跟踪 |
| `realtime.py` | AKShare实时行情：全市场快照 / 个股报价 / 涨跌榜 / 板块数据 |
| `live_runner.py` | 实盘试跑：双轨执行(主+影子) + 每日报告 + SessionReport |
| `qmt_utils.py` | QMT工具：symbol映射 / 错误码 / 订单状态转换 / 成交解析 |

**交易流程**：
```
AKShare实时行情 → 多因子信号(动量+低波+RSI反转+MACD) → 目标组合权重
→ 风控预检(RiskMonitor.check_pre_trade) → 下单(SimulatedBroker/QMTBroker)
→ 实时P&L → EventBus广播 → WebSocket推送
```

### 10. LLM增强模块 `agent/` — 面试差异化亮点

- **LLMSentimentFactor**：财经新闻标题 → 情绪因子，Strategy模式(KeywordAnalyzer ↔ OpenAIAnalyzer)，30条中文模板，JSON缓存
- **ResearchAgent**：RAG风格研报分析 + 因子假设生成 + 归因分析 + 风险叙述

### 11. 合规与基金运营

| 模块 | 功能 |
|------|------|
| `compliance/exporter.py` | 合规导出：交易/委托/风控日志CSV/Excel，中英对照字段名 |
| `operations/nav.py` | NAV计算：日频净值，管理费(年化/252)，业绩报酬(高水位法) |
| `operations/investor.py` | 投资人视图：净值曲线/收益/回撤/Sharpe/月度收益矩阵(隐藏持仓细节) |

### 12. 多策略管理 `strategy/`

机构级 Multi-Pod 结构：
- 策略注册/移除/资本分配
- 聚合P&L：加权收益、策略相关性矩阵
- 风控告警：per-strategy回撤限额检测

### 13. Web界面

| 视图 | 组件 | 功能 |
|------|------|------|
| **Terminal** | TerminalDashboard | Bloomberg风格主仪表盘 (11行×20+面板) |
| **交易** | LiveTrading | 实盘交易引擎：Paper+QMT+实时行情+持仓P&L+状态机+风控+审计 |
| **实时组合** | LivePortfolio | 实时组合追踪 (Baostock) |
| **订单管理** | OrderBlotter | 订单管理+持仓+TCA |
| **因子排名** | FactorRanking | 因子IC排名 |
| **多策略对比** | StrategyCompare | 多策略并排对比 |
| **参数扫描** | ParamSweep | 参数网格搜索 |
| **监控大屏** | MonitorDashboard | Bloomberg风格监控：风控/TCA/因子/容量/配置6面板 |

**97个REST API端点 + WebSocket实时推送**

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行完整流水线

```bash
# 合成数据（无需API key，约3分钟）
python main.py run

# 强制重算（忽略缓存）
python main.py run --force

# Baostock 实盘数据（免费，无需API key）
python main.py run --use-baostock
```

### 策略对比与扫描

```bash
# 对比3种优化器
python main.py compare --optimizers equal_weight,mean_variance,risk_parity

# 参数网格搜索
python main.py sweep --optimizers equal_weight,mean_variance --frequencies monthly,weekly
```

### ML Alpha信号

```bash
# 训练ML模型
python main.py ml train --model lightgbm

# 生成ML信号
python main.py ml signal --model xgboost
```

### 实盘交易

```bash
# Paper Trading（默认，30天仿真）
python main.py trade

# 自定义参数
python main.py trade --broker paper --days 60 --universe "600519,000858,000001,300750"

# QMT实盘（需miniQMT + xtquant）
export QMT_PASSWORD="your_sim_password"
python main.py trade --broker qmt --days 30

# 单交易周期执行
python main.py trade --run-once
```

### 启动Web服务

```bash
python main.py web
# 访问 http://localhost:8000
# API文档 http://localhost:8000/api/docs
```

### 运行测试

```bash
# 全部1077个测试
pytest tests/ -v

# 按模块测试
pytest tests/test_core/ -v
pytest tests/test_trading/ -v
pytest tests/test_execution/ -v
```

---

## CLI 命令

```bash
python main.py run                      # 完整流水线
python main.py run --force              # 强制重算
python main.py run --use-baostock       # Baostock实盘数据
python main.py analyze                  # 分析已有结果
python main.py compare                  # 策略对比
python main.py sweep                    # 参数网格搜索
python main.py ml train                 # 训练ML模型
python main.py ml signal                # 生成ML信号
python main.py trade                    # 实盘试跑 (Paper Trading)
python main.py research report          # LLM研究分析
python main.py profile                  # 流水线性能分析
python main.py web                      # 启动Web服务
python main.py cache list               # 查看缓存
python main.py cache clear              # 清除缓存
```

---

## A股实盘陷阱处理

| # | 陷阱 | 处理方案 |
|:--|:-----|:---------|
| 1 | **复权** | Tushare前复权(qfq)；合成数据生成adj_factor |
| 2 | **停牌** | 短停牌(≤30天)前向填充；长停牌移出股票池 |
| 3 | **幸存者偏差** | 跟踪上市/退市日期，时间点股票池构建 |
| 4 | **涨跌停** | 日收益截断±10%；涨跌停标记 |
| 5 | **ST股票** | is_st标记，默认排除(±5%涨跌停/高退市风险) |
| 6 | **T+1** | 月频调仓天然规避；日频用shift(-1)次日执行 |
| 7 | **交易成本** | 佣金0.03%双边 + 印花税0.1%仅卖出 + 滑点 |
| 8 | **手数限制** | 100股=1手，优化器向下取整 |
| 9 | **除权除息** | 前复权将分红调整嵌入历史价格 |
| 10 | **行业漂移** | 取最新行业分类；动态中性化处理 |

---

## 未来函数防范

| # | 问题 | 防护方案 |
|:--|:-----|:---------|
| 1 | IC权重全量数据泄漏 | Point-in-time IC加权：每个时间点只用之前数据计算权重 |
| 2 | IC计算shift链条 | returns已在pipeline做过shift(-1)，IC计算不再重复shift |
| 3 | Walk-Forward信号泄漏 | 每个fold内用train-only数据重新计算信号 |
| 4 | 合成数据人造Alpha | Alpha强度降至IC~0.015-0.02，接近真实A股水平 |
| 5 | 默认配置过拟合 | 默认合成方法改为equal_weight，避免全量优化参数 |

---

## 技术栈

| 层级 | 技术 |
|:-----|:-----|
| 语言 | Python 3.10+ |
| 异步 | asyncio, aiohttp |
| 数据处理 | Pandas, NumPy |
| 优化 | cvxpy, SciPy |
| 机器学习 | XGBoost, LightGBM, SHAP, scikit-learn |
| 性能加速 | Numba (6个JIT内核) |
| Web框架 | FastAPI, Vue 3, Vite, ECharts |
| 数据存储 | SQLite (WAL), PostgreSQL/TimescaleDB, asyncpg |
| 消息队列 | Redis, Kafka (可插拔) |
| 实时行情 | AKShare, WebSocket, Level 2盘口 |
| 监控 | Prometheus, Grafana (16面板) |
| 容器化 | Docker, Docker Compose |
| CI/CD | GitHub Actions (Python 3.10/3.11/3.12矩阵) |

---

## 项目结构

```
quant_platform/
├── main.py                     # CLI入口
├── app.py                      # FastAPI应用
├── CLAUDE.md                   # 完整架构文档
├── BEGINNER_GUIDE.md           # 量化新手入门
├── INTERVIEW_CHEATSHEET.md     # 面试指南
│
├── core/                       # ★ 核心架构层
│   ├── events.py               # EventBus: topic pub/sub, 通配符, 死信队列
│   ├── store.py                # SQLite持久化: WAL模式, 8张表
│   ├── state_machine.py        # 状态机: 8生命周期状态
│   ├── scheduler.py            # 交易调度: A股开市时间
│   ├── audit.py                # 合规审计: 三路输出
│   ├── context.py              # 多租户上下文
│   └── instrument.py           # 跨资产抽象: 股票/ETF/期货/期权
│
├── data/                       # 数据层
│   └── providers/              # 8个数据源提供者
│
├── factors/                    # 因子引擎 (15因子)
├── alpha/                      # Alpha模型 (IC/ICIR/ML)
├── portfolio/                  # 组合优化 (EW/MVO/RP)
├── backtest/                   # 回测引擎
├── execution/                  # 执行层 (OMS/算法/TCA/PaperBroker)
├── risk/                       # 风险管理 (VaR/Barra/风控/压力)
├── trading/                    # 实盘交易 (Broker/Engine/LiveRunner)
├── strategy/                   # 多策略管理
├── compliance/                 # 合规导出
├── operations/                 # 基金运营 (NAV/投资者门户)
├── agent/                      # LLM增强模块
├── api/                        # FastAPI (97端点)
├── frontend/                   # Vue 3 (37组件)
├── monitoring/                 # Grafana监控
├── utils/                      # 工具 (Numba/缓存/配置/指标)
│
└── tests/                      # 1077个单元测试
    ├── test_core/              # 核心架构 (137测试)
    ├── test_data/              # 数据层 (105测试)
    ├── test_factors/           # 因子引擎 (95测试)
    ├── test_alpha/             # Alpha模型 (27测试)
    ├── test_portfolio/         # 组合优化 (6测试)
    ├── test_backtest/          # 回测引擎 (24测试)
    ├── test_risk/              # 风险管理 (51测试)
    ├── test_execution/         # 执行层 (94测试)
    ├── test_trading/           # 实盘交易 (112测试)
    ├── test_api/               # API (33测试)
    ├── test_compliance/        # 合规 (21测试)
    ├── test_operations/        # 基金运营 (40测试)
    ├── test_strategy/          # 多策略 (7测试)
    ├── test_agent/             # LLM (32测试)
    ├── test_reporting/         # 报告 (14测试)
    └── test_utils/             # 工具 (33测试)
```

---

## 面试亮点

### 核心技术

1. **事件驱动核心架构** — EventBus(pub/sub+通配符+死信队列) + SQLite持久化(8表+WAL) + StateMachine(8状态) + AuditLog
2. **实盘交易引擎** — AKShare实时行情 + 多因子信号 + 实时风控熔断 + Paper/QMT实盘 + WebSocket实时推送
3. **多源数据流水线** — Tushare + Baostock + AKShare + WebSocket + Level 2，前复权，HDF5缓存
4. **10个A股实盘陷阱全处理** — 有文档、有代码、能讲清楚
5. **Numba JIT加速** — 6个计算内核LLVM编译，Pandas vs Numba 加速比输出
6. **LLM Agent集成** — 财经新闻情感因子 + RAG研究Agent，展示Agent/RAG→量化迁移
7. **未来函数防范** — Point-in-time IC加权 + Walk-Forward折内重算 + 合成数据真实IC水平
8. **跨资产接口** — Instrument统一抽象，消除lot_size=100硬编码，支持股票/ETF/期货/期权

### 机构级模块

- Walk-Forward验证 / 蒙特卡洛模拟 / Barra风险模型 / 因子风险分解
- TWAP/VWAP/Iceberg执行算法 / SmartRouter智能路由
- 实时风控熔断 / Kill Switch / 5级风险等级
- 行情状态检测 / IC自动衰减 / 多策略组合管理
- 数据质量监控 / 开盘前系统自检 / 合规审计
- Prometheus指标 + Grafana监控面板

---

## 许可证

MIT
