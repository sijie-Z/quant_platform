# A-Share Multi-Factor Quantitative Trading Platform

> **A 股多因子量化研究 + 交易平台** —— 从数据到回测到实盘的完整流水线
> 面向量化开发面试，展示**机构级架构设计**能力

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Tests-610%20Passed-brightgreen?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/Modules-84-orange" alt="Modules">
  <img src="https://img.shields.io/badge/Lines-27%2C000%2B-yellow" alt="Lines">
  <img src="https://img.shields.io/badge/API-91%20Endpoints-red?logo=fastapi" alt="API">
  <img src="https://img.shields.io/badge/Factors-15-purple" alt="Factors">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="https://sijie-z.github.io/quant-platform/">Live Documentation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#modules">Modules</a> &bull;
  <a href="#cli-commands">CLI</a>
</p>

---

## Highlights

```
  Event-Driven Architecture    Real Order Book (LOB)       Tick-Level Backtest
  ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
  │ AsyncEventBus       │     │ Red-Black Tree      │     │ Event-Driven        │
  │  per-handler queue  │     │  Price-Time FIFO    │     │  Tick Replay        │
  │  backpressure       │     │  IOC/FOK            │     │  Market Impact      │
  │  P50/P99/P999       │     │  Partial Fill       │     │  TWAP/VWAP          │
  │  DLQ + WAL          │     │  VPIN Metrics       │     │  3 Impact Models    │
  └─────────────────────┘     └─────────────────────┘     └─────────────────────┘

  Real-Time Risk Engine        Cython Hot Path             Distributed Bus
  ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
  │ Per-Tick Greeks     │     │ 4 Cython Kernels    │     │ MessageBus ABC      │
  │ Pre-Trade Check     │     │  momentum           │     │  LocalBus           │
  │ Auto Delta-Hedge    │     │  volatility         │     │  RedisBus           │
  │ Kill Switch         │     │  rank_ic            │     │  KafkaBus           │
  │ 12 Stress Scenarios │     │  zscore + fallback  │     │  ServiceRegistry    │
  └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

| Capability | Implementation | Level |
|:-----------|:---------------|:------|
| **Event-Driven Architecture** | AsyncEventBus (backpressure + P50/P99/P999 latency + DLQ retry + WAL event sourcing) | Jane Street |
| **Real Order Book** | Red-black tree + Price-Time FIFO + IOC/FOK + partial fills + VPIN microstructure | Jane Street |
| **Tick-Level Backtest** | Event-driven tick replay + 3 market impact models (Almgren-Chriss / Square-Root / Kyle) | Jane Street |
| **Real-Time Risk** | Per-tick Greeks + pre-trade check + auto delta-hedge + Kill Switch + 12 stress scenarios | Jane Street |
| **Cython Hot Path** | 4 kernels (rolling_momentum / volatility / rank_ic / zscore) + Python fallback | Performance |
| **Distributed Message Bus** | MessageBus ABC + LocalBus / RedisBus / KafkaBus + ServiceRegistry | Production |
| **Microservice Skeleton** | BaseService lifecycle + RiskService / ExecutionService / DataService | Production |
| **Multi-Factor Signals** | 15 factors (10 technical + 5 fundamental) + 4-factor composite + ML signal | Research |
| **LLM Enhancement** | Sentiment factor from financial news + RAG research agent | Differentiated |
| **A-Share Pitfalls** | 10 real-world traps handled (adjacency / suspension / ST / limit-up / T+1 / costs) | Production |
| **Look-Ahead Prevention** | Point-in-time IC weighting + IC shift fix + Walk-Forward fold-recompute + realistic synthetic alpha | Production |

> Documentation: [CLAUDE.md](CLAUDE.md) | Interview Guide: [INTERVIEW_CHEATSHEET.md](INTERVIEW_CHEATSHEET.md) | Jane Street Gap Analysis: [JANE_STREET_GAP_ANALYSIS.md](JANE_STREET_GAP_ANALYSIS.md)

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │         Core Architecture (core/)          │
                        │  AsyncEventBus · Store · StateMachine      │
                        │  AuditLog · Scheduler · MessageBus         │
                        └──────────────┬──────────────────────────────┘
                                       │ All components communicate via EventBus
                                       │ All state persisted via Store
    ┌──────────────────────────────────┼──────────────────────────────────┐
    v                                  v                                  v
Data Layer  -->  Factor Engine  -->  Alpha Model  -->  Portfolio Optimizer
(Synthetic/       (15 Factors +      (IC/ICIR/ML      (EW/MVO/RP)
 Tushare/          Cython Hot Path)    Signal)
 Baostock/LLM)
                                                          |
                                                          v
  Order Book  <--  Backtest Engine  -->  Real-Time Risk  -->  Execution Layer
  (Red-Black LOB)  (Tick Event-Driven)  (Greeks+PreCheck+   (TWAP/VWAP/Iceberg)
                                          Kill Switch)
                                                          |
                                                          v
                   Live Trading Engine  <--  Multi-Strategy  <--  Report Engine
                   (AKShare+Paper+QMT)     (Capital/P&L)        (HTML/Prometheus)
                                                          |
                                                          v
                                      Web Dashboard (Vue 3 + ECharts)
                                      REST API (91 endpoints)
                                      WebSocket (Real-time Push)
```

---

## Modules

<details>
<summary><b>1. Core Architecture</b> <code>core/</code> — The nervous system</summary>

| Module | Function | Key Features |
|--------|----------|--------------|
| `event_bus_v2.py` | Async Event Bus | Per-handler asyncio.Queue, backpressure, P50/P99/P999 latency histograms, DLQ exponential backoff retry, WAL event sourcing |
| `events.py` | Event Bus Bridge | Backward-compatible `get_event_bus()`, auto-detect sync/async handlers |
| `store.py` | SQLite Persistence | WAL mode, 8 tables, thread-safe |
| `state_machine.py` | Portfolio State Machine | 8 lifecycle states (INIT->READY->TRADING->REBALANCING->POST_MARKET) |
| `audit.py` | Compliance Audit | Every decision recorded: who/what/when/why/result |
| `scheduler.py` | Trading Scheduler | A-share market hours detection, auto state transitions |
| `message_bus.py` | Distributed Message Bus | MessageBus ABC + LocalBus/RedisBus/KafkaBus + ServiceRegistry |

</details>

<details>
<summary><b>2. Data Layer</b> <code>data/</code> — Multi-source real-time data</summary>

| Module | Function |
|--------|----------|
| `providers/synthetic.py` | Synthetic A-share data (500 stocks / 5 years / reproducible / embedded alpha) |
| `providers/tushare_loader.py` | Tushare Pro live data (CSI 300 / forward-adjusted / HDF5 cache) |
| `providers/baostock_provider.py` | Baostock free data (no API key required) |
| `providers/postgres_provider.py` | PostgreSQL/TimescaleDB (connection pool + asyncpg + SQLite fallback) |
| `providers/websocket_provider.py` | WebSocket real-time quotes (Eastmoney/Sina push) |
| `providers/level2_provider.py` | Level 2 order book (10-level bid/ask + tick data + VWAP + microstructure factors) |
| `providers/fundamental_realtime.py` | Real-time fundamentals (PE/PB/ROE + TTL cache + screener) |
| `pipeline.py` | ETL pipeline (ST filter / suspension handling / adjustment / alignment) |
| `quality.py` | Data quality monitoring (8 checks + severity levels) |

</details>

<details>
<summary><b>3. Factor Engine</b> <code>factors/</code> — 15 factors + Cython acceleration</summary>

**10 Technical Factors**: momentum_1m/3m/6m/12m, volatility_20d/60d, turnover_20d, rsi_14d, macd, amplitude_20d

**5 Fundamental Factors**: log_market_cap, pb_ratio, pe_ratio, roe, asset_growth

**Pipeline**: Raw -> Winsorize (1%/99%) -> Standardize (zscore/rank) -> Neutralize (industry + market cap)

**Evaluation**: Rank IC / ICIR / Quantile Returns / Correlation Matrix / IC Decay Curve

**Network Factors** (`network.py`): Stock association network + 4 centrality measures (PageRank / Eigenvector / Betweenness / Degree)

**IC Monitor** (`ic_monitor.py`): Rolling IC/ICIR + decay detection + half-life estimation + adaptive weights

</details>

<details>
<summary><b>4. Alpha Model</b> <code>alpha/</code> — Signal generation</summary>

- **3 Combination Methods**: equal_weight / ic_weighted / icir_weighted
- **ML Signal** (`ml_signal.py`): XGBoost/LightGBM + Walk-Forward CV + SHAP interpretability
- **LLM Sentiment** (`agent/sentiment_factor.py`): Financial news headlines -> sentiment factor, Strategy pattern with pluggable OpenAI

</details>

<details>
<summary><b>5. Execution Layer</b> <code>execution/</code> — Institutional-grade order handling</summary>

| Module | Function |
|--------|----------|
| `order_book.py` | **Real LOB**: Red-black tree bid/ask + FIFO PriceLevel + IOC/FOK + partial fills + L1/L2/L3 snapshots + VPIN |
| `market_impact.py` | **Market Impact Models**: Almgren-Chriss + Square-Root + Kyle's Lambda + weighted ensemble |
| `algorithms.py` | TWAP / VWAP / Iceberg + SmartRouter intelligent routing |
| `oms.py` | Order Management System: order lifecycle + SimulatedExchange |

</details>

<details>
<summary><b>6. Backtest Engine</b> <code>backtest/</code> — Event-driven tick replay</summary>

| Module | Function |
|--------|----------|
| `engine.py` | Vectorized monthly backtest + daily position drift |
| `tick_engine.py` | **Tick-level event-driven backtest**: tick replay + real LOB matching + market impact simulation + TWAP/VWAP |
| `cost_model.py` | A-share costs: commission 0.03% + stamp tax 0.1% (sell) + slippage |
| `walkforward.py` | Walk-Forward validation (rolling/expanding window OOS testing) |
| `distributed.py` | Parallel backtest (ProcessPoolExecutor parameter sweep) |

</details>

<details>
<summary><b>7. Risk Management</b> <code>risk/</code> — Real-time Greeks + Kill Switch</summary>

| Module | Function |
|--------|----------|
| `realtime_engine.py` | **Real-Time Risk Engine**: Per-tick Greeks update + pre-trade check + auto delta-hedge + Kill Switch + 12 stress scenarios |
| `greeks.py` | Black-Scholes full Greeks (Delta/Gamma/Vega/Theta/Rho) + portfolio aggregation + delta-hedge calculation |
| `circuit_breaker.py` | RiskMonitor: position/industry/loss/drawdown limits + 5 risk levels |
| `var.py` | VaR (historical/parametric/Monte Carlo) + CVaR |
| `stress.py` | Stress testing: 2008 Financial Crisis / 2015 A-Share Crash / 2020 COVID |
| `barra.py` | Barra 10-factor risk model: cross-sectional regression + Ledoit-Wolf shrinkage + risk attribution |
| `regime.py` | Market regime detection: volatility / trend / correlation |

</details>

<details>
<summary><b>8. Live Trading</b> <code>trading/</code> — Production-ready</summary>

| Module | Function |
|--------|----------|
| `broker.py` | **SimulatedBroker** (real LOB matching) + QMTBroker (xtquant live trading) |
| `engine.py` | **Live Trading Engine**: AKShare real-time quotes -> multi-factor signal -> RealTimeRiskEngine pre-check -> LOB order -> P&L tracking |
| `realtime.py` | AKShare real-time quotes: full market snapshot / individual quotes / gainers/losers |

</details>

<details>
<summary><b>9. Performance</b> <code>utils/cyext/</code> — Cython + Numba acceleration</summary>

**4 Cython Kernels** (.pyx source + Python fallback):
- `rolling_momentum`: Rolling momentum (log returns)
- `rolling_volatility`: Rolling volatility (Welford single-pass algorithm)
- `rank_ic`: Spearman Rank IC
- `zscore_cross_section`: Cross-sectional Z-Score normalization

**6 Numba JIT Kernels**: Rolling cumulative returns / Max drawdown / Winsorize / Rank IC / Ledoit-Wolf / Z-Score

</details>

<details>
<summary><b>10. Monitoring</b> <code>utils/metrics.py</code> + <code>monitoring/</code></summary>

- **Prometheus Metrics**: Counter/Gauge/Histogram + Timer decorator + `/api/metrics` endpoint
- **Grafana Template**: 16-panel one-click import (Pipeline / API / Risk / Factors / EventBus)
- **Structured Logging**: JSON format + log level configuration

</details>

<details>
<summary><b>11. Web Interface</b> <code>app.py</code> + <code>frontend/</code></summary>

- **FastAPI**: 91 REST API endpoints
- **Vue 3 Frontend**: Bloomberg Terminal-style dark dashboard + 8 views
- **WebSocket**: EventBus -> WebSocket bridge, real-time trading event push

</details>

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# Synthetic data (no API key, ~3 minutes)
python main.py run

# Force recalculate (ignore cache)
python main.py run --force

# Baostock live data (free, no API key)
python main.py run --use-baostock

# Tushare live data (requires token)
export TUSHARE_TOKEN=your_token
python main.py run
```

### Strategy Comparison

```bash
# Compare 3 optimizers
python main.py compare --optimizers equal_weight,mean_variance,risk_parity

# Parameter grid search
python main.py sweep --optimizers equal_weight,mean_variance --frequencies monthly,weekly
```

### ML Alpha Signal

```bash
# Train ML model
python main.py ml train --model lightgbm

# Generate ML signal
python main.py ml signal --model xgboost
```

### Web Service

```bash
# Start FastAPI + Vue frontend
python main.py web

# API docs
open http://localhost:8000/api/docs
```

### Test

```bash
# Run all 610 tests
pytest tests/ -v

# Run core architecture tests only
pytest tests/test_core/ -v

# Run new module tests
pytest tests/test_core/test_event_bus_v2.py tests/test_execution/test_order_book.py tests/test_risk/test_realtime_engine.py tests/test_backtest/test_tick_engine.py tests/test_utils/test_cyext.py tests/test_core/test_message_bus.py -v
```

---

## Configuration

Edit `config/default.yaml`:

```yaml
portfolio.optimizer: "equal_weight" | "mean_variance" | "risk_parity"
alpha.method: "equal_weight" | "ic_weighted" | "icir_weighted"
universe.n_stocks: 100 | 300 | 500
backtest.rebalance_frequency: "daily" | "weekly" | "monthly"
portfolio.covariance.method: "sample" | "ledoit_wolf" | "ewma"
risk.var.method: "historical" | "parametric" | "monte_carlo"
```

---

## CLI Commands

```
python main.py run                      # Full pipeline
python main.py run --force              # Force recalculate
python main.py run --use-baostock       # Baostock data
python main.py analyze                  # Analyze existing results
python main.py compare                  # Strategy comparison
python main.py sweep                    # Parameter grid search
python main.py ml train --model lightgbm  # Train ML model
python main.py ml signal --model xgboost  # Generate ML signal
python main.py research report          # LLM research analysis
python main.py profile                  # Performance profiling
python main.py web                      # Start web service
python main.py cache list               # View cache
python main.py cache clear              # Clear cache
```

---

## A-Share Pitfalls

The platform explicitly handles 10 real-world A-share trading traps:

| # | Trap | Solution |
|:--|:-----|:---------|
| 1 | Forward Adjustment | Tushare qfq; synthetic data generates adj_factor |
| 2 | Suspension | Short suspension (<=30d) forward-fill; long suspension removed from universe |
| 3 | Survivorship Bias | Track listing/delisting dates, point-in-time universe construction |
| 4 | Limit Up/Down | Daily returns capped at +/-10%; limit flags marked |
| 5 | ST Stocks | is_st flag, excluded by default |
| 6 | T+1 | Monthly rebalance naturally avoids; daily uses shift(-1) next-day execution |
| 7 | Trading Costs | Commission 0.03% bilateral + stamp tax 0.1% (sell only) + slippage |
| 8 | Lot Size | Optimizer rounds down to 100-share multiples |
| 9 | Ex-Dividend | Forward adjustment embeds dividend adjustments into historical prices |
| 10 | Industry Drift | Use latest industry classification; dynamic neutralization |

---

## Project Structure

```
quant_platform/
├── main.py                          # CLI entry point
├── app.py                           # FastAPI application
├── CLAUDE.md                        # Full architecture documentation
├── INTERVIEW_CHEATSHEET.md          # Interview Q&A guide
├── JANE_STREET_GAP_ANALYSIS.md      # Jane Street gap analysis
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker deployment
├── docker-compose.yml               # Docker Compose
├── .github/workflows/ci.yml         # CI/CD
│
├── config/
│   ├── default.yaml                 # Default configuration
│   └── schema.py                    # Typed config validation
│
├── core/                            # Core Architecture
│   ├── event_bus_v2.py              # AsyncEventBus (backpressure+DLQ+WAL+latency monitoring)
│   ├── events.py                    # Event bus bridge (backward compat)
│   ├── message_bus.py               # Distributed message bus (Local/Redis/Kafka)
│   ├── store.py                     # SQLite persistence
│   ├── state_machine.py             # Portfolio state machine
│   ├── audit.py                     # Compliance audit
│   └── scheduler.py                 # Trading scheduler
│
├── data/                            # Data Layer
│   ├── providers/                   # 8 data providers
│   ├── pipeline.py                  # ETL pipeline
│   └── quality.py                   # Data quality monitoring
│
├── factors/                         # Factor Engine
│   ├── technical.py                 # 10 technical factors
│   ├── fundamental.py               # 5 fundamental factors
│   ├── processing.py                # Cross-sectional processing
│   ├── evaluation.py                # IC evaluation
│   ├── ic_monitor.py                # IC monitoring
│   └── network.py                   # Graph network factors
│
├── alpha/                           # Alpha Model
│   ├── combination.py               # 3 combination methods
│   ├── pipeline.py                  # AlphaPipeline
│   └── ml_signal.py                 # ML signal
│
├── portfolio/                       # Portfolio Optimization
│   ├── optimizers.py                # EW/MVO/RP
│   ├── covariance.py                # Covariance estimation
│   └── constraints.py               # Constraints
│
├── backtest/                        # Backtest Engine
│   ├── engine.py                    # Vectorized backtest
│   ├── tick_engine.py               # Tick-level event-driven backtest
│   ├── cost_model.py                # Cost model
│   ├── walkforward.py               # Walk-Forward validation
│   └── distributed.py               # Parallel backtest
│
├── execution/                       # Execution Layer
│   ├── order_book.py                # Real LOB (red-black tree+FIFO+IOC/FOK+VPIN)
│   ├── market_impact.py             # Market impact models (AC/SR/Kyle)
│   ├── algorithms.py                # TWAP/VWAP/Iceberg
│   └── oms.py                       # Order management
│
├── risk/                            # Risk Management
│   ├── realtime_engine.py           # Real-time risk (Greeks+precheck+Kill Switch)
│   ├── greeks.py                    # Black-Scholes Greeks
│   ├── circuit_breaker.py           # RiskMonitor
│   ├── var.py                       # VaR/CVaR
│   ├── stress.py                    # Stress testing
│   ├── barra.py                     # Barra 10-factor model
│   └── regime.py                    # Market regime detection
│
├── trading/                         # Live Trading
│   ├── broker.py                    # SimulatedBroker(LOB) + QMTBroker
│   ├── engine.py                    # Live trading engine
│   └── realtime.py                  # AKShare real-time quotes
│
├── services/                        # Microservice Skeleton
│   ├── base.py                      # BaseService lifecycle
│   ├── risk_service.py              # RiskService
│   ├── execution_service.py         # ExecutionService
│   └── data_service.py              # DataService
│
├── agent/                           # LLM Module
│   ├── sentiment_factor.py          # Sentiment factor
│   └── research_agent.py            # RAG research agent
│
├── utils/                           # Utilities
│   ├── cyext/                       # Cython hot path (3 .pyx + setup.py)
│   ├── numba_accelerator.py         # 6 Numba JIT kernels
│   ├── metrics.py                   # Prometheus metrics
│   ├── cache.py                     # Pipeline cache
│   └── config.py                    # YAML config loading
│
├── api/                             # Web API
│   ├── routes.py                    # 91 FastAPI endpoints
│   └── schemas.py                   # Pydantic models
│
├── frontend/                        # Vue 3 Frontend
│   └── src/components/              # 37 Vue components
│
├── monitoring/
│   └── grafana_dashboard.json       # Grafana 16-panel template
│
├── docs/
│   └── index.html                   # GitHub Pages architecture documentation
│
└── tests/                           # 610 Unit Tests
    ├── test_core/                   # EventBus(13)+EventBus v2(25)+MessageBus(16)+Store(16)+StateMachine(15)+Audit(10)+Scheduler(2)
    ├── test_execution/              # OrderBook(22)+OMS(17)+Algorithms(13)+MarketImpact(10)
    ├── test_risk/                   # RealTimeRisk(12)+Greeks(8)+VaR(7)+Barra(16)+...
    ├── test_backtest/               # TickEngine(15)+Cost(4)+Metrics(7)+WalkForward(2)
    ├── test_utils/                  # Cython(18)+Numba+Cache+Config+Metrics
    └── ...                          # 610 tests total
```

---

## Jane Street Comparison

| Dimension | This Project | Jane Street | Gap |
|:----------|:-------------|:------------|:----|
| Event Bus | AsyncEventBus (backpressure+DLQ+WAL) | Similar architecture | **Matched** |
| Order Book | Red-black tree + FIFO + IOC/FOK + VPIN | Similar architecture | **Matched** |
| Backtest | Tick-level event-driven + market impact | Tick-level event-driven | **Matched** |
| Risk | Per-tick Greeks + pre-check + Kill Switch | Per-tick pre-check | **Matched** |
| Latency | ~10ms (Python) | ~10us (OCaml+FPGA) | **1000x** |
| Language | Python (runtime types) | OCaml (compile-time types) | **Fundamental** |
| Data | Level 2 | Level 3 + cross-market | **Resource** |
| Strategy | Monthly multi-factor stock selection | Microsecond market-making + arbitrage | **Category** |

> Detailed analysis: [JANE_STREET_GAP_ANALYSIS.md](JANE_STREET_GAP_ANALYSIS.md)

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python 3.10+ |
| Async | asyncio, aiohttp, aiokafka, redis.asyncio |
| Data | Pandas, NumPy, SQLAlchemy, asyncpg |
| Optimization | cvxpy, SciPy |
| ML | XGBoost, LightGBM, SHAP, scikit-learn |
| Performance | Cython, Numba |
| Web | FastAPI, Vue 3, Vite, ECharts |
| Monitoring | Prometheus, Grafana |
| Container | Docker, Docker Compose |
| CI/CD | GitHub Actions (Python 3.10/3.11/3.12 matrix) |

---

## License

MIT
