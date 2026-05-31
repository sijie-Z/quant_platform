# A-Share Multi-Factor Quant Platform · A股多因子量化交易平台

> **From data to backtest to live trading** — Event-driven · Real-time risk controls · Multi-factor signals · A-Share production trading engine
> 
> **从数据到回测到实盘的完整量化流水线** —— 事件驱动架构 · 实时风控熔断 · 多因子信号 · A股实盘交易引擎

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Tests-1205%20passed-brightgreen?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/Factors-26-orange" alt="Factors">
  <img src="https://img.shields.io/badge/Python%20modules-105-blueviolet" alt="Modules">
  <img src="https://img.shields.io/badge/Code%20lines-34K%2B-yellow" alt="Lines">
  <img src="https://img.shields.io/badge/API%20endpoints-97-red?logo=fastapi" alt="API">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#cli-commands">CLI</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#factors">Factors</a> •
  <a href="#risk-controls">Risk</a> •
  <a href="#execution">Execution</a> •
  <a href="#project-structure">Structure</a>
</p>

---

## 🌟 Highlights

| Capability | Implementation | Level |
|:-----------|:---------------|:------|
| **1205 unit tests** | 105 Python modules, 34K+ lines, 37 Vue components | Enterprise |
| **26 factors** | 22 technical + 4 fundamental + expression engine (35 functions) | Research |
| **Event-driven core** | EventBus (topic pub/sub + wildcards + dead-letter queue) | Enterprise |
| **Real-time risk controls** | 8-layer: VaR/Barra/MC/Regime/CircuitBreaker/KillSwitch/HealthCheck/Connection | Enterprise |
| **A-share traps handled** | 10 pitfalls: adjusted prices, suspensions, ST, T+1, survival bias, price limits, costs, lot sizes, ex-rights, industry drift | Production |
| **Lookahead bias prevention** | Point-in-time IC weighting + Walk-Forward per-fold recompute + Lookahead Detector tool | Production |
| **Live trading engine** | Paper/QMT/XTP brokers + real-time risk monitor + WebSocket push | Production |
| **ML alpha signals** | XGBoost/LightGBM + Walk-Forward CV + SHAP explainability | Research |
| **Multi-source data** | Baostock, Tushare, Adata, AKShare, WebSocket, Level 2, PostgreSQL, Synthetic | Production |
| **Expression factor engine** | String-based factor formulas: `ts_rank(ts_sum(close_pct,5)/ts_std(close_pct,20),10)` | Research |
| **LLM integration** | DeepSeek sentiment factor + research agent + multi-model depot | Differentiator |
| **Barra risk model** | 10-factor cross-sectional regression + Ledoit-Wolf shrinkage + risk attribution | Enterprise |
| **Web dashboard** | Bloomberg-terminal style + WebSocket real-time + 97 REST API endpoints | Production |

---

## 🇨🇳 中文说明

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流水线（合成数据，无需API key）
python main.py run

# Baostock 实盘数据（免费，无需API key）
python main.py run --use-baostock

# 启动Web服务
python main.py web
# 访问 http://localhost:8000  |  API文档 http://localhost:8000/api/docs
```

### 测试

```bash
# 全部1205个测试
pytest tests/ -q

# 按模块
pytest tests/test_factors/ -q
pytest tests/test_execution/ -q
```

### CLI 命令

```bash
python main.py run                      # 完整流水线
python main.py run --force              # 强制重算（忽略缓存）
python main.py run --use-baostock       # Baostock 实盘
python main.py run --description "v2"   # 带版本描述

python main.py compare                  # 多策略对比
python main.py sweep                    # 参数网格搜索
python main.py ml train --model lightgbm  # 训练ML模型
python main.py ml signal --model xgboost  # 生成ML信号

python main.py screen --rules "pe_ratio=lt:30,roe=gt:0.15"  # 因子条件选股
python main.py check-lookahead          # 未来函数检测
python main.py config list              # 配置版本管理
python main.py config diff v1 v3        # 版本对比
python main.py config rollback v2       # 回滚配置
python main.py execute                  # 全流程执行（数据→因子→信号→订单→持仓）

python main.py trade                    # Paper Trading 实盘
python main.py trade --broker qmt --days 30  # QMT实盘

python main.py web                      # Web控制台
python main.py profile                  # 性能分析
python main.py cache clear              # 清除缓存
```

### 26个因子一览

**技术因子（22个）：**
| 类别 | 因子 | 说明 |
|------|------|------|
| 动量 | momentum_1m/3m/6m/12m | 1-12个月累计收益 |
| 波动率 | volatility_20d/60d | 20/60日波动率 |
| 基础技术 | turnover_20d, rsi_14d, amplitude_20d, macd | 换手率/RSI/振幅/MACD |
| 趋势质量 | efficiency_ratio | 路径效率比，区分流畅趋势vs噪音震荡 |
| 突破信号 | breakout_ignition | 放量+异动复合信号 |
| 趋势阶段 | trend_stage | 价格在120日区间位置（鱼头/鱼身/鱼尾） |
| 均线汇聚 | ma_convergence | MA5/10/20趋近程度，积蓄突破能量 |
| 突破临近 | breakout_proximity | 距20日高点距离 |
| 特质波动率 | pure_volatility | FF3回归残差标准差，去噪后的纯特质波动 |
| 多周期共振 | mtf_resonance | 日/周/月线趋势一致性 |
| K线形态 | kmid/klen/kup/klow/ksft | 5个蜡烛图特征因子 |
| **基本面（4个）：** | log_market_cap, pb_ratio, pe_ratio, roe, asset_growth | 市值/PB/PE/ROE/资产增长 |

**表达式引擎（35个函数可无限组合）：**
```python
# 例：过去10天（5天收益/20天波动率）的百分位排名
"ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"
```

### 因子筛选模式

三种信号合成方式：
| 模式 | 命令 | 说明 |
|------|------|------|
| Ranking（排名） | `main.py run` | ICIR加权 → 优化器 → 组合权重 |
| Screener（筛选） | `main.py screen --rules "pe<30,roe>0.15"` | Bool规则 → 等权入选 |
| Vote（投票） | `config: alpha.method: vote` | 因子独立投票 → 多数决 |

---

## 🇬🇧 English

### Quick Start

```bash
pip install -r requirements.txt
python main.py run                          # Full pipeline (synthetic data)
python main.py run --use-baostock           # Real A-share data (free)
python main.py web                          # Web UI: http://localhost:8000
```

### Tests

```bash
pytest tests/ -q                            # All 1205 tests
pytest tests/test_execution/test_engine.py -v  # Specific module
```

### CLI Commands

```bash
python main.py run                      # Full pipeline
python main.py run --force              # Bypass cache
python main.py run --use-baostock       # Baostock real data
python main.py run --description "v2"   # Tag config version

python main.py compare                  # Strategy comparison
python main.py sweep                    # Parameter grid search
python main.py ml train --model lightgbm  # Train ML model
python main.py ml signal --model xgboost  # Generate ML signals

python main.py screen --rules "pe_ratio=lt:30,roe=gt:0.15"  # Boolean screening
python main.py check-lookahead          # Lookahead bias detection
python main.py config list              # Config version list
python main.py config diff v1 v3        # Version diff
python main.py config rollback v2       # Rollback config
python main.py execute                  # Full execution chain

python main.py trade                    # Paper trading
python main.py trade --broker qmt --days 30  # QMT live trading

python main.py web                      # Web dashboard
python main.py profile                  # Performance profiling
python main.py cache clear              # Clear pipeline cache
```

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │         Core Architecture (core/)           │
                        │  EventBus · Store · StateMachine · Audit    │
                        │  Scheduler · RiskMonitor · CircuitBreaker  │
                        └──────────────┬──────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    v                                  v                                  v
 Data Layer ──→ Factor Engine ──→ Alpha Model ──→ Portfolio Optimizer
 (6 providers)  (26 factors +    (ICIR/Vote/     (MVO/RP/EW)
                  expr engine)     Screener/ML)
                                                                         |
                                                                         v
    Enhanced ←── Backtest Engine ←── Risk Module ←── Execution Layer
    Backtest     (vectorized/      (VaR/Barra/      (OMS/Algorithms/
    (WalkFW/MC/    cost model)       MC/CircuitBr)     TCA/Broker)
     ParamSweep)
                                                                         |
                                                                         v
                    Live Trading Engine  ←── Portfolio  ←── Report Engine
                    (Paper/QMT/XTP)       Orchestrator     (Dashboard/
                                                           Prometheus/
                                                           Alphalens)
                                                                         |
                                                                         v
                              Web UI (Vue 3 + ECharts)
                              REST API (97 endpoints)
                              WebSocket (real-time push)
```

### Design Principles

- **Event-driven**: EventBus decouples all components, topic pub/sub, wildcards, dead-letter queue
- **Full persistence**: SQLite WAL mode, 8 tables (orders/positions/trades/pnl/signals/sessions/events/config)
- **State machine**: 8 lifecycle states with validated transitions
- **Compliance audit**: Every decision logged (who/what/when/why/result)
- **Pluggable**: DataProvider / BaseFactor / PortfolioOptimizer ABC interfaces
- **Config-driven**: All parameters in YAML, zero hard-coded values

---

## Factors

The platform computes **26 factors** (22 technical + 4 fundamental) and provides an **expression engine** with 35 functions for custom factor creation.

### Technical Factors (22)

| Category | Factor | Description |
|----------|--------|-------------|
| Momentum | momentum_1m/3m/6m/12m | Cumulative return over 1-12 months |
| Volatility | volatility_20d/60d | 20/60-day rolling std of returns |
| Basic | turnover_20d, rsi_14d, amplitude_20d, macd | Turnover, RSI, amplitude, MACD |
| Trend Quality | efficiency_ratio | Path efficiency: direction/total path [0,1] |
| Breakout | breakout_ignition | Volume + return shock composite signal |
| Trend Stage | trend_stage | Price position in 120-day range → fish head/body/tail |
| MA Convergence | ma_convergence | MA5/10/20 clustering — coiling pattern |
| Breakout Proximity | breakout_proximity | Distance to 20-day high [0,1] |
| Pure Volatility | pure_volatility | FF3 residual IVOL, orthogonalized & AR-filtered |
| MTF Resonance | mtf_resonance | Daily/weekly/monthly trend alignment |
| Candle Patterns | kmid/klen/kup/klow/ksft | 5 K-line shape features |
| Candle Pattern Recognition | candle_patterns.py | 12 patterns: engulfing, morning/evening star, hammer, etc. |

### Fundamental Factors (4)

log_market_cap, pb_ratio, pe_ratio, roe, asset_growth

### Expression Engine (35 functions)

Define factors as string formulas — no Python classes needed:

```python
# Time-series (17): ts_sum, ts_mean, ts_std, ts_rank, ts_corr, ts_slope, ts_delay...
# Cross-section (6):  cs_rank, cs_zscore, cs_scale...
# Math (10):         log, abs, sign, if_else, greater, less...
```

```yaml
# In config/default.yaml:
factors:
  expression:
    my_alpha: "ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"
```

### Signal Combination Methods

| Method | Description |
|--------|-------------|
| `equal_weight` | Simple average of all factor z-scores |
| `ic_weighted` | Weight by historical Rank IC (point-in-time) |
| `icir_weighted` | Weight by IC/IC_std (point-in-time) |
| `vote` | Each factor votes long/short/pass → majority decision |

---

## Risk Controls

| Layer | Module | Description |
|-------|--------|-------------|
| VaR/CVaR | `risk/var.py` | Historical/parametric/Monte Carlo VaR |
| Stress Testing | `risk/stress.py` | 2008 crisis, 2015 crash, 2020 COVID scenarios |
| Exposure | `risk/exposure.py` | Sector concentration, HHI, effective N |
| Factor Risk | `risk/factor_risk.py` | Systematic vs idiosyncratic risk decomposition |
| Monte Carlo | `risk/monte_carlo.py` | Block bootstrap + parametric simulation + trade shuffle |
| Circuit Breaker | `risk/circuit_breaker.py` | Position/sector/loss/drawdown limits + Kill Switch |
| Regime Detection | `risk/regime.py` | Volatility + trend + correlation → risk_on/off |
| Barra Model | `risk/barra.py` | 10-factor regression + Ledoit-Wolf + risk attribution |
| Health Check | `risk/healthcheck.py` | Pre-market system check → blocks trading on failure |
| Profile Classifier | `risk/profile_classifier.py` | Per-stock regime (trend_follower/breakseeker/defender) |
| Lookahead Detector | `risk/lookahead_detector.py` | Detects future data leakage in factor pipeline |
| Dynamic Stop-Loss | `risk/stop_loss.py` | 3-tier stop (monitor→half→clear) + rebound protection |
| Time Segmentation | `risk/time_segment.py` | A-share intraday session risk adjustment |
| Financial Health | `risk/financial_health.py` | 24-rule fraud detection + ST risk + Buffett Owner Earnings |

---

## Execution

| Layer | Module | Description |
|-------|--------|-------------|
| Order FSM | `execution/engine.py` | Validated state transitions, EventBus events |
| Execution Engine | `execution/engine.py` | Unified order lifecycle (backtest + live) |
| Order Book | `execution/order_book.py` | Price-time priority matching |
| OMS | `execution/oms.py` | Order management, blotter |
| Algorithms | `execution/algorithms.py` | TWAP/VWAP/Iceberg + SmartRouter |
| TCA | `execution/tca.py` | Implementation shortfall, arrival price |
| Paper Broker | `execution/paper_broker.py` | Latency simulation, partial fills, L2 replay |
| Connection | `execution/connection.py` | Connection lifecycle manager (Hummingbot-style) |

### Portfolio Orchestrator

```
Signal → PortfolioOrchestrator → ExecutionEngine → Broker
                    ↕
        MultiStrategyManager (capital + risk)
```

---

## Project Structure

```
quant_platform/
├── main.py                     # CLI entry (15 commands)
├── app.py                      # FastAPI entry
├── config/                     # YAML config + schema
│
├── core/                       # ★ Core architecture
│   ├── events.py               # EventBus: topic pub/sub, wildcards, dead-letter
│   ├── store.py                # SQLite: WAL mode, 8 tables
│   ├── state_machine.py        # Portfolio lifecycle FSM
│   ├── scheduler.py            # A-share trading calendar
│   ├── audit.py                # Compliance audit log
│   └── instrument.py           # Cross-asset abstraction
│
├── data/                       # Data layer (6 providers)
│   ├── providers/              # Baostock/Tushare/Adata/Synthetic/Postgres/WebSocket
│   ├── board_scanner.py        # Real-time A-share market scanner
│   ├── pipeline.py             # ETL pipeline
│   └── quality.py              # Data quality monitor
│
├── factors/                    # Factor engine (26 factors + expression engine)
│   ├── technical.py            # 22 technical factors
│   ├── fundamental.py          # 5 fundamental factors
│   ├── expression_engine.py    # String-based factor formulas
│   ├── expressions/            # 35 built-in functions
│   ├── candle_patterns.py      # 12 K-line pattern recognition
│   ├── shareholder.py          # Shareholder structure analysis
│   ├── evaluation.py           # IC analysis + Alphalens export
│   ├── processing.py           # Winsorize / standardize / neutralize
│   └── registry.py             # Factor registry
│
├── alpha/                      # Alpha signal (ICIR/Vote/ML/expression)
├── portfolio/                  # Portfolio optimization (EW/MVO/RP/BL)
├── backtest/                   # Backtest engine + WalkForward + MC
│
├── execution/                  # Execution layer
│   ├── engine.py               # Order FSM + ExecutionEngine
│   ├── oms.py                  # Order management
│   ├── algorithms.py           # TWAP/VWAP/Iceberg
│   ├── tca.py                  # Transaction cost analysis
│   ├── connection.py           # Connection lifecycle
│   └── order_book.py           # Limit order book
│
├── trading/                    # Live trading
│   ├── broker.py               # Simulated/QMT/XTP brokers
│   ├── engine.py               # Live trading engine
│   └── signal_generator.py     # Live signal generation
│
├── strategy/                   # Multi-strategy management
│   ├── multi_strategy.py       # Multi-pod capital allocation
│   └── portfolio_orchestrator.py  # Signal → order bridge
│
├── risk/                       # Risk management
│   ├── circuit_breaker.py      # Real-time risk monitor + Kill Switch
│   ├── var.py / stress.py / exposure.py
│   ├── barra.py                # 10-factor Barra model
│   ├── regime.py               # Market regime detection
│   ├── monte_carlo.py          # MC simulation + trade shuffle
│   ├── profile_classifier.py   # Per-stock regime (5 profiles)
│   ├── lookahead_detector.py   # Future data leakage detection
│   ├── stop_loss.py            # Dynamic tiered stop-loss
│   ├── time_segment.py         # Intraday session rules
│   └── financial_health.py     # 30-rule fraud + ST risk + capital cycle
│
├── agent/                      # LLM integration
│   └── sentiment_factor.py     # DeepSeek sentiment factor
│
├── api/                        # FastAPI (97 endpoints)
├── frontend/                   # Vue 3 (37 components)
├── utils/                      # Numba, cache, config, metrics
│
└── tests/                      # 1205 unit tests
```

---

## External Absorptions

This project has absorbed design patterns from 15+ open-source quant projects:

| Source | Absorption | Lines |
|--------|-----------|-------|
| vnpy | Expression-based Factor Engine | 1248 |
| BlackOil-OmniAlpha | Factor Screener (boolean selection) | 1165 |
| 悟道真英雄 | Config Version Manager | 680 |
| KF Timing App | Profile Classifier + Tradability Gate | 599 |
| freqtrade | Lookahead Bias Detector | 408 |
| PyPortfolioOpt | Black-Litterman Model | 220 |
| Hummingbot | ConnectionManager | 153 |
| Zipline | Factor Filters + ScreenFilter | ~150 |
| Qlib | K-line Candlestick Factors | ~96 |
| Jesse | Trade-shuffle Monte Carlo | ~52 |
| Alphalens | to_alphalens() export | ~53 |
| FinRL | Ensemble Voting Signal | ~44 |
| Backtrader | print_summary | ~70 |
| financial-report-minesweeper | 24-rule fraud detection + ST risk | ~400 |
| a-share-trend-strategy | TrendStage factor | ~25 |
| a-share-hybrid-strategy | MA convergence + breakout proximity | ~55 |
| AlphaPilot Pro | Dynamic stop-loss + time segment | ~300 |
| Senior Analyst | Multi-source validated provider | ~280 |
| Buffett Skill | Owner Earnings + Moat Score | ~120 |
| PureVolatility research | Pure Volatility factor | ~80 |
| a-share-signal | Multi-timeframe resonance | ~45 |
| stock-analyst | K-line pattern recognition | ~275 |
| Adata | Adata provider | ~200 |
| **Total** | **~5000+ lines** | |

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Language | Python 3.10+ |
| Async | asyncio |
| Data | Pandas, NumPy, Polars-compatible |
| Optimization | cvxpy, SciPy |
| ML | XGBoost, LightGBM, SHAP, scikit-learn |
| Acceleration | Numba (6 JIT kernels) |
| Web | FastAPI, Vue 3, Vite, ECharts, Lightweight-Charts |
| Storage | SQLite (WAL), PostgreSQL |
| Data Sources | Baostock, Tushare, Adata, AKShare, East Money, Sina |
| Monitoring | Prometheus, Grafana |
| Container | Docker, Docker Compose |
| CI/CD | GitHub Actions (Python 3.10/3.11/3.12) |

---

## License

MIT
