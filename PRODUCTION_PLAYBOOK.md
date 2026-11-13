# Production Playbook

> 如何从零开始用这个平台做真正的量化研究和实盘交易。
> 本文档是为了确保一件事：**你在回测里看到的收益，和实盘拿到的收益是一致的。**

---

## 核心理念

```
研究 → 回测 → 前测(Paper) → 实盘
  ↑         ↑         ↑          ↓
  └─────────┴─────────┴──────────┘
        持续反馈、迭代
```

每个阶段使用**完全相同**的因子定义、信号合成逻辑和风险模型。唯一变化的是数据源和执行环境。

---

## P0 防线：别在回测里骗自己

### 1. 合成数据的嵌入式 Alpha

**问题**：`SyntheticDataProvider` 默认开启 `embedded_alpha=True` 时，会在收益率中硬编码动量/价值/规模的预测模式。在这上面跑回测 Sharpe 2.0，换真实数据就是 0.5。

**规则**：
- `embedded_alpha=True` 是**仓库默认**（`config/default.yaml`，演示用的人造信号）→ 仅限面试/集成演示，**永远不要用来验证策略**
- 做研究前必须显式改成 `embedded_alpha=False` → 纯噪声，只用来测代码不报错

```yaml
# config/default.yaml
data:
  provider: baostock       # ❌ 不要用 synthetic
  synthetic:
    embedded_alpha: false  # 保持 false
```

### 2. 回测信号 = 实盘信号（部分成立）

**问题**：历史上回测用 `AlphaPipeline` + 15 个因子，实盘用另一套 4 因子逻辑，两套逻辑，回测结果没有参考价值。

**当前状态（2026-10-20）——必须分清两条实盘路径**：
```
回测:         数据 → FactorEngine(默认21因子) → Process → AlphaPipeline → Optimizer → BacktestEngine
Web API 引擎:  数据 → LiveSignalGenerator(同一因子注册表, 7因子) → Process → AlphaPipeline → 下单
CLI trade:    数据 → LiveRunner 自算加权4因子(0.35/0.25/0.20/0.20, 阈值0.005) → 下单
```

- **Web API** 的 `LiveTradingEngine`（`POST /api/trading/start` 启动）使用
  `LiveSignalGenerator`，与回测同源（7 因子复合，`min_signal_score=0.3`）。
- **CLI** `python main.py trade` 走 `LiveRunner`，它仍使用独立的加权 4 因子复合
  （动量/波动/RSI/MACD），**不是** `LiveSignalGenerator`。
- 因此"回测 = 实盘"目前只对 Web API 引擎成立；CLI 实盘路径是**已知不一致**，
  上线前需统一（或改用 Web API 引擎）。

### 3. 交易成本 = 市场冲击

**问题**：回测用固定0.1%滑点，实盘的冲击成本随资金量非线性增长。

**修复后**：
```yaml
costs:
  slippage_model: "impact"   # 使用 Almgren-Chriss + Square-Root 模型
  impact_model: "composite"  # 考虑参与率、波动率、价差
```

`CostModel` 只有在 `slippage_model='impact'` **且调用方提供 `daily_volume` / `volatility`** 时才会调用 `ExecutionCostCalculator`。当前 `main.py run` 的主回测路径只传执行日门禁（`tradable_mask`/涨跌停），不传 `daily_volume`，因此会回退到 `fixed` 并打印一次警告——`config/default.yaml` 因此保持 `slippage_model: "fixed"`。要真正启用 impact，需要先把成交量/波动率面板接入 `BacktestEngine.run`；在此之前不要把 impact 当作已生效的成本模型。

---

## P1: 研究流水线

### 标准流程

```bash
# 1. 数据准备：数据源与日期在 config/default.yaml 的 data 段配置。
#    Baostock 可以用 CLI 显式切换（没有 fetch-data 命令）：
python main.py run --use-baostock                       # 全流水线（Baostock 真实数据）

# 2. 因子研究
python main.py run                                      # 全流水线（config 指定数据源）
python main.py sweep --optimizers mean_variance,risk_parity  # 参数扫描
python main.py compare                                   # 策略对比

# 3. Walk-Forward 验证 (最重要的一步)
python main.py walkforward --method expanding --folds 6

# 4. 前测 (Paper Trading)
python main.py trade --broker paper --days 60 --cash 1000000

# 5. 实盘（QMT 连接失败会直接中止，不会回退模拟盘；真实券商每次只支持 1 个交易所日）
python main.py trade --broker qmt --days 1 --cash 1000000
```

### Walk-Forward 是黄金标准

Walk-Forward 验证确保每个测试集用的信号**只基于该时间点之前的数据训练**：

```
训练期1(504天) → 测试期1(126天) → 指标1
训练期2(630天) → 测试期2(126天) → 指标2
...
```

所有 fold 的 OOS 指标聚合 → 真实的预期表现。

**不要跳过的步骤**：
1. 至少 4 个 fold
2. 检查 fold 间的稳定性（不要只看平均 Sharpe）
3. 如果各 fold 差异巨大 → 策略过拟合

---

## P2: 数据源

### 推荐配置

```yaml
data:
  provider: baostock   # baostock（免费，无需API key）
  # provider: tushare  # Tushare Pro（更全，需要 token）
  # provider: postgres # 生产环境，PostgreSQL 存储
```

**三级路由机制**（`connection_pool.py`）：
```
Tushare → Baostock → Synthetic
```
按优先级和健康状态路由。注意：配置的真实数据源不可用时，`run` 默认 **fail closed**
（非零退出并给出明确原因）；只有显式 `--allow-synthetic-fallback` 或
`data.allow_synthetic_fallback: true` 才回退合成数据，且 `metrics.json` 的
`provenance` 会记录 requested/effective provider 与 `synthetic: true`。

### PostgreSQL 生产配置

`docker-compose.yml` **不包含** PostgreSQL 服务（只有 `api`）。需要自行部署
PostgreSQL/TimescaleDB，并在 config 中选择 `data.provider: postgres`（连接参数见
`config/default.yaml` 的 postgres 段）。`PostgresDataProvider` 连接失败会抛
`ConnectionError`，由调用方按 fail-closed 策略处理；不会把 SQLite 或合成数据
静默伪装成 PostgreSQL。

---

## P3: 过拟合控制清单

每个策略上线前，逐项检查：

| # | 检查项 | 通过条件 |
|---|--------|---------|
| 1 | Point-in-time IC 加权 | 信号权重只用 t 日之前的数据计算 |
| 2 | Walk-Forward OOS | OOS Sharpe > 0，且各 fold 一致 |
| 3 | Deflated Sharpe | 考虑多因子多重检验后的 Sharpe |
| 4 | 合成数据禁用 | `embedded_alpha: false` |
| 5 | 成本模型 | 佣金+印花税+滑点已计入；impact 需先接入成交量/波动率面板 |
| 6 | 实盘信号一致 | Web API 引擎使用 `LiveSignalGenerator`（与回测同源）；CLI `LiveRunner` 仍为独立 4 因子逻辑，**未统一前不要用 CLI 做实盘信号验证** |
| 7 | 前测验证 | Paper Trading 至少跑 60 个交易日 |

---

## 已知限制

1. **SQLite 并发**：生产环境请自行部署 PostgreSQL 并配置 `data.provider: postgres`（docker-compose 不包含数据库）
2. **Level 2 数据**：仅回放/模拟，无真实券商 L2 接入
3. **日内回测**：`tick_engine.py` 支持 tick 级回测，但默认走日频
4. **QMT 接口**：需要开通券商 miniQMT + 安装 xtquant 包

---

## 部署

### Docker

```bash
# QUANT_API_TOKEN 在 .env 中设置（compose 用 :? 强制要求），或按下面方式内联传入：
QUANT_API_TOKEN=your-token docker compose up -d --build
# 访问 http://localhost:8000
# API: http://localhost:8000/api/health
```

`QUANT_API_TOKEN` **必填**（容器发布端口后不是回环地址，没有 token 时所有 `/api`
会被拒绝）。持久化卷：`/app/data`、`/app/results`、`/app/.quant_cache`、
`/app/.quant_versions`、`/app/.cache`、`/app/.agent_cache`、`/app/.sentiment_cache`；
`stop_grace_period: 30s` 用于引擎停止、状态落盘与 SQLite WAL checkpoint。

### 手动

```bash
python -m venv .venv && .venv\Scripts\activate   # Python 3.11/3.12
pip install -r requirements.txt
pip install -e ./packages/quant_core   # core/store.py re-exports quant_core
pip install -e .                        # 使 quant_platform 可导入
cd frontend && npm ci && npm run build && cd ..   # Node 22；产物 dist 已 gitignore
python main.py web
```

---

## 架构决策记录

### EventBus vs MessageBus

两套事件系统，各自解决不同的问题：

| 特性 | EventBus (core/events.py) | MessageBus (core/message_bus.py) |
|------|--------------------------|----------------------------------|
| 范围 | 进程内，同步 | 跨进程/跨机器，异步 |
| 后端 | 内存 dict | Local/Redis/Kafka |
| 使用方 | Engine/Store/Scheduler | services/ 微服务 |
| 场景 | 核心组件通信 | 生产分布式部署 |

**演进路径**：
```
开发: EventBus (进程内)
       ↓
过渡: EventBus + MessageBus(Local)
       ↓
生产: MessageBus(Kafka) 微服务
```

`services/` 目录是未来的微服务骨架，当前零运行时导入（见 `docs/DEAD_CODE_STATUS.md`），不参与生产路径。当需要多进程部署时，将核心组件拆成独立服务即可。
