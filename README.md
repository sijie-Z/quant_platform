# A-Share Multi-Factor Quantitative Research & Trading Platform

<p align="center">
  <a href="https://github.com/sijie-Z/quant_platform/actions/workflows/ci.yml"><img src="https://github.com/sijie-Z/quant_platform/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/sijie-Z/quant_platform/actions/workflows/codeql.yml"><img src="https://github.com/sijie-Z/quant_platform/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/API-99%20endpoints-brightgreen" alt="API">
  <img src="https://img.shields.io/badge/CLI-21%20commands-blueviolet" alt="CLI">
  <img src="https://img.shields.io/badge/前端-Vue%203%20%2B%20ECharts-orange" alt="Frontend">
  <img src="https://img.shields.io/badge/实盘-Paper%20%2F%20QMT-yellow" alt="Live Trading">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

**A 股多因子量化平台：从数据研究到实盘交易的一条完整流水线。**

数据 → 因子 → 信号 → 组合 → 回测 → 风控 → 执行 → 实盘 → Web 监控

---

## 项目定位

一个覆盖 **研究 → 验证 → 实盘** 全链路的 A 股量化平台：

- **研究层**：20+ 因子引擎、IC/ICIR 评估、因子合成、Walk-Forward 验证、组合优化
- **数据层**：Tushare 指数成分历史掩码接入 PIT 门禁（`universe_history_available` 为真才应用；单快照回退显式拒绝并告警，`metadata.attrs["universe_pit"]` 留痕），因子与信号截面同样按日剔除非成分（`excluded_mask`，执行层卖出语义不变）
- **验证层**：Oracle IC / Known Alpha / Rank IC 对照等验证工具；No-Lookahead 契约 8 条均已实现并有回归测试守护，残余风险见「研究验证」与 HANDOVER
- **实盘层**：实时行情、Paper Trading 模拟盘、QMT 实盘接口、实时风控熔断、执行算法
- **展示层**：FastAPI 99 个 HTTP 端点 + WebSocket 实时推送 + Vue 3 Bloomberg 风格仪表盘

> 设计原则：**先验证，后交易；先诚实，后优化。** 所有研究记录都携带 Trust Metadata（数据源、复权方式、PIT 状态、偏差警告），不做粉饰。

---

## 功能全景

### 📊 因子研究

| 能力 | 说明 |
|------|------|
| 技术因子 | Momentum(1/3/6/12m)、Volatility(20/60d)、Turnover、RSI、MACD、Efficiency Ratio、Breakout、Candle 系列、Trend Stage、MA Convergence 等 20+ |
| 基本面因子 | log_market_cap、PB、PE、ROE、Asset Growth |
| 因子处理 | 缩尾(1%/99%) → zscore 标准化 → 行业+市值中性化 |
| 因子评估 | Rank IC / Pearson IC / ICIR / 分位数收益 / IC 衰减 / 相关性矩阵 |
| 图网络因子 | 股票关联网络 + 4 种中心性度量（PageRank/特征向量/介数/度） |
| LLM 因子 | 财经新闻情绪因子（Strategy 模式：关键词 ↔ DeepSeek（OpenAI 兼容接口）可插拔） |

### 📈 回测与验证

| 能力 | 说明 |
|------|------|
| 向量化回测 | 月频调仓 + 日频持仓漂移，横截面运算向量化（回测按交易日/调仓日推进，非逐股票循环） |
| A 股成本模型 | 佣金 0.03%（双边，每笔最低佣金）+ 印花税按日期（2023-08-28 前 0.1%、之后 0.05%，仅卖出）+ 过户费 + 冲击成本滑点（默认 impact 模型；缺量价数据时告警并回退固定滑点） |
| Walk-Forward | 滚动/扩展窗口 OOS 验证，每个 fold 内用 train-only 数据重算信号，因子按 fold 训练窗逐窗选择（IC t 检验 + BH-FDR）；支持 purge/embargo 隔离（训练/测试窗之间留空隙，测试窗后的行情不回流入训练） |
| 多重检验校正 | `/api/sweep` 与 CLI `sweep` 对 N 次参数试验做 DSR + BH-FDR（best 不显著时给出警告）；Walk-Forward 因子筛选先做 IC t 检验 + BH-FDR |
| 研究 holdout | `research.holdout.start_date/end_date` 预留最终评估窗口（闭区间，空 = 关闭）。数据窗口与 holdout 重叠时 `/api/sweep` 返回 422、CLI `sweep` 非零退出且不先跑；研究验证在因子筛选与 Walk-Forward 前剔除 holdout 交易日及紧邻的前向标签行（重算 IC 矩），报告记录 `holdout: {spec, excluded_days}`；剩余样本不足时显式 skipped |
| 系统性验证 | Oracle Factor / Known Alpha 诊断工具（`tools/diagnose_ic.py`、`tools/sanity_check.py`）、Rank IC 对手算、MVO/成本模型回归测试（历史 60/60 报告已作废，见「研究验证」） |
| 风险建模 | VaR/CVaR、蒙特卡洛模拟、Barra 10 因子风险模型、压力测试 |

### 💹 实盘交易

| 能力 | 说明 |
|------|------|
| 实时行情 | AKShare 全市场快照 / 东方财富 WebSocket 推送 / Level 2 盘口 |
| Paper Trading | 延迟模拟、部分成交、撤单失败、LOB 撮合、A 股规则（T+1/涨跌停/手数） |
| 实盘接口 | QMT（xtquant）实盘 broker，代码格式自动转换；断线后懒重连（指数退避，重连失败抛 BrokerConnectionError 且绝不下单）+ 重连后/轮询时从券商幂等回查当日委托与成交（query_stock_orders/query_stock_trades，get_orders 2 秒 TTL）；连接失败直接中止（fail closed），不会自动回退模拟盘；异步成交按 delta 每 cycle 回写 Store 成交流水（order.synced，幂等） |
| 实时风控 | 仓位/行业/日亏损/回撤限额 + 5 级风险等级 + Kill Switch 熔断 |
| 执行算法 | TWAP / VWAP / Iceberg + SmartRouter 智能路由 + TCA 成本分析 |
| 运行内核 | EventBus（异步/背压/死信队列）+ StateMachine + 审计日志 + 调度器 |
| 每日对账 | 每个交易日 EOD 自动账实对账（expected = 引擎启动时券商基线 + 成交流水推导的独立账本，actual = 券商柜台；无基线记 unknown），按 `QUANT_RECON_*` 容差判定 passed/failed/unknown 并持久化结果 |

### 🖥️ Web 平台

- **FastAPI 99 个 HTTP 端点 + 1 个 WebSocket**：pipeline / 分析 / 风控 / OMS / 实盘 / 监控 全覆盖；分析/监控端点从有界 TTL 缓存的真实研究上下文现算（并发饱和返回 429；无数据时明确返回 `available:false` 及原因）
- **WebSocket 实时推送**：EventBus → WS 桥接，交易事件实时到达前端
- **Vue 3 + ECharts 仪表盘**：Bloomberg 风格 12 行面板（净值曲线/因子热力图/风险仪表/持仓/归因）
- **监控**：Prometheus 指标 + Grafana 16 面板模板 + 合规审计导出

---

## 快速开始

```bash
# 1. 环境（Python 3.11 / 3.12；CI 与 Docker 使用 3.12）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# core/store.py re-exports the local quant_core package; the editable
# installs are required (CI and the Dockerfile do the same).
.venv\Scripts\pip install -e ./packages/quant_core
.venv\Scripts\pip install -e .

# 2. 跑完整流水线（默认合成数据，无需联网）
#    注意：默认 config 的合成数据启用了 embedded_alpha（演示用的人造信号）。
#    做研究/回测结论时请把 data.synthetic.embedded_alpha 设为 false。
.venv\Scripts\python main.py run

# 3. 构建前端（可选；Node 22，产物 frontend/dist 已 gitignore，不随仓库分发）
cd frontend
npm ci
npm run build
cd ..

# 4. 启动 Web 平台（FastAPI + 已构建的 Vue 静态文件）
.venv\Scripts\python main.py web
# 浏览器打开 http://localhost:8000
# 没有 frontend/dist 时仍可启动，但只提供 API（控制台会提示先构建前端）。
# 默认只监听 127.0.0.1。远程访问请设置 QUANT_API_TOKEN，并在 UI Settings
# 填入同一 token；未设置 token 时所有 /api（除 /api/health）仅允许回环访问。

# 5. 运行所有测试
.venv\Scripts\python -m pytest tests/ -q
```

前端开发模式：`cd frontend && npm ci && npm run dev`（Vite 5173，`/api` 代理到 127.0.0.1:8000）。

> 上面的 Windows 示例使用 cmd/PowerShell 的 `.venv\Scripts\...`；在 Git Bash 中请改为
> `.venv/Scripts/python`、`.venv/Scripts/pip`（或先 `source .venv/Scripts/activate`）。

### Docker 部署（仅 API 服务）

```bash
# QUANT_API_TOKEN 必填：容器发布 8000 端口后不再是回环地址，
# 未设置 token 时所有 /api 请求都会被拒绝。
QUANT_API_TOKEN=your-token docker compose up -d --build
```

`docker-compose.yml` 只定义 `api` 服务，**不包含 PostgreSQL**。持久化卷：
`/app/data`（交易库/审计/EOD/NAV/引擎状态）、`/app/results`、`/app/.quant_cache`、
`/app/.quant_versions`、`/app/.cache`、`/app/.agent_cache`、`/app/.sentiment_cache`；
`stop_grace_period: 30s` 为引擎停止、状态落盘和 SQLite WAL checkpoint 留出时间。
运维相关环境变量：`ALERT_WEBHOOK_URL`（告警 JSON POST）、`ALERT_WEBHOOK_TIMEOUT`、
`QUANT_RETENTION_*_DAYS`（scheduler 每日保留任务；手动 retention 命令用 CLI 参数）；
用法见 `docs/OPS_RUNBOOK.md` 第 12 节。
PostgreSQL 由 `data/providers/postgres_provider.py` 支持，但需自行部署并在配置中选择
`data.provider: postgres`，compose 不会替你启动数据库。

### CLI 命令（21 个）

```bash
# 研究/回测流水线
python main.py run                      # 完整流水线（数据→因子→信号→回测→报告）
python main.py run --use-baostock       # 使用 Baostock 真实数据
python main.py run --force              # 跳过缓存重算
python main.py walkforward --folds 6 --method expanding   # Walk-Forward OOS 验证
python main.py analyze --results-dir ./results            # 分析已有结果
python main.py compare --optimizers equal_weight,risk_parity
python main.py sweep --optimizers equal_weight,mean_variance --frequencies monthly,weekly
python main.py profile                  # 各阶段耗时分析
python main.py gate                     # 策略门禁评估
python main.py check-lookahead          # 前视偏差检测

# 因子/研究工具
python main.py factor list              # 单因子研究报告
python main.py factor-store stats       # 因子健康度/研究存储
python main.py ml train --model lightgbm
python main.py ml signal --model xgboost
python main.py research report          # LLM 归因报告
python main.py research validate --mode quick
python main.py screen --rules "pe_ratio=lt:30,roe=gt:0.15"  # 选股筛选

# 交易/执行
python main.py trade                    # Paper Trading 仿真（默认 30 天、5 只蓝筹）
python main.py trade --broker paper --days 60 --cash 5000000
python main.py trade --broker qmt --universe "600519,000858,000001" --days 1  # QMT 实盘（需 miniQMT；真实券商每次只支持 1 个交易所日）
python main.py execute                  # 因子→信号→订单→成交全链路
python main.py strategy list            # 多策略管理

# 配置/缓存/运行记录
python main.py config list              # 配置版本管理
python main.py cache list               # 流水线缓存
python main.py cache refresh            # 重算并原子覆盖当前配置的缓存条目
python main.py cache clear              # 清空缓存
python main.py runs list                # 历史运行与 validity
python main.py data retention --archive-dir ./archive  # 归档并清理超过保留期的行

# Web
python main.py web
```

---

## 实时数据接入（默认不是实时行情）

**默认 `data.provider: synthetic`**：离线、可复现，用于测试与研究演示，不是实时行情。真实/实时数据有三条路径，按需开启：

| 路径 | 入口 | 说明 |
|---|---|---|
| 交易事件流 | `WS /api/ws`（`bearer.<token>` 子协议） | 引擎运行时推送 `order.*`/`trade.*`/`risk.*`/`portfolio.snapshot`，前端交易页实时刷新；无事件时保持长连接 |
| 实时行情 | `GET /api/market/*`（AKShare 快照）、`GET /api/portfolio/live`（Baostock 报价）、`POST /api/ws-quotes/start` + `GET /api/ws-quotes`（东财/新浪 WebSocket） | 响应携带数据源与 `price_basis`；数据源不可用返回 503/明确原因，**不会伪装成实时** |
| 真实研究/交易数据 | `data.provider: baostock`（免费无 key）或 `tushare`（`TUSHARE_TOKEN`），生产可用 `postgres` | 首次全量拉取较慢（500 只逐只拉取），之后命中缓存；`python main.py run --incremental` 只追加新交易日并刷新复权因子 |

```yaml
data:
  provider: baostock        # synthetic | baostock | tushare | postgres
  start_date: "2024-01-01"
  end_date: "2026-10-21"
```

远程部署需设置 `QUANT_API_TOKEN`（前端 Settings 填同一个值）；UI 与 API 不同源时设置 `QUANT_ALLOWED_ORIGINS`。浏览器用 `localhost` 打开时 WebSocket 会自动改连 `127.0.0.1`（服务端默认只绑 IPv4）。

实测备注（2026-10-22）：本机 Baostock 登录可达，但受限网络下 500 只全量串行拉取会超时；真实行情依赖网络与数据源可用性，取不到时系统明确报错/回退，不伪造实时数据。

---

## 目录结构

```
quant_platform/
│
├── main.py                 # CLI 入口（21 命令）
├── app.py                  # FastAPI 应用
│
├── core/                   # 运行内核：EventBus v2 / Store / StateMachine / Audit / Scheduler
├── data/                   # 数据层：Synthetic/Tushare/Baostock/AkShare/PostgreSQL/WebSocket/Level2
│   └── providers/
├── factors/                # 因子引擎：20+ 技术/基本面因子 + 处理 + 评估 + 图网络
├── alpha/                  # Alpha 合成：等权/IC/ICIR + ML 信号
├── portfolio/              # 组合优化：EqualWeight / MVO / RiskParity（MVO 支持成本/换手惩罚与 ADV 容量上限）
├── backtest/               # 回测：向量化引擎 + 成本模型 + WalkForward + 逐笔回测
├── risk/                   # 风险：VaR/Barra/蒙特卡洛/熔断/Kill Switch/Regime 检测
├── execution/              # 执行：OMS / TWAP/VWAP/Iceberg / TCA / 订单簿
├── trading/                # 实盘：实时行情 / Paper / QMT / LiveEngine / SignalGenerator
├── strategy/               # 多策略管理
├── agent/                  # LLM：情绪因子 + RAG 研究 Agent
├── api/                    # FastAPI 路由（99 HTTP 端点 + 1 WebSocket）
├── frontend/               # Vue 3 仪表盘（37 组件）
├── reporting/              # 报告：文本/图表/自包含 HTML
├── operations/             # 基金运营：NAV / 投资者门户
├── compliance/             # 合规审计导出
├── monitoring/             # Grafana 仪表盘模板
│
├── framework/              # 能力层契约（Protocols）：Factor/Evaluator/MarketData/Broker
│   └── contracts/
├── lab/                    # 研究实验室：诚实因子运行 + Registry + 自动报告
│   ├── runs/
│   ├── registry/
│   └── reports/
├── tools/                  # 研究工具：common_runner / sanity_check / attribution
│
├── docs/                   # 文档：架构 / 里程碑 / 研究报告 / 协议
├── tests/                  # 单元测试（236 个测试文件，3,342 条用例，覆盖全部模块）
├── config/                 # YAML 配置（零硬编码）
└── requirements.txt        # Python 依赖（版本锁定；adata 例外见下文）
```

---

## 研究验证

2026-09-13 的巡检曾列出若干前视偏差与成本模型缺陷（BUG-21/23/03 等）。这些缺陷已在 2026-09/10 的修复轮次中修复，并各自有回归测试守护：

| 历史缺陷 | 当前状态 | 实现 / 回归测试 |
|----------|----------|-----------------|
| Walk-Forward OOS 指标覆盖 train+test（BUG-21） | 已修复：OOS 指标只取测试窗 | `backtest/walkforward.py` |
| 财务数据未按 publish_date 过滤 | 已修复：合成数据按公告日 reindex；provider 提供公告日时按公告日 | `data/providers/synthetic.py`、`data/pipeline.py` |
| 行业中性化只用静态 map | 已修复：按日期的 `sector_panel` 接入中性化 | `factors/processing.py`、`main.py` |
| ST 过滤用全样本 `is_st` | 已修复：按公告日构建 `tradable_mask`；Baostock 由日线 `isST` 派生 ST episode，无 ST 历史的数据源显式标记不可用 | `data/pipeline.py`、`data/providers/*` |
| ML CV 按 (日期, 股票) 拍平、训练/测试共享日期（BUG-23） | 已修复：按日期切分 + purge/embargo | `alpha/ml_signal.py` |
| 成本模型系统性减半（BUG-03） | 已修复：印花税仅卖出、按资产拆分买卖名义额 | `backtest/cost_model.py`、`backtest/engine.py` |

当前仍然存在的限制（不属于上述历史缺陷）：多租户隔离（#75 产品定位未定；共享 token 不是租户边界，`X-Tenant-Id` 是可信头约定）、Adata 不提供价格面板（仅作 metadata/财务的跨源校验，价格路径明确 fail closed）、增量更新仅对支持 `update()` 的真实数据源生效（合成数据源会回退全量）。

> `docs/VALIDATION_REPORT.md` 是 2026-06-20 的历史报告，数字早于上述修复，已标注 SUPERSEDED，不应作为当前证据引用。当前状态以 `docs/HANDOVER.md` 的第十轮记录为准。

**诚实研究记录**（`lab/`）：每次因子运行都会在 SQLite Registry 中记录数据源、复权方式、PIT 状态、偏差警告，失败运行同样记录——系统从不隐藏自身的不确定性。

---

## 研究里程碑

| 阶段 | 内容 | 结论 |
|------|------|------|
| M1 | Research OS MVP：3 个诚实因子运行 | 平台可信度验证通过 |
| M2 | Low Vol vs Momentum 逐年 IC 诊断 | Low Vol 胜在年际稳定性 |
| M3 | 10 因子 Factor Zoo | **A 股是反转市场**，动量因子 IC 全负 |
| M4 | volatility_20d 组合回测 | IC=+0.0334 不能直接转成组合收益（成本/换手吃掉 alpha） |

**核心发现**：A 股市场存在强均值回归 Alpha（40–80 交易日连续时间结构），但固定频率执行（月频）下因采样混叠系统性失效——信号存在 ≠ 策略可投资。这正是平台"诚实研究"的价值：不粉饰数据。

> 上述里程碑是 2026 年早期的研究记录。涉及回测收益/IC 的数字早于 2026-09 的修复轮次，需在修正后的代码与数据上重跑后才能引用；判据见 `docs/HANDOVER.md` 第十节。

---

## 已知限制

| 限制 | 说明 |
|------|------|
| 部分数据源需要 API key | Tushare（token）、DeepSeek（LLM 因子，OpenAI 兼容接口） |
| QMT 实盘需环境 | 需券商 miniQMT + xtquant 包 + 实盘账号；连接失败直接中止（fail closed），不会回退模拟盘 |
| Level 2 数据 | 仅回放/模拟，无真实券商 L2 接入 |
| 生产存储 | 默认 SQLite；PostgreSQL 由 `data/providers/postgres_provider.py` 支持，需自行部署（docker-compose 不包含数据库） |
| 多租户隔离 | API 不提供租户隔离（#75 产品定位未定） |
| 分析端点计算量 | 7 个分析/监控端点使用有界研究上下文现算（TTL 缓存、并发饱和 429）；默认配置首次调用需构建上下文，可能耗时数十秒 |
| WebSocket token 传输 | 仅接受 `bearer.<token>` 子协议（时序安全比较）；`?token=` 查询串已移除，非 RFC-7230 字符的 token 需更换 |
| 价格基准 | 已统一：provider 交付 raw OHLCV + 复权因子（Tushare/Baostock 同一约定），`get_raw_close()` 供成交/估值，`get_adjusted_close()` 供研究；仅 Postgres/Adata 尚未声明基准，货币路径 fail closed |
| 数据更新 | `run --incremental` / `cache refresh --incremental` 增量追加（边界去重、复权因子全量刷新、完成标记与新鲜度门禁）；Tushare/Baostock 支持，合成数据源回退全量 |
| 财务公告日 | Baostock 财报必须带公告日（`pubDate`）；缺失/不可解析的行会被丢弃并告警，不会用期末日伪造可见日（否则构成前视） |
| 交易所日历 | 内置休市表覆盖至 2026 年；2027 年休市日需通过 `QUANT_EXCHANGE_HOLIDAYS` 追加 |
| 回测交易性 | 执行日已按 `tradable_mask` / 涨跌停阻断成交（成分 `universe_mask` 仅拦买、退出成分仍可卖）；整手约束在执行层生效，向量化回测仍为权重层面 |
| 可选依赖 | `adata` 为 `>=1.0.0`（其余依赖精确锁定）。仅当 `data.validated: true` 时作为第二数据源做 metadata/财务跨源校验；其 `get_prices()` 明确不支持（无价格基准声明），不可用时记录警告并跳过校验（不阻断主数据源） |
| **仅支持单进程部署** | 见下方"部署约束：单 worker" |

### 部署约束：单 worker

运行注册表（run registry）、2 路 pipeline 并发上限、实盘引擎句柄、Kill Switch、
OMS 账本和 WebSocket 客户端集合**全部是进程内状态**。因此必须以**单个 uvicorn
worker** 运行：不要给 `uvicorn` / `main.py web` 加 `--workers N` 或 `--reload`
（`--reload` 也会启动额外进程）。多 worker 会导致：

- `POST /api/run` 的状态/结果在其它 worker 上 404；
- 并发上限变成 N×2，两个 worker 可能同时启动实盘引擎（同一 broker 上重复下单）；
- Kill Switch 只作用于收到请求的那个 worker；
- WebSocket 客户端分散在不同 worker，事件只推送给其中一部分；
- 多个进程独立写同一个 SQLite 文件（各自持有进程内锁，写入互不知情）。

正确做法：单 worker 部署；如需横向扩展，先把上述状态外置到 Redis/数据库
（本阶段未实现，仅在此记录约束）。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/HANDOVER.md` | 当前状态、历次修复记录与第十轮交接（**先读这份**） |
| `docs/NO_LOOKAHEAD_CONTRACT.md` | 前视偏差零容忍契约（8 条铁规） |
| `docs/VALIDATION_REPORT.md` | 2026-06-20 历史验证报告（已 SUPERSEDED，仅存档） |
| `docs/KERNEL_ARCHITECTURE.md` | 实盘运行内核（控制平面）设计（提案，未接入运行路径） |
| `PRODUCTION_PLAYBOOK.md` | 从回测到实盘的完整操作手册 |
| `docs/PRODUCTION_ARCHITECTURE_V1.md` | 生产架构 v1（目标设计，未实现） |
| `data/ASHARE_PITFALLS.md` | 10 大 A 股实盘陷阱及处理 |
| `docs/OPS_RUNBOOK.md` | 运维手册（启停、故障、备份/恢复、升级迁移） |
| `docs/research/` | 系统研究报告（Alpha Discovery / 市场结构 / Regime） |

---

## 许可

MIT License — 仅供教学和研究使用，不构成投资建议。
