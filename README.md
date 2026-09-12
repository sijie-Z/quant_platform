# A-Share Multi-Factor Quantitative Research & Trading Platform

<p align="center">
  <a href="https://github.com/sijie-Z/quant_platform/actions/workflows/ci.yml"><img src="https://github.com/sijie-Z/quant_platform/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/sijie-Z/quant_platform/actions/workflows/codeql.yml"><img src="https://github.com/sijie-Z/quant_platform/actions/workflows/codeql.yml/badge.svg" alt="CodeQL"></a>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/API-97%20endpoints-brightgreen" alt="API">
  <img src="https://img.shields.io/badge/CLI-18%20commands-blueviolet" alt="CLI">
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
- **验证层**：Oracle IC / Known Alpha / MVO Audit 六项系统性验证，No-Lookahead 零前视契约
- **实盘层**：实时行情、Paper Trading 模拟盘、QMT 实盘接口、实时风控熔断、执行算法
- **展示层**：FastAPI 97 个 REST 端点 + WebSocket 实时推送 + Vue 3 Bloomberg 风格仪表盘

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
| LLM 因子 | 财经新闻情绪因子（Strategy 模式：关键词 ↔ OpenAI 可插拔） |

### 📈 回测与验证

| 能力 | 说明 |
|------|------|
| 向量化回测 | 月频调仓 + 日频持仓漂移，热路径零 for 循环 |
| A 股成本模型 | 佣金 0.03%（双边）+ 印花税 0.1%（仅卖出）+ 滑点 + 手数约束 |
| Walk-Forward | 滚动/扩展窗口 OOS 验证，每个 fold 内用 train-only 数据重算信号 |
| 系统性验证 | Oracle Factor IC=1.0、Known Alpha Recovery、MVO 60/60 成功、Rank IC 手动对比 |
| 风险建模 | VaR/CVaR、蒙特卡洛模拟、Barra 10 因子风险模型、压力测试 |

### 💹 实盘交易

| 能力 | 说明 |
|------|------|
| 实时行情 | AKShare 全市场快照 / 东方财富 WebSocket 推送 / Level 2 盘口 |
| Paper Trading | 延迟模拟、部分成交、撤单失败、LOB 撮合、A 股规则（T+1/涨跌停/手数） |
| 实盘接口 | QMT（xtquant）实盘 broker，代码格式自动转换，失败自动回退模拟盘 |
| 实时风控 | 仓位/行业/日亏损/回撤限额 + 5 级风险等级 + Kill Switch 熔断 |
| 执行算法 | TWAP / VWAP / Iceberg + SmartRouter 智能路由 + TCA 成本分析 |
| 运行内核 | EventBus（异步/背压/死信队列）+ StateMachine + 审计日志 + 调度器 |

### 🖥️ Web 平台

- **FastAPI 97 端点**：pipeline / 分析 / 风控 / OMS / 实盘 / 监控 全覆盖
- **WebSocket 实时推送**：EventBus → WS 桥接，交易事件实时到达前端
- **Vue 3 + ECharts 仪表盘**：Bloomberg 风格 11 行面板（净值曲线/因子热力图/风险仪表/持仓/归因）
- **监控**：Prometheus 指标 + Grafana 16 面板模板 + 合规审计导出

---

## 快速开始

```bash
# 1. 环境
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. 跑完整流水线（默认合成数据，无需联网）
.venv\Scripts\python main.py run

# 3. 启动 Web 平台（FastAPI + Vue）
.venv\Scripts\python main.py web
# 浏览器打开 http://localhost:8000

# 4. 运行所有测试
.venv\Scripts\python -m pytest tests/ -q
```

### CLI 命令（18 个）

```bash
python main.py run                 # 完整流水线（数据→因子→信号→回测→报告）
python main.py analyze             # 分析已有结果
python main.py compare             # 多策略对比
python main.py sweep               # 参数网格搜索
python main.py walkforward         # Walk-Forward OOS 验证
python main.py ml train/signal     # ML Alpha 信号（XGBoost/LightGBM）
python main.py trade               # Paper Trading 仿真
python main.py trade --broker qmt  # QMT 实盘（需 miniQMT）
python main.py web                 # Web 服务
python main.py research            # 研究 Agent
python main.py screen              # 选股筛选
python main.py execute             # 执行算法
python main.py factor-store        # 因子健康度
python main.py gate                # 策略门禁评估
python main.py strategy            # 多策略管理
python main.py check-lookahead     # 前视偏差检测
python main.py config              # 配置管理
python main.py profile             # 性能分析
```

---

## 目录结构

```
quant_platform/
│
├── main.py                 # CLI 入口（18 命令）
├── app.py                  # FastAPI 应用
│
├── core/                   # 运行内核：EventBus v2 / Store / StateMachine / Audit / Scheduler
├── data/                   # 数据层：Synthetic/Tushare/Baostock/AkShare/PostgreSQL/WebSocket/Level2
│   └── providers/
├── factors/                # 因子引擎：20+ 技术/基本面因子 + 处理 + 评估 + 图网络
├── alpha/                  # Alpha 合成：等权/IC/ICIR + ML 信号
├── portfolio/              # 组合优化：EqualWeight / MVO / RiskParity
├── backtest/               # 回测：向量化引擎 + 成本模型 + WalkForward + 逐笔回测
├── risk/                   # 风险：VaR/Barra/蒙特卡洛/熔断/Kill Switch/Regime 检测
├── execution/              # 执行：OMS / TWAP/VWAP/Iceberg / TCA / 订单簿
├── trading/                # 实盘：实时行情 / Paper / QMT / LiveEngine / SignalGenerator
├── strategy/               # 多策略管理
├── agent/                  # LLM：情绪因子 + RAG 研究 Agent
├── api/                    # FastAPI 路由（97 端点）
├── frontend/               # Vue 3 仪表盘（35 组件）
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
├── tests/                  # 单元测试（81 个测试文件，覆盖全部模块）
├── config/                 # YAML 配置（零硬编码）
└── requirements.txt        # 精确版本锁定
```

---

## 研究验证

本平台核心研究链路经过系统性验证（详见 `docs/VALIDATION_REPORT.md`）：

| 验证项 | 方法 | 结果 |
|--------|------|------|
| Oracle Factor | 已知未来收益作为因子 → Rank IC | **IC = 1.000000**（计算与数据对齐正确） |
| Known Alpha Recovery | 生成含已知 Alpha 的数据 → 因子引擎恢复 | **IC 与理论值一致** |
| Rank IC 对比 | 手动 vs 官方计算 | **差异 < 0.001%** |
| MVO Audit | 60 次调仓全日志 | **60/60 Success, 0 Fallback** |
| WalkForward | 多 fold 滚动 OOS | **全部通过（无前视偏差）** |
| No-Lookahead | 8 条不可协商契约 | **全部实现** |

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

---

## 已知限制

| 限制 | 说明 |
|------|------|
| 部分数据源需要 API key | Tushare（token）、OpenAI（LLM 因子） |
| QMT 实盘需环境 | 需券商 miniQMT + xtquant 包 + 实盘账号 |
| Level 2 数据 | 仅回放/模拟，无真实券商 L2 接入 |
| 生产存储 | 默认 SQLite；生产环境建议切换 PostgreSQL（docker-compose 已包含） |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/NO_LOOKAHEAD_CONTRACT.md` | 前视偏差零容忍契约（8 条铁规） |
| `docs/VALIDATION_REPORT.md` | 六项系统性验证报告 |
| `docs/KERNEL_ARCHITECTURE.md` | 实盘运行内核（控制平面）设计 |
| `docs/PRODUCTION_PLAYBOOK.md` | 从回测到实盘的完整操作手册 |
| `docs/PRODUCTION_ARCHITECTURE_V1.md` | 生产架构 v1（80d 反转 + Vol Filter） |
| `docs/ASHARE_PITFALLS.md` | 10 大 A 股实盘陷阱及处理 |
| `docs/OPS_RUNBOOK.md` | 运维手册 |
| `docs/research/` | 系统研究报告（Alpha Discovery / 市场结构 / Regime） |

---

## 许可

MIT License — 仅供教学和研究使用，不构成投资建议。
