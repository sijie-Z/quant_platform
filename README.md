# A股多因子量化交易平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/%E6%B5%8B%E8%AF%95-1205%20%E9%80%9A%E8%BF%87-brightgreen?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/%E5%9B%A0%E5%AD%90-26%20%E4%B8%AA-orange" alt="Factors">
  <img src="https://img.shields.io/badge/Python%20%E6%A8%A1%E5%9D%97-105-blueviolet" alt="Modules">
  <img src="https://img.shields.io/badge/%E4%BB%A3%E7%A0%81%E8%A1%8C-34K%2B-yellow" alt="Lines">
  <img src="https://img.shields.io/badge/API%20%E7%AB%AF%E7%82%B9-97-red?logo=fastapi" alt="API">
  <img src="https://img.shields.io/badge/%E5%BC%80%E6%BA%90%E5%8D%8F%E8%AE%AE-MIT-green" alt="License">
</p>

<p align="center">
  <a href="#%E7%9B%AE%E5%BD%95">目录</a> •
  <a href="#%E7%AE%80%E4%BB%8B">简介</a> •
  <a href="#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B">快速开始</a> •
  <a href="#%E6%9E%B6%E6%9E%84">架构</a> •
  <a href="#%E5%9B%A0%E5%AD%90%E7%B3%BB%E7%BB%9F">因子</a> •
  <a href="#cli-%E5%91%BD%E4%BB%A4%E9%9B%86">CLI</a> •
  <a href="#api-%E6%8E%A5%E5%8F%A3">API</a> •
  <a href="#%E9%A3%8E%E9%99%A9%E7%AE%A1%E6%8E%A7">风控</a> •
  <a href="#%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84">结构</a>
</p>

---

## 目录

- [简介](#简介)
- [核心能力](#核心能力)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [运行完整流水线](#运行完整流水线)
  - [运行测试](#运行测试)
- [架构](#架构)
- [因子系统](#因子系统)
  - [26个内置因子](#26个内置因子)
  - [表达式引擎](#表达式引擎)
  - [信号合成方式](#信号合成方式)
  - [因子筛选模式](#因子筛选模式)
- [CLI 命令集](#cli-命令集)
- [API 接口](#api-接口)
- [风控体系](#风控体系)
  - [12层风控](#12层风控)
  - [动态止损](#动态止损)
  - [财务排雷](#财务排雷)
- [执行层](#执行层)
  - [订单状态机](#订单状态机)
  - [Portfolio Orchestrator](#portfolio-orchestrator)
- [交易引擎](#交易引擎)
- [数据源](#数据源)
- [配置管理](#配置管理)
- [项目结构](#项目结构)
- [外部吸收](#外部吸收)
- [技术栈](#技术栈)
- [许可证](#许可证)

---

## 简介

这是一个面向 **A 股市场**的多因子量化交易平台，覆盖从数据获取、因子计算、信号合成、组合优化、回测验证到实盘交易的完整流水线。

跟市面上其他量化框架的区别：

- **不是 backtrader 那样的逐 K 线回测框架**，而是向量化多因子平台，更适合日频选股策略
- **不是 Qlib 那样的纯 ML 研究平台**，而是把因子研究、回测、实盘执行打通了
- **不是 vnpy 那样的交易执行系统**，而是从因子出发的策略研究平台，附带执行能力
- **专为 A 股设计**，不是把美股框架改几个参数就拿来用

整个项目大概 34,000 行 Python + 9,800 行 Vue，1205 个单元测试，105 个 Python 模块。

---

## 核心能力

### 已经做到的

| 模块 | 状态 | 说明 |
|------|------|------|
| 多数据源接入 | ✅ 6 个 | Baostock / Tushare / Adata / AKShare / 新浪 / 合成数据 |
| 因子计算 | ✅ 26 个内置 + 35 函数表达式引擎 | 22 技术 + 4 基本面，支持字符串公式定义因子 |
| 信号合成 | ✅ 4 种方式 | 等权 / IC 加权 / ICIR 加权 / 投票 |
| 因子筛选 | ✅ 3 种模式 | Ranking / Screener / Vote |
| 组合优化 | ✅ 3 种优化器 | 等权 / 均值方差 / 风险平价 |
| 回测 | ✅ 完整 | 向量化月频回测 + Walk-Forward + 蒙特卡洛 + 参数扫描 |
| 风控 | ✅ 12 层 | VaR / Barra / 压力测试 / 熔断 / Regime / 排雷 / 止损 / 时段 / 连接健康 |
| 实盘交易 | ✅ Paper / QMT / XTP | 多因子信号 → 风控预检 → 自动下单 → P&L 跟踪 |
| 执行层 | ✅ 完整 | 订单 FSM / OMS / TWAP/VWAP / TCA / PortfolioOrchestrator |
| Web 界面 | ✅ Vue 3 + ECharts | Bloomberg 风格仪表盘 + WebSocket 实时推送 |
| API | ✅ 97 个端点 | REST + WebSocket |
| 测试 | ✅ 1205 个 | 覆盖全部模块 |
| 配置版本管理 | ✅ | 自动备份 + diff + rollback |
| LLM 情绪因子 | ✅ DeepSeek | 财经新闻 → 情绪评分 |
| 未来函数检测 | ✅ | 自动化工具，不是只靠文档承诺 |
| 跨源数据验证 | ✅ | 多数据源交叉验证 + 置信度评分 |
| K 线形态识别 | ✅ 12 种 | 吞没/启明星/黄昏星/锤头等 |

### 还没做的

- 日内高频交易（本平台定位日频多因子）
- 期权/期货策略引擎（只有基础接口）
- 日内高频交易（本平台定位日频多因子）
- 多账户管理（目前单账户）

---

## 快速开始

### 安装

```bash
# 克隆
git clone https://github.com/sijie-Z/quant_platform.git
cd quant_platform

# 安装依赖
pip install -r requirements.txt
```

### 运行完整流水线

合成数据模式，不需要任何 API key，直接跑：

```bash
python main.py run
```

三分钟跑完，你会看到这样的输出：

```
[1/6] Loading data...         → 500只股票，5年数据
[2/6] Computing factors...    → 20个因子计算 + IC评价
[3/6] Generating alpha...     → 等权合成信号
[4/6] Portfolio optimization...
[5/6] Running backtest...     → 月频调仓，完整成本模型
[6/6] Generating report...    → 净值曲线 / IC排名 / 压力测试
```

用实盘数据跑：

```bash
# Baostock（免费，无需API key）
python main.py run --use-baostock

# 或者改用 Tushare（需 TUSHARE_TOKEN 环境变量）
export TUSHARE_TOKEN=your_token_here
python main.py run
```

### 运行测试

```bash
# 全部 1205 个测试（跑一次大概 3 分钟）
pytest tests/ -q

# 只看某个模块
pytest tests/test_factors/test_technical.py -v
pytest tests/test_execution/test_engine.py -v
pytest tests/test_risk/test_financial_health.py -v
```

---

## 架构

```
                        ┌─────────────────────────────────────────────┐
                        │         Core Architecture (core/)           │
                        │  EventBus · Store · StateMachine · Audit    │
                        │  Scheduler · RiskMonitor · CircuitBreaker  │
                        └──────────────┬──────────────────────────────┘
                                       │ 全部通过 EventBus 通信
                                       │ 全部状态由 Store 持久化
    ┌──────────────────────────────────┼──────────────────────────────────┐
    v                                  v                                  v
 数据层 ────→ 因子引擎 ────→ Alpha 模型 ────→ 组合优化器
 (6 数据源)   (26 因子 +     (ICIR/Vote/     (MVO/RP/EW)
               表达式引擎)     Screener/ML)
                                                                         |
                                                                         v
 增强回测 ←──── 回测引擎 ←──── 风控模块 ←──── 执行层
 (WalkFW/MC/    (向量化/       (VaR/Barra/     (OMS/算法/
  参数扫描)       成本模型)       MC/熔断/排雷)     TCA/Broker)
                                                                         |
                                                                         v
                 实盘交易引擎  ←──── 多策略管理  ←──── 报告引擎
                 (Paper/QMT/       (资本分配/      (Dashboard/
                  XTP)             聚合P&L)          Alphalens)
                                                                         |
                                                                         v
                              Web 界面 (Vue 3 + ECharts)
                              REST API (97 个端点)
                              WebSocket (实时推送)
```

### 事件驱动

所有组件通过 EventBus 通信，互不了解对方存在：

```python
from quant_platform.core.events import get_event_bus

bus = get_event_bus()
bus.subscribe("order.*", handler)  # 通配符，订阅所有订单事件
bus.subscribe("order.filled", specific_handler)  # 精确匹配优先
bus.publish("order.filled", {"order_id": "...", "price": 10.5})

# 死信队列：处理不了的自动进入死信
# 拦截器链：可以在事件传播中插入过滤/转换
# 事件历史：可以回溯最近 N 条事件
```

### 状态机

```python
INIT → READY → PRE_MARKET → TRADING ↔ REBALANCING → POST_MARKET → READY
任何状态 → HALTED（熔断）
任何状态 → ERROR（不可恢复错误）
```

---

## 因子系统

### 26 个内置因子

**动量因子（4 个）：**
| 因子 | 参数 | 计算公式 |
|------|------|---------|
| `momentum_1m` | period=21 | 过去 21 交易日累计收益 |
| `momentum_3m` | period=63 | 过去 63 交易日累计收益 |
| `momentum_6m` | period=126 | 过去 126 日累计收益 |
| `momentum_12m` | period=252, skip=21 | 过去 252 日收益（跳过最近 21 日避免反转） |

**波动率因子（2 个）：**
| 因子 | 参数 | 说明 |
|------|------|------|
| `volatility_20d` | period=20 | 20 日日收益率标准差 |
| `volatility_60d` | period=60 | 60 日波动率 |

**基础技术因子（4 个）：**
| 因子 | 参数 | 说明 |
|------|------|------|
| `turnover_20d` | period=20 | 20 日平均换手率 |
| `rsi_14d` | period=14 | 相对强弱指数，>70 超买，<30 超卖 |
| `amplitude_20d` | period=20 | 20 日平均振幅 |
| `macd` | fast=12, slow=26, signal=9 | MACD 柱状图值 |

**新因子（7 个，从外部项目吸收）：**
| 因子 | 来源 | 核心思路 |
|------|------|---------|
| `efficiency_ratio` | KF Timing App | 方向/总路径，衡量趋势流畅度 [0,1] |
| `breakout_ignition` | KF Timing App | 放量 + 异动同时触发 → 突破信号 |
| `trend_stage` | a-share-trend-strategy | 价格在 120 日高低点区间的位置 → 鱼头/鱼身/鱼尾 |
| `ma_convergence` | a-share-hybrid-strategy | MA5/10/20 三条均线的间距 → 是否在积蓄突破能量 |
| `breakout_proximity` | a-share-hybrid-strategy | 距 20 日高点的距离 [0,1] |
| `pure_volatility` | 东吴证券研报复现 | FF3 回归残差标准差 → 正交化换手率 → AR(30) 去序列相关 |
| `mtf_resonance` | a-share-signal skill | 日/周/月三周期趋势是否同向 |

**K 线形态因子（5 个，来自 Qlib Alpha158）：**
| 因子 | 公式 | 含义 |
|------|------|------|
| `kmid` | (close - open) / open | 阳线/阴线实体强度 |
| `klen` | (high - low) / open | 日内振幅 |
| `kup` | (high - max(open,close)) / open | 上影线长度 |
| `klow` | (min(open,close) - low) / open | 下影线长度 |
| `ksft` | (2*close - high - low) / open | 收盘价在当日区间位置 |

**K 线形态识别（12 种，单独的 `candle_patterns.py` 模块）：**

```
单根： 十字星、锤头线、射击之星、光脚阳/阴线
双根： 阳包阴、阴包阳、多头母子、空头母子
三根： 早晨之星、黄昏之星、三白兵、三只乌鸦
```

用法：

```python
from quant_platform.factors.candle_patterns import CandlePatternRecognizer
recognizer = CandlePatternRecognizer()
patterns = recognizer.recognize_all(ohlc_df)
for p in patterns:
    print(f"{p.name} ({p.type.value}) 强度={p.strength}")
```

**基本面因子（5 个）：**
`log_market_cap`（对数市值）、`pb_ratio`（市净率）、`pe_ratio`（市盈率）、`roe`（净资产收益率）、`asset_growth`（资产增长率）

**股东结构因子（单独的 `shareholder.py` 模块）：**

从 akshare 获取前十大股东数据，分类识别股东性质：

```python
from quant_platform.factors.shareholder import classify_shareholders

# 获取数据
import akshare as ak
df = ak.stock_gdfx_top_10_em(symbol="sh600519", date="20241231")

# 分析
structure = classify_shareholders(df, code="600519")
print(structure.concentration_label)      # "高度集中" / "相对分散"
print(structure.has_state_owned)          # 是否有国资
print(structure.has_foreign)              # 是否有外资
print(structure.z_index)                  # 第一大 / 第二大股东比
```

### 表达式引擎

受 vnpy 启发，支持**用字符串公式定义因子**，不用写 Python 类：

```python
from quant_platform.factors.expression_engine import ExpressionFactor

# 实现一个 WorldQuant 风格的 alpha 因子
factor = ExpressionFactor(
    name="my_alpha",
    expression="ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)",
)
result = factor.compute(prices)
```

在 YAML 配置里定义：

```yaml
# config/default.yaml
factors:
  expression:
    my_momentum: "ts_sum(close_pct, 21)"
    my_vol: "ts_std(close_pct, 20)"
    complex_alpha: "ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"
```

**内置 35 个函数：**

| 类别 | 函数 | 数量 |
|------|------|------|
| 时序 | ts_sum, ts_mean, ts_std, ts_rank, ts_min, ts_max, ts_corr, ts_cov, ts_slope, ts_delay, ts_delta, ts_decay_linear, ts_product, ts_rsquare, ts_argmax, ts_argmin, ts_quantile | 17 |
| 截面 | cs_rank, cs_mean, cs_std, cs_sum, cs_scale, cs_zscore | 6 |
| 数学 | log, abs, sign, pow, sqrt, less, greater, if_else, scale, neg | 10 |
| 辅助 | 支持 +, -, *, /, >, <, == 等运算符 | |

### 信号合成方式

在 `config/default.yaml` 中设置：

```yaml
alpha:
  method: "icir_weighted"  # 可选: equal_weight, ic_weighted, icir_weighted, vote
```

- **`equal_weight`**：所有因子等权平均，简单但不容易过拟合
- **`ic_weighted`**：按历史 Rank IC 加权（point-in-time，只用过去数据）
- **`icir_weighted`**：按 IC/IC_std 加权（point-in-time），过滤低 ICIR 因子
- **`vote`**：每个因子独立判断多/空/观望，多数决（来自 FinRL 理念）

### 因子筛选模式

除了常规的排名加权，还有两种额外模式：

**Screener（条件筛选模式）：**

```bash
python main.py screen --rules "pe_ratio=lt:30,roe=gt:0.15"
```

返回所有同时满足 PE < 30 和 ROE > 15% 的股票，等权分配。适合快速条件选股。

**Tradability Gate（可交易门控）：**

设置 `alpha.tradability_gate: true` 后，信号会乘以一个"可交易分数"（基于个股的行情状态分类），低 tradability 的股票信号自动归零。适合过滤掉噪音股。

**ScreenFilter（Zipline 风格筛选器）：**

```python
from quant_platform.factors.expression_engine import ScreenFilter, top, bottom

momentum_mask = top(momentum_factor_result, 20)
volatility_mask = bottom(volatility_factor_result, 30)
screen = ScreenFilter(momentum_mask) & ScreenFilter(volatility_mask)

pipe = AlphaPipeline(screen_filter=screen)
signal = pipe.run(factors, returns)
```

---

## CLI 命令集

### 回测与研究

```bash
# 基础回测
python main.py run                                    # 合成数据
python main.py run --use-baostock                     # 实盘数据
python main.py run --force                            # 忽略缓存重算
python main.py run --description "加了新因子 v2"      # 给版本加描述

# 策略对比与参数扫描
python main.py compare --optimizers equal_weight,mean_variance,risk_parity
python main.py sweep --optimizers equal_weight,mean_variance --frequencies monthly,weekly

# 高级分析
python main.py walkforward --folds 6 --method expanding
python main.py ml train --model lightgbm
python main.py ml signal --model xgboost
```

### 因子筛选与排雷

```bash
# 条件选股
python main.py screen --rules "pe_ratio=lt:30,roe=gt:0.15"
python main.py screen --rules "momentum_1m=gt:0.05,volatility_20d=lt:0.03" --logic and

# 检查未来函数（验证数据泄露）
python main.py check-lookahead --threshold 1e-4 --max-dates 20
```

### 配置管理

```bash
python main.py config list           # 查看所有版本
python main.py config show v3        # 查看某版本配置
python main.py config diff v1 v3     # 对比两个版本
python main.py config rollback v2    # 回滚到 v2
python main.py config delete v1      # 删除版本（需确认）
```

### 实盘交易

```bash
# Paper Trading（默认）
python main.py trade
python main.py trade --broker paper --days 60 --cash 5000000

# QMT 实盘（需 miniQMT + xtquant）
export QMT_PASSWORD="your_password"
python main.py trade --broker qmt --universe "600519,000858,000001" --days 30
```

### 全流程执行

```bash
python main.py execute
```

执行完整的"数据 → 因子 → 信号 → 订单创建 → 成交 → 持仓"链，验证所有模块能串起来跑通。

### 其他

```bash
python main.py web                    # 启动 Web 控制台
python main.py profile                # 性能分析
python main.py cache list             # 缓存列表
python main.py cache clear            # 清除缓存
python main.py research report        # LLM 分析报告
python main.py analyze                # 分析已有结果
```

---

## API 接口

FastAPI 提供 97 个 REST 端点 + WebSocket。启动后访问 `/api/docs` 自动生成 Swagger 文档。

```bash
python main.py web
# http://localhost:8000/api/docs  ← Swagger UI
# http://localhost:8000/          ← Vue 3 前端
```

![API Docs](https://img.shields.io/badge/Swagger-UI-blue?logo=swagger)

### 核心端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/run` | POST | 运行完整流水线 |
| `/api/run/{id}/result` | GET | 获取运行结果 |
| `/api/factors` | GET | 因子列表 |
| `/api/screen` | POST | 条件选股 |
| `/api/risk/status` | GET | 风控状态 |
| `/api/trading/start` | POST | 启动交易引擎 |
| `/api/ws` | WS | WebSocket 实时推送 |
| `/api/health` | GET | 健康检查 |

完整列表见 `api/routes.py`，每个端点都有 Pydantic 请求/响应模型和详细的类型注解。

---

## 风控体系

### 12 层风控

| 层次 | 模块 | 机制 |
|------|------|------|
| 1 | VaR | 历史/参数/蒙特卡洛三种 VaR + CVaR |
| 2 | 压力测试 | 2008 金融危机 / 2015 股灾 / 2020 新冠三种场景 |
| 3 | Barra 模型 | 10 因子横截面回归，因子风险 vs 特异性风险分解 |
| 4 | 蒙特卡洛 | Block Bootstrap + Student-t 参数化 + 交易序列洗牌 |
| 5 | 熔断器 | 仓位/行业/亏损/回撤/订单频率限额 + Kill Switch |
| 6 | 行情检测 | 波动率+趋势+相关性三维度 → risk_on/off |
| 7 | 健康检查 | 开盘前自检（数据/资金/持仓/路由/风控） |
| 8 | 行情分类 | 个股级 5 profile 分类器 + tradability 门控 |
| 9 | 未来函数检测 | 截断数据重算信号 vs 全量信号 -> 差异即泄露 |
| 10 | 动态止损 | 三层止损（监控→半仓→清仓）+ 反弹保护 |
| 11 | 时段风险 | 不同交易时段不同风险乘数（开盘尾盘最危险） |
| 12 | 财务排雷 | 24 条规则检测造假 + ST 退市风险 + 资本周期 + 护城河评分 |

### 动态止损

```
10:45 之前 → 不做止损（避免开盘剧烈波动误触发）
10:45-14:50 → 正常执行
14:50 之后 → 不做止损（避开尾盘集合竞价）

亏损达到 -0.5% → 进入监控
亏损达到 -1.2% → 减半仓（创业板/科创板 -1.6%）
亏损达到 -2.5% → 清仓（创业板/科创板 -3.5%）

反弹保护：触发半仓止损后，如果价格反弹回成本价以上 → 重置止损状态
```

### 财务排雷

```python
from quant_platform.risk.financial_health import FraudDetector, assess_st_risk

# 24 条规则排雷
detector = FraudDetector()
report = detector.analyze(financials_df)
print(report.summary)   # 排雷评分: 15分 (中风险)
for rule in report.rules:
    if rule.status == 'fail':
        print(f"  FAIL: {rule.name} — {rule.detail}")

# ST 退市风险评估
st = assess_st_risk(financials_df)
if st.total_score >= 5:
    print(f"ST 风险 {st.risk_level}")  # 高风险 / 极高风险

# 巴菲特 Owner Earnings
from quant_platform.risk.financial_health import owner_earnings, capital_cycle_stage
oe = owner_earnings(net_income=1000, depreciation=200, maintenance_capex=150)
cc = capital_cycle_stage(capex=[200, 150, 100], depreciation=[100, 95, 90])
print(cc['stage'])  # 投资扩张期 / 稳态期 / 产能出清期
```

---

## 执行层

### 订单状态机

```python
PENDING → SUBMITTED → PARTIAL → FILLED
                  ↘ CANCELLED / REJECTED / EXPIRED
```

每个状态迁移都经过 `transition_order()` 函数校验，非法迁移直接抛异常。每次迁移自动向 EventBus 发布 `order.status` 事件。

```python
from quant_platform.execution.engine import ExecutionEngine, OrderSide

engine = ExecutionEngine()
order = engine.create_order("600519", OrderSide.BUY, 100)
engine.submit_order(order)
engine.process_fill(order, price=150.0, quantity=100, commission=4.5)

pos = engine.get_position("600519")
print(f"持仓: {pos.quantity} 股, 成本: {pos.avg_cost}")

snapshot = engine.portfolio_snapshot({"600519": 155.0})
print(f"未实现盈亏: {snapshot['total_unrealized_pnl']}")
```

### Portfolio Orchestrator

连接因子流水线和执行引擎的桥梁：

```
Alpha 信号 → PortfolioOrchestrator → ExecutionEngine → Broker
                    ↕
        MultiStrategyManager（资本分配 + 风控）
```

处理信号→目标仓位→买卖订单的转换：

```python
from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator
from quant_platform.strategy.multi_strategy import MultiStrategyManager, StrategyConfig

ms = MultiStrategyManager(total_capital=1_000_000)
sid = ms.add_strategy(StrategyConfig(name="test", allocation_pct=1.0))
orchestrator = PortfolioOrchestrator(ms)

orchestrator.on_signal("2025-01-01", alpha_signal, strategy_id=sid)
orchestrator.rebalance()
orchestrator.process_fills(prices)

summary = orchestrator.portfolio_summary()
print(f"持仓数: {summary['n_positions']}, PnL: {summary['total_pnl']}")
```

---

## 数据源

| 名称 | 类型 | 需要 API Key | 覆盖内容 |
|------|------|-------------|---------|
| Baostock | 免费 A 股数据 | ❌ | 日K/周K/月K/财务/行业分类 |
| Tushare | 免费+付费 A 股数据 | ✅ 需注册 | 日K/分钟K/财务/资金流/龙虎榜 |
| Adata | 免费 A 股数据 | ❌ | 日K/概念板块/资金流/龙虎榜/北向 |
| AKShare | 免费全市场 | ❌ | A 股实时行情/板块/新闻 |
| 东方财富 | HTTP 直连 | ❌ | 实时行情/龙虎榜/资金流/涨停板 |
| 新浪财经 | HTTP 直连 | ❌ | 实时报价/历史K线 |
| PostgreSQL | 自有数据 | ❌ | 存储回测结果/策略状态 |
| Synthetic | 合成数据 | ❌ | 500只/5年/可复现 |

多源交叉验证（可选）：

```yaml
# config/default.yaml
data:
  validated: true   # 同时从 Baostock + Adata 拉数据，对比一致性
```

启用后系统会自动从两个源分别取数、对比偏差、给出置信度分数。

---

## 配置管理

每次运行 `python main.py run` 时，系统会自动保存当前配置为一个版本（v1, v2, v3...）。

```bash
python main.py config list
# v3  2026-05-31T12:00:00   Run: alpha=icir_weighted optimizer=risk_parity
# v2  2026-05-30T10:30:00   Run: alpha=equal_weight optimizer=mean_variance
# v1  2026-05-29T18:00:00   Initial config

python main.py config diff v1 v3      # 查看两次运行之间改了什么
python main.py config rollback v2     # 回滚到 v2 版本的配置
```

---

## 项目结构

```
quant_platform/
├── main.py                     # CLI 入口（15 个命令）
├── app.py                      # FastAPI 入口
├── config/default.yaml         # 所有可配置参数（零硬编码）
├── config/schema.py            # 类型化 dataclass 配置校验
│
├── core/                       # ★ 核心架构（事件驱动）
│   ├── events.py               # EventBus: topic pub/sub
│   ├── store.py                # SQLite WAL: 8张表
│   ├── state_machine.py        # 8 状态 FSM
│   ├── scheduler.py            # A 股开市时间调度
│   ├── audit.py                # 合规审计三路输出
│   └── instrument.py           # 跨资产抽象
│
├── data/                       # 数据层
│   ├── providers/              # 6 个数据源
│   ├── pipeline.py             # 清洗/对齐/过滤
│   ├── quality.py              # 8 项数据质量检查
│   ├── board_scanner.py        # 实时全市场扫描（涨停/强势/连板）
│   └── providers/validated_provider.py  # 多源交叉验证
│
├── factors/                    # ★ 因子引擎
│   ├── technical.py            # 22 个技术因子
│   ├── fundamental.py          # 5 个基本面因子
│   ├── expression_engine.py    # 表达式引擎 + 35 个函数
│   ├── expressions/            # ts_* / cs_* / math_* 函数库
│   ├── candle_patterns.py      # K 线形态识别（12 种）
│   ├── shareholder.py          # 股东结构分析
│   ├── registry.py             # 因子注册表
│   ├── evaluation.py           # IC 分析 + Alphalens 导出
│   ├── processing.py           # 缩尾/标准化/中性化
│   ├── ic_monitor.py           # IC 衰减监控
│   ├── network.py              # 图网络因子
│   └── orthogonalization.py    # 因子正交化
│
├── alpha/                      # Alpha 信号
│   ├── pipeline.py             # 信号生成管道
│   ├── combination.py          # 4 种合成方法
│   └── ml_signal.py            # XGBoost/LightGBM
│
├── portfolio/                  # 组合优化
│   ├── optimizers.py           # 等权 / MVO / 风险平价
│   ├── black_litterman.py      # Black-Litterman 模型
│   ├── constraints.py          # A 股约束
│   ├── screener.py             # 因子条件筛选器
│   └── covariance.py           # 协方差估计（Ledoit-Wolf Numba）
│
├── backtest/                   # 回测
│   ├── engine.py               # 向量化多期回测
│   ├── cost_model.py           # A 股成本模型
│   ├── metrics.py              # 绩效指标 + print_summary
│   ├── walkforward.py          # Walk-Forward 验证
│   └── distributed.py          # 并行参数扫描
│
├── execution/                  # 执行层
│   ├── engine.py               # Order FSM + ExecutionEngine
│   ├── oms.py                  # 订单管理系统
│   ├── algorithms.py           # TWAP/VWAP/Iceberg
│   ├── tca.py                  # 交易成本分析
│   ├── connection.py           # 连接生命周期管理
│   └── order_book.py           # 限价订单簿
│
├── trading/                    # 实盘交易
│   ├── broker.py               # Simulated/QMT/XTP Broker
│   ├── engine.py               # 实盘交易引擎
│   ├── realtime.py             # AKShare 实时行情
│   └── live_runner.py          # 实盘试跑
│
├── risk/                       # ★ 风控体系（12 层）
│   ├── circuit_breaker.py      # 实时风控 + Kill Switch
│   ├── var.py                  # VaR/CVaR
│   ├── stress.py               # 压力测试
│   ├── barra.py                # 10 因子 Barra 模型
│   ├── regime.py               # 行情状态检测 + 自适应参数
│   ├── profile_classifier.py   # 个股行情分类
│   ├── lookahead_detector.py   # 未来函数检测
│   ├── stop_loss.py            # 动态三层止损
│   ├── time_segment.py         # 交易时段风控
│   └── financial_health.py     # 24 条排雷 + ST 风险 + 资本周期 + 护城河
│
├── strategy/                   # 策略管理
│   ├── multi_strategy.py       # 多策略资本分配 + P&L 聚合
│   └── portfolio_orchestrator.py  # 信号→订单桥梁
│
├── agent/                      # LLM
│   ├── sentiment_factor.py     # DeepSeek 情绪因子
│   └── research_agent.py       # 研报分析 Agent
│
├── api/routes.py               # 97 个 REST 端点 + WebSocket
├── api/schemas.py              # Pydantic 模型
├── frontend/                   # Vue 3 + ECharts
├── monitoring/                 # Grafana 仪表盘
├── utils/                      # Numba / PipelineCache / Config / Metrics
│
├── tests/                      # 1205 个测试
│   ├── test_core/              # 137 测试
│   ├── test_data/              # 154 测试（含 validated provider）
│   ├── test_factors/           # 131 测试
│   ├── test_alpha/             # 42 测试
│   ├── test_portfolio/         # 36 测试
│   ├── test_backtest/          # 51 测试
│   ├── test_execution/         # 141 测试（含 engine FSM）
│   ├── test_trading/           # 112 测试
│   ├── test_risk/              # 55 测试（含 financial health）
│   ├── test_strategy/          # 14 测试（含 orchestrator）
│   └── test_utils/             # 33 测试
│
└── docs/ABSORPTION_NOTES.md    # 外部项目吸收笔记
```

---

## 外部吸收

这个项目从 20+ 个外部开源项目中吸收设计理念。每一行代码都标明了来源。

| 来源项目 | 吸收内容 | 差不多行数 | 类型 |
|---------|---------|-----------|------|
| vnpy | 表达式驱动因子引擎 | 1248 | 架构级 |
| freqtrade | 未来函数检测工具 | 408 | 架构级 |
| KF Timing App | 个股行情分类 + Tradability Gate | 599 | 架构级 |
| BlackOil-OmniAlpha | Factor Screener 条件选股 | 1165 | 新功能 |
| Zipline | Factor Filters + ScreenFilter | ~150 | 新功能 |
| FinRL | 投票信号组合 | ~44 | 新功能 |
| PyPortfolioOpt | Black-Litterman 模型 | ~220 | 新功能 |
| 悟道真英雄 | 配置版本管理 | 680 | 基建 |
| Hummingbot | 连接生命周期管理 | ~153 | 基建 |
| Backtrader | print_summary 格式化输出 | ~70 | 基建 |
| Jesse | 交易序列洗牌 MC | ~52 | 基建 |
| Qlib | K 线形态因子 | ~96 | 因子 |
| a-share-trend-strategy | 趋势阶段因子 | ~25 | 因子 |
| a-share-hybrid-strategy | 均线汇聚 + 突破临近因子 | ~55 | 因子 |
| financial-report-minesweeper | 24 条排雷规则 + ST 风险 | ~400 | 风控 |
| AlphaPilot Pro | 动态止损 + 时段风控 | ~300 | 风控 |
| Senior Analyst | 多源数据交叉验证 | ~280 | 数据 |
| Buffett Skill | Owner Earnings + 护城河评分 | ~120 | 风控 |
| PureVolatility 研报 | 特质波动率因子 | ~80 | 因子 |
| a-share-signal skill | 多周期共振因子 | ~45 | 因子 |
| stock-analyst | K 线形态识别 | ~275 | 因子 |
| Adata | Adata 数据源 | ~200 | 数据 |
| **合计** | **~5000+ 行** | | |

详细记录见 [docs/ABSORPTION_NOTES.md](docs/ABSORPTION_NOTES.md)。

---

## 技术栈

| 层面 | 技术选型 |
|:-----|:---------|
| 编程语言 | Python 3.10+ |
| 异步框架 | asyncio |
| 数据处理 | Pandas, NumPy |
| 优化求解 | cvxpy, SciPy |
| 机器学习 | XGBoost, LightGBM, SHAP, scikit-learn |
| 性能加速 | Numba（6 个 JIT 内核：Rank IC / 缩尾 / zscore / Ledoit-Wolf / 动量 / 回撤） |
| 表达式引擎 | 自研，35 个函数，Python eval() 沙箱 |
| Web 框架 | FastAPI |
| 前端 | Vue 3, Vite, ECharts, Lightweight-Charts |
| 数据存储 | SQLite (WAL), PostgreSQL (asyncpg) |
| 数据源 | Baostock, Tushare, Adata, AKShare, East Money, Sina |
| 文档 | Swagger (自动生成), 中英双语 README |
| 监控 | Prometheus, Grafana（16 面板） |
| 容器化 | Docker, Docker Compose |
| CI/CD | GitHub Actions（Python 3.10/3.11/3.12 矩阵） |

---

## 许可证

MIT License

Copyright (c) 2026

本软件仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
