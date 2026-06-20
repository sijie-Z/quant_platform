# 项目快照 — 可验证的量化研究与策略开发平台

> **定位**：一个可验证（Verifiable）的 A 股多因子量化研究与策略开发平台。
> 
> 与普通"多因子平台"的区别：核心研究链路已通过系统性验证——Oracle Factor IC=1.0、Known Alpha Recovery、WalkForward、MVO Audit。
>
> 最后更新：2026-06-20

---

## 一、核心能力（已验证）

### 研究验证体系 — Research Validation

这是本平台区别于其他量化项目的核心竞争力。

| 验证项 | 方法 | 结果 | 意义 |
|-------|------|------|------|
| **Oracle Factor** | 用已知未来收益作为因子计算 IC | **IC = 1.000000** | IC 计算与数据对齐完全正确 |
| **Known Alpha Recovery** | 生成含已知预测性 Alpha 的数据，验证因子引擎能否恢复 | **IC 与理论值一致** | 因子链路完整 |
| **WalkForward** | 多 fold 滚动 OOS 验证 | **全通过** | 回测无前视偏差 |
| **MVO Audit** | 60 次调仓全部日志记录 | **60/60 Success, 0 Fallback** | 优化器工作正常 |
| **Rank IC 对比** | 手动 vs 官方计算 | **一致（差异 < 0.001%）** | evaluation.py 正确 |
| **No-Lookahead 防护** | Point-in-time IC 加权 + publish_date 过滤 + fold 内重算 | **6 道防线全部实现** | 研究过程严格因果 |

**详细验证报告**：`docs/VALIDATION_REPORT.md`

### No-Lookahead Contract

8 条不可协商的铁规，确保研究过程零前视偏差：

```
1. 价格因子：只能用 signal_date 当天及之前的数据
2. Alpha 权重：每期权重只用该期之前的 IC 历史
3. 基本面因子：只能用 report_date，不用 fiscal_period_end
4. WalkForward：每个 fold 用 train-only 数据重算信号
5. 合成数据嵌入式 Alpha：仅用于演示
6. IC 计算：shift 链条正确
7. 行业分类：用 effective_date
8. ST 状态：用 announce_date
```

**详细文档**：`docs/NO_LOOKAHEAD_CONTRACT.md`

### Strategy Gates

5 道评估门禁，产出 PASS/WARNING/FAIL/REJECTED：

| 门禁 | 检查内容 | 集成组件 |
|------|---------|---------|
| IC Quality | 因子 IC/ICIR 是否达阈值 | factors/evaluation.py |
| WalkForward | OOS Sharpe、fold 数、train/test gap | backtest/walkforward.py |
| Drawdown Risk | 最大回撤、Kill Switch | risk/circuit_breaker.py |
| Complexity | 因子数量、参数数量 | 策略定义检查 |
| Data Coverage | 价格/基本面覆盖率 | data/quality.py |

**代码路径**：`strategy/gates.py`

### Synthetic Benchmark Suite

可配置的合成数据基准，用于验证和演示：

| Strength | IC 水平 | 用途 |
|----------|---------|------|
| 0 (off) | ≈ 0.000 | 研究验证（纯噪声） |
| 0.03 (weak) | ≈ 0.025 | 默认，模拟真实 A 股 IC |
| 0.06 (normal) | ≈ 0.052 | 演示友好 |
| 0.12 (strong) | ≈ 0.091 | 强信号测试 |
| 0.50 (oracle) | ≈ 0.115 | Pipeline 正确性测试 |

Alpha 信号设计：`alpha[t] → return[t+1]`（预测性，非同期），信号加噪以模拟真实信噪比。

**代码路径**：`data/providers/synthetic.py`

---

## 二、研究平台能力（已验证+可用）

### 因子引擎 — Factor Engine

- **20+ 因子**：Momentum(1/3/6/12m)、Volatility(20/60d)、Turnover、RSI、MACD、Efficiency Ratio、Breakout Ignition、Candle 系列、Trend Stage、MA Convergence、Breakout Proximity、Multi-Timeframe Resonance、Pure Volatility
- **基本面因子**：log_market_cap、PB、PE、ROE、Asset Growth
- **因子处理**：缩尾(1%/99%) → zscore 标准化 → 行业+市值中性化（线性回归残差）
- **图网络因子**：股票关联网络 + 4 种中心性度量
- **Numba JIT 加速**：6 个计算内核，5-20x 加速比
- **IC 实时监控**：衰减检测 → 半衰期估计 → 自适应权重

### Alpha 信号合成 — Alpha Pipeline

- 3 种合成方法：等权 / IC 加权 / ICIR 加权（均为 Point-in-time，无前视）
- 集成投票法
- ML 信号（XGBoost / LightGBM + Walk-Forward CV + SHAP）

### 组合优化 — Portfolio Optimizer

- EqualWeight / MeanVariance(cvxpy) / RiskParity(cvxpy)
- MVO 已验证 60/60 成功求解
- 约束：纯多头、权重上限、行业上限、换手上限、手数

### 回测引擎 — Backtest Engine

- 向量化多期回测（月频/周频/日频）
- A 股成本模型：佣金 0.03%（双边）+ 印花税 0.1%（仅卖出）+ 滑点
- Walk-Forward 验证（滚动/扩展窗口）
- 蒙特卡洛模拟 + 压力测试

### 风险管理 — Risk Management

- VaR/CVaR（三种方法）
- 实时风控熔断：仓位/行业/日亏损/回撤限额 + Kill Switch
- Barra 10 因子风险模型
- 行情状态检测（波动率/趋势/相关性三维度）
- 开盘前系统自检（数据/资金/持仓/路由/风控）

### 策略管理 — Strategy DSL

- YAML 策略定义，可版本化、可复现
- 因子权重/组合参数/风控阈值/执行设置 全部 DSL 化
- 版本注册 + 运行历史

### 因子研究持久化 — Factor Research Store

- 8 张 SQLite 表：因子定义 / 值 / IC 历史 / 回测历史 / WalkForward / 稳定性 / 行情状态 / 版本
- `python main.py factor-store rank` 查看因子健康度

---

## 三、扩展能力（非核心，可独立部署）

### 数据层
- 数据源：Synthetic / Tushare / Baostock / PostgreSQL / WebSocket / Level 2
- 实时行情：AKShare 全市场快照
- 实时基本面：PE/PB/ROE 缓存 + 批量获取

### 执行与交易
- OMS：订单全生命周期（PENDING→FILLED）
- 执行算法：TWAP / VWAP / Iceberg + SmartRouter
- TCA：Implementation Shortfall / Arrival Price
- 实盘：SimulatedBroker(LOB) / QMT(xtquant) / XTP

### API 与前端
- FastAPI 97 REST 端点
- WebSocket 实时推送
- Vue 3 Bloomberg 风格仪表盘

### 监控与合规
- Prometheus 指标 + Grafana 16 面板
- 合规审计：三路输出（SQLite+EventBus+Logger）
- 基金运营：NAV 计算 + 投资者门户

---

## 四、项目规模

| 维度 | 数据 |
|------|------|
| Python 模块 | 143 |
| Vue 组件 | 35 |
| 代码行数（Python） | ~30,000 |
| 代码行数（Vue） | ~9,800 |
| 单元测试 | **1,222** |
| 测试通过率 | **99.9%（1 个已知环境兼容性问题）** |
| 验证实验 | 6 项全部通过 |

---

## 五、已知限制

| 问题 | 级别 | 说明 |
|------|------|------|
| Pure Volatility 因子计算失败 | P2 | shape mismatch，不影响其他因子 |
| FastAPI Router 版本兼容 | P3 | 不影响核心研究链路 |
| 无统一配置校验 | P3 | 配置拼写错误静默生效 |
| RiskParity 未在验证中覆盖 | P3 | 默认使用 MVO |

---

## 六、v1.0 验收标准

本版本已通过以下验收标准：

- [x] Oracle Factor IC ≈ 1.0
- [x] Known Alpha IC ≈ 理论值
- [x] Random Factor IC ≈ 0
- [x] MVO 回测成功率 > 95%（实际 100%）
- [x] WalkForward 全通过
- [x] No-Lookahead Contract 文档化并验证
- [ ] 真实 A 股数据验证（P1，下一阶段）

---

## 七、项目发展路线

### Phase 1（当前）— 平台验证完成 ✅
- 核心研究链路验证通过
- 新增 Factor Store / Strategy DSL / Gates / Research Validation
- No-Lookahead Contract 文档化

### Phase 2（下一阶段）— 真实数据验证
- ✅ 已有 VALIDATION_REPORT 作为基线
- 接入真实 A 股数据跑一次完整的 Known Alpha 验证
- 将验证流程集成到 CI

### Phase 3（远期）— 策略研究与 Agent 集成
- 基于验证平台的策略研究
- MCP 接口规划
- Agent 驱动的因子挖掘
