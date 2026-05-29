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
- `embedded_alpha=False`（默认）→ 纯噪声，只用来测代码不报错
- `embedded_alpha=True` → 仅限面试演示，**永远不要用来验证策略**

```yaml
# config/default.yaml
data:
  provider: baostock       # ❌ 不要用 synthetic
  synthetic:
    embedded_alpha: false  # 保持 false
```

### 2. 回测信号 = 实盘信号

**问题**：之前回测用 `AlphaPipeline` + 15个因子，实盘用自己另外算的4个因子。两套逻辑，回测结果毫无参考价值。

**修复后**：
```
回测: 数据 → FactorEngine(15因子) → Process → AlphaPipeline → Optimizer → BacktestEngine
实盘: 数据 → LiveSignalGenerator(同15因子) → Process → AlphaPipeline → 下单
```

`LiveSignalGenerator` (`trading/signal_generator.py`) 使用和回测完全相同的因子类和 AlphaPipeline。

### 3. 交易成本 = 市场冲击

**问题**：回测用固定0.1%滑点，实盘的冲击成本随资金量非线性增长。

**修复后**：
```yaml
costs:
  slippage_model: "impact"   # 使用 Almgren-Chriss + Square-Root 模型
  impact_model: "composite"  # 考虑参与率、波动率、价差
```

`CostModel` 在 `slippage_model='impact'` 时会调用 `ExecutionCostCalculator`，根据订单规模/日均成交量/波动率动态计算冲击成本。

---

## P1: 研究流水线

### 标准流程

```bash
# 1. 数据准备
python main.py fetch-data --provider baostock --start 2020-01-01 --end 2025-12-31

# 2. 因子研究
python main.py run --provider baostock                  # 全流水线
python main.py sweep --optimizers mean_variance,risk_parity  # 参数扫描
python main.py compare                                   # 策略对比

# 3. Walk-Forward 验证 (最重要的一步)
python main.py walkforward --method expanding --folds 6

# 4. 前测 (Paper Trading)
python main.py trade --broker paper --days 60 --cash 1000000

# 5. 实盘
python main.py trade --broker qmt --days 30 --cash 1000000
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

**三级回退机制**（`connection_pool.py`）：
```
Tushare → Baostock → Synthetic
```
自动检测 API 可用性，无需手动切换。

### PostgreSQL 生产配置

```bash
# docker-compose.yml 已经包含 PostgreSQL 服务
docker-compose up -d postgres

# 平台自动检测 PostgreSQL 连接，可用则使用，不可用回退 SQLite
```

---

## P3: 过拟合控制清单

每个策略上线前，逐项检查：

| # | 检查项 | 通过条件 |
|---|--------|---------|
| 1 | Point-in-time IC 加权 | 信号权重只用 t 日之前的数据计算 |
| 2 | Walk-Forward OOS | OOS Sharpe > 0，且各 fold 一致 |
| 3 | Deflated Sharpe | 考虑多因子多重检验后的 Sharpe |
| 4 | 合成数据禁用 | `embedded_alpha: false` |
| 5 | 成本模型 | `slippage_model: impact` |
| 6 | 实盘信号一致 | `LiveSignalGenerator` 使用和回测相同的因子 |
| 7 | 前测验证 | Paper Trading 至少跑 60 个交易日 |

---

## 已知限制

1. **SQLite 并发**：生产环境请用 PostgreSQL（docker-compose up -d）
2. **Level 2 数据**：仅回放/模拟，无真实券商 L2 接入
3. **日内回测**：`tick_engine.py` 支持 tick 级回测，但默认走日频
4. **QMT 接口**：需要开通券商 miniQMT + 安装 xtquant 包

---

## 部署

### Docker

```bash
docker-compose up -d --build
# 访问 http://localhost:8000
# API: http://localhost:8000/api/health
```

### 手动

```bash
pip install -r requirements.txt
pip install -e .  # 使 quant_platform 可导入
python main.py web
```
