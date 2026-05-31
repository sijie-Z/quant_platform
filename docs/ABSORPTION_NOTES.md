# 外部项目吸收笔记

## 概述

本文记录了从外部开源项目吸收的设计理念和具体实现。

| 项目 | 吸收 | 新增行数 | 状态 |
|------|------|---------|------|
| BlackOil-OmniAlpha | Factor Screener | 1165 | ✅ merged |
| 悟道真英雄 | Config Version Manager | 680 | ✅ merged |
| KF Timing App | Profile Classifier + Tradability Gate | 599 | ✅ merged |
| vnpy | Expression-based Factor Engine | — | 🏗️ WIP |

---

## 1. BlackOil-OmniAlpha — Factor Screener

### 项目定位
简单的 A 股选股工具，~2930 行。不是量化平台，是条件筛选器。

### 吸收：Factor Screener（因子布尔筛选）

**核心设计**：一个 `StockStrategy.check(code, data_df) -> (bool, dict)` 接口，每只股票逐一判定，支持 AND/OR 组合。

**我们学到**：之前的平台只有"多因子→ICIR加权→排名→优化器"一条路。加一个 **条件筛选模式** 让用户可以直接写规则（pe < 30 AND roe > 0.15）做快速选股。

**实现**：`portfolio/screener.py` + CLI `python main.py screen` + `POST /api/screen`

**关键代码**：
```python
@dataclass
class ScreenRule:
    factor: str
    operator: str  # gt, lt, gte, lte, eq, ne, between
    value: float | list[float]

class FactorScreener:
    def screen(self, processed_factors, rules, logic="and"):
        cross = self._build_cross_section(processed_factors, rules, target_date)
        mask = self._apply_rules(cross, rules, logic)
        return cross.index[mask].tolist()
```

---

## 2. 悟道真英雄 — Config Version Manager

### 项目定位
Claude Code Skill，交易心理辅导 + 悟道人格创建，不是量化工具。

### 吸收：Config Version Manager（配置版本管理）

**核心设计**：每次进化前自动备份配置文件，支持版本列表查看、回滚、版本间 diff。

**我们学到**：虽然 Store 已经有 `config_snapshots` 表，但缺少显式版本号、回滚命令、版本 diff。

**实现**：`utils/version_manager.py` + CLI `python main.py config list|show|diff|rollback|delete`

**关键代码**：
```python
class VersionManager:
    def save(self, config_dict, description="") -> str  # 自动 v1/v2/v3
    def list(self) -> list[ConfigVersion]
    def show(self, version_id) -> dict
    def diff(self, v1, v2) -> str                       # unified diff
    def rollback(self, version_id, target_path)
```

---

## 3. KF Timing App — Profile Classifier + Tradability Gate

### 项目定位
单股票 Kalman Filter 择时系统，~2000 行。用状态空间模型做趋势/周期分解。

### 吸收一：Efficiency Ratio 因子

**原设计**：
```python
er = abs(close.diff(win)) / abs(close.diff()).rolling(win).sum()
```
值域 [0,1]，1 = 完美趋势，0 = 纯噪音。

**我们原有**：momentum 因子衡量"趋势大小"，但无法区分"流畅趋势 vs 剧烈震荡"。

**实现**：`factors/technical.py` 新增 `EfficiencyRatioFactor`

### 吸收二：Breakout Ignition 因子

**原设计**：同时监测 return shock + volume shock，两者同时超过阈值视为突破启动。

**实现**：`factors/technical.py` 新增 `BreakoutIgnitionFactor`

### 吸收三：Per-Stock Profile Classifier

**原设计**：5 个 profile（Trend_follower/Breakseeker/Defender/Activist/All_other），
通过 efficiency + coherence + curvature + breakout 特征进行个股级市场状态分类。
用 tradability score = 0.65*efficiency + 0.35*coherence 量化"可交易程度"。

**我们学到**：之前的 `regime.py`（CompositeRegimeDetector）只做市场级别的风险分类，
没有个股级别的状态检测。

**实现**：`risk/profile_classifier.py` — 5-profile 分类器 + tradability gate
集成到 `AlphaPipeline` — 低 tradability 股票信号自动压制

**关键代码**：
```python
class ProfileClassifier:
    def classify(self, prices, volume) -> dict[asset -> {profile, tradability, ...}]
    def compute_tradability_gate(self, prices, volume, min_tradability=0.3) -> DataFrame

def apply_tradability_gate(signal, prices, volume, min_tradability):
    gate = classifier.compute_tradability_gate(prices, volume, min_tradability)
    return signal * gate  # 低 tradability 股票信号归零
```

---

## 4. vnpy — Expression-based Factor Engine

### 项目定位
中国最知名的开源量化交易框架。数十万行代码，数十个交易接口，
7+ 数据库适配器，完整的 AI 策略模块（vnpy.alpha）。

### 为什么选这个吸收

vnpy 体量太大，不可能整体借鉴。但有一个设计理念值得单独抽出来：

**Expression-based Factor Engine**（表达式驱动的因子计算引擎）

### 原设计分析

vnpy.alpha 的 `dataset` 模块提出了一种**因子 = 字符串公式**的范式。

你不需要写 Python 类来定义因子：

```python
# vnpy 的做法
dataset.add_feature("momentum_1m", "ts_sum(returns, 21)")  
dataset.add_feature("volatility", "ts_std(returns, 20)")
dataset.add_feature("rank_pe", "cs_rank(pe_ttm)")
dataset.add_feature("complex", "ts_rank(ts_sum(returns, 5) / ts_std(returns, 20), 10)")
```

背后的 `DataProxy` 类通过运算符重载 + 函数注册表将字符串表达式编译为实际计算：

```
calculate_by_expression(df, "ts_sum(close / ts_delay(close, 1) - 1, 21)")
                      → close.pct_change().rolling(21).sum()
```

**内置函数库包含**：

| 类别 | 函数 | 数量 |
|------|------|------|
| 时序 | ts_sum, ts_mean, ts_std, ts_rank, ts_min, ts_max, ts_corr, ts_cov, ts_slope, ts_delay, ts_delta, ts_decay_linear, ts_product... | ~20 |
| 截面 | cs_rank, cs_mean, cs_std, cs_sum, cs_scale | 5 |
| TA | ta_rsi, ta_atr | 2+ |
| 数学 | log, abs, sign, less, greater, pow | 10+ |

### 我们有什么

```python
# 当前做法：每个因子是一个 Python 类
class Momentum1M(MomentumFactor):
    def __init__(self):
        super().__init__(period=21, name="momentum_1m", skip=0)

# 注册
registry.register(Momentum1M)
# 计算
raw_factors["momentum_1m"] = Momentum1M().compute(prices)
```

### 差距

| 维度 | 当前（硬编码因子类） | vnpy（表达式驱动） |
|------|---------------------|-------------------|
| 定义新因子 | 写 Python 类，注册 | 写字符串公式 |
| 迭代速度 | 改代码 → 重启 | 改文本 → 重跑 |
| 用户门槛 | 需要 Python 开发 | 会写公式即可 |
| 探索效率 | 1 分钟/因子 | 5 秒/因子 |
| 集成 | 无法动态组合 | 可任意组合已有函数 |

### 实现方案

新增 `factors/expression.py`，核心组件：

```
factors/expression.py        —— DataProxy + calculate_by_expression + ExpressionFactor
factors/expressions/
    __init__.py              —— 函数注册表
    ts_functions.py          —— 时序函数 (ts_sum, ts_mean, ts_std, ts_rank...)
    cs_functions.py          —— 截面函数 (cs_rank, cs_mean, cs_std...)
    math_functions.py        —— 数学函数 (log, abs, sign, pow...)
    ta_functions.py          —— TA 函数 (ta_rsi, ta_atr...)
```

**ExpressionFactor**（和现有 BaseFactor 体系共存）：
```python
class ExpressionFactor(BaseFactor):
    """因子表达式 → 自动计算"""
    def __init__(self, name, expression, params=None):
        self._name = name
        self._expression = expression
        ...
    def compute(self, prices, **kwargs):
        return calculate_by_expression(self._build_df(prices), self._expression)
```

**注册方式**（在 `config/default.yaml` 中即可定义）：
```yaml
factors:
  expression:
    momentum_1m: "ts_sum(close_pct, 21)"
    volatility_20d: "ts_std(close_pct, 20)"
    custom_factor: "ts_rank(ts_sum(close_pct, 5) / ts_std(close_pct, 20), 10)"
```

**关键接口**：
```python
def calculate_by_expression(df, expression) -> pd.DataFrame
    # 解析表达式 → 建立计算图 → 逐层求值 → 返回结果
```

### 价值

1. **零代码定义新因子**。在 YAML 里写个公式就行，不需要写 Python 类
2. **快速迭代**。改一行配置等于加了一个新因子
3. **复杂因子组合**。`ts_rank(ts_sum(returns, 5) / ts_std(returns, 20), 10)` = 过去 10 天的"5 天收益 / 20 天波动率"的百分位排名——这是 WorldQuant 101 里的一个典型 Alpha
4. **与现有系统完全兼容**。ExpressionFactor 继承自 BaseFactor，可以跟普通因子一起注册、一起处理、一起进 Alpha 流水线
