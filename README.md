# A-Share Multi-Factor Quantitative Trading Research Platform

A股多因子量化研究平台 —— 从数据到回测的完整流水线。面向量化开发面试，展示架构设计、性能优化、真实市场处理、LLM增强选股。

**状态**：105 单元测试全部通过。合成数据端到端 ~3 分钟，Tushare 实盘数据 ~5 分钟。

> 📖 完整文档见 [CLAUDE.md](CLAUDE.md)（架构详解/文件树/模块说明/面试亮点/扩展指南/改进记录）

## 架构

```
config/*.yaml  -->  Data Layer  -->  Factor Engine  -->  Alpha Model
                                                          |
                                                          v
Reports  <--  Backtest Engine  <--  Portfolio Optimizer
                 |
            Risk Module (VaR, CVaR, Stress)
```

## 功能

- **数据层**：抽象 DataProvider 接口 + 合成A股数据生成器（500只/5年/可复现/内含alpha结构）+ Tushare 实盘接入
- **因子引擎**：10个技术因子 + 5个基本面因子 + LLM情感因子，横截面缩尾/标准化/中性化
- **因子评估**：Rank IC / Pearson IC / ICIR / 分位数收益 / 相关性矩阵 / IC衰减
- **Alpha模型**：等权/IC加权/ICIR加权 三种合成法
- **组合优化**：等权基准 / 均值方差(cvxpy) / 风险平价，含行业+换手约束
- **回测引擎**：向量化月频回测，完整A股成本模型（佣金+印花税+滑点）
- **风险管理**：VaR/CVaR / 压力测试 / 行业暴露分析
- **报告**：净值曲线 / 回撤 / 滚动Sharpe / 月度热力图 / 文本仪表盘
- **LLM增强**：财经新闻情感因子，Strategy模式可插拔OpenAI

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流水线（合成数据，无需API key）
python main.py run

# 强制重算（忽略缓存）
python main.py run --force

# 分析已有结果
python main.py analyze --results-dir ./results

# 管理缓存
python main.py cache list
python main.py cache clear

# 运行测试
pytest tests/ -v
```

## 配置热切换

在 `config/default.yaml` 中修改：

```yaml
portfolio.optimizer: "equal_weight" | "mean_variance" | "risk_parity"
alpha.method: "equal_weight" | "ic_weighted" | "icir_weighted"
universe.n_stocks: 100 | 300 | 500
backtest.rebalance_frequency: "daily" | "weekly" | "monthly"
```

## 项目结构

```
quant_platform/
├── main.py              # CLI入口
├── config/              # YAML配置 + schema验证
├── data/                # 数据层 (Synthetic/Tushare + ETL)
├── factors/             # 因子引擎 (15因子 + 处理 + 评估)
├── alpha/               # Alpha模型 (因子合成 → 信号)
├── portfolio/           # 组合优化 (协方差 + 3种优化器)
├── backtest/            # 回测引擎 (向量化 + 成本模型)
├── risk/                # 风险管理 (VaR/CVaR/压力测试)
├── reporting/           # 报告 (图表 + 仪表盘)
├── agent/               # LLM模块 (情感因子)
├── utils/               # 工具 (缓存/Numba/配置)
└── tests/               # 105单元测试
```

## A股实盘陷阱

平台显式处理了10个A股特有的实盘陷阱：前复权、停牌、幸存者偏差、涨跌停、ST股票、T+1、交易成本、手数限制、除权除息、行业漂移。详见 `data/ASHARE_PITFALLS.md`。

## 扩展

- **实盘数据**：实现 `DataProvider`（已有 TushareProvider 参考）
- **新因子**：继承 `BaseFactor` 并注册到 `FactorRegistry`
- **新优化器**：实现 `optimize()` 方法并注册
