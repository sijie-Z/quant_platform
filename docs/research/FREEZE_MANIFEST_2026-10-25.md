# 策略冻结清单（2026-10-25）

- 仓库提交：`a1c3dc2`
- 生成时间：2026-10-25 14:18

## 1. 冻结的两个 forward 策略

| 策略 | 因子 | 组合约束 | 目标文件 sha256[:16] |
|---|---|---|---|
| EP+换手 | ep,turnover20d（行业中性化） | MVO，单票5%，行业30% | `2f85b5c029e6eaf8` |
| 换手单因子 | turnover20d（行业中性化） | MVO，单票5%，行业30% | `b8067856456be21a` |

## 2. 代码与配置版本（sha256[:16]）

| 文件 | 哈希 |
|---|---|
| `tools/daily_signal.py` | `b602dc538a22e25b` |
| `tools/paper_track.py` | `1bad2b0c6bd97be8` |
| `config/default.yaml` | `4422d2cdca99a719` |
| `backtest/engine.py` | `77f107ad82eeffef` |
| `backtest/walkforward.py` | `ff513b393448f260` |
| `research/validation.py` | `ff726fdf020ab8a6` |
| `portfolio/optimizers.py` | `56a41be56075cb1b` |

## 3. 数据版本

- 合并面板：`.scratch/orchestrate/merged_wide.parquet`，563782 行 × 300 只，2018-01-02..2026-10-23
- 价格源：新浪 `stock_zh_a_daily`（后复权，2018-01-02 起）；估值源：百度 PE(TTM)/PB（半月频，按日 ffill）
- 抓取与合并脚本：`.scratch/orchestrate/forward_update.py`、`fetch_sina_2026.py`、`fetch_baidu_valuation.py`

## 4. 冻结与变更规则（预注册）

1. 自本清单生成起，上述两个策略的参数、因子定义、组合约束、成本模型不得修改；任何修改必须建立新版本（新目标目录、新台账、新清单），不得覆盖旧记录。
2. 执行规则固定：目标 as_of 为交易所月末最后交易日；执行日为下一交易日收盘；成本口径见 `docs/OPS_RUNBOOK.md` §13。
3. 每个交易日收盘后由 `quant-forward-paper-daily` 任务更新台账；不得因短期表现调整策略。
4. 预注册评估节点：满 3 个月、6 个月各做一次一次性评估，报告内容固定为：净收益、标准年化 Sharpe、相对同池等权/市值加权/buy&hold 的超额、最大回撤、成交与成本偏差；评估前不得改参。
5. 终止标准：若 6 个月评估时标准 Sharpe < 0.2 或相对全部基准无正超额，停止该策略的 forward 跟踪，转入新研究版本。
