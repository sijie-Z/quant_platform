# 策略冻结清单（2026-10-25）

- 仓库提交：`a1c3dc2`（v1）；v2 见 §5
- 生成时间：2026-10-25 14:18（v1）；v2 2026-10-25 16:10

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

## 5. v2 —— ST 过滤修复（2026-10-25）

**触发原因（缺陷修复，非业绩驱动）**：v1 目标由 `daily_signal.py` 生成，但该工具没有任何 ST 过滤；核对 Baostock 2026-10-23 的 `isST` 发现 v1 的 40 只持仓中有 5 只是 ST（002168、002650、300419、603189、603825），与生产口径“ST 不进买入池”不符。

**修复**：
- `tools/daily_signal.py` 新增 `--exclude-codes-file`（PIT 排除名单），并在 `provenance/diagnostics` 记录排除名单与实际剔除的代码；`_normalize_code` 支持 `sh600053/600053.SH` 形式。
- `forward_update.py` 在每次月末生成目标前抓取新浪全市场快照，生成当日 ST 名单并传入；抓取失败会显著记录并**拒绝**把“无 ST 过滤”的目标当作合规目标。
- 两个 2026-10-23 目标已用 **2026-10-23 的 PIT ST 名单**重建（排除上述 5 只）；v1 文件保留在各自 `superseded/target_2026-10-23_v1_no_st_filter.json`（子目录不在 `paper_track` 的 targets 扫描范围内，避免同一 as_of 重复读入），不覆盖、不删除。

**v2 哈希（sha256[:16]）**：

| 文件 | 哈希 |
|---|---|
| `tools/daily_signal.py`（v2） | `78d48e763aabe3f0`（修 `600053.SH` 后缀解析后；重生成目标权重逐字节一致） |
| 换手目标 `target_2026-10-23.json`（v2） | `afbe650fa8525417` |
| EP+换手目标 `target_2026-10-23.json`（v2） | `b69e05d89324bbc0` |
| PIT ST 名单 `st_codes_2026-10-23.txt` | `bee723388c0e953d` |
| 换手目标 v1（保留） | `b8067856456be21a` |
| EP+换手目标 v1（保留） | `2f85b5c029e6eaf8` |

**状态**：forward 台账尚未开始（无 NAV 记录），因此 v2 作为**起点**，不构成对已跟踪记录的事后修改。评估节点仍为满 3/6 个月；参数与规则不变。新 tag：`forward-freeze-2026-10-25`（v1 tag `forward-freeze-2026-10-23` 保留）。

**已知残余**：v2 的 ST 名单来自 2026-10-23 的 Baostock `isST`；今后每日任务用当日新浪快照（名称含 ST），两者口径在“当日是否 ST”上一致，但供应商切换会导致名单边界略有差异，属可接受运维误差。
