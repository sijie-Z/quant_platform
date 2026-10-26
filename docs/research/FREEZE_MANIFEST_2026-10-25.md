# 策略冻结清单（2026-10-25）

- 仓库提交：`a1c3dc2`（v1）；后续修正见 §5（v2）与 §6（v3，当前有效）
- 生成时间：2026-10-25 14:18（v1）；v2 2026-10-25 16:10；v3 2026-10-26

## 1. 冻结的两个 forward 策略

| 策略 | 因子 | 组合约束 | v1 目标文件 sha256[:16]（已 superseded） |
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

## 6. v3 —— 完整 ST 快照与首日执行防护（2026-10-26，当前有效）

**触发原因（仍是缺陷修复，非业绩驱动）**：v2 只排除了 v1 持仓中的 5 只 ST，没有排除候选池中其余 ST。逐一读回 300 只股票的 PIT `isST` 后，2026-10-23 实际有 **13 只** ST；遗漏的 8 只为 002055、002198、002717、002822、300087、301030、603008、603359。v2 未产生任何真实 NAV，因此不得作为 forward 起点。

**处置**：
- 用完整 13 只名单重建两个目标：换手 **33 只**、EP+换手 **32 只**，均无当日 ST；v2 JSON/CSV 与旧 ST 文件已保留在 `superseded/`。
- `daily_signal --exclude-codes-file` 现在拒绝空文件、空列表、null/浮点数/对象及非六位代码；`forward_update` 拒绝空/候选池覆盖不全的快照，并禁止把当日快照用于历史目标。
- `paper_track` 拒绝首日零成交时发布全现金台账；零金额买入不再生成幽灵持仓。
- 回测冲击成本增加无效 ADV 的固定滑点回退，并改用 60 日中位成交额（shift 1）；这是工程正确性修正，不改变因子、参数或组合约束。

**v3 当前有效哈希（sha256[:16]）**：

| 文件 | 哈希 |
|---|---|
| 换手目标 `target_2026-10-23.json` | `738fca0071edbc99` |
| EP+换手目标 `target_2026-10-23.json` | `e466c21411405372` |
| 完整 ST 名单 `st_codes_2026-10-23.txt` | `b42b7ab7a67a28f6` |
| `tools/daily_signal.py` | `d353668be760cba2` |
| `tools/paper_track.py` | `672c8aa58ef9956f` |
| `tools/forward_update.py` | `f68b9cca15ec612d` |
| `backtest/engine.py` | `324b29c392994090` |
| `backtest/cost_model.py` | `69e1f38af626c36f` |

**状态**：截至 v3 生成，正式 forward 台账仍无 NAV；v3 是唯一有效起点。3/6 个月评估节点与终止标准不变。
