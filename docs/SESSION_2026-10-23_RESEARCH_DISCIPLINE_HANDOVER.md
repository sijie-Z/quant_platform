# 会话交接 · 研究纪律与实盘闭环（2026-10-23）

> 本文件记录 2026-10-22 ~ 10-23 这一轮“按短板报告执行”的全部工作、验证结果、残余风险与续做入口。用户因其他事务暂停，后续可从这里继续。

## 0. 一句话状态

在分支 `feature/system-hardening`（基线 `c19d28b`）上连续完成 **13 个执行轮（R1–R13，约 60 个提交）**，HEAD `4a8a639`，全部本地提交、**未 push**。研究纪律（purge/embargo、DSR/FDR、holdout、逐 fold 选因子）、数据 PIT（Baostock 公告日、Tushare 成分股、因子/信号截面、Baostock ST）、成本感知优化器、主链路真实成本、QMT 重连、每日对账与独立账本、异步成交回写均已落地并有测试守护。

## 1. 工作方式（监督—执行环）

- 监督方（主会话）：侦察 → 写任务书（`.scratch/orchestrate/taskbook_R*.md`）→ 守卫检查 → 发车执行方 subagent → **独立读回 + 亲跑测试**验收；不采信自报。
- 执行方：按任务书施工、每小节 commit、起全新上下文 subagent 做对抗评审，迭代到无未决分歧；报告落 `.scratch/orchestrate/cc_report_R*.md`，评审落 `review_R*.md`。
- 台账：`.scratch/orchestrate/_CHAIN_STATE.md`（逐轮发车/验收记录，绝对时间）。
- 验收命令统一 `-p no:codspeed`；内存不足时 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（需要 `-p asyncio`）；量化包测试加 `PYTHONPATH=packages/quant_core/src`。

## 2. 完成的轮次（均经监督方亲验）

| 轮次 | 内容 | 关键提交 | 监督验收 |
|---|---|---|---|
| R1 | Walk-Forward 增加 purge_gap / embargo；默认逐位一致 | `a01114f..a91a8ca` | 21+46 passed |
| R2 | DSR/BH-FDR 接入 `/api/sweep`、CLI sweep、因子筛选（`validate_sweep_trials`） | `9ba6735..cfc09f0` | 50+32 passed |
| R3 | 正式 holdout：配置 + sweep 守卫（422/exit 2）+ 研究路径剔除（含前向标签边界行） | `dbe1183..166e70d` | 350 passed |
| R4 | 因子选择移入每个 fold 的训练窗（`factor_selector`），pooled WF 修正 | `89030b9..68916b3` | 28+259+13 passed，基线 sha 一致 |
| R5 | Baostock 缺 pubDate 不再回退期末（删前视），缓存 v5，asset_growth 同季锚定 | `9b9e3ba..6d9b7e4` | 468 passed |
| R6a | provider 逐日 membership 接入 pipeline（provenance） | `559f0d0..84dbbc5` | 51 passed |
| R6b | 修 provider ffill 吞退出；membership 拆为 `universe_mask`（买拦卖不拦）；缓存 schema 6 | `870e97b..4752918` | 101 passed |
| R6c | 因子/信号截面 PIT：`excluded_mask = ~tradable | ~universe`；输入侧保留历史热身 | `eaf9a21..e89b18d` | 339 passed |
| R7a | Baostock 日线 isST → ST episode 表；缓存 schema 7 | `6171b4f..2ccee7b` | 34 passed |
| R8 | 优化器成本感知：`transaction_cost_bps`、`turnover_penalty`、`max_participation` 容量上限 | `8a4921e..78e54df` | 126 passed，默认路径逐位一致 |
| R9 | 主链路接入 daily_volume + raw_prices，启用 impact 成本；印花税按日期（null=法定表） | `854bbdf..e86a4b3` | 102 passed；成本对照：年化拖累 1.18pp → 2.08pp |
| R10 | QMT 懒重连（退避、fail closed）+ `query_stock_orders/trades` 回查 + get_orders TTL | `5eaef48..229d014` | 155 passed |
| R11 | Scheduler EOD 每日账实对账 + `reconciliation_results` 表 + 事件 | `2942352..50f3f50` | 90 passed |
| R12 | 独立账本：启动基线 + 成交流水推导 expected（消除同源自比较）；无基线=unknown | `f80be55..5e78128` | 144 passed |
| R13 | 实盘异步成交回写 Store（`_sync_broker_fills`，累计量幂等，order.synced） | `a39c15f..4a8a639` | 101 passed |
| R14 | Tushare ST 历史（namechange → episode 表）；缓存 schema 8 | `3ed18d1..c00a056` | 72 passed |
| R15 | 优化器数值稳定性：`cp.psd_wrap` 修 cvxpy ARPACK 认证失败；PSD 加脊与告警 | `d50342e..9e11329` | 129 passed |

### 真实数据研究（2026-10-24）

本机网络不可用（Baostock 登录超时），改用本机既有真实日线缓存（101 只 × 2018–2025）跑完整研究闭环，详见 `docs/research/REAL_DATA_EVIDENCE_2026-10-24.md`。核心结果（净成本、12 折 OOS、逐 fold 选因子）：

- 等权因子组合 Sharpe −0.506（方向用反）；IC 加权 +0.211；
- **低换手单因子（5% 单票上限）：Sharpe 0.669、年化 13.8%、最大回撤 −17.7%**；同池等权基准 0.222 / 4.5% / −25.7%；
- `turnover_20d` 年度 IC 六年全为负（均值 −0.047，ICIR −0.225），低风险异象稳定；
- 成本敏感性：impact vs fixed 几乎一致；R15 修复后优化器回退 0 次；
- 诚实限制：DSR p≈0.33 不显著、股票池可能有幸存者偏差、无财务/ST、未做 paper——定位为“有希望的候选”，不是已证实 alpha。

### 关键数字与结论

- **成本真实性**：R9 前后对照（40 只合成股、2022–2024、月频、10M）：旧默认年化成本拖累 1.181%，新默认（冲击成本 + 按日期印花税）2.076%，年化收益 2.149% → 1.237%。方向正确：旧回测系统性低估成本。
- **PIT**：Baostock 缺公告日的行已丢弃；Tushare 成分按日 mask 生效（买入门禁 + 因子截面），卖出不受成分退出限制；Baostock ST 历史可用（真实数据 ST 过滤不再是空转）。
- **研究纪律**：sweep/因子筛选已带 DSR/BH-FDR；holdout 独立成配置并有守卫；WF 因子逐 fold 训练窗选择，全样本选择偏差消除。
- **实盘闭环**：QMT 断线可恢复、委托/成交可回查；每日 EOD 对账用“启动基线 + 成交流水”作为独立 expected；异步成交落库，账本不再漂移。

## 3. 分支与远端状态（重要）

- 本地分支 `feature/system-hardening`，HEAD `4a8a639`；**未 push**。
- 远端同名分支仍停在 `c19d28b`（本轮开始前的提交）。
- 会话内 `gh` 未登录（交互式提示），GitHub 写操作未执行；PR #84 未在本会话绑定/更新。
- 因此：本轮全部成果仅存在于本地工作树。恢复后第一件事若是交付，需 `git push` 并更新 PR（注意 GitHub 限流钩子，聚合操作）。

## 4. 已知残余与风险（评审记录在案）

1. **Tushare ST 历史**仍未实现（namechange 路径未做）；CSI300 真实数据回测的 ST 过滤仍不可用（Baostock 已可用）。
2. **TCA 回写/校准**未做：R9 接通了冲击模型，但系数仍是硬编码默认值，未用实盘成交校准。
3. **holdout 最终评估器**未实现（只有窗口纪律与守卫）。
4. **策略闭环**未做：尚未用真实数据跑通“一条策略研究 → paper → 小资金”。这是“赚钱”目标的下一关键步。
5. R13 评审残余：XTP 无 resync（重启后停机成交不可恢复）；QMT `order_time` 格式未在真机确认；idle sync 与 EOD 首跑之间存在晚到成交窄窗口。
6. R11/R12 残余：多租户 EOD 快照共用 default 租户（既有口径）；公司行为未纳入 ledger（分红/拆分会让现金/持仓对账出现已知差异，docstring 已注明）。
7. 环境：`pytest-codspeed` 会让回测类测试无声死亡（exit 127）；低内存时 pytest 插件自动加载也会被杀。见 memory `pytest-codspeed-crash-local`。

## 5. 续做路线（按“距赚钱远近”排序）

1. **推进策略闭环**：选一条策略（小市值反转/周频），用真实数据（Baostock/Tushare）走 研究 → WF → paper → 小资金；把 R9 成本与 R1/R2/R3/R4 纪律用在同一条策略上，产出第一份“净成本后”证据。
2. **TCA 校准**：用实盘/paper 成交回写冲击模型系数；R9 已留好参数入口。
3. **Tushare ST（namechange）**：补齐 CSI300 的 ST 历史；参考 R7a 的 episode 契约与缓存版本纪律。
4. **holdout 最终评估器**：一次性评估工具 + 结果落 lab registry。
5. **推送与 PR**：聚合一次 push，更新/新开 PR，跑 CI（六项）。
6. 之后：另类数据真实管道、tick 引擎接入主链路、投资者门户等（对赚钱优先级低）。

## 6. 断点续做入口

- 台账：`.scratch/orchestrate/_CHAIN_STATE.md`
- 任务书：`.scratch/orchestrate/taskbook_R*.md`
- 执行报告/评审：`.scratch/orchestrate/cc_report_R*.md`、`review_R*.md`
- 进度：`.scratch/orchestrate/progress_R*.md`
- 本报告：`docs/SESSION_2026-10-23_RESEARCH_DISCIPLINE_HANDOVER.md`
- 短板总报告（Word）：`docs/量化平台短板诊断与实盘路线图_2026-10-22.docx`

## 7. 验收命令速查

```bash
# 研究纪律
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=packages/quant_core/src \
  python -m pytest tests/test_backtest/test_walkforward.py tests/test_research tests/test_strategy -q -p no:codspeed

# 数据 PIT
PYTHONPATH=packages/quant_core/src python -m pytest tests/test_data -q -p no:codspeed

# 成本/优化器/交易
PYTHONPATH=packages/quant_core/src python -m pytest tests/test_backtest tests/test_portfolio tests/test_trading tests/test_core -q -p no:codspeed

# 全量（内存充足时）
PYTHONPATH=packages/quant_core/src python -m pytest tests -q -p no:codspeed
```
