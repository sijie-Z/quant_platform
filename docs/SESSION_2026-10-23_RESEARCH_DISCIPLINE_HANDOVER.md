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

### 真实数据研究（2026-10-24，R19b 重跑后）

本机网络不可用（Baostock 登录超时），改用本机既有真实日线缓存（101 只 × 2018–2025）跑完整研究闭环，详见 `docs/research/REAL_DATA_EVIDENCE_2026-10-24.md`。经三份对抗评审，R19a 修复门禁前视/低波前向/大小盘全样本分组/基准/DSR 口径、R18 修复 gross 超配后，R19b 已用修正代码重跑（结果 JSON 在 `.scratch/orchestrate/r19b_*_2026-10-24.json`）：

- **净成本 OOS（2020-02-11..2025-12-31，1433 会话）**：EP+换手（行业中性）Sharpe 0.470、年化 10.40%、回撤 −19.72%；行业+成交额中性 0.365 / 8.78%；60 日低换手 0.682 / 13.92%；60 日低换手行业中性 0.363；低波组合 0.550。Sharpe 均为平台口径 `(CAGR−3%)/年化波动`。
- **同窗口基准**：日频等权 0.294 / 9.00% / −25.71%；等权 buy&hold（constant-share，退市价值结转）0.243 / 8.06% / −33.52%。旧报告的 0.222 / 4.50% 不可复现且标签错误（4.50% 是基准自身的超额年化）。注意：R19a 工具的 buy&hold 用 skipna 截面均值、会隐性剔除退市股（得到 0.310 / 9.52%），本轮在 `.scratch` 用 carry-forward 修正，工具本体待下一轮修复。
- **风格归因（市场+低换手+低波+EP+规模，Newey-West）**：R² 0.86–0.88；残余 alpha −1.0% ~ +2.6%/年，|t|≤1.14，**没有一个显著**。行业中性化后的“保守候选”alpha t=0.23–0.45。
- **DSR（超额收益，N=48）**：p=0.71–0.89，全部不显著；旧报告的低 p 来自 gross 输入与低估 trial 数。
- **PIT 大小盘拆分**（2018–19 成交额分组）：大盘半池 Sharpe 0.345、小盘半池 0.633——旧结论“大盘更强”方向反转，低换手溢价在当年小盘中更强。
- **容量（因果波动率估计器）**：1M Sharpe 0.594、5 亿 0.443，阈值（≥0.3）在 5 亿处仍未触及；估计器有重复扣费/非 WF 基线等已知局限（见报告 §7）。
- **结论（诚实版）**：策略本质是低换手/价值/低波的**风格暴露实现**，不是已证实的 alpha；理性预期年化 5–9%、Sharpe 0.13–0.4（“5–7%/0.2–0.4”只在 10–12% 波动假设下同时成立）；不应据此投入真金白银，下一步是 forward paper 跟踪与正交 alpha 研究。
- 目标组合（`docs/research/target_portfolio_2025-12-31.{json,csv}`）已用 `tools/daily_signal.py` 重新生成（权重与上一版一致），并标注“style tilt 演示，不是已验证 alpha”。

### R20–R30 更新（2026-10-25）

- **R20 行业上限与 rank 对齐**：真实行业映射传入优化器，30% cap 第一次真正生效（0 违规）；WF 增加 rank 归一化选项（默认关闭）。重跑后 Sharpe：ep_turn_sector 0.490、ep_turn_ss 0.355、turnover60_sector 0.429、lowvol_sector 0.577；DSR 0.78–0.89。
- **R22 波动率目标**：6 组预注册实验中仅 1 组过回撤门槛，信息比率不变 → 机械降杠杆，不推荐。
- **R23 正交因子预注册筛选**：彩票效应、异质波动、Amihud、日内收益、异常成交量 5 个因子 0/5 通过；与现有策略日收益相关 0.94–0.98，组合层面退化为同一多头篮子。
- **R25 组合构建对比**：等权 top-20/30、信号比例、风险平价全部不合格（换手 3.4–4 倍、回撤差 2.8–7.2pp）；保留 MVO。
- **R26 生产工具**：`tools/daily_signal.py` 补齐组合层 30% 行业上限，目标与 2020–2025 滚动台账重生成。
- **R27 宽池工具**：`tools/fetch_universe.py`（akshare/baostock/新浪/腾讯，断点续传）。
- **R28 交付前修复**：选择器迁入 CORE 层（架构边界），T+1 重启测试消除周末日历依赖；全量 3,862 passed。
- **R30 宽池验证**：新浪抓到 **297 只跨板块股票（60:106 / 00:99 / 30:92）**，2018–2025。低换手方向复现（Sharpe 0.807 vs 同池等权 0.493，+0.31），但优势主要来自降波动（IR vs 等权为负），且宽池 0 退市股（幸存者偏差，明确披露）；101 池的超额更强（IR +0.30、CAGR +4.1pp）但仅限深市 000xxx。
- **决策备忘**：`docs/research/DEPLOYMENT_DECISION_2026-10-25.md` —— 当前不投真金白银；门槛为 6 个月 forward paper、≥300 只含退市/ST 的 PIT 池、正交信号 alpha t≥2 且 DSR p<0.1、对账/Kill Switch 演练。

### R31–R33 与 forward paper（2026-10-25 凌晨）

- **R31 宽池价值**：300 只 2018–2026-10-23 新浪后复权面板 + 百度 PE/PB（半月频 ffill，无前视）。宽池 EP+换手 Sharpe 0.674/0.606，**低于换手单因子 0.805**——EP 在宽池无增量证据；生成了 as_of 2026-10-23 的实盘目标（32 只、单票 5%、行业最大 13.5%、首建换手 50%，带免责声明）。
- **R32 paper_track 原始价口径**：`--raw-price-mode auto` 自动识别后复权面板（新浪走 `amount/volume`，Baostock 走 close），旧路径逐字节一致。
- **R33 2026 真 holdout**（2026 数据从未参与选择，生产链路、零调参）：换手单因子 **−4.89% vs 基准 −11.63%（超额 +6.73pp）**，Sharpe −0.57 vs −0.83，回撤 −17.5% vs −22.2%；EP+换手 −8.49%。防御性风格 tilt 在真样本外延续，但绝对收益为负。
- **趋势过滤 overlay（预注册）**：负结果（Sharpe 0.683→0.527；2026 −0.57→−1.14），不采纳。
- **Forward paper 已就绪**：双策略目标（`target_portfolio_2026-10-23.json` EP+换手、`target_portfolio_2026-10-23_turnover.json` 换手单因子），执行日 2026-10-26；已建工作日 16:20 自动任务 `quant-forward-paper-daily`（运行 `.scratch/orchestrate/forward_update.py --track`）。**注意**：该任务未保存本会话权限模式，首次运行可能停在权限确认；请在桌面端 “Scheduled” 页对任务点一次 “Run now” 预授权。

### 关键数字与结论

- **成本真实性**：R9 前后对照（40 只合成股、2022–2024、月频、10M）：旧默认年化成本拖累 1.181%，新默认（冲击成本 + 按日期印花税）2.076%，年化收益 2.149% → 1.237%。方向正确：旧回测系统性低估成本。
- **PIT**：Baostock 缺公告日的行已丢弃；Tushare 成分按日 mask 生效（买入门禁 + 因子截面），卖出不受成分退出限制；Baostock ST 历史可用（真实数据 ST 过滤不再是空转）。
- **研究纪律**：sweep/因子筛选已带 DSR/BH-FDR；holdout 独立成配置并有守卫；WF 因子逐 fold 训练窗选择，全样本选择偏差消除。
- **实盘闭环**：QMT 断线可恢复、委托/成交可回查；每日 EOD 对账用“启动基线 + 成交流水”作为独立 expected；异步成交落库，账本不再漂移。

## 3. 分支与远端状态（重要）

- 本地分支 `feature/system-hardening`，HEAD 已推送；远端同名分支 head = `def17e5`（2026-10-24 聚合推送，`c19d28b..def17e5`）。
- PR **#84**（base `develop`）已更新描述，包含 R1–R16 的摘要与真实数据研究结论；CI 由推送触发。
- 说明：本会话 `gh` CLI 未登录，PR 更新通过本机 git 凭据调 GitHub REST API 完成；后续如需 CI 结果请查看 PR 页面。
- 另有 R16 工具化提交（`tools/real_data_research.py`）在同一分支上。

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
