# 交接说明（2026-09-23，更新至 2026-10-21 第十一轮）

> 这份文档写给接手这个仓库的人。它记录**当前状态、改过什么、还剩什么、以及那些代码里看不出来的坑**。
> 未完成的工作全部开在 [Issues](https://github.com/sijie-Z/quant_platform/issues) 上，编号在下面对应。
>
> **如果你只读一段，读第一节。**

---

## 一、一句话现状

这个仓库在 2026-09 经历了一次覆盖全仓库的诚实性清算（20+ 个 PR），修掉了让研究结论失去意义的缺陷。**但清算同时暴露了一件事：这个仓库真正的病不是 bug，是"宣称"与"实际"之间的落差，而且它同时存在于代码、治理、记录、定位四个层面。**

修完代码层的缺陷之后，另外三层还在。它们全部开在 Issues 上，**第十三节是完整索引**。

---

## 二、已经修掉的（诚实性清算）

2026-09 的两周里，下面这些不是"小 bug"，而是**让研究结论失去意义**的问题：

| 问题 | 事实 | 处置 |
|---|---|---|
| **alpha 信号是空的** | `ma_convergence` 是 21 个默认因子里唯一返回 `Series` 的，它的字面量列名 `'factor'` 通过 pandas 的标签并集把**整个截面**变成 NaN。信号只有 0.69% 有值，60 个调仓日里 59 个走等权回退 —— **每一次回测实际都是等权买入持有** | 已修（#54） |
| **`embedded_alpha: false` 从来没生效过** | `load_config()` 不把嵌套 dict 转成声明的 dataclass，`getattr` 全部拿回退值，`embedded_alpha or (alpha_strength > 0)` 恒为真。三种配置产出的收益矩阵 **SHA1 完全相同** | 已修（#65） |
| **四个嵌套配置段全是坏的** | 修上面那个时按 schema 枚举 18 个嵌套字段，发现 **4 个**没被读取：`data.synthetic`（dict 而非 dataclass）、**`screener`**、**`execution`（含 QMT）**、**`instruments`**。后三个 YAML **从没被读过**，而类型检查看不出来（`default_factory` 让类型正确、YAML 被丢掉）——**跨资产 `instruments` 那个卖点配置从来没生效过** | 已修（#65） |
| **成交时点与四处文档矛盾** | 引擎按信号当日收盘成交，而 docstring、`ASHARE_PITFALLS.md`、`CLAUDE.md` 全写的是次日成交。那个判据来自初始提交，从没兑现过 | 已修（#53/#56） |
| **股票池用未来信息** | `is_st` 的语义是"**整个样本内会不会**变 ST"，触发日随机落在样本内 —— 第一天就按几年后的事件剔除股票 | 已修（#55） |
| **契约第 7、8 条是假的** | 文档写着"已实现的位置"，实际那两个 PIT 访问器**没有任何生产调用方** | 已修文档（#48），第 8 条代码已修（#55） |

另有十来个**端点伪造数据**的问题（伪造的 IC 衰减曲线、随机相关矩阵、硬编码的风险分解、样本内的 ML 信号、前端 Vue 面板自己造数据），以及一批资金账务错误（同周期买入按 0 计价造成 ¥8,330 假亏损、经纪商现金可为负、部分成交塌成 1 股、多租户持仓互相覆盖）。

**后果**：这些修复之前的所有历史数字**全部作废**。这不是口径微调 —— 那时候描述的是一个从没选过股的组合。

### 已经证明的事情

**管线不会凭空造 alpha。** 2026-09-23 做了一次 null control：同一套代码、同一批 seed，只翻 alpha 开关跑 9 次：

| | 超额收益 | Sharpe | IR |
|---|---|---|---|
| null（纯噪声） | **−0.99%/年**（9 个里 8 个为负） | +0.21 | −0.26 |
| 对照（内嵌 alpha） | **+25.3%/年**（9/9） | +1.49 | +2.53 |

阳性对照是关键 —— 同一套链路确实能找到植进去的 alpha。所以"噪声下什么都没出来"是可证伪的结论，不是空话。

---

## 三、这个仓库的病：四个层面上的同一个问题

代码层的缺陷修完了，但**同一件事**还在另外三个层面上。把它们放在一起看，形状是这样的：

| 层面 | 现象 | Issue |
|---|---|---|
| **代码** | 三代实现共存：两份 `Order`、三条实盘路径、两套 registry、7 个零导入的死目录 | #69 #74 #77 |
| **治理** | 宣称的强制手段不存在：代码注释引用的 `CONSTITUTION.md` 从未创建、"enforced by import-linter" 而 import-linter 不存在 | #67 #68 |
| **记录** | 诚实性机制建好但从未启用：Registry 有 `validity`/`affected_by` 列和 `mark_validity()`，42 行**全部**标 `valid`，方法一次没被调用过 | #71 #72 |
| **定位** | **这个仓库为谁服务，没有答案** —— 2026-07-31 被提出，至今未回答 | #75 |

前三行都是"宣称了但没兑现"。**如果不需要向任何人交付，"宣称"就没有代价。** 这就是为什么这三行会同时出现。

**目标漂移不是结果，是原因。** 每一代都会重新发明一套架构，上一代立的规矩没人引用、上一代写的协议层没人实现 —— 因为没有"做完"的定义，就没有理由回头对齐上一代。

---

## 四、架构层的核查结论（2026-09-23 新增）

仓库里有 5 份"架构决策"性质的文档。逐条量过之后：

**`docs/ARCHITECTURE_V2.md` 开头写着「后续所有 PR、重构、拆分都以这一页为判断标准」。它有 4 条 Principle + 2 条 ADR。今天有 1 条在代码里成立。**

| 声称 | 判定 |
|---|---|
| P1 Truth First（偏差必须机器可读记录） | **假** —— 机制建好了没用（#71） |
| P2 Point-in-time | **假** —— 见 #55 #48 |
| P3 Knowledge compounds（全部进 Registry） | **半** —— 6 月那批报告根本没有 Registry 行 |
| P4 Protocol over Implementation（接口先锁） | **假** —— `framework/contracts/` 6 个 Protocol，**符合的实现 0 个**（#70 #77） |
| ADR-0003 Single Live Runner | **假** —— 实盘引擎 3 条、风控引擎 3 个 |
| ADR-0004 Protocol Before Plugin | **假** —— 两份 `Order`，同一个概念两个字段名（#74） |
| §5 三层目录 `core/`-`live/`-`lab/` | **假** —— `live/` 不存在；且仓库的 `core/` 是 EventBus/Store，**不是**文档说的"研究内核"，同名不同物 |
| §3「跨层依赖由 import-linter 强制」 | **机制假、实质真** —— import-linter 不存在，但 `tests/test_architecture/test_core_boundaries.py` 确实在 CI 里强制了依赖方向，**反向对照实验验证过它真能抓**（临时造违规目录树，四个方向全抓住） |
| §6-P3 `main.py factor <name>` | **✅ 真的** |
| §6-P2「本文件落地 + 修掉悬空引用」 | **假** |

另外 4 份文档的逐条核查在 #78。其中一条值得单独记住：

> **`PRODUCTION_ARCHITECTURE_V1.md` 写的风控数字在代码里不存在**——文档说最大回撤限额 **−30%**，实际是 **15%/25%**；文档给的滑点公式全仓 grep 零命中。**照那份文档理解系统风险，会得到偏松一倍的印象。**

### 失配是两种，修法完全不同

**(A) 先立契约、再也没人回头对齐实现** —— 代表 `framework/contracts/`。修法：要么让实现 conform 并加测试卡住，要么删掉契约。

**(B) 代码从未漂移到位，但文档用现在时写成了已完成** —— 代表 `KERNEL_ARCHITECTURE.md`、`PRODUCTION_ARCHITECTURE_V1.md`。这里根本没有契约，只有一个提案被写成了状态报告。**修法：没有任何代码需要改**，要改的是文档的语气。

**混在一起处理会同时做错两件事。**

### 死代码层（AST 实测 206 个运行时文件）

| 目录 | 运行时导入点 |
|---|---|
| `framework/` | **6，全部是它自己引用自己** |
| `services/` | **3，全部自引用** |
| `kernel/` | **1，自引用** |
| `shared/` `regime_router/` | **0** |
| `compliance/` `research/` | 0（只被测试导入） |

对照 `factors` 73 / `data` 67 / `backtest` 41。**被宣传为核心模块的 `kernel/`、`services/`，导入点全是自己。** 完整清单见 #77。

---

## 五、那个未回答的问题（最重要的一条）

2026-07-31 的一场架构讨论里，被反复追问的只有一句：

> **"MiQi 到底是什么？"** → "不是目录。不是模块。不是架构。而是：**一句话。**"
> "这句话**如果今天定不了，以后目录永远改不完**。"

具体问法是三选一：**"如果只能保留一个，这个项目是给谁用的？A. 给面试官看的　B. 给你自己用的　C. 给未来用户看的"**

**这个问题至今没有被回答。**

而且那一轮讨论里有一条很硬的判断，与本仓库的 `CLAUDE.md` 直接冲突：

> "注意：**没有'面试官'这个 use case**。面试官不是用户 —— 他们是评估你项目的人，不是用你项目的人。把'给面试官看'作为 use case 是个陷阱，因为它会驱动你去做**表演型代码**（花哨但不实用）。"

**而 `CLAUDE.md` 是围绕面试组织的**（"面试亮点"整章、"企业级就绪度评估"整章、"面试视角"专节）。

这两者不必然冲突，但**不能同时假装对方不存在**。完整讨论见 #75。

---

## 六、还没做的事（按优先级）

**全部开在 Issues 上。这是完整索引。**

### P0 —— 回答"alpha 到底存不存在"

1. **#63 第一个真实研究** —— 这个仓库从来没回答过的问题。协议是外审定的：真实 A 股数据 → PIT 股票池 → **单一预注册因子**（明确不要"全仓库因子一起测"）→ IC → 严格 OOS → 成本后回测。**注意不要下载当前成分股再回溯十年**，那会重新引入幸存者偏差。

### P1 —— 影响研究结论可信度

2. **#58 合成数据会伪造反转信号** —— 随机置 NaN 再前向填充，导致相邻日带着被夸大的多日收益。null control 里测到 `IC(过去1日→未来1日) = −0.0116 (t=−7.63)`，排除停牌相邻样本对后降到 `+0.0007 (t=+0.47)`。它到不了组合层面，但**任何在合成数据上做的因子研究都被污染**。
3. **#59 MVO 约 10% 的调仓回退等权，且回退无视 `max_turnover`** —— 18 次运行里回退 289 次，实际换手 0.44 vs 配置 0.30。而且它是**静默的**：summary 仍然写 `optimizer: mean_variance`。
4. **#71 Registry 的 validity 机制从未启用** —— 42 行全 `valid`。这是"所有不可信必须机器可识别"这条原则的落点。
5. **#72 研究产物分档** —— 哪些报告作废、哪些干净、哪些待定，已经查清并归档。

### P2 —— 结构性问题，需要决定而不是顺手修

6. **#74 两份 `Order`** —— `trading/broker.py:59` 用 `code`/`price`/`filled_price`；`execution/models.py:50` 用 `ticker`/`limit_price`/`fills`。三个枚举各定义两遍。默认值语义相反（限价单 vs 市价单）。**收敛到哪一份是一次真正的迁移，需要先定接口。**
7. **#70 / #77 `framework/contracts/` 与死代码层** —— 6 个 Protocol 符合的实现 0 个；7 个目录零运行时导入。每条都要做一次知情决定：**接入 / 删除 / 标注为提案**。
8. **#69 双导入身份** —— 同一个文件会作为两个模块对象加载，状态分裂。实测 `data.providers.synthetic.SyntheticDataProvider is not quant_platform...SyntheticDataProvider` → `same class: False`；两个 registry 分别有 0 和 22 个条目。
9. **#67 / #68 治理债务** —— 代码引用了不存在的 `CONSTITUTION.md` / `ARCHITECTURE.md` / `ROADMAP.md` / `SUCCESS_METRICS.md` / `docs/ADR/`。选一条路：写一份精简的 `docs/governance.md` 落实原则，或删掉代码里的引用。
10. **#73 / #78 文档状态标注** —— 那 5 份架构文档需要加状态说明（哪些是当前事实、哪些是愿望）。**一份用现在时描述的愿望文档，比没有文档更危险。**
11. **#75 那个未回答的问题** —— 见第五节。

### P3 —— 边角

12. **#61 纸面交易的锁价快照有同样的自成交问题**
13. **#62 `test_ic_monitor.py` 11 个测试跑 4 分钟，且在内存压力下崩溃**（CI 跑 Linux 正常）
14. **#60 历史产物作废标注** —— `results/backtest_report.html` 是一份 **Sharpe 1.45 但 `Rebalances 0`** 的报告
15. **#3 / #4 实盘引擎的状态持久化** —— `daemon/runner.py` 每次运行重建 `LiveEngine`，交易与风控状态从不持久化

---

## 七、环境上的坑（历史开发机备注）

- 按 README 创建 `.venv`（Python 3.11/3.12；CI 与 Docker 使用 3.12）。`.venv312` / `.venv` 是旧开发机的本地目录，不在仓库里，不要当作仓库指令。
- **pytest 如需自定义 `--basetemp`**：这台开发机曾出现 `pytest-of-admin` 临时目录 ACL 损坏；在正常机器上不需要这个 workaround。
- **ruff 版本以 CI 的 pin 为准**（0.15.12）。
- **CI 跑 `ruff check .`（整仓库）**，只查自己改的文件会漏掉 import 块规则 —— 已经因此挂过一次 CI。
- **代理**：这台历史开发机上曾是 `HTTPS_PROXY=http://127.0.0.1:7899`；仅作环境备注。
- **这台机器在内存压力下会崩解释器**（WMI 服务 + pandas access violation，以及 `bash: fork: Resource temporarily unavailable`），同时跑多个 agent 的测试套时特别明显。
- **所有 agent 提交的 git 作者都是 `Builder`** —— 从 `git log` 分不出哪次是哪个 agent 做的。

---

## 八、判断问题的方式（这次清算留下的最重要遗产）

这次最有价值的发现不是那几十个 bug，而是：**它们全都是靠人读代码发现的，没有一个是测试发现的。**

主端点 `POST /api/run` 从来就没跑通过，却没有任何测试碰过它。`pure_volatility` 因子在整个平台里静默消失，也没人发现。

所以补了三道**系统性**守护，而不是修完就算：

- `tests/test_factors/test_factor_causality.py` —— 遍历**每一个注册因子**做"截断未来 / 替换未来，看过去是否改变"的检验，新增因子自动被覆盖。它在修复前的代码上会**指名道姓**报出 `pure_volatility`。
- `tests/test_factors/test_factor_output_shape.py` —— 每个因子必须返回列与输入一致的矩阵。缺了它，一个畸形因子能让整个信号变空。
- `tests/test_data/test_pipeline_pit_st_mask.py::TestNoLookAhead` —— 改动 D 之后的 ST 事件，**不能**改变 D 之前的股票池。把"无未来函数"从一句宣称变成机器可验证的性质。

### 接手后请保持这个标准

**任何一个"宣称"如果不能用测试表达，它就还不是真的。**

2026-09-23 的核查给这条加了一个具体做法：**反向对照**。不要只问"这条约束在不在"，而是**临时构造一个违规样本，看它抓不抓得住**。

`tests/test_architecture/test_core_boundaries.py` 通过了这个检验 —— 所以它说的边界是真的。`import-linter` 没能通过，因为它不存在。**以后任何"X 被强制"的声称，都应该附一个反向对照。**

---

## 九、仓库同步状态（2026-09-23 交接时）

- `develop` / `main` 与 GitHub **完全一致**（各 0/0）
- 工作区**干净**，无未跟踪文件
- **本地不存在任何 GitHub 上没有的提交**；tag 4/4
- 无未合并的分支，**无开着的 PR**
- `docs/SESSION_HANDOFF_20260731.md` 此前一直是未跟踪状态（只存在于一台机器上），已于本次交接提交（PR #76）

---

## 十、这个仓库里，什么东西是可信的

给接手的人一条实用判据：

| 可信 | 需要核实 | 不可信 |
|---|---|---|
| 有测试守护的性质（第八节那三套） | 有 Registry 行但 `validity` 未标注的结论 | 任何 2026-09 修复**之前**产生的回测性能数字 |
| 真实数据 + 纯 IC 测量（无交易） | 合成数据上的因子研究（#58） | `docs/ARCHITECTURE_V2.md` 里的方针陈述（#73） |
| `MARKET_STRUCTURE_DISCOVERY_V3_REPORT.md`（80d IC −0.081, t=−4.58） | 2026-07 之前的所有 `docs/research/` 报告 | `PRODUCTION_ARCHITECTURE_V1.md` 的风控数字（#78） |
| 该次提交的 git 历史 | | `CLAUDE.md` 的计数（2026-10-20 已按 2,662 用例 / 99 HTTP + 1 WS / 21 命令校正；仍以实测为准） |

完整的研究产物分档在 #72。

---

## 十一、2026-10-19 修复记录

这一轮处理的是交接时仍列在 Issues 上、且能用测试表达的问题。每项都有回归测试守护：

| Issue | 修复 | 位置 |
|---|---|---|
| #58 | 合成数据停牌前向填充不再伪造反转信号：provider 标记 `is_suspended`，收益与因子价格在填充边界置 NaN，涨跌停标记跳过复牌跳空 | `data/providers/synthetic.py`、`data/pipeline.py`、`main.py` |
| #59 | MVO 回退等权不再静默：记录回退次数/原因并写入 summary 与 API；等权不再把权重放大到 `max_weight` 之上；换手按索引并集计算并真正钳制（首次建仓豁免）；RiskParity 同样受换手约束 | `portfolio/optimizers.py`、`backtest/engine.py`、`api/schemas.py` |
| #71 | Registry validity 机制启用：`list_runs(validity=...)`、`is_trusted()`、JSON 导入/导出、`runs import/export` CLI；43 条历史记录可导入并标记 | `lab/registry/__init__.py`、`main.py` |
| #3/#4 | 实盘状态持久化：Store 新增 `engine_state` 表；`LiveEngine` 快照/恢复 + 已处理日期水位；daemon 每次运行恢复并落盘，日盈亏不再硬编码 0，损坏快照拒绝覆盖 | `packages/quant_core/src/quant_core/store.py`、`trading/live_engine.py`、`daemon/runner.py` |
| #60 | 历史产物作废标注：`ABLATION_001` 顶部加 SUPERSEDED 说明，`results/backtest_report.html` 加可见 VOID 横幅，新增 `results/README.md` 说明整目录不可引用 | `docs/research/`、`results/` |
| #61 | 纸面交易快照播种不再自成交：L2 快照档位以 passive 挂单进入订单簿，锁价/交叉快照保留原始深度；后续真实订单仍可成交 | `execution/paper_broker.py` |
| #67/#68/#73/#78 | 文档状态标注：`framework/__init__.py` 去掉不存在的 CONSTITUTION/ARCHITECTURE/import-linter 引用与 `registry`/`trust` 导出；4 份架构文档加 STATUS 横幅，修正 import-linter 表述、风险阈值与 README 中的错误路径 | `framework/__init__.py`、`docs/*.md`、`README.md` |
| #62 | IC 监控滚动计算从 O(n·window) 改为“每日 IC 只算一次 + 滚动平均”；测试耗时 233s → 7s，Spearman 调用 14,931 → 237 | `factors/ic_monitor.py` |
| #69 | 双导入身份：`pyproject.toml` 改为把仓库根目录安装为 `quant_platform`（不再把 46 个子目录装成顶层包）；移除 flat 别名 sys.path 与导入；CI/Docker 去掉 PYTHONPATH 垫片；新增 `tests/test_architecture/test_import_identity.py` 守卫 | `pyproject.toml`、`.github/workflows/ci.yml`、`Dockerfile`、`tests/conftest.py` 及多处导入 |
| #74 | 订单模型收敛：broker 层 `Order` 更名为 `BrokerOrder`（保留 `Order` 别名），三组枚举统一到 `execution.models`，新增与 OMS 模型的双向转换 `to_execution_order()` / `from_execution_order()`；broker 特有字段保留在 DTO 层 | `trading/broker.py` |
| #70/#77 | 未接线模块标注：7 个零运行时导入的目录在 `__init__.py` 加 `STATUS:` 横幅，新增 `docs/DEAD_CODE_STATUS.md` 与守卫测试；删除/接入留作显式决策 | `framework/contracts/`、`services/`、`kernel/`、`shared/`、`regime_router/`、`compliance/`、`research/` |

另外修掉了三处 pandas 3.x 兼容问题（`resample("M")`、`rolling(axis=1)`、零价格 `pct_change`），它们在本机全局 pandas 3.0.6 下会让 17 个测试失败；在仓库锁定的 2.1.3 下同样成立。

验证：全量测试 **1678 passed / 3 skipped / 0 failed**（3m58s；#62 修复前同一套件 6m53s），ruff 全绿。

### 2026-10-19 第二轮加固（审计驱动）

在上述 issue 之外，对回测、实盘、风控、API、性能做了一轮多 agent 审计并修复：

| 领域 | 修复 |
|---|---|
| 回测正确性 | 权重随价格漂移（此前隐含每日再平衡）；基准按日截面等权（消除全样本筛选的未来数据）；Sortino 改用下行偏差；最大回撤从初始资金起算；协方差保留有停牌的资产（≥50% 覆盖）；RiskParity 改为凸对数障碍模型；容量模型冲击成本乘以股数、改为次日收盘执行并让权重漂移；并行 sweep 修复签名不匹配；执行后钳制权重回写 `weights_history` |
| 实盘/OMS | 周期锁阻止并发双下单；T+1 日切换释放（引擎/runner）；SimulatedBroker 未成交余量撤单、市价单参考价、行情更新重播做市盘口；OMS 成交前校验现金/可卖；报价 NaN/陈旧防护；QMT 注册 `on_stock_order`/`on_stock_trade` 并把成交正确回填，ACK 不再降级 SUBMITTED；连接失败阻断引擎启动 |
| 数据 | 复权修正：合成数据把公司行为应用到原始价格；Baostock 改用后复权（adjustflag=1）；Tushare 合并并应用 `adj_factor`；基本面接口修正 f116/f117 的市值映射（此前把市值当成净利率/营收增速） |
| 风控 | 卖出降风险不再被仓位/日亏/回撤/熔断阻断；扇区检查计入本次订单；引擎用当日盈亏而非累计盈亏；安全模块盈利不再触发日亏熔断；Greeks 新增持仓即计算、对冲数量修正；压力情景改为累计冲击 + 多路径中位数/最差 |
| API | 无 token 时交易/OMS 端点仅限回环；WebSocket 校验 token；修复 `data/quality`、`ml/predict`、run result 的 500；订单类型字段兼容；重端点移出事件循环；Monte Carlo 参数边界；WS 进度广播 |
| 性能 | 中性化 5.67s→0.15s/因子；breakout 因子 8.6s→0.09s；Barra fit 9.2s→0.24s；SQLite 写入 12ms→0.13ms；订单簿成交后清理 |
| #3 | LiveEngine 到期仓位按收盘价卖出并计成本，不再直接从权益中消失 |

验证：全量测试 **1732 passed / 3 skipped / 0 failed**（1m57s），ruff 全绿。

### 2026-10-19 第三轮审计（alpha / 报告 / 运营 / 前端）

| 领域 | 修复 |
|---|---|
| Alpha/因子 | 组合层改为逐资产 NaN 感知加权（一个因子缺一格不再清空该资产；默认因子集不再产出空信号）；水平类因子用前向填充价格计算（停牌不再让 momentum_12m 覆盖率归零）；PE/PB 非正分母剔除（亏损股/负净资产不再排成"最便宜"）；numba 动量窗口与 pandas 对齐（此前滞后一天）；numba Rank IC 平均并列名次（离散因子 IC 不再依赖列序）；正交化按共同资产对齐（NaN 模式不同也能正交）；RSI 无亏损窗口返回 100；NetworkFactor 正确初始化基类；纯波动 beta 统一 ddof |
| 运营/报告 | NAV 管理费累计计提并逐日扣减（不再永久高估净值）；业绩报酬只在创新高时收一次、HWM 用费前净值；跨周末/节假日日收益不再归零；投资者年化按实际经过时间；归因换手用并集（完整换仓不再报 0）；HTML 月度热图按 `{years,data}` 结构渲染；matplotlib 月份标签/空月/无限 Sharpe 修复 |
| 前端/契约 | 订单簿与持仓正确解包响应 envelope；实盘风控面板字段对齐（kill_switch/recent_breaches/limits）；市价成交使用真实报价（后端拒绝无价成交，删除 `Math.random()` 价格）；数据质量不再用无关合成数据冒充某次运行；前端 bundle 过期时启动告警；监控端点显式返回 `available/reason`；行情缓存改为共享实例；OMS 快照与 PaperBroker 记录有上限 |
| 数据 | Baostock 后复权 + Tushare adj_factor 应用 + 合成数据公司行为落到原始价格；基本面 f116/f117 市值映射修正 |

验证：全量测试 **1753 passed / 3 skipped / 0 failed**（3m08s），ruff 全绿。

### 2026-10-19 第四轮审计（门禁 / 配置 / 多策略 / 前端）

| 领域 | 修复 |
|---|---|
| 策略门禁 | 默认 `minimum_icir=0.0` 让 `abs(icir) >= 0` 对所有因子恒真，门禁永远不可能失败却仍报告"通过 ICIR 门禁"；默认改为 0.2，阈值 0 显式报告为"已禁用"。覆盖率门禁的调用方把 `n_assets_total` 和 `n_assets_with_data` 传成同一个数（自比较恒 PASS）；改为从价格面板实测覆盖率，宇宙未知时 SKIPPED 而不是伪造通过。删除从未被任何检查读取的配置字段；新增交易成本门禁，使用回测 summary 新增的 `annual_turnover` / `annual_cost_drag`（引擎按执行日累计成本） |
| 配置接线 | `portfolio.risk_aversion`、`portfolio.covariance.ewma_half_life`、`risk.var.confidence` / `horizon` 此前只存在于 schema/YAML，没有任何调用方读取；现在分别接线到约束、协方差估计和报告 |
| 多策略 | orchestrator 此前用**共享持仓簿**与每个策略的目标比较：两个策略各目标 100 股时，一笔成交让双方都以为已持有，下一次调仓双方各卖 100 股来回震荡。新增按 `order.strategy` 记账的逐策略持仓台账，调仓只看自己的持仓；逐策略 P&L 由自有份额的已实现+盯市计算，不再把整本账的盈亏记到每个策略头上 |
| API/前端 | `/api/demo` 的回撤与月度收益改为小数，与 `/api/run` 及前端 `×100` 的约定一致；Monte Carlo 返回下采样后的 `terminal_values`（分布图此前永远空白）；MonitorDashboard 消费后端的 `available/reason` 并显示不可用原因，同时复用 ECharts 实例——此前每 15 秒轮询都会重新 `echarts.init` 同一节点，一个打开的页面每轮泄漏 8 个图表实例 |
| 合规导出 | 带时区偏移的成交/委托时间与本地朴素日期窗口比较会抛 `TypeError`，被旧的 `except` 当作"无法解析"静默丢弃；现在先转换为本地时间再比较 |

验证：全量测试 **1767 passed / 3 skipped / 0 failed**（3m21s），ruff 全绿。

### 2026-10-19 第五轮审计（数据源 / 研究统计 / 运营 / 实盘风控）

这一轮由四个并行审计 agent 覆盖此前未深挖的模块，修复均带回归测试。

| 领域 | 修复 |
|---|---|
| 组合优化 | MVO 只尝试 `[ECOS, SCS]`，而 cvxpy 1.7 的轮子里没有 ECOS；同时换手上限在信号轮换超过预算时本身不可行。默认配置 **60 次调仓中 54 次静默回退等权**。改为只尝试已安装的求解器（CLARABEL 优先），并把既有持仓保留在候选集里，使换手约束可行——同一回测现在 **0 次回退**，`effective_optimizer=mean_variance` |
| 基本面 | 实时接口失败时回退合成数据却没有任何标记，`stats` 仍报告 `eastmoney`；`hash(code)` 每个进程加盐，同一只股票的 PE/PB 每次重启都变。现在 `source`/`is_synthetic` 随响应返回、种子改用 `crc32`、EastMoney 改读正确的 f162/f167（含 /100 缩放）；筛选器把 0 当作"未知"而不是真实值（`pe_max=20` 不再通过所有股票） |
| 行情/盘口 | 行情缓存过期后仍以"现在的 `datetime.now()`"作为响应时间；WebSocket 解析把 f17（开盘）当最新价、f16（最低）当开盘，成交量未统一为股；Level 2 全部为模拟数据却无任何标记，空 tick 时 VWAP 返回 `0.0`、流向返回 `50`。现在快照与报价携带真实抓取时间与 `stale`、WebSocket 字段映射与单位修正、所有 L2/WS 响应带 `simulated/source`，空数据返回 404 |
| PostgreSQL | `get_prices` 只 pivot 第一个字段且多字段时列名重复（违反 `(date, asset)` MultiIndex 契约）；连接失败被构造函数吞掉，`main.py` 记录"Using PostgreSQL provider"后第一次调用崩溃。现在按契约返回 MultiIndex，构造失败抛 `ConnectionError` 触发真正的合成数据回退 |
| 连接池 | 去重缓存键忽略 `kwargs` 与数据源，60 秒内把 close 请求的结果返回给 volume 请求；provider 返回 `None` 同时记成功与失败；缓存对象被调用方原地修改。键现在包含 kwargs，`None` 记为失败，缓存进出都做拷贝 |
| NAV/运营 | 业绩报酬只在当天扣减、此前已计提的费用第二天又加回净值（每次结晶后出现一次假反弹）；管理费按工作日计数却除以 252（一年实际收 2.08%）；同一日期重算会再次计提。现在净值始终扣除累计费用负债、按日历天/365 计提、同日重算从该日之前的持久状态重新开始 |
| 投资者/报告 | 年化收益用"全生命周期收益 ÷ 3 年窗口"，回撤只看窗口内；净值图回撤从首个收益日而不是初始 1.0 起算；月度热图把无数据月份画成 0.0%，HTML 热图按数组下标当月（3 月数据画在 1 月）。均已修正 |
| 合规导出 | `get_trades/get_orders` 先取最新 10 万行再按日期过滤，历史区间可能导出空文件；P&L 固定只看 5 年。范围过滤已下推到 SQL |
| 研究统计 | Deflated Sharpe 把年化 SR 与单位方差次序统计量、逐期标准误混用（真实年化 SR=2.0、1260 天、N=10 被算成 p=0.59）；ROE 研究把财报"报告期结束日"当作"公告日"，最多提前 4 个月使用（现已按法定披露截止日映射并更新 trust metadata）；ML 的 IC 用整面板 pooled correlation（只有市场择时的预测也能拿 0.7）、ICIR 是 2-5 个 fold 均值的离散度；LightGBM 模型只写了 JSON 元数据、加载后 `predict` 重新引入样本内信号；walk-forward 的测试窗互相重叠且 embargo 在 expanding 模式下无效。均已修正 |
| Registry/门禁 | `is_trusted` 只看 validity，`begin_run` 默认 `status=failed, validity=valid`，失败/未完成的 run 也算可信；验证报告的 `overall` 只看步骤是否崩溃，门禁 FAIL/REJECTED 与跳过的 walk-forward 仍报 `pass` 并退出 0。现在可信要求 `status=success`，overall 由门禁与必需步骤共同决定 |
| IC 监控 | ICIR 与"IC 为正比例"在 63 日移动平均序列上计算（真实 0.19 被报成 1.34、56% 报成 92%）；现在离散度统计使用原始逐日 IC，平滑序列只用于展示与趋势 |
| 实盘/API | `POST /api/risk/kill-switch` 不在无 token 时的回环白名单内（docker-compose 默认无 token 且端口公开），远程可解除熔断；`GET /api/risk/status` 会用 OMS 账本覆盖实盘风控状态（下一次引擎周期看到 90% 回撤并触发熔断）；回撤/日亏只在有订单时检查；扇区检查对新股票一律 "Unknown" 且同批订单不累计；熔断后引擎直接跳过周期、从不执行其声称的"flatten"；盘前持仓同步检查从不比较任何东西；6 个交易端点用 `async def` 执行阻塞 RPC。均已修正：`/api/risk` 纳入敏感前缀、GET 不再改状态、组合更新即评估权益熔断、扇区映射与在途订单投影、熔断时卖出可用份额、持仓同步真实比较并在收盘后清理幽灵持仓、阻塞端点改同步、风险概览返回 `available/reason` 而不是 200 + GREEN |
| 其他 | 分位数收益的 Q1 方向与文档相反（`qcut` 的 0 是最低组）；`factor_stability_history` 从未写入导致因子排名永远为空，且保存的 `pearson_ic/coverage` 恒为 0；SHAP 永远返回 "no data"；`max_total_positions` / `max_gross_exposure` / `max_net_exposure` / `max_orders_per_second` 声明但从未检查；orchestrator 接收了 `risk_monitor` 却从不喂组合状态。均已修复，先卖后买的执行顺序保证同一批调仓的卖出释放敞口 |

验证：全量测试 **1813 passed / 4 skipped / 0 failed**（3m10s），ruff 全绿。

### 2026-10-20 第六轮审计（核心架构 / 回测内核 / 执行撮合 / 前端契约）

四个只读审计 agent + 五个并行执行 agent（每个执行包都跑了内部对抗评审）覆盖上一轮未深挖的模块。

| 领域 | 修复 |
|---|---|
| 核心架构 | 调度器用宿主机本地时间判断 A 股时段（官方容器是 UTC：真实盘中判为 closed、17:30 判为 trading_am）；改为 Asia/Shanghai，并新增 2024–2026 交易所休市表（含法定节假日，可用 `QUANT_EXCHANGE_HOLIDAYS` 追加），容器设置 `TZ`。EventBus 消费者按 handler 名字注册，同名 handler / 退订后重订会丢事件并最终把队列塞满；改为按订阅计数身份注册、退订时取消任务。两套事件总线的通配符语义不一致（`market.*` 在旧总线匹配 `market_data.tick`，`order.**` 在新总线从不匹配）；统一为一个段感知匹配器。审计日志不传租户，订单/成交/信号全落到 default，且平仓会删除所有租户的同名持仓；现在显式携带租户，signals/events 增加 tenant_id 列并迁移。`save_order` 的 INSERT OR REPLACE 会把未提供的列清空；改为按键 UPSERT，同一订单多次成交获得唯一 trade_id 并在同一事务写入。EOD 对账无幂等标记，重启后同日重复写快照；新增 `eod_reconciliation` 表。`Instrument.from_dict` 给期货/期权套用了股票默认（印花税 0.1%、t_plus=1）；按资产类型取默认 |
| 回测内核 | 固定版本的 numba 不支持 `returns.mean(axis=0)`，Ledoit-Wolf 内核从未编译成功：异常被引擎吞掉后 MVO 用单位阵、RiskParity 每个调仓回退等权，而 summary 仍写配置的风险模型。重写为正确的常相关 Ledoit-Wolf 公式（与 sklearn 误差 <1e-3），并让协方差失败记录 `n_covariance_failures` 而不是静默。容量模型只把成本扣在内部 sizing 资金上、报告仍旧是毛收益，且参与率超限只记一个冲击数、从不缩减交易，所有 AUM 档位收益完全相同；现在真实缩减交易、成本/冲击从报告收益扣除，指标改用与主回测相同的 Sharpe/MDD 定义。冲击成本路径把已含佣金/印花税的 total_cost 再加一遍、且按股的冲击与货币金额直接相加；改为只返回冲击并统一折算为收益率。印花税对买入也收 0.05%、清仓只收一半；引擎改为按资产拆分买卖名义额。并行 sweep 的 Sharpe/MDD 与主回测口径不同，且文档示例的 `optimizer`/`n_stocks` 平铺键被 setattr 成死属性（每格配置相同）；改为复用 metrics 并校验/映射参数键。Black-Litterman 用年化风险厌恶配日协方差，先验日均约 5.7%；统一为单位后年化超额约 6% |
| 数据/因子 | 长停牌会把资产从**整个样本**删除（2024 年停牌导致 2021–2023 也没有它；晚上市 40 天也被删）；改为只删从未有价格的资产，长缺口保留 NaN。合成数据的晚上市股票首日 `close` 由 NaN 前值乘出，之后永远 NaN，30 日 ffill 还制造上市前价格；修正上市日种子并遮蔽上市前。`BaseFactor.run` 的全面板 ffill 把 ST/上市遮蔽重新填上，ST 股票重新进入每个截面；现在只对停牌缺口 ffill 并按 PIT ST 掩码回填 NaN，成交量/换手也在 ST 日置空。合成财报按原始公告日 reindex 到交易日，周末/节假日发布的报告被静默丢弃（20 份只剩 14–16）；映射到下一个交易日。中性化用静态行业 map，PIT 行业变更（143 条记录）从未生效；新增 `get_sector_panel()` 并在因子处理中按日期使用（默认数据 522×120 面板约 1.1s）。市值因子被回归到自身，处理后只剩 1.3% 方差；跳过自身市值中性化。PCA/对称正交化用全样本估计（未来函数）且 PCA 对特征向量求平均；改为截至 t 的扩张窗口与确定性符号。数据质量检查把 `(date, asset)` MultiIndex 当宽表，跨股票算 pct_change（4 万条假异常）、新鲜度永远 0 天；入口统一 unstack 并只用价格列做异常检测 |
| 实盘执行 | QMT 状态映射用了另一套 API 的 0/1/2/…/50，真实 xtquant 的 48–57 全部落到 PENDING（56 已成交也会被陈旧回调降级）；改用真实枚举并保护终态。QMT 忽略 `order_stock` 的 -1 与 `cancel_order_stock` 的非 0 返回，失败被报告成功。OMS 成交接受负数量（买 -100 股反而增加现金、生成负持仓）；在任何账务变更前拒绝。成交后不更新 `current_price/market_value`，刚买入的组合快照出现等于全部买入额的假亏损；立即 mark-to-market，未定价持仓按成本估值。PaperBroker 在订单簿已撮合后改写 `filled_quantity`，导致后续成交超量 28%；改为入簿前缩减。ExecutionEngine 接受超量成交、并向已撤销订单追加成交记录；append 前校验状态与剩余量。模拟交易所用 NaN/0 行情成交、可成交限价按限价而不是更优价成交；已修正。TCA 的 dataframe 入口用全样本均值与最后一行收盘价作基准（未来函数），文档列布局下基准为 0；要求执行日期并只取当日数据 |
| Alpha/工具 | 情感因子缓存键不含 asset/analyzer/model，不同股票共享分数、API 失败的关键词回退被永久缓存、非原子写会清空全部缓存；改为完整键、失败不落盘、原子替换。组合层缺失因子在分子分母同时剔除，缺主因子的资产反而排名更高；改为缺失填 0、分母固定。Prometheus 指标从未被调用（Grafana 全空），导出无锁；新增 HTTP 中间件并在 app 接入。缓存键不含 provider/回退/代码版本，`--use-baostock` 会命中合成缓存；新增 schema/键版本与 provider/override 入参，回退数据不写请求 provider 的键。`config.data.provider` 被默认 `use_tushare=True` 覆盖（有 token 时合成/postgres 配置失效）；改为配置决定、CLI 显式覆盖。因子 `params` 与 `factors.processing` 在解析时被丢弃（改 YAML 无任何效果），空 section 还会重新启用默认因子集；已接线并让固定周期因子接受覆盖 |
| 打包/前端 | 非 editable 的 `pip install .` 产出的 wheel 只有 4 个顶层 .py，所有子包缺失、配置 YAML 也不在包里；显式列出 33 个子包并声明 package-data，新增清单守卫测试。WebSocket 在无 token 时对远程开放（携带订单/风控事件）；纳入回环限制。数据质量面板对每次已完成 run 都 422（run 记录不存价格面板）；改为 run 时计算并随结果保存。OMS blotter 只显示已完成订单，挂单/部分成交永远不出现；改为返回全部订单与剩余量。LivePortfolio 把 200+`{"error"}` 当成功、WS 重连无法停止、因子排名用绝对 ICIR 重复计数、参数热图显示 “NaN”、风控标签键名不匹配；均已修正。`QUANT_API_TOKEN` 模式补充前端构建期 token 支持，docker-compose 的 `OPENAI_API_KEY` 改为代码真正读取的 `DEEPSEEK_API_KEY` |

验证：全量测试 **1983 passed / 4 skipped / 0 failed**（2m55s），ruff 全绿。

已知仍未处理：多租户隔离（#75 产品定位未定）；`framework/contracts/`、`kernel/` 等零导入目录的去留（#70/#77）仍是显式决策项。

### 2026-10-20 第七轮（研究脚本与文档真实性）

| 领域 | 修复 |
|---|---|
| 研究脚本 | `run_factor.py` / `tools/common_runner.py` 的 12 个月动量用 `.shift(-21)`，把未来 21 天收益嵌进因子（与 IC 的目标窗口完全重叠）；改为 `.shift(21)` 并加静态守卫测试。`run_factor.py` 回测在调仓日把当日已实现收益记到新权重上（一日前视）；改为次日生效，基准与策略同持有期对齐。`tools/run_xv_synthetic_vs_real.py` 把收益 DataFrame 当 benchmark Series 传给 `all_metrics`，每行 regime Sharpe 被静默记为 0；改传 benchmark Series 并记录异常 |
| 真实数据股票池 | Baostock 的“500 只蓝筹”实际是排序截断后的 101 只深市主板（000001–000500）。改为优先 `query_stock_basic` 跨板分层抽样，失败时回退 60/68/00/30 交错候选池；缓存键加入完整代码列表哈希。历史 real-A-share 结论的样本范围需按新股票池重跑 |
| 统计口径 | `run_market_structure_v3.py` 的重叠前向窗口用 `sqrt(n)` 计算 t 值；改为 Newey-West HAC（lag=ceil(H/21)）并保留 naive 列，规模分组从全样本均价改为当日价格横截面三分组（明确标注 price proxy）。`run_rq1_cluster_attribution.py` 原来对每日横截面均值求相关（测的是覆盖率噪声）；改为逐日资产交集上的横截面相关。`run_rq3_stability.py` 的 regime 标签被 ffill 到下月、`%Time` 用日历天（占比合计 70.2%）；改为当月标签与交易日分母 |
| 文档 | README 快速开始补上 `quant_core` 与项目 editable 安装（否则 import 直接失败）。CLAUDE.md/AGENTS.md 校正：EventBus 只覆盖运行时核心、实盘为 7 因子、默认启用 21/注册 27、管理费 /365、LLM 是 DeepSeek、CI 是 3.11/3.12 + ruff/mypy、Store 11 张表、端点/视图/组件计数、涨跌停与整手在回测层的真实限制，并据此下调相关 A 评级。默认 alpha 方法改为 equal_weight，合成数据 embedded_alpha 标注为演示用途 |

验证：全量测试 **2019 passed / 4 skipped / 0 failed**（2m55s），ruff 全绿。

历史 `results/*.csv` 中的 RQ5/RQ6/RQ7 与 run_factor 报告需要用修正后的脚本在真实数据上重跑后才能引用；docstring/注释已注明口径。

### 2026-10-20 第八轮（并发一致性 / 测试有效性 / CLI 与端点契约）

这一轮以“实测”为主：并发一致性压测、变异检验（mutation testing）、CLI 全命令冒烟、当时 98 HTTP + 1 WS 端点全扫。所有修复带回归测试。

| 领域 | 修复 |
|---|---|
| 并发 | `OrderManager.fill_order` 无锁：两笔 60 股成交同时通过 100 股订单的校验，最终成交 120 股且现金为负；加 `RLock` 并把校验/记账放进同一临界区。`RiskMonitor` 在持续违约时每次检查都追加同一条 breach（800 条重复且无上限）；只记录新签名，上限 1000/保留 500。`PipelineCache` 直接写目标文件，Windows 上并发读者会读到半写 pickle 或 `PermissionError`；改为临时文件 + fsync + `os.replace`，读取失败清理不抛异常 |
| 测试有效性 | 变异审计（8 个变异，7 killed）发现默认 Ledoit-Wolf 协方差被换成单位阵仍全绿；新增非单位阵/量纲/PSD/numba+sklearn 双路径断言。补强 NAV 费用负债精确等式、OMS 每次成交现金校验与部分成交、MVO/RP 换手上限、成本与印花税精确值、中性化行业残差、门禁状态。7 个空转测试改为真实读回/契约断言（PostgreSQL fallback 写后读回、WS 无客户端、run consumer 输出），删除 skip 占位 |
| IC 评估 | 硬编码“每个截面至少 30 个有效资产”使 30 股票池 21/21 因子 IC 全 NaN（n_obs 0–3），门禁误报“弱 ICIR”；改为 `min(30, max(10, panel_width // 2))` 并接入 numba 与 pandas 两条路径，宽度不足时输出明确 warning；全部因子无观测时门禁 SKIPPED 并说明观测不足。30 股票池现在 19/21 因子可评估（N=133–250），门禁 FAIL 是真实的弱 ICIR |
| ML | 260 天样本下 `ml train` / `ml signal` 曾以退出码 0 输出零样本与空信号；现在短历史自适应首个训练折（≥30 日期、≥60 样本、≥2 特征），样本不足抛 `InsufficientTrainingData`，CLI 退出 1、API 返回 422；全 NaN 特征显式丢弃，训练失败不落模型与 sidecar |
| CLI | `check-lookahead` 已注册但 dispatch 缺失（只打印 help 并退出 1）；接入并新增“所有注册命令均有 dispatch”的静态守卫。`run` 从不生成 HTML 与 metrics.json，`research report` 因无生产者而死路；现在 `run` 写出自包含 HTML 与 metrics.json，`research report` 可回退读取 daily_returns.csv。`walkforward --folds` 只打印不生效；校验器新增 `n_folds` 并截断。`screen` 把 z-score 当 PE 阈值；CLI 与 API 均改用原始因子面板，未知因子返回 422。前视检测器在 Windows GBK 控制台因 ⚠/✓ 字符崩溃，输出改为 ASCII；排名对 1e-14 求和噪声敏感导致同一日期信号随面板长度变化（检测器把自己的因果信号判为有偏），排名前统一 `round(12)` |
| API | 端点全扫（当时 98 HTTP + 1 WS）确认 12 条缺陷。P0：已完成 run 因 `exposure.top_holdings.pnl_pct` 与不可评估因子统计为 NaN，`GET /result`、`/factors` 返回 500；pnl 只在两个价格都有限时计算、跳过不可评估因子，并在入库前用 `json.dumps(allow_nan=False)` 守卫。Barra 横截面回归未剔除停牌 NaN 收益，1236/1236 个因子收益全 NaN、协方差 100/100 NaN，两个端点每次 500；过滤非有限行并对协方差 `nan_to_num`。再分配资金不重置 `current_value`，1 亿 +1% 被算成 -99%；重置基线并拒绝未知策略。`trading/stop` 后 `run-once` / `order` 仍执行，检查 `_running`。OMS 与 smart-route 接受空 ticker，改 422。parallel sweep 非法参数 500→422；regime 数据不足 200+error→409；`portfolio/live` 回退最近 7 天并返回 `data_date/stale`；`data/quality` 标注 `source/synthetic/data_date_range`；QMT 不可用与缺 akshare 返回 503 |
| 日志/文档 | `get_logger(__name__)` 双重前缀（`quant_platform.quant_platform.*`）修正；HANDOVER 补充本轮记录 |

验证：全量测试 **2074 passed / 3 skipped / 0 failed**（3m20s），ruff 全绿。CLI 实测：30 股票 `run` 得到 19/21 因子有限 IC（N=133–250）、HTML+metrics 落盘；`check-lookahead` 报告 0 偏差并退出 0；`ml train` 训练 3434 样本、`ml signal` 非空；`walkforward --folds 1` 实际执行 1 折。

### 2026-10-20 第九轮（实盘生命周期 / PIT 数据 / 策略-OMS 集成 / 前端契约 / API 安全）

第二轮实测审计：5 个只读 agent 分别覆盖实盘交易、PIT 数据、策略与 OMS 集成、前端契约、API 安全；随后 3 个执行 agent 修复后端缺陷，主线程修复 API/前端契约。全部修复带回归测试。

| 领域 | 修复 |
|---|---|
| 实盘交易 | SimulatedBroker 的 check-then-act 无锁：并发卖出双倍入账、并发首次买入互相覆盖，现金/持仓不再原子；加 `RLock` 覆盖下单/账户/盯市/T+1/恢复。部分成交此前不落库、不审计、不计数；现在写 order(partial)+trade+审计并按实际成交量计数。券商成交后审计写失败会永久丢单；改为先原子写 order+trade（确定性 trade_id 防重），审计失败只记日志。重启时恢复/对账 store 持仓与现金（paper broker 重建，QMT/XTP 不覆盖），kill switch/峰值/日基线/交易日/现金/持仓写入 `engine_state` 并恢复；持久化 kill 在显式解除前阻断下单。空持仓查询视为“未知”而非“空仓”，不再重复买入或清空 store。T+1 与日 P&L 基线改用 Asia/Shanghai 交易所日历（不再被宿主午夜/周末解锁），`_trading_day` 持久化。LiveRunner 持久化 signal/order/trade，单日券商异常不再中止全程，`finally` 关闭 session；双轨 PaperBroker 用主券商成交价播种流动性。逐 symbol 行情时间戳，过期报价不用于下单/盯市。`stop()` 等待在途周期、`start()` 拒绝复活旧线程；默认风控后端改为共享 `RiskMonitor` |
| PIT 数据 | 统一 `tradable_mask`（ST + 上市/退市 + 停牌），同时作用于价格因子、财务因子、volume/turnover、`fin_unstacked`、信号与 `get_financials_as_of`；ST/停牌/退市/上市前不再获得因子值、信号或权重。退市按 `delisting_date` 与价格终止双路径识别。停牌改为从缺失/零成交量/整行 ffill 推导，不依赖可选 `is_suspended` 列；复牌首日收益保留。合成数据日期、基准与财报公告映射改用交易所日历（节假日不再成为交易日，节假日公告映射到下一交易日），`DATA_SCHEMA_VERSION` 4→5。停牌日除权因子冻结，估值标记不再跳变。短样本合成数据不再因 `rng.integers(low>=high)` 崩溃 |
| 策略/OMS | `PortfolioOrchestrator` 接入真实现金台账、T+1 可用量、成本基础与已实现盈亏；`rebalance()` 对未成交订单做净额投影，重复调用不再重复下单，反向下单会撤单并释放资金锁；`(1+signal)` 目标被压缩到策略资金预算内，买入前校验合并现金，成交时引擎硬拒绝透支。全局先卖后买（创建与成交两阶段）。风控收到 `cash+盯市市值` 的真实权益并在日期推进时重建日基线。策略移除会撤单、按 T+1 可卖量清仓或转移、释放资金并清理台账；未知策略 id 拒绝。资金分配改为绝对比例（省略项归零、和≤1、拒绝负数/未知 id），再分配时重置 P&L 与风险统计，零资金期不记录收益；聚合指标排除 inactive 并改用市值加权组合收益序列。lot_size 真正生效，零数量订单跳过；手续费/印花税进入引擎现金与成本 |
| 前端契约 | `exposure_report` 返回 `sector_exposure`（扇区面板此前永远为空）。因子散点/归因改用资产级前向收益面板（此前传入组合收益 Series，异常被吞，两个面板永远为空），并去掉第二次 `shift(-1)`。WalkForward 在 `available:false` 时显示原因而非 0 folds；MonitorDashboard 统一风险等级大小写并从后端播种 kill switch/实时限额；WS 完成不再与轮询重复应用、取消后不再拉取结果，失败 run 会广播；DataQuality/MonteCarlo/RiskDecomposition 正确识别错误/不可用；Compare/Sweep 用并集列，首行失败不再隐藏后续成功结果；LivePortfolio 显示 `data_date/stale`，OrderBlotter 拒绝用非当日收盘价成交；Regime 相关性标记不可用、推荐语映射修正；RiskMonitor 圆点/日亏条、超额峰度阈值、压力情景名、命令面板、Vite WS 代理、QMT 服务地址字段、Settings 明确为浏览器本地偏好 |
| API 安全/资源 | 无 token 时**所有** `/api`（除 health）仅回环可访问，有 token 时全部鉴权；`/docs`、`/openapi.json` 纳入保护；CORS 预检不再被鉴权中间件拦截；默认监听改为 127.0.0.1（容器显式 0.0.0.0 + 必须设置 token）。`/api/screen` 的 config 路径限制在服务端 `config/` 内（路径穿越/配置注入）。OMS `limit_price`、策略 allocation/daily_return、`risk/check-order` 拒绝 NaN/Inf，校验错误响应统一清洗非有限值。运行窗口 ≤10 年且 start<end，sweep/parallel grid/worker/基本面代码数量设上限，Store 的 limit/days 统一钳制。失败 run 不再返回 traceback/绝对路径。前端 API token 改为 localStorage 运行时读取，不再打包进公开 bundle |

验证：全量测试 **2195 passed / 3 skipped / 0 failed**（3m06s），ruff 全绿，前端生产构建通过。CLI 30 股票实测：新交易日历下 run 正常（Total Return 14.29%、Sharpe 0.56），HTML+metrics.json 落盘。Round 9 时已知未处理：多租户隔离（#75 产品定位未定，当前 API 不提供租户隔离）；7 个分析端点仍显式 `available:false`；WS token 仍走查询串（uvicorn 访问日志可见）；回测尚未按执行日可交易性阻断；交易所日历表覆盖至 2026 年。**Round 10 已接入执行日交易性门禁；其余见第十轮。**

### 2026-10-20 第十轮（全仓审计后的系统加固 + 文档准确性）

第十轮以「只读审计 → 并行修复 → 内部对抗评审」的方式覆盖整个仓库，基线提交
`40a5bcf`，全量测试 **2659 passed / 3 skipped / 0 failed**（基线 `40a5bcf` 为 2567 passed / 3 skipped；本波次 +92 条用例），ruff 全绿。
这一轮不是单点修 bug，而是把「宣称」与「实现」逐项对齐。

| 波次 | 主要内容 |
|---|---|
| API 性能/并发/生命周期 | 移除 async 路径里的阻塞 sleep；Store/pandas 计算移入线程池；pipeline 与 analytics 分离的有界池（饱和返回 429 + Retry-After）；`/trading/start` 并发锁；WebSocket 每客户端有界队列、慢客户端丢弃、断开清理；Store 幂等索引；lifespan 有序关闭并可在下次启动重建；run registry 加锁与状态感知清理；响应边界（limit/矩阵单元）；懒加载单例锁；500/503 错误脱敏；持仓批量落库；合规导出分页（>1000 行）；WS 子协议 token |
| 运维 wave 1 | `trade` 真实成功/失败语义、QMT fail closed、真实行情校验、交易所日历驱动；docker-compose 持久化卷 + `stop_grace_period` + `QUANT_API_TOKEN` 必填；Grafana 面板只绑定真实指标；真实数据源不可用默认 fail closed 并把 requested/effective provider 写入 `metrics.json`；缓存/日志/配置版本有上限与清理；配置加载校验；配置版本原子写 + 密钥脱敏；每 run 独立输出目录；daemon watermark；监控脚本非零退出；Prometheus 格式 |
| 运维 wave 2 | 调度器只在收盘后跑 EOD、不再使用硬编码 100 万权益、启动补齐遗漏交易日；`AlertSink`（日志常开 + `ALERT_WEBHOOK_URL`，5 分钟去重）；`GET /api/readiness` dead-man switch；`main.py data retention`（CSV 归档 → 删除 → checkpoint/VACUUM，`QUANT_RETENTION_*_DAYS`）；优雅停机；风控状态持久化；新写入的 P&L/会话/事件时间戳统一为 Asia/Shanghai naive（旧行不迁移） |
| 数据 | Tushare 批次/复权失败 fail closed、不写部分缓存；Baostock 财报面板、失败不缓存、退市名单、显式 `st_available=False`；Tushare 历史成分 + 退市名单；Postgres 契约；单位统一（volume 股、turnover 分数、amount CNY）；Adata 契约；连接池键与空帧语义；pipeline 合约校验、重复行、`adj_factor`、空财务、PIT 掩码、板块限价、退市推断；合成数据停牌/上市/公告日 |
| 回测 | 执行日可交易性门禁（`tradable_mask` / `limit_up` / `limit_down`，按执行日而非信号日）；ISO 周年分组；Walk-Forward 接受完整引擎配置、OOS 指标只取测试窗；成本模型（最低佣金、过户费、按资产名义额、日期化印花税）；市值加权基准；per-instrument 费率；成交量/波动率进入冲击模型；tick 引擎成交记账/T+1；回撤日期映射；年化口径；首月现金 warm-up；`weights_history` 只记录已执行调仓 |
| 组合/风险数学 | RiskParity 改为凸对数障碍；Barra Ledoit-Wolf 强度；MVO 单位/风险项/目标波动；因子风险 Euler 分解；Monte Carlo 路径统计；bootstrap/Student-t 修正；归因 t 统计；投资者 Sharpe 统一口径；VaR 随机种子；long-only 校验；缺失收益协方差；Barra Newey-West；多重检验修正 |
| 测试有效性 | 变异审计 **11/11 KILLED**；新增精确公式、边界、账本、门禁测试；z-score numba 与 pandas 的 ddof 对齐 |
| 文档（本轮） | README / AGENTS / CLAUDE / HANDOVER / VALIDATION_REPORT / PRODUCTION_PLAYBOOK / OPS_RUNBOOK / ASHARE_PITFALLS / BEGINNER_GUIDE / NO_LOOKAHEAD_CONTRACT 按当前代码校正：QMT fail closed、可运行 CLI、Docker/PostgreSQL 事实、7 个 `available:false` 端点、PIT 真实状态、后复权 hfq（原文档误写前复权 qfq）、测试/端点/命令/组件计数、备份恢复与升级迁移、`metrics.json` provenance；VALIDATION_REPORT 标注 SUPERSEDED |

**当前计数（2026-10-20 交接实测）**：`python main.py --help` 列出 **21 个命令**
（新增 `data retention`）；`create_app` 暴露 **99 个 HTTP 端点 + 1 个 WebSocket**；
pytest 收集 **2,662 条用例**（2,659 passed / 3 skipped；基线提交 40a5bcf 为 2567 passed / 3 skipped）；测试文件 193 个；
Vue 组件 37 个。计数会随代码变化，以 `main.py --help`、`pytest --collect-only -q`
和 `create_app().routes` 为准。

**已知仍未处理（诚实清单，交接时）**：

1. **多租户隔离**（#75）：产品定位未定，API 不提供租户隔离；`core/context.py` 的
   TenantContext 不是端到端隔离。
2. **分析/监控端点已接线**：`/api/walkforward`、`/api/risk/decompose`、
   `/api/analysis/ic-decay`、`/api/analysis/correlation`、`/api/monitor/tca-summary`、
   `/api/monitor/factor-status`、`/api/monitor/capacity-gauge` 现在从有界 TTL 缓存的
   研究上下文现算（真实 validator / Barra / IC 衰减 / 相关矩阵 / TCA / 容量），
   并发饱和返回 429；确实无数据时仍返回 `available:false` 及精确原因。
3. **WebSocket token**：仅接受 `bearer.<token>` 子协议（时序安全比较）；`?token=` 查询串已移除。
4. **回测整手限制**：执行层（TWAP/VWAP/Iceberg、paper broker、OMS）强制整手，
   向量化日频回测仍是权重层面（已明确标注，未隐藏）。
5. **交易所日历**：内置休市表覆盖至 2026 年；2027 年休市日需通过
   `QUANT_EXCHANGE_HOLIDAYS` 追加。
6. **价格基准**：已统一。provider 交付 raw OHLCV + 复权因子（Tushare `adj_factor=1/native_cum`、
   Baostock raw+hfq 双取），`get_raw_close()` 供成交/估值、`get_adjusted_close()` 供研究，
   货币路径在只有复权价时 fail closed；符号统一为 `CODE.EXCHANGE`（含北交所 30% 限价）。
   残余：Adata 不提供价格面板（仅跨源校验，价格路径 fail closed）；公司行动已在 paper 引擎实现并接线（见本轮“公司行动”）。
7. **增量数据**：已实现（D32）。`run --incremental` / `cache refresh --incremental` 只追加新交易日，边界重叠一天并 `keep="last"` 去重；每次更新全量刷新复权因子（除权日不产生假跳变）；数据写成功后才推进完成标记；部分响应不推进并保留旧缓存；`data.max_staleness_days` 对研究告警、对决定性运行 fail closed。边界：合成数据源 `supports_incremental=False`（回退全量）；财务/基准在追加时仍全量刷新；停牌与截断的区分偏保守。
8. **ST 历史**：Tushare/Baostock 不提供 ST 公告历史（`st_available=False`，
   不伪造全 False 列）；合成数据提供，用于 PIT 验证。
9. **产品定位**（#75）与 `framework/contracts/`、`kernel/`、`services/` 等零运行时
   导入目录的去留仍是显式决策项（#70/#77）。

> 判据不变：任何"宣称"如果不能用测试或可运行命令表达，它就还不是真的。
> 本轮文档校正的原则是**删掉无法验证的数字、标注不可用能力、把 roadmap 写成 roadmap**。

---

### 2026-10-21 第十一轮（缓存 provenance + 价格基准/符号统一）

基线提交 `8f8caec`。这一轮补上第十轮留下的两块数据层缺口：

| 波次 | 主要内容 |
|---|---|
| 缓存 provenance（D12/D13/D33-D36） | pipeline 缓存键在 provider 解析之后生成，包含物化配置（含 schema 默认值）、有效 provider 链（Validated + secondary）、实际采样符号集哈希、复权基、日期范围与 `DATA_SCHEMA_VERSION`；Tushare/Baostock provider 缓存带 schema/复权版本清单，不匹配即重取；`--force` 重算并原子覆盖，新增 `cache refresh`；仅真正损坏才删除缓存（瞬时 IO 错误保留）；完整 provenance 清单并入 `metrics.json`，缓存命中不会把 synthetic 伪装成真实 provider；跨进程锁 + 唯一临时文件 + 陈旧临时文件清理，保存失败保留旧条目 |
| 价格基准/符号（D15/D41/D18/D19/D30） | 面板携带 `price_basis` / `adjusted_price_basis` / `adjustment_factor_available`；Tushare 改为 raw + `adj_factor`、Baostock raw(`adjustflag=3`)+hfq 双取、合成数据显式声明；`get_raw_close()` 用于 paper/live 成交与估值，只有复权价时 `DataQualityError`（fail closed）；`get_adjusted_close()` 供研究且与旧 hfq 序列逐点一致；双重复权守卫；`shared/symbols.py` 统一 `CODE.EXCHANGE`（前后缀、`sh.688`、北交所 4/8/920 与 30% 限价），pipeline 资产标签与 QMT/realtime/runner/API 边界统一规范化 |
| 增量数据（D32） | provider 契约新增 `supports_incremental` / `update()` / `data_status()`；Tushare/Baostock 增量追加（完成标记 +1 交易日、重叠一天、`keep="last"` 去重、全量复权因子刷新、符号/行覆盖检查、原子 data→manifest 发布）；部分响应不推进标记并保留旧缓存；`run --incremental` / `cache refresh --incremental`；`data.max_staleness_days` 研究告警、决定性运行 fail closed；freshness 写入 provenance 与 `metrics.json`；合成数据源显式不支持并回退全量 |
| 实盘信号统一 | 新增 `trading/signals.py::composite_signal`（因果 7 因子等权复合 + 横截面百分位 [0,1]）；Web `LiveTradingEngine`、CLI `LiveRunner`、`LiveSignalGenerator` 等权分支共用；删除 CLI 的 4 因子加权与短历史 RNG 兜底；127 日窗口预热，真实券商路径通过 `set_history_provider`（`main.py trade` 已接线 Baostock 日线），同一历史两条路径分数逐点一致 |
| 分析端点接线 | 7 个原本 `available:false` 的分析/监控端点改为从有界 TTL 缓存的研究上下文现算：真实 Walk-Forward（折内 train-only 重算信号、仅取 OOS）、与 `/api/barra/decompose` 共用的 Euler 风险分解、IC 衰减、因子相关矩阵、TCA（成交+订单参考价）、滚动 IC 状态、ADV 容量；并发饱和 429，缺失输入仍给精确原因与所需样本 |
| 前端契约 | LivePortfolio 挂载即加载服务端持仓；完整 en-US 语言包 + `check-i18n` 脚本；Settings 后端 URL 真正驱动 axios/WS；Ctrl+K 单一归属；行业条 100% 轴；不可用/失败 KPI 显示 `—`/N/A+原因；WS 退避+鉴权停止；OMS 行级成交；策略配比全量提交且 >100% 阻断；readiness 指示灯；上海时间解析；NaN/Inf 防护 |
| 研究数字/lab | P1 因子报告最大回撤（复利权益，旧公式显示 -0.00%）、P1 回撤 PNG 缺 `cumprod`；健康分改为带符号 ICIR；报告/比较展示 validity/affected_by；`is_trusted` 要求完整 lineage；零 IC 运行不再 success；run_id 用 uuid；迁移补齐全部列；脚本前视/买卖双边换手/硬编码结束日期/TOP8 重复/失败写成 0/成交量字段等修正 |
| LLM 诚实性 | 请求后端与实际后端一致并可观测；真实 LLM 无新闻时 fail closed（演示路径显式 `is_synthetic`）；调用方新闻入口；缓存 v4 校验/TTL；关键词去重；LLM JSON（数组/包装/围栏）解析与字段校验；输入长度上限 |
| 合规/运营 | 租户过滤（默认取 TenantContext，跨租户需显式）；风险日志 P&L 五列真实计算；CSV/XLSX 公式注入中和；导出审计；NAV 读取失败 fail closed、零净值不落库、回溯日期不继承未来费率；对账缺失腿按价格估值；投资者口径统一 |
| 研究残留 | R11：因子库 writer 以面板实际最后观测日为 `signal_date`，稳定性指标由真实 walk-forward 折计算，无折时写 NULL+`stability_unavailable`（不再伪造 0）；R85/R87 热力图与 RQ3 统一同期月份标签/实现日收益；R86/R74 成本进入日收益序列（净口径名副其实）、终期不可执行调仓一致跳过、`metric_basis` 记录；RQ1-3 输出带 provider/seed/窗口 manifest |
| provider 基准 | Postgres 显式声明 raw+`adj_factor`（可配置 `close_adjustment`，双重复权/非法因子报错）；ValidatedProvider 转发主源基准并保留 `adj_factor`，基准不一致拒绝融合；Adata 价格路径明确不支持 |
| 报告一致性 | R90 总收益口径统一（全历史 headline，公共窗口/覆盖单列）；R91 基准按日期对齐；R92 滚动 Sharpe 用日期轴且 <252 显示空态；R93 <10 样本显示 n/a；R97 VaR/CVaR 标注配置置信度；R89/R96/R98–R105 标签、优化器回退、PNG 月末分桶、空数据横幅、格式化一致、恢复期、归因截距、UTC 时间戳等 |
| 安全修复（P0） | 修复 Host 头鉴权绕过（Starlette BadHost：`Host: x/..` 使 `request.url.path` 变为 `/../api/config`，绕过 token/loopback）：鉴权改用 ASGI `scope["path"]` 并归一化 `root_path`/斜杠/编码，新增最外层 `RequestGuardMiddleware` 校验 Host（可配 `QUANT_ALLOWED_HOSTS`）、10MB 请求体上限、计算端点每客户端令牌桶 429；token 比较改为 `secrets.compare_digest`；修复 422 序列化 500。新增 31 条 Host 变体/WS/root_path 回归测试（旧代码 200 泄漏，新代码 403/401） |
| 安全加固（P1/P2） | sweep/parallel-sweep/研究窗口全部有界（组合 ≤200、n_stocks 50–1000、日期 2000–2100/≤10 年、workers≤4）；`/api/core/store/*` 与审计按租户过滤（`X-Tenant-Id`/`tenant_id`，跨租户需 loopback+显式 `all_tenants`），`delete_position` 必须有租户；HTML 报告转义+CSP、保留归档 CSV 公式注入中和、配置版本后缀密钥脱敏、WS 查询 token 移除、指标标签基数有界、QMT 密码禁止请求体、Tushare 客户端强制 HTTPS；`python-multipart`/`lightgbm` 升级到修复版本 |
| 公司行动 | provider 契约 `get_corporate_actions`（送转/派息，规范化/组合事件拆分）、pipeline 访问器、Synthetic/Tushare/Baostock 数据源；paper broker/ExecutionEngine/LiveEngine/LiveRunner/dual-track PaperBroker 按除权日幂等应用（拆股数量/成本、现金分红入账、T+1、持久化 `(asset, ex_date)` 标记），真实券商不本地应用；Web 引擎/CLI trade/daemon 全部接线，报告 `corporate_actions_applied` |
| 验证 | 新增 53 条价格基测试（`test_price_basis*`、`test_symbol_normalization`）；全量 **3,282 passed / 3 skipped / 0 failed**；ruff 全绿；CI 另修两处环境差异（fresh-boot 首条告警被 `monotonic` 节流吞掉、无 pyarrow 时缓存回退 CSV） |

**当前计数（2026-10-21 实测）**：2,989 个测试函数、3,285 条用例
（3,282 passed / 3 skipped）、234 个测试文件、约 233 个非测试 Python 文件、
99 HTTP + 1 WebSocket、21 个 CLI 命令、37 个 Vue 组件。

**仍未处理（诚实清单）**：多租户隔离（#75）；
Postgres/Adata 价格基准声明（Postgres 已声明、Adata 价格不支持）；
交易所日历 2027 年需通过 `QUANT_EXCHANGE_HOLIDAYS` 追加。
