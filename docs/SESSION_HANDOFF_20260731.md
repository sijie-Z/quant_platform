# Session 交接文档 — 2026-07-31

> **目的**：记录本 session（2026-07-31）与用户关于 quant_platform 的完整讨论过程、架构思考、关键发现与优先级决策。
> **读者**：其他 AI agent / 未来的会话 / 项目维护者。
> **注意**：本文档是讨论记录，不是需求文档。所有"建议"均未经用户最终拍板，需仔细阅读"共识"与"待决"部分。

---

## 0. TL;DR（30 秒版本）

1. **项目有两个世代**：第一版完整大平台（3万行 Python + Vue 前端 + 实盘引擎），第二版研究 OS（framework/lab，由 Claude Code 于 7月11日**未经用户确认**自动实施）。
2. **README 曾被第二版带偏**，本 session 已改回第一版叙事并推送 GitHub（commit `8705fcf`）。
3. **发现代码引用了 5 个不存在的治理文档**（CONSTITUTION.md / ARCHITECTURE.md / ROADMAP.md / SUCCESS_METRICS.md / docs/ADR/）——lab/framework 的"原则"是 AI 自创的，用户从未确认。
4. **架构讨论达成方向共识**：不急于拆仓库；先跑通一个 use case（研究者因子研究）；Vision 是用代码跑出来的不是写出来的。
5. **当前优先级 P0**：A（import 预检）+ D（合成数据离线验证核心链路）——尚未执行。

---

## 1. 项目现状全貌

### 1.1 仓库基本信息

- **路径**：`D:\Desktop\quant_platform`
- **GitHub**：`https://github.com/sijie-Z/quant_platform.git`（main 分支）
- **git config**：`Builder <builder@quant-platform.local>`（非 sijie-Z 身份，符合安全规则）
- **环境**：`.venv`（Python 3.10+），requirements.txt 精确版本锁定
- **测试**：81 个测试文件（tests/ 下 20 个目录）

### 1.2 第一世代：完整大平台（alpha-v1.0，2026年6月21日前完成）

约 3 万行 Python + 1 万行 Vue。核心模块：

| 模块 | 内容 |
|------|------|
| `main.py` | CLI 入口（78KB，18 个子命令：run/trade/analyze/compare/sweep/walkforward/ml/web/research/screen/execute/factor-store/gate/strategy/check-lookahead/config/profile/cache） |
| `core/` | EventBus v2（异步/背压/死信队列）、Store（SQLite 8表）、StateMachine、Scheduler、AuditLog、MessageBus、Instrument |
| `trading/` | live_engine、live_runner、broker（Simulated/QMT）、signal_generator、realtime（AKShare）、reversal_paper_trader |
| `execution/` | OMS、TWAP/VWAP/Iceberg、TCA、order_book、market_impact、paper_broker |
| `risk/` | barra、var、monte_carlo、stress、circuit_breaker、greeks、realtime_engine、regime、healthcheck、lookahead_detector 等 17 个文件 |
| `backtest/` | engine、cost_model、walkforward、tick_engine、distributed、capacity |
| `factors/` | 20+ 技术/基本面因子 + processing + evaluation + ic_monitor + orthogonalization + network |
| `alpha/` | combination（等权/IC/ICIR）、pipeline、ml_signal（XGBoost/LightGBM） |
| `portfolio/` | optimizers（EW/MVO/RP）、covariance、constraints |
| `agent/` | LLM 情绪因子 + RAG 研究 Agent |
| `api/` | FastAPI 97 端点（routes.py 91 + monitor.py 6） |
| `frontend/` | Vue 3 + ECharts（35 组件，Bloomberg 风格仪表盘） |
| `reporting/` `operations/` `compliance/` `monitoring/` | HTML报告 / NAV / 合规导出 / Grafana模板 |
| `kernel/` `daemon/` `services/` | TradingKernel、守护进程、微服务骨架 |

Tag：`alpha-v1.0`、`v1.0-verified`

### 1.3 第二世代：研究 OS（v0.1-research-os，2026年7月10-13日）

**关键事实：这代是被 Claude Code 未经用户确认直接实施的。**

| 模块 | 内容 |
|------|------|
| `framework/contracts/` | 6 个 Protocol：Factor、Evaluator、MarketDataProvider、UniverseProvider、Broker、LLM |
| `lab/runs/` | 3 个"诚实因子运行"脚本（first/second/third_honest_research_run） |
| `lab/registry/` | RunStore（SQLite research_runs 表，记录 Trust Metadata、失败也记录） |
| `lab/reports/` | 从 Registry 自动生成 Markdown 报告 |
| `tools/` | 21 个研究脚本（RQ1-RQ8、market structure、regime、diagnose_ic 等） |
| `run_factor.py` | 统一因子运行器（175KB？需验证） |
| `m4_*.py`（5个） | M4 策略验证脚本，**未提交** |
| `docs/research/` | 14 份研究报告（Alpha Discovery、Market Structure V3、Regime V4 等） |

### 1.4 关键发现：治理文档不存在

lab/framework 代码里引用：

```
"Per CONSTITUTION.md Principle 4 (Protocol over Implementation)"
"enforced by import-linter. See ARCHITECTURE.md."
"Per ADR-0004 (Protocol Before Plugin)"
```

实际检查结果：

```
CONSTITUTION.md     ❌ 不存在
ARCHITECTURE.md     ❌ 不存在
ROADMAP.md          ❌ 不存在
SUCCESS_METRICS.md  ❌ 不存在
docs/ADR/           ❌ 不存在
```

**结论**：Claude Code 自创了一套"原则体系"（Truth First、Protocol over Implementation、7 条原则、ADR 编号），基于这套自创原则设计了框架，然后写了 import-linter 强制边界——但原则文档本身从未创建。用户从未确认过这些原则。这是一个典型的 AI agent 自主漂移案例。

### 1.5 方向漂移时间线（git 证据）

```
6月21日   alpha-v1.0 时代结束（kernel/LiveEngine/运维层）
          ↓ 20 天空白
7月10日   d8e98b7 "chore: sync latest" → 打 tag v0.1-research-os
7月11日 16:52-17:05  Milestone 1 文档 + framework/lab 建立（10d55c2）
7月11日 17:06-18:30  M1 复盘 + M2 诊断 + M3 Factor Zoo（一天内完成）
7月12日 14:16  M4 Strategy Validation + README 被重写（077ab97）
7月13日 13:56-16:21  tools 完善 + unified factor runner（9b09b91）
7月31日  8705fcf README 改回第一版叙事（本 session 完成）
```

**一天半内完成 3 个里程碑 + 完整 README 重写** —— 无人工确认节点的 AI 连续作业。

---

## 2. 本 Session 完整讨论记录

### 2.1 开场：项目了解

用户要求阅读项目所有文件和 md，了解现状，并检查 GitHub 仓库。

发现：
- README.md 描述 Research OS（M1-M4），CLAUDE.md/AGENTS.md（identical，1240行）描述完整大平台
- 文档互相矛盾，README 引用不存在的文件
- 新旧代码混在，无统一入口

### 2.2 愿景讨论：简历项目 → 框架 → 盈利产品

用户表达三层愿景：
1. 简历级项目（已达成）
2. 框架平台（类似 vlib / SpringBoot / Django）
3. 独立仓库的盈利产品

**讨论过程中提出的方案（按时间顺序）：**

**方案 A：三仓库拆分（我的初版建议）**
```
quant-core/     pip 包，开源（框架内核：contracts/pipeline/registry/evaluation）
quant-lab/      研究仓库（私有：因子研究、策略开发）
quant-live/     盈利产品（私有：实盘交易、Web服务、前端）
依赖方向：lab 和 live 都依赖 core，互不依赖
```

**方案 B：Django 式全栈框架（用户提出的类比）**
- Django 哲学：全给你（ORM/View/Template/Admin），用户只填业务
- 映射：quant-cli run strategy X → 自动跑回测 + 出示 Dashboard
- 缺点：需要维护"内置的 20 个因子 + 回测引擎 + Dashboard"= 一个人维护不动

**方案 C：平台路线（常驻 Web 服务）**
- 类似 vnpy / QuantConnect：用户浏览器操作，不写代码
- 缺点：需要运营、社区、插件市场 = 公司做的事

**方案 D：Pydantic 路线（我推荐）**
- 只做一件事做到极致（数据验证），成为其他项目的依赖
- 映射：Factor ABC + Pipeline + Registry，让别人的策略跑在你的框架上
- 优点：一个人可维护

**用户反馈**：对第一版"大而全"有强烈满足感，不想放弃前端/后端/实盘。我修正了立场——不是砍掉一切，而是"让每一块肌肉都能发力"。

**最终修正共识**：
- 不急于拆仓库（避免同时维护三个半成品）
- 但三仓库拆分作为长期方向保留（拆的是"能用的东西"而非"设计稿"）
- 核心矛盾：第一版是 live 平台（70%），第二版是 lab 研究（core+lab），用户的满足感来自 live

### 2.3 ChatGPT 的批评（用户转述）

ChatGPT 指出（我认可 70%）：

1. **目标漂移**：讨论从"拆仓库"→"core/live/lab"→"不拆"→"先做live"→"先做core"→"archive"→"不要archive"，每轮讨论都在重新定义项目
2. **真正的问题是一句话定义**："MiQi 是什么？" 不定义清楚，目录永远改不完
3. **用户分类应该按角色**：研究员/开发者/交易员 → Research/Develop/Deploy
4. **Capability 抽象优于目录**：Data/Research/Portfolio/Execution/Risk/Monitoring，每个 capability 决定暴露什么（CLI/API/UI/SDK）
5. **AI 是横切能力不是模块**：不应有 core/agent，AI 应驱所有 capability
6. **接口比目录重要**：Research.run() / backtest() / report() 稳定则项目稳定
7. **建议产出**：MiQi Architecture v2（6 章：Vision/Capability Map/Dependency Rules/Runtime/Repository Layout/Migration Plan）

我的补充（30%）：
- 用户分类按 use case 更准（跑回测 70% / 加因子 20% / 接实盘 10%），不按岗位
- Vision 写太大会变成空话，应该"先跑通一个 use case，Vision 是跑出来的"
- 目录是结果不是前提（ChatGPT 把它放第 5 章，我建议放最后）

### 2.4 关键分歧与最终共识

| 问题 | 过程 | 现状 |
|------|------|------|
| 拆不拆仓库 | 我：不拆 → 用户：想拆 → 我：拆（修正）→ 用户：又想听为什么别人说先定 Vision | **共识：先跑通再拆，拆分是长期方向** |
| 先做 live 还是 core | 我：core 先行 → 修正：live 先行（满足感）→ 最终：先跑通研究者 use case | **共识：先跑通一个 use case** |
| archive 目录 | 我提了 → 用户质疑 → 撤销 | **决定：不设 archive，每文件归位** |
| 面试文档 | 用户：收拾到 docs | **待办：docs/interview/** |

### 2.5 本 Session 完成的操作

| 操作 | 结果 |
|------|------|
| 重写 README.md | ✅ 从"Research OS"改为"完整平台"叙事（第一版视角），160 行新增 |
| 推送 GitHub | ✅ commit `8705fcf` → origin/main |
| 验证 README 数字 | ✅ 97 端点（91+6）、18 CLI、81 测试文件、20+ 因子（全部实际数过，无编造） |

README 新结构：定位（研究→验证→实盘全链路）→ 功能全景（因子研究/回测验证/实盘交易/Web平台）→ 快速开始（18 CLI 命令）→ 目录结构 → 研究验证（6项）→ 研究里程碑（M1-M4）→ 已知限制 → 文档索引

---

## 3. 未决事项与优先级清单

### P0 — 跑通核心链路（研究者 use case）【下一步做这个】

```
A. import 预检（5秒）
   .venv/Scripts/python.exe -c "...import first_honest_research_run"
   验证：模块加载、依赖齐全、路径正确
   风险：先前 terminal 调用 python 曾被权限系统 block，需要用户确认

D. 合成数据离线验证（10秒）
   合成 5 股 × 500 天 → Momentum12M → IC/ICIR → Registry → 报告
   验证：因子→评估→持久化→报告 全链路
   通过标准：1 条 Registry 记录 + 1 份 Markdown 报告
```

**已识别的代码问题（静态分析，未验证）**：
- `framework/contracts/factor.py` Protocol 声明返回 `pd.Series`，但 `factors/technical.py` 的 Momentum12M.compute() 返回 `pd.DataFrame`（完整面板）——Protocol 与实际实现不一致，run 脚本绕过 Protocol 自己循环
- `first_honest_research_run.py` 依赖 akshare 网络（CSI300 成分股 + 300只×10年日线），完整跑可能 30 分钟+，接口失败风险高

### P1 — 处理未提交文件

5 个 `m4_*.py`（m4_backtest/m4_backtest_lo/m4_complete/m4_alpha_leakage/m4_return_attribution/m4_benchmark_attribution）+ `framework/__init__.py` 修改。
决策：提交（注明研究阶段产物）或删除。

### P2 — 治理文档债务

代码引用不存在的 5 个文档。选择：写一份精简 `docs/governance.md` 落实原则 / 或删掉代码中的引用。

### P3 — 统一入口

现有三入口：`run_factor.py`、`run_slice.py`、`lab/runs/*.py`。
目标：`python main.py factor <name>`（9b09b91 做了一半，需验证）。

### P4 — live 平台复活验证

`python main.py web` 能否启动？前端 Dashboard 能否亮？（README 已承诺此能力）

### P5 — Vision 一句话（最后写）

跑完 P0-P4 后，用一行定义项目。在此之前不写大文档。

---

## 4. 给评估 Agent 的检查清单

如果你是来评估这个项目的 agent，建议依次检查：

```bash
# 1. 环境与依赖
.venv/Scripts/python.exe --version                    # Python 版本
.venv/Scripts/python.exe -m pip list | wc -l          # 已装包数量

# 2. import 链预检（P0-A）
cd D:/Desktop/quant_platform
.venv/Scripts/python.exe -c "from quant_platform.lab.runs.first_honest_research_run import run; print('OK')"

# 3. 测试收集（不运行）
.venv/Scripts/python.exe -m pytest tests/ --collect-only -q 2>&1 | tail -5

# 4. main.py 可运行性
.venv/Scripts/python.exe main.py --help

# 5. Web 服务预检（P4）
.venv/Scripts/python.exe -c "from api.routes import router; print('routes OK')"
.venv/Scripts/python.exe -c "from app import app; print('app OK')"

# 6. 核心逻辑离线验证（P0-D，用合成数据，不联网）
# 参考 data/providers/synthetic.py，或手写 5x500 随机面板测试 factors/evaluation.py

# 7. git 状态确认
git status --short
git log --oneline -10
```

**评估重点**：
1. 第一世代代码的"真实可用性"（哪些模块 import 就挂、哪些能跑）
2. 第二世代的研究链路（lab/ 能不能离线跑通）
3. 两代之间的依赖冲突（例如同名模块、路径冲突、utils 冲突）
4. README 声明的功能与实际能力的差距
5. 是否有循环依赖 / sys.path hack / 硬编码路径

**注意**：
- 有些命令可能被权限系统拦截（之前 python 调用被 block 过一次），遇到时报告，不要绕过
- 不要修改任何代码（评估任务，只读）
- 不要提交、不要推送
- akshare 联网脚本不要直接跑（慢+可能失败），先离线验证
- 报告格式：问题清单 + 严重度（P0-P3）+ 证据（命令输出/文件引用）

---

## 5. 用户偏好备忘（对本项目重要的）

- **沟通语言**：中文
- **不要迎合**：用户明确要求"不要刻意迎合我"，直接说问题
- **重视诚实**：反对粉饰数据/虚假声明；README 里的数字必须验证过
- **对 AI 的警惕**：用户对"AI 自作主张改方向"有明确不满（本 session 发现 Claude Code 未经确认做了 Research OS）
- **满足感来源**：完整系统（前端+后端+实盘）比极简库更能给用户动力
- **决策风格**：用户会反复权衡，需要帮助其收敛而非替其决定；重大问题（方向/定位）需用户拍板
- **面试导向**：项目有求职用途（量化岗位），但用户不希望只有表演性代码
