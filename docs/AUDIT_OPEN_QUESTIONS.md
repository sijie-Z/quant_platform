# quant_platform — 代码与 CI 审计：待决问题清单

> **文档目的**：本文件是一次代码/CI 审计后遗留的**不确定问题**清单，用于外部审阅。
> 每个问题都包含：现象、代码证据、我已做的处理、我不确定的点、候选方案与权衡、我的倾向。
> **审阅者不需要访问仓库**，所有必要上下文都在正文中。
>
> 生成日期：2026-09-12
> 代码基线：`main` @ `1caba6d`

---

## 0. 背景（给审阅者）

### 0.1 项目是什么

**quant_platform** —— 一个 A 股（中国股市）多因子量化研究 + 交易平台，单人项目，公开在 GitHub（`sijie-Z/quant_platform`，8 stars）。定位是"研究 → 验证 → 实盘"全链路。

规模：

| 指标 | 数值 |
|---|---|
| Python 文件 | 320 个 |
| Python 代码行 | 57,683 |
| Vue/JS 文件 / 行 | 42 / 10,107 |
| 测试用例 | 1,247 个（收集数） |
| 测试文件 | 86 个 |
| REST API 端点 | 97 |
| CLI 命令 | 19 |

架构上分成两层（这是理解后续问题的关键）：

1. **v1 平台层**（2026-05/06）：完整的研究→回测→实盘→Web 平台。目录：`core/ data/ factors/ alpha/ portfolio/ backtest/ risk/ execution/ trading/ api/ frontend/` 等。
2. **v2 研究 OS 层**（2026-07 起）：转向"诚实研究"。新增 `framework/`（Protocol 契约）、`lab/`（研究运行 + Registry）、`tools/`。核心原则是"Truth First / Point-in-time / Knowledge compounds"，每次因子运行都把数据源、复权方式、PIT 状态、偏差警告写进 SQLite Registry，报告里的告警由这些字段**机器生成**而非手写。

最近一次重构（2026-08-05）把 5 个共享模块抽成独立包 `packages/quant_core`，原位置的 `core/store.py` 等 **17 个文件变成转发 shim**：

```python
# core/store.py （完整内容）
"""Compatibility shim: moved to quant_core.store."""
from quant_core.store import *  # noqa: F401,F403
from quant_core.store import Store  # noqa: F401
```

这个 shim 机制是下面 Q1 的直接起因。

### 0.2 本次审计的起因

用户要求："GitHub 上最后一次 CI 是红的，修掉，并且把 CI 基础设施按大项目标准补齐。"

上一次 CI 失败的根因已定位并修复：**Dockerfile 从未安装 `packages/quant_core`**，容器启动即 `ModuleNotFoundError: No module named 'quant_core'`。CI 的 `test` job 里有一行 `pip install -e packages/quant_core`，所以只有容器能暴露这个问题。

---

## 1.5 外部裁定执行记录（2026-09-12 更新）

本文件经过一轮外部审阅，以下是**已按裁定落地的改动**：

| 裁定项 | 动作 | 状态 |
|---|---|---|
| **Q2** 测试污染 Registry | `run_synthetic_factor()` 增加 `db_path` / `out_dir` 参数（默认值保持生产行为）；测试全部改用 `tmp_path`，并新增"不得写入真实 `data/` 树"的回归守卫 | ✅ 已改，4 个测试通过 |
| **Q8a** `None → All_other` 过度降级 | **回退**。改为显式 `raise TypeError(...)` —— 与原始行为同样是 TypeError（原来从 `np.isfinite(None)` 抛出），但不吞掉调用方 bug，同时给 mypy 提供类型收窄 | ✅ 已改 |
| **Q8b** benchmark 静默跳过 | **回退**。改为 `raise RuntimeError`。该函数在压测前已 `get_or_create()` 全部 symbol，`None` 不可达；静默跳开会悄悄缩小样本、改变吞吐数字 | ✅ 已改 |
| **Q8c** lambda 闭包 `H=H` | 保留（归类为 robustness cleanup，不声称修复了研究结果） | ✅ 保留 |
| **Q10a** `portfolio_snapshot` 缺 `cash` | **调用点核实完成**：`PortfolioOrchestrator.portfolio_snapshot()` 只取 `positions_value / n_positions / total_unrealized_pnl / total_realized_pnl / total_pnl`，现金用它自己的 `cash_available`；测试只取 2 个字段。**无任何调用方需要 `cash`** → 判定为 docstring 过时，改为准确描述 | ✅ 已改 docstring |
| **Q10b** `zip(strict=False)` | **改为 `strict=True`**。`turnover_records` 是 `append((date, turnover))` 累积的 2-元组列表，`zip(*records)` 各序列长度必然相等，invariant 恒成立 | ✅ 已改 |
| **Q10g** Python 版本上限 | `requires-python` 改为 `>=3.11,<3.13`，并在 pyproject 里写明上限理由；README 徽章同步改为 `3.11 \| 3.12` | ✅ 已改 |
| **Q1** LiveEngine 生命周期 | **不并入本次改动**，拆成两个独立 issue（见 §4 草稿） | ⏸ 待建 issue |
| **Q3 / Q5 / Q6** 依赖与研究基线 | 维持现状，不升级、pip-audit 保持 report-only、覆盖率不设阈值 | ✅ 维持 |
| **Q4** 全仓 `ruff format` | 不并入本次，另开独立机械 commit | ⏸ 待办 |
| **Q9** 未跟踪研究文件 | 不批量删除，改为分类归档（分类建议见 §5） | ⏸ 待项目所有者决定 |

> **注意 Q3 的 `uv.lock` 例外**：裁定建议"采用 uv 就提交 `uv.lock`"，但**当前工作区里的
> `uv.lock` 不能直接提交** —— 它是本地 Python 3.14 精简 venv 的产物（见 Q3 表格），
> 并不代表项目的 research baseline。提交它等于把错误的环境锁死。正确顺序是先确定
> canonical 依赖来源，用 3.12 环境重新 `uv lock`，再提交。详见 §5。

---

## 1. 本次已修复的内容（供参考，不需要建议）

| 类别 | 内容 |
|---|---|
| **CI 红根因** | Dockerfile 增加 `pip install ./packages/quant_core` |
| **依赖体积** | `xgboost==2.1.4` → `xgboost-cpu==2.1.4`。前者在 Linux x86_64 上**硬依赖 `nvidia-nccl-cu12`**，而本平台是纯 CPU 的（无任何 GPU 代码路径）。实测 Docker 构建日志确认该包被下载。`xgboost-cpu` 是同版本同 API 的 CPU-only 发行版，wheel 内同样是 `xgboost/__init__.py`（已用 HTTP range 读取 wheel 文件清单确认导入名一致）。**实测体积差**：GPU wheel 223.6 MB + NCCL 342.1 MB = **565.7 MB**，CPU wheel 仅 **4.5 MB**，**每次安装省 561 MB**（CI 两个 matrix leg ≈ 省 1.12 GB/次） |
| **Docker 基建** | 多阶段构建（前端在 node 镜像构建）；去掉 `build-essential/gfortran/libopenblas-dev`（实测所有依赖都有 manylinux wheel）；新增 `.dockerignore`（原先 `COPY . .` 会把 350MB 的 Windows `.venv` 塞进 Linux 镜像）；新增 `HEALTHCHECK`（用 stdlib，因为 slim 镜像没有 curl） |
| **docker-compose** | healthcheck 原先调用 `curl`，而镜像里没有 curl → 服务永远不可能 healthy，配合 `restart: unless-stopped` 会无限重启 |
| **真 bug** | `api/routes.py`: `POST /api/screen` 调用未定义的 `_load_config`（应为 `load_config`），且在 try 块之外 → 每次请求必然 500 |
| **真 bug** | `quant_core/regime.py`: `get_execution_params()` 调用 `detect()` 时不传参数，而 `detect(returns, prices)` 两个参数都是必填 → 必然 `TypeError` |
| **语法兼容** | 2 个 `tools/` 脚本在 f-string 表达式里用反斜杠（PEP 701，仅 3.12+ 可解析），而项目声明支持 3.10+ |
| **版本声明** | `requires-python` 从 `>=3.10` 改为 `>=3.11`——`scipy==1.16.0` 与 `tables==3.11.1` 都要求 ≥3.11，3.10 装不上 |
| **Lint** | ruff 从 256 个错误清零（202 个自动修复）；mypy 修复 `quant_core` 的 13 个类型错误 |
| **CI 加固** | 独立 lint job（硬门槛）、Python 3.11/3.12 矩阵、覆盖率 artifact、pip-audit（报告制）、buildx + GHA 层缓存、concurrency、least-privilege permissions、逐 job timeout |
| **新增** | CodeQL workflow（`security-and-quality`，每周）、Dependabot（pip/npm/actions/docker） |
| **假门槛** | 原 CI 的 `mypy quant_platform/` 指向一个**不存在的路径**，且被 `continue-on-error: true` 掩盖——从来没检查过任何东西。pre-commit 的 mypy hook 也不传任何目标文件 |

### 已验证的证据

```
ruff check .                    → All checks passed!              (256 → 0)
ruff check . (仅 git 跟踪文件)   → All checks passed!              (等价于 CI 条件)
mypy packages/quant_core/src    → Success: no issues found in 6 source files
main.py --help                  → 19 个命令全部完好
```

**测试结果（重要更正）**

本机 `pytest` 的 `tmp_path` 基础目录 `C:\Users\admin\AppData\Local\Temp\pytest-of-admin`
存在**损坏的 ACL**（连所有者都 `Access is denied`，`takeown` 也被拒绝）。这导致所有用
`tmp_path` 的测试在 fixture setup 阶段就 ERROR，**测试根本没跑起来**。

用 `--basetemp=<可写目录>` 绕过之后，真实数字是：

| | `tmp_path` 坏掉时 | 修复 basetemp 后 |
|---|---|---|
| passed | 1092 | **1198** |
| failed | 34 | **49** |
| errors | 126 | **7** |

那 7 个 error 才是真正的缺包（scipy / sklearn / matplotlib / cvxpy 的 collection error）。
此前的 126 个 error 中有 ~119 个是 `tmp_path` 权限问题，**不是缺包** —— 这一点我最初判断错了。

多出来的 15 个 failure 全部来自 `tests/test_operations/test_investor.py`，根因是本地 venv 装的
是 pandas 3.0.3（而非 pin 的 2.1.3），见下方 Q3 的实证。

回归测试有效性验证：把 `_load_config` 这个 bug 临时改回去 → 新加的回归测试立刻失败（2 failed）。

---

## 2. 待决问题

> **状态提示**：Q2 / Q8a / Q8b / Q10a / Q10b / Q10g 已经按外部裁定处理完毕，结论见 §1.5。
> 下面保留它们的完整分析过程，因为"当时为什么不确定"本身就是有价值的记录。
> **仍然开放**：Q1、Q3、Q4、Q5、Q6、Q7、Q9、Q10f。

### Q1 ★最严重★ `LiveEngine` 到期仓位不回款，且现金永不回收

**位置**：`trading/live_engine.py`（`LiveEngine`，docstring 自称 "alpha-v1.0 运行时系统"，对应 `docs/PRODUCTION_ARCHITECTURE_V1.md` 的"80d 反转 + vol filter 生产主线"）

**现象**：`_execute_rebalance()` 里"平到期仓位"这一行：

```python
    def _execute_rebalance(self, date: pd.Timestamp) -> str:
        """执行调仓."""
        ...
        # ── 平到期仓位 ──
        date_str = str(date)[:10]
        self.state.positions = [p for p in self.state.positions if p.exit_date > date_str]
```

到期仓位被**直接从列表里过滤掉**——没有卖出动作、没有把市值加回现金、没有扣任何交易成本。

**证据（穷尽核对）**：全文件中所有对 `cash` 的写操作只有两处：

```
236:  self.state.cash -= total_spent          # 买入
252:  self.state.cash += pos.shares * px * (1 - cost)   # 只在 _emergency_flat() 里（kill switch 紧急平仓）
```

第 236 行还带着注释 `# 买入不扣成本, 卖出时扣`（买入不扣成本、卖出时扣）——**但正常调仓路径根本不卖出**，只有 kill switch 的 `_emergency_flat()` 才卖。

而权益计算是：

```python
    def _update_equity(self, date: pd.Timestamp):
        pos_value = 0.0
        ...
            for pos in self.state.positions:
                pos_value += pos.shares * px[pos.asset]
        self.state.current_equity = self.state.cash + pos_value
```

只给**当前还在列表里的**持仓估值。

**后果**：到期仓位被移除的瞬间，它的市值从权益里凭空消失，而现金从未被补回。按设计（`HOLD_H = 80`，调仓周期也是 80 天），**每一个调仓周期都会发生一次，等于系统性地销毁资本**。

**但是——当前的实际使用方式把它掩盖了**。这是我不确定的关键点：

```python
# daemon/runner.py
        # 2. 运行策略
        self.engine = LiveEngine()                       # ← 每次都新建实例
        self.engine.load_history(pipeline.returns, pipeline.get_close())
        last_date = pipeline.returns.index[-1]
        self.engine.run_once(last_date)                  # ← 只调用一次
```

而 `LiveEngine.run_once()` 的 docstring 明确写着设计意图是持续运行：

```python
    def run_once(self, date: pd.Timestamp) -> dict[str, Any]:
        """单日运行 (每次调用 = 一个交易日).

        这是 live engine 的核心——每次市场收盘后调用一次.
        """
```

也就是说：**这个引擎被设计成跨日持有状态的持久进程，但唯一的调用方每次都 new 一个新实例、只跑一天**。

- 按**当前 daemon 用法**：每次都是全新引擎（现金回到 `INITIAL_CAPITAL`），持仓从未活到 `exit_date`，所以 bug **不会显现**。
- 按**设计意图用法**（一个实例跨多日）：bug 每个调仓周期发作一次。
- 顺带：`equity_peak`（用于 15%/25% 回撤熔断）在这个用法下也永远无法累积，所以**回撤熔断实际上永远不会触发**。

**我不确定的点**：

1. 这算"生产路径上有严重 bug"，还是"一个还没真正接线的模块里的潜在 bug"？取决于这个引擎到底有没有被当作生产主线跑过。
2. 修复方向是哪一个：
   - **(A) 补上卖出逻辑**：到期时按当日价格卖出，`cash += shares * px * (1 - cost)`，并扣除成本（与 `_emergency_flat` 一致）。
   - **(B) 把"到期"改成显式的卖出信号**，走和买入对称的路径。
   - **(C) 承认它就是一个单日模拟器**，删掉跨日状态的伪装，把 `exit_date` 逻辑移除。
3. daemon 每次 new 实例、只跑一天——这本身是不是也是 bug？（即：本该持久化的状态被丢了）

**补充**：`trading/live_engine.py` 里还有一个我已删除的死代码：`_execute_rebalance()` 里有一行 `cost = COST_BPS / 10000` 赋值后从未使用（真正的使用在 `_emergency_flat()` 里）。我**只删了死代码，没有改任何交易逻辑**。

---

### Q2 测试会污染真实数据目录（无 `tmp_path` 隔离）

**位置**：`tests/test_lab/test_synthetic_factor_run.py`

```python
def test_run_synthetic_factor_persists_report():
    run_id = run_synthetic_factor("momentum_12m", n_stocks=3, n_days=120, seed=1)
    assert run_id.startswith("run_")
    report = Path("data/reports") / f"{run_id}.md"
    assert report.exists()
```

被调用的 `run_synthetic_factor()` 内部硬编码了两个真实路径，没有注入点：

```python
# lab/runs/synthetic_factor_run.py
75:    store = RunStore(DEFAULT_DB)
89:    report_path = generate_report(store.get_run(run_id), out_dir="data/reports")
```

**后果**：每跑一次测试就会
1. 在 `data/reports/` 留下一个新的 `run_*.md` 文件（导致 `git status` 永远不干净）
2. 往 `data/trading.db` 的 `research_runs` 表插入记录（我这两轮测试把它从 38 条涨到 41 条）

而 `research_runs` 正是这个项目"Knowledge compounds"原则的核心资产——研究知识库。用测试数据污染它，和该原则直接冲突。

**我已经做的**：把我这两轮测试产生的 `run_*.md` 删掉了（DB 里的 3 条记录**没有动**，我不确定该不该删）。

**候选方案**：

- **(A) 给 `run_synthetic_factor()` 加 `db_path` / `out_dir` 参数**，测试传 `tmp_path`。改动小，语义清晰。
- **(B) 测试里 monkeypatch `DEFAULT_DB` 和 `generate_report`**。不改产品代码，但测试变得更"魔法"。
- **(C) 用 `conftest.py` 的 autouse fixture 全局 chdir 到 tmp**。范围最大，可能影响其他测试。
- **(D) 不管**——承认这是"集成测试"，本来就要写真库。

**我的倾向**：(A)。但**不确定**该不该顺手清掉 registry 里已有的那些 synthetic 测试记录（现在 41 条里可能有 7 条是历次测试留下的），因为那毕竟是一个"研究知识库"，删记录需要考虑可追溯性。

---

### Q3 本地 venv 与 `requirements.txt` 完全脱节

**现象**：本地 `.venv` 和项目声明的依赖是两套完全不同的世界。

| | 本地 `.venv` | `requirements.txt` / CI |
|---|---|---|
| Python | **3.14.6** | 3.12 |
| numpy | 2.5.1 | 1.26.2 |
| pandas | 3.0.3 | 2.1.3 |
| pytest | 9.1.1 | 7.4.3 |
| scipy / scikit-learn / cvxpy / matplotlib / sqlalchemy / pytest-asyncio | **全部缺失** | 已锁版本 |

这个 venv 是 2026-08-04 用 `uv` 建的（`uv.lock` 至今未提交、未 gitignore），装的是一个精简子集。

**后果**：本地 `pytest tests/` 的结果是 `1092 passed, 34 failed, 126 errors`，但**失败全部是缺包导致，不是代码问题**（CI 上同一个 commit 的 test job 是绿的）。这会持续误导任何本地跑测试的人——包括我自己在审计初期。

**候选方案**：

- **(A) 重建 3.12 venv，严格按 `requirements.txt` 安装**。和 CI 一致，但 `numpy==1.26.2` / `pandas==2.1.3` 这些 2023 年的版本现在看已经很旧。
- **(B) 保留 uv 工作流，把缺的 6 个包补上，并把 `uv.lock` 提交**。但这样本地和 CI 的依赖版本仍然不一致，"本地绿 / CI 红"或反之的风险一直在。
- **(C) 全面升级依赖到新版本**（numpy 2.x / pandas 3.x），同步更新 `requirements.txt`。工作量大，可能引入 breaking change，而且**会改变所有既有回测结果**，对一个"研究可信度"导向的项目是重大决定。
- **(D) 什么都不做，只写进文档**。

**实证：升级 pandas 会破坏什么（这是本次审计新拿到的硬证据）**

本地 venv 装的是 pandas 3.0.3，而项目 pin 的是 2.1.3。两者的差异直接导致
`tests/test_operations/test_investor.py` 里 **15 个测试失败**，报错：

```
ValueError: Invalid frequency: M. Failed to parse with error message:
ValueError("'M' is no longer supported for offsets. Please use 'ME' instead.")
```

（pandas 2.2 弃用 `'M'` 频率别名，3.0 移除。）

**而 `'M'` 出现在 4 处生产代码，不只是测试**：

| 位置 | 代码 |
|---|---|
| `operations/investor.py:163` | `df["nav_per_unit"].resample("M").last()` |
| `reporting/performance.py:111` | `strategy_returns.resample("M").apply(...)` |
| `api/routes.py:747` | `sr.resample("M").apply(...)` |
| `api/routes.py:748` | `monthly_ret.index.to_period("M")` |

后两处所在的测试文件因为缺 matplotlib 而 collection error，**根本没有执行到**，
所以升级 pandas 的真实影响面**大于**观测到的 15 个失败。这正是"依赖升级会不受控地改变
研究结果"的具体形态 —— 不是一个抽象的担心。

**我不确定的点**：这是一个研究平台，历史回测结果的可复现性很重要（numpy/pandas 的升级会改变浮点行为、`pct_change` 的 fill 语义、resample 行为等）。**升级依赖 = 让所有已有研究结论失效**。这个代价 vs 用 2023 年旧版本的安全风险，我不知道该怎么权衡。

---

### Q4 是否要执行 `ruff format`（会重排 276 个文件）

**现象**：

```
ruff format --check .
→ 276 files would be reformatted, 100 files already formatted
```

`pyproject.toml` 里已经配好了 `[tool.ruff.format] quote-style = 'single'`，`.pre-commit-config.yaml` 里也挂了 `ruff-format` hook，但**从来没跑过**。注意这个项目和 ruff 默认风格的关键差异：代码里大量使用单引号，而 `ruff-format` 会按配置保持单引号——所以主要是换行、缩进、括号、尾逗号的改动。

**候选方案**：

- **(A) 跑一次 `ruff format .`，单独一个 commit**，然后 CI 加 `ruff format --check` 门槛。一次性 276 文件的 diff，但之后永久一致。
- **(B) 只格式化新代码**（比如 `packages/quant_core`、`framework/`），老代码不动。门槛只在新目录生效。
- **(C) 不做**，接受风格不统一。

**我的倾向**：(A) 或 (B)。**不确定**的是：这个仓库有大量研究脚本（`tools/` 下 20 个 `run_rq*.py` / `run_market_structure*.py`），它们承载着已经发表的研究结论。虽然纯格式化不改变行为，但如果有人对照历史 commit 阅读这些脚本，大规模重排会让 `git blame` 变得难用。另外我**不确定**格式化会不会在别处产生语义变化（理论上不会，但 276 个文件的规模值得谨慎——我倾向于先跑，然后 `pytest` 验证）。

---

### Q5 pip-audit 应该只报告还是做硬门槛？

**现状**：我把它做成 **report-only**（`continue-on-error: true` + 上传 JSON artifact），理由是：

`requirements.txt` 里的 pin 相当旧（numpy 1.26.2 / pandas 2.1.3 / fastapi 0.128.0 / matplotlib 3.8.4 …），大概率会命中已知 CVE。如果直接做硬门槛，CI 会**永久变红**，那和没有门槛没区别。

**不确定的点**：

1. 正确的做法是"**先把依赖升上去，再开硬门槛**"，还是"**开硬门槛 + `--ignore-vuln` 白名单**"，还是维持 report-only？
2. 如果走升级路线，又回到 Q3 的问题：升级会改变回测结果。
3. 一个公开的量化项目，安全审计的实际优先级有多高？（这个项目不处理用户凭据、不联网服务，最坏情况是供应链投毒。）

---

### Q6 覆盖率应该设门槛吗？设多少？

**现状**：CI 在 3.12 这一 leg 产出 `coverage.xml` 并上传 artifact，但**没有设 `--cov-fail-under`**，因为我不知道基线是多少（本地跑不出可信的覆盖率，见 Q3）。

**不确定的点**：

1. 对一个 57k 行、单人维护的研究平台，覆盖率门槛是**有价值的纪律**还是**纯粹的负担**？
2. 如果要设，合理的初始值是多少？（我倾向于先从"不低于当前基线"开始，即先测出基线再锁死）
3. 更根本的问题：**这个项目的测试结构是"每个模块一个单测文件"**（86 个测试文件覆盖 20+ 模块）。有没有可能"覆盖率"这个指标在这里会激励错误的优化？（例如为了提升覆盖率去测 `tools/` 下的一次性研究脚本，而那些脚本的价值在于结论而不在于可测性。）

---

### Q7 是否配置分支保护 / required status checks？

**现状**：仓库是 public，`main` 上没有分支保护。所有提交（包括我的审计修复）都会直接推 `main`。

**候选方案**：

- **(A) 给 `main` 加分支保护**：要求 CI 通过才能合并，禁止 force push。代价是单人项目也得走 PR 流程（或者至少不能直接 push）。
- **(B) 只要求 `lint` job 通过**（快，几十秒），其余保持宽松。
- **(C) 不设**，保持单人直推的流畅度。

**我不确定的点**：这是单人项目，PR 流程可能是纯粹的仪式感；但"CI 绿了才能进 main"确实能防止再次出现"最后一次提交是红的"这种状态。另外我**不确定**：如果设了 required checks，而某个 job（比如 docker，需要 5-10 分钟）偶发失败，会不会反而拖慢迭代。

---

### Q8 我在本次审计中做的几处**行为变更**，是否可接受？

为了让 ruff/mypy 通过，我做了三处**不只是格式化**的改动。我认为它们都是改进，但都是行为变更，需要确认：

**(a) `profile_classifier.py`：把崩溃改成优雅降级**

```python
# 改前
tradability = features.get("tradability")
if not np.isfinite(tradability) or not np.isfinite(transition):
    return "All_other"
# 传入 None 时 np.isfinite(None) → TypeError

# 改后
if tradability is None or transition is None:
    return "All_other"
tradability = float(tradability)
transition = float(transition)
if not np.isfinite(tradability) or not np.isfinite(transition):
    return "All_other"
```

合法输入行为不变；缺失输入从"抛 TypeError"变成"返回 All_other"。**不确定**：这是不是掩盖了一个本该暴露的调用方错误？

**(b) `order_book.py` benchmark 辅助函数：None 保护**

```python
sym_book = manager.get(sym)
if sym_book is None:
    continue        # 新增
```

这是 `benchmark_order_book()` 里的压测循环，`manager.get()` 可能返回 None（该 symbol 没有 book）。原来会 `AttributeError`。**不确定**：静默跳过是不是比报错更好？（我个人认为压测脚本里跳过合理。）

**(c) `regime.py`：`get_execution_params()` 签名从可选改成必填**

```python
# 改前（必然 TypeError，且零调用者）
def get_execution_params(self, returns: pd.Series | None = None) -> dict:
    if returns is not None:
        result = self.detect(returns)
    else:
        result = self.detect()

# 改后
def get_execution_params(self, returns: pd.Series, prices: pd.Series,
                         returns_matrix: pd.DataFrame | None = None) -> dict:
    result = self.detect(returns, prices, returns_matrix)
```

虽然零调用者，但这毕竟是一个公开类的公开方法签名变更。**不确定**：是不是应该保留向后兼容（比如 `prices` 给个默认值然后内部降级）？

**另外**：`data/providers/validated_provider.py` 里的异常类 `DataDiscrepancy` 被我重命名为 `DataDiscrepancyError`（PEP 8 要求异常类以 Error 结尾）。全仓库搜索确认**零引用**，但同样是公开 API 变更。

---

### Q9 未跟踪文件该怎么处理？

当前工作区里有一批**从未提交**的文件，我没有擅自处理：

| 文件 | 日期 | 说明 |
|---|---|---|
| `m4_alpha_leakage.py` | 2026-07-12 | M4 分析脚本。**我修了里面一个 `cost_pct` 未定义的 `NameError`**（否则必然崩溃） |
| `m4_backtest_lo.py` `m4_benchmark_attribution.py` `m4_complete.py` `m4_return_attribution.py` | 2026-07-12/13 | 同批 M4 后续分析 |
| `data/reports/run_1785833*.md` ×4 | 2026-08-04 | synthetic 空跑报告（IC 全是 `None`） |
| `uv.lock` | 2026-08-04 | uv 的锁文件 |
| `docs/` 里若干 | — | — |

**背景**：这个项目的 M1-M4 研究结论（`README.md` 里"研究里程碑"那一节）来自这些脚本。但 M4 的核心脚本 `m4_backtest.py` **已经提交了**，这几个是后续的补充分析。

**候选方案**：

- **(A) 全部提交**，作为研究过程的一部分归档。
- **(B) 只提交有结论价值的，删掉中间产物**（比如那 4 个 IC 全为 `None` 的空跑报告）。
- **(C) 全部删掉**，保持仓库干净。
- **(D) 移到 `lab/` 或 `docs/research/` 下归档**，和已有的研究文档放一起。

**我不确定的点**：这是**项目所有者的研究过程记录**，我不清楚哪些结论最终被写进了报告、哪些只是中间探索。而且这个项目明确重视"诚实研究记录"（`lab/` 的整个设计哲学就是这个），所以轻率删除可能违背项目意图。

**另外**：`uv.lock` 未提交且**不在 `.gitignore` 里**——它应该被提交（锁文件的意义就在于进版本控制），还是删掉（承认 uv 只是临时用过）？

---

### Q10 其他较小的不确定项

| # | 问题 | 我的处理 | 不确定点 |
|---|---|---|---|
| a | `execution/engine.py::portfolio_snapshot()` 的 docstring 写 "Dict with **cash**, positions_value, unrealized_pnl, etc."，但返回的字典里**没有 `cash` 键** | 未改（只删了一个未使用的 `total_value = 0.0`） | 是 docstring 过时，还是真的漏了 `cash` 字段？调用方需要 cash 吗？ |
| b | `zip(*turnover_records)` 加了 `strict=False`（而非 `strict=True`） | 选 `strict=False` | 我刻意选保守值以**保证 lint 清理不改变行为**。但严格说这两个序列长度必然相等，`strict=True` 更能暴露问题。该不该用 True？ |
| c | `tools/run_market_structure_v3.py` 里 lambda 闭包捕获循环变量 `H`，我加了 `lambda x, H=H:` 显式绑定 | 已修 | 实际上 pandas 的 `rolling().apply()` 是**立即求值**的，所以原代码结果一直是对的。这属于"修了一个不影响结果的脆弱写法"——值不值得改动研究脚本？ |
| d | `.dockerignore` 排除了 `docs/` 和 `*.md`（保留 `README.md`，因为 `pyproject.toml` 的 `readme` 字段需要它） | 已加 | 会不会有运行时依赖某个 md 文件的情况？（我检查过没有，但不排除遗漏） |
| e | CI 触发范围从 `on: push`（所有分支）收窄为 `push: branches: [main]` + `pull_request` | 已改 | 如果有分支不建 PR 直接推，就不会跑 CI。对单人项目这是优化还是隐患？ |
| f | Docker 镜像仍以 **root** 运行 | 未改 | 生产级镜像应该用非 root 用户。但这会增加文件权限的复杂度，而且这个镜像主要是本地/demo 用途。值不值得改？ |
| g | CI 矩阵只有 3.11 / 3.12，没有 3.13 / 3.14 | 未加 | 加新版本能提前发现兼容问题，但依赖（numpy 1.26.2 等）大概率没有 3.13/3.14 的 wheel，加了必然红。所以实际上**被依赖锁死了 Python 版本上限**——这算不算 Q3 升级依赖的又一个理由？ |

---

## 3. 我最想得到的建议（按优先级）

1. **Q1**：`LiveEngine` 这个 bug，修复方向选哪个？以及"设计成持久进程、实际被当单日模拟器用"这个矛盾，是不是才是真正的问题？
2. **Q3 + Q5 + Q10g**：这三个其实是同一个问题——**依赖版本被锁死在 2023 年**。对一个以"研究可复现性"为核心价值的项目，升级依赖 vs 保留旧版本，该怎么权衡？
3. **Q2**：测试污染研究 Registry，该修到什么程度？已污染的记录要不要清理？
4. **Q7**：单人公开项目，分支保护是纪律还是负担？
5. **Q4**：276 个文件的格式化，一次性做掉还是只约束新代码？
6. **Q8**：我那三处行为变更有没有过头的地方？

---

## 4. Q1 拆出的两个独立 issue（草稿，尚未提交到 GitHub）

> **公开措辞红线**（裁定明确要求）：这个问题**公开建到 GitHub 是合适的**（public repo，
> 且已有完整证据链），但表述只能停在
> **"如果 `LiveEngine` 按设计的跨日生命周期运行，这条路径会触发资金不守恒"**。
> **不得**写成"已造成真实资金损失"或任何暗示生产事故的说法 —— 目前没有证据支持后者，
> 而且现有 daemon 调用方式根本走不到这条路径。

按裁定，Q1 不并入 CI PR，拆成两个必须一起看的 issue。

### Issue A — `LiveEngine` loses expired-position proceeds

**Severity**: **P1 — High**
**Location**: `trading/live_engine.py::_execute_rebalance`
**Related to**: <!-- Issue B -->（见下方，两者互相关联，但各自独立成立）

到期仓位被直接从 `self.state.positions` 过滤掉，没有卖出、没有回款、没有扣成本：

```python
self.state.positions = [p for p in self.state.positions if p.exit_date > date_str]
```

`_update_equity()` 只给仍在列表中的持仓估值，因此**到期仓位的市值从权益中凭空消失**。
全文件对 `cash` 的写入只有两处：买入扣减（`_execute_rebalance`）与 kill-switch 紧急平仓回款
（`_emergency_flat`）。正常调仓路径**没有任何卖出动作**。

**影响**：资金不守恒 → 权益/收益/回撤/peak 全部失真 → 后续仓位规模错误 → 回撤熔断失效。

**建议方向**（短期 A + 中期 B，**不采用 C**）：
- 短期：到期时按当日价格结算，`cash += shares * px * (1 - cost)`
- 中期：把卖出抽成统一的内部执行路径，让"到期退出 / 风控退出 / 策略退出 / kill switch"
  全部汇聚到同一个 `_execute_sell()`
- **不采用**：把它降级成"单日模拟器"、删掉 `exit_date` 与状态模型 —— 代码里有 `equity_peak`、
  `kill switch`、`next_rebalance` 等大量信号，说明它被设计成跨交易日的持久状态机；
  为了迁就当前接线错误而删除产品设计属于反向修复

### Issue B — `daemon/runner.py` recreates `LiveEngine` every run

**Severity**: **P1 — High**
**Location**: `daemon/runner.py`
**Related to**: <!-- Issue A -->（见上方）

```python
self.engine = LiveEngine()            # 每次新建
self.engine.load_history(pipeline.returns, pipeline.get_close())
self.engine.run_once(last_date)       # 只调用一次
```

而 `LiveEngine.run_once()` 的契约是：`"""单日运行 (每次调用 = 一个交易日)."""`

**后果**（每次 daemon 运行都会丢失全部状态）：
- `positions` 重置为空
- `cash` 重置为 `INITIAL_CAPITAL`
- `equity_peak` 重置 → **在当前 daemon 每次重新实例化 `LiveEngine` 的调用方式下，依赖实例状态
  累计的跨日回撤峰值无法持续，因此 15% / 25% 回撤熔断按设计无法跨运行周期累计**

（注意措辞：不能写成"熔断永远无法触发"—— 那把它泛化成了一个未经证明的结论。
可证明的只是"无法跨运行周期累计"。）

**两者互相关联，但各自独立成立**（不是"必须一起修"关系的两个 bullet）：

- Issue A 即使在没有当前这个 daemon 的情况下**仍然是一个真 bug** —— 未来任何
  persistent runtime 调用者都会踩到
- Issue B 也独立成立 —— 它本身就是一个"没有实现持续 live runtime"的缺陷
- 反过来，**只修 A 不修 B**：修的是一个"当前调用方式碰不到、未来接持久 runtime 就爆炸"的 bug
- **只修 B 不修 A**：会让 A 立刻开始实际发生

所以两个都应记为 **P1**，互相 `Related to`，各自独立修复。

---

## 5. Q9 未跟踪文件的分类建议（等项目所有者决定）

裁定要求"不批量删除、要分类"。以下是按可判定性给出的分类，**尚未执行任何操作**：

| 文件 | 可判定性 | 建议 |
|---|---|---|
| `data/reports/run_1785833*.md` ×4 | **可判定**：`data_source=synthetic`、IC 字段全为 `None`、无 universe provider | 属于测试/空跑产物，可删 |
| `m4_backtest.py` | 已提交 | — |
| `m4_alpha_leakage.py` | **不可判定**：含真实 CSI300 数据拉取逻辑，是 M4 的 IC→收益泄漏审计 | 需所有者判断是否属于"最终研究证据" |
| `m4_backtest_lo.py` `m4_benchmark_attribution.py` `m4_complete.py` `m4_return_attribution.py` | **不可判定**：同上，M4 后续归因分析 | 同上；建议归档到 `docs/research/` 而非删除 |
| `uv.lock` | **可判定但不该提交** | 见下 |

### 关于 `uv.lock`

裁定建议"采用 uv 就把 `uv.lock` 提交"。方向同意，但**当前这个文件不能直接提交**：

它是在 2026-08-04 由本地 `.venv` 生成的，而那个 venv 是 **Python 3.14.6 + uv 精简依赖子集**
（numpy 2.5.1 / pandas 3.0.3，且缺 scipy / scikit-learn / cvxpy / matplotlib / sqlalchemy /
pytest-asyncio）。它**不代表项目的 research baseline**，把它锁进版本控制等于把一个错误的
环境固化下来。

**正确顺序（经第二轮审阅修正为 9 步，关键是第 5 步的"零差异门"）**：

```
1. 确定 canonical 依赖模型（长期：pyproject.toml → uv.lock）
2. 用 Python 3.12 构建干净环境
3. 保留当前 requirements.txt 作为【冻结约束】(constraints)
4. 生成 uv.lock
5. ★ diff：逐包确认没有未经批准的版本变化 ★
6. uv sync
7. pytest
8. 与 CI 结果对照
9. 才提交 uv.lock
```

> **为什么第 5 步是必须的**：`uv lock` **不保证**解析结果等于旧 `requirements.txt` ——
> uv 默认倾向最新的可满足版本。如果直接 `uv lock` 再提交，很可能得到
> `numpy 2.x / pandas 3.x`，这就是一次**隐性依赖变更**，直接违反 Q3 的冻结原则。
> 官方推荐的迁移路径是把旧的 locked requirements 作为 constraints 传入。
>
> 第 5 步必须明确回答：`direct dependency changed?` / `transitive dependency changed?` /
> `Python marker changed?` / `platform resolution changed?`
> —— **任何一项出现版本变化就停下调查，不要直接提交。**

### 关于已污染的 Registry

`data/trading.db` 的 `research_runs` 表在测试过程中从 38 条增长到 41 条（其中至少 3 条是本轮
测试写入的 synthetic 记录）。裁定意见是**不要盲删** —— 如果无法可靠识别哪些是测试记录，
保留并新增 `origin` 字段（`production | experiment | test`）区分，比误删研究资产更安全。
本轮**未对数据库做任何删除**。

---

## 附录：本次审计的完整验证输出

```
# ruff（与 CI 同版本 0.15.12）
$ ruff check .
All checks passed!

# ruff，仅在 git 跟踪的文件上（等价于 CI 实际看到的文件集）
$ git ls-files -z -- '*.py' | xargs -0 ruff check
All checks passed!

# mypy
$ mypy packages/quant_core/src --config-file=pyproject.toml
Success: no issues found in 6 source files

# 测试（改动前后对比）
改动前：34 failed, 1087 passed, 7 skipped, 126 errors
改动后：34 failed, 1092 passed, 7 skipped, 126 errors   (+5 = 新增的回归测试)
        → 失败集合完全一致，全部为本地缺包所致

# 回归测试有效性（把 bug 改回去验证）
$ sed -i 's/load_config(req.config)/_load_config(req.config)/' api/routes.py
$ pytest tests/test_api/test_screen_endpoint.py
FAILED test_screen_reaches_the_data_load_guard
FAILED test_screen_passes_request_config_path_through
2 failed, 1 passed

# .dockerignore 生效证据（Docker 构建日志）
# 修复前：COPY . . 会传输 .venv(350MB) + node_modules(100MB)
# 修复后：
#5 [internal] load build context
#5 transferring context: 366.93kB 0.1s done
```

### Docker 容器启动验证（已完成 ✅）

在本地环境网络恢复后，镜像构建成功并完成端到端验证。**镜像大小 1.71 GB。**

| 验证项 | 结果 |
|---|---|
| ① 原始 CI 失败点：`import quant_core` | ✅ `/usr/local/lib/python3.12/site-packages/quant_core/__init__.py` |
| ② `core.store → Store` 转发 shim 链路 | ✅ `<class 'quant_core.store.Store'>` ← **当初 CI 正是炸在这一行** |
| ③ `import xgboost`（换包后模块名不变） | ✅ `xgboost 2.1.4` |
| ④ CUDA 残留 | ✅ `nvidia-nccl` 未安装 |
| ⑤ 前端 dist（多阶段构建产物） | ✅ 已被服务，返回 `<html lang="en">...` |
| ⑥ Docker `HEALTHCHECK` 指令 | ✅ `healthy`（stdlib 探测，slim 镜像无 curl 的问题已解决） |
| ⑦ `/api/health` 响应 | ✅ 启动后 **5 秒**返回 `{"status":"ok",...}` |

此前的失败全部源于本机网络损坏下载（哈希不匹配 / `IncompleteRead` / pip `JSONDecodeError`），
与 Dockerfile 无关。

**本次改动规模**：91 个已跟踪文件修改（+688 / -415），5 个新文件
（`.dockerignore`、`.github/workflows/codeql.yml`、`.github/dependabot.yml`、
`tests/test_api/test_screen_endpoint.py`、`docs/AUDIT_OPEN_QUESTIONS.md`）。

**尚未推送**：所有改动都在本地工作区，**没有 commit、没有 push**。
