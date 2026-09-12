# quant_platform 审计 — 第二轮评审请求

> **用途**：本文件是提交给外部审阅的第二轮材料。第一轮审阅（针对 `docs/AUDIT_OPEN_QUESTIONS.md`）
> 已经给出裁定，本文件汇报**裁定执行结果**＋**执行过程中新发现的事实**＋**需要重新裁定的问题**。
>
> **本文件可独立阅读**：§0 用最小篇幅重建全部上下文，不需要先读第一轮文档。
>
> 日期：2026-09-12　　代码基线：`main` @ `1caba6d`（未推送任何改动）

---

## §0 前情提要（重建上下文用）

### 0.1 项目

**quant_platform** —— 单人维护的 A 股多因子量化研究 + 交易平台，公开在 GitHub（8 stars）。
320 个 Python 文件 / 57,683 行；42 个 Vue/JS 文件；1,247 个测试用例；97 个 REST 端点；19 个 CLI 命令。

架构分两层：

1. **v1 平台层**：完整的研究→回测→实盘→Web 平台
2. **v2 研究 OS 层**（2026-07 起）：转向"诚实研究"，新增 `framework/`（Protocol 契约）与
   `lab/`（研究运行 + SQLite Registry）。核心原则是 **Truth First / Point-in-time /
   Knowledge compounds** —— 每次因子运行都把数据源、复权方式、PIT 状态、偏差警告写入
   `data/trading.db` 的 `research_runs` 表，报告里的告警由这些字段**机器生成**而非手写。

2026-08-05 的重构把 5 个共享模块抽成独立包 `packages/quant_core`，原位置留下 **17 个转发 shim**：

```python
# core/store.py（完整内容）
"""Compatibility shim: moved to quant_core.store."""
from quant_core.store import *  # noqa: F401,F403
from quant_core.store import Store  # noqa: F401
```

### 0.2 第一轮审计的起点

用户要求："GitHub 上最后一次 CI 是红的，修掉，并按大项目标准补齐 CI 基础设施。"

失败的根因：**Dockerfile 从未安装 `packages/quant_core`**，容器启动即
`ModuleNotFoundError: No module named 'quant_core'`。CI 的 `test` job 里有一行
`pip install -e packages/quant_core`，所以**只有容器能暴露这个问题**。

审计过程中还发现若干真实缺陷（`POST /api/screen` 调用未定义的 `_load_config`、
`get_execution_params()` 必然 TypeError、2 个 `tools/` 脚本用了 3.12+ 专有语法等），
连同 256 个 ruff 错误一并清理，并把 CI 重做（lint 硬门槛 / 3.11·3.12 矩阵 / CodeQL /
Dependabot / concurrency / least-privilege / timeout / 覆盖率 artifact / pip-audit 报告制）。

### 0.3 第一轮审阅的裁定（已收到）

第一轮文档列了 Q1–Q10 十个待决问题。收到的裁定要点：

| 裁定 | 内容 |
|---|---|
| **Q1** | `LiveEngine` 到期仓位不回款是**真 bug**，但真正的问题是**生命周期断裂**（类被设计成跨日持久状态机，唯一调用方却每次新建实例只跑一天）。**不并入 CI PR**，拆成两个独立 issue。**不采用**"降级成单日模拟器"的方案 |
| **Q2** | 测试污染 `research_runs` 违反项目自己的 Truth First 原则，**必须修**；采用"参数注入 `db_path`/`out_dir`"，测试传 `tmp_path`。**不要盲删**已被污染的记录 |
| **Q3** | **绝对不要在这次审计里升级 numpy/pandas**。依赖版本本身就是实验环境的一部分。应建立"冻结的 Research Baseline"，未来另开 modernization 并做 golden result 差分 |
| **Q4** | 全仓 `ruff format`（276 文件）正确但**必须独立成 PR**，不混入 |
| **Q5** | pip-audit 维持 report-only。**不要**搞永久白名单，也**不要**现在硬门槛 |
| **Q6** | 覆盖率**暂不设绝对阈值**，先测可信 baseline；建议未来只要求 changed-lines coverage |
| **Q7** | **应该现在就做 branch protection**，但不需要企业级重流程；Docker 是否 required 取决于它是不是部署产物 |
| **Q8a** | `None → All_other` 的降级**过度**，建议回退（会吞掉调用方 bug，对量化研究尤其危险）|
| **Q8b** | benchmark 里 `None → continue` **静默吞错**，建议回退 |
| **Q8c** | lambda `H=H` 闭包绑定**可以保留**，但归为 robustness cleanup，**不要声称修复了研究结果** |
| **Q9** | 未跟踪研究文件**不要批量删除**，要分类归档。`uv.lock` 若采用 uv 则**应提交** |
| **Q10a** | `portfolio_snapshot` 缺 `cash` —— 唯一正确做法是**搜索所有调用方**再定 |
| **Q10b** | `zip(strict=False)` 若 invariant 成立应改 `True` |
| **Q10f** | Docker 非 root 值得做，但独立提交 |
| **Q10g** | **不要**为了"看起来现代"加 3.13/3.14 矩阵；应明确声明支持上限 |

### 0.4 第二轮审阅的裁定（已收到，正在执行）

第二轮收到的裁定要点：

| 项 | 裁定 |
|---|---|
| **QA** | ✅ **保留显式 `TypeError`**，不做字节级回退。理由：第一轮的真实要求是"不能把异常输入吞成合法结果"，而不是"必须保留 NumPy 内部的报错实现" |
| **QB** | ❌ **明确否决选项 B**。理由：GitHub 的 "Require status checks" 是**合并门槛**语义，"Require pull request" 才是真正砍掉直推路径的规则 —— B 看似兼得直推与门禁，边界实际不清晰。**应采用 A 的单人轻量版**：Require PR + required checks（lint / test / **docker**）+ 禁止 force push + 禁止删除分支，**不要求 approving review** |
| **QB-Docker** | ✅ **Docker 必须设为 required**。理由不是"它是部署产物"，而是**它是本项目唯一验证 packaging boundary 的 CI 层** —— 这次事故正是 `pytest 绿 / Docker 红` |
| **QC** | ✅ 不提交现有 `uv.lock`；长期倾向 `pyproject.toml + uv.lock`。**关键纠正**：`uv lock` **不保证**自动等于旧 `requirements.txt` 的解析结果（uv 默认倾向最新可满足版本）→ 必须把旧 requirements 作为 **constraints** 生成 lock，再逐包 diff，**加一道"零差异门"**，有版本变化就停下调查 |
| **QD** | ✅ **前移**。但必须与 dependency modernization **切开**：`环境一致性现在做，依赖升级以后再做` |
| **QE** | 两个 issue 均为 **P1**（非 P0）；**两个独立 issue 互相 related**，不是"必须一起修才成立"；**公开建到 GitHub**，但只能写"设计路径会触发"，**不得声称已有真实资金损失** |
| **QF1** | 损坏 ACL 无普通用户态解，需管理员 |
| **QF2** | ✅ `RuntimeError` 接受 |
| **QF3** | ✅ coverage baseline 以 **CI 3.12 leg** 为 canonical |

**修正后的执行顺序（本轮严格遵守）**：

```
① 修完当前工作区的 Q2 / 审计整改              ✅ 已完成
② 建干净 Python 3.12 canonical venv          ← 进行中（见 §6）
③ 用当前 requirements.txt 完整安装（不升级）    ← 进行中
④ 在该环境重跑 pytest / coverage              ← 待办
⑤ 确认本次审计改动未引入真实 failure            ← 待办
⑥ 提交当前 CI/审计 PR                         ← 待用户授权
⑦ main 开 Require PR + lint/test/docker       ← 必须在 ⑥ 之后（见下方约束）
⑧ 建 Q1-A / Q1-B 两个 P1 issue                ← 待用户授权
⑨ 独立的 dependency-lock migration（带零差异门） ← 后续
⑩ 最后才考虑 numpy/pandas modernization        ← 后续
```

> **一条第 ⑦ 步的硬约束（裁定时未提及，但会影响可行性）**：
> GitHub 的 required status checks **只能从"已经在仓库上运行过的 check 名称"里选**。
> 本次重写的 CI 引入了全新的 job 名（`Lint & type check` / `Test (py3.11)` / `Test (py3.12)` /
> `Docker image` 等），这些名称**目前还不存在于 GitHub 上** —— 因为改动尚未推送。
> 所以顺序上 ⑦ **必然**在 ⑥ 之后，无法提前配置。

---

## §1 裁定执行结果

| 裁定项 | 执行动作 | 证据 |
|---|---|---|
| **Q2** | `run_synthetic_factor()` 增加 `db_path: str \| Path = DEFAULT_DB` 与 `out_dir: str \| Path = "data/reports"` 参数（默认值完全保持生产行为）。测试全部改传 `tmp_path`，并新增一条"不得写入真实 `data/` 树"的回归守卫 | `tests/test_lab/test_synthetic_factor_run.py`（2 → 4 个测试）；实测 `tests/test_lab/ tests/test_api/` **46 passed**，且 `data/reports/` 文件数**零新增** |
| **Q8a** | **见 §3-QA，已获裁定：保留显式 TypeError，不做字节级回退**。准确表述是：
**"保持异常语义和合法输入行为不变；仅将原先由 `np.isfinite(None)` 触发的隐式 TypeError
改为显式、可诊断的 TypeError。"** —— 不能写成"行为完全不变"，因为异常消息与抛出位置确实变了 | — |
| **Q8b** | 改为 `raise RuntimeError`。该函数在压测前已对全部 symbol 执行 `get_or_create()`，`None` 不可达。准确表述是：**"不可达异常分支仍保持 fail-fast 语义，仅将隐式 `AttributeError` 改为显式 `RuntimeError`，以表达内部 invariant violation。"** —— 同样不能写成"行为完全不变" | `packages/quant_core/src/quant_core/order_book.py` |
| **Q8c** | 保留，文档里明确标注为 robustness cleanup | — |
| **Q10a** | **调用点核实完成**：`strategy/portfolio_orchestrator.py:315` 只取 `positions_value / n_positions / total_unrealized_pnl / total_realized_pnl / total_pnl`，现金用它自己的 `cash_available` 字段；`tests/test_execution/test_engine.py` 只取 2 个字段。**无任何调用方读取 `cash`** → 判定为 docstring 过时，改为准确描述而非新增字段 | `execution/engine.py` |
| **Q10b** | 改 `strict=True`。`turnover_records` 由 `append((next_rdate, turnover))` 累积，`zip(*records)` 各序列长度必然相等 | `backtest/engine.py:244` |
| **Q10g** | `requires-python` 改为 `">=3.11,<3.13"`，并在 pyproject 里写明上限理由；README 徽章同步为 `3.11 \| 3.12` | `pyproject.toml`, `README.md` |
| **Q1 / Q3 / Q4 / Q5 / Q6** | 按裁定不并入本次改动。Q1 的两个 issue 草稿见 §4 | — |
| **Q9** | 未做任何删除。分类建议见 §5 | — |

**结论：第一轮裁定基本已全部落地，未扩大范围。**

---

## §2 执行过程中新发现的事实（这部分可能影响既有裁定）

### 2.1 ⚠️ 我此前报告给你的测试数字是**错的**

**起因**：我在本机跑 `pytest` 得到 `34 failed / 1092 passed / 126 errors`，并把 126 个 error
全部归因为"本地 venv 缺 scipy/sklearn/cvxpy/matplotlib"。**这个归因是错的。**

**真相**：本机 `C:\Users\admin\AppData\Local\Temp\pytest-of-admin` 的 **NTFS ACL 已损坏**
——连所有者都 `Access is denied`，`takeown /f ... /r` 也被拒绝，`icacls` 同样拒绝。
这导致**所有使用 `tmp_path` fixture 的测试在 setup 阶段就 ERROR，根本没有执行**。

用 `pytest --basetemp=<可写目录>` 绕过后的真实数字：

| | 我此前报告的 | **真实数字** |
|---|---|---|
| passed | 1092 | **1198** |
| failed | 34 | **49** |
| errors | 126 | **7** |
| skipped | 7 | 7 |

那 7 个 error 才是真正的 collection error（缺包）：

```
tests/test_backtest/test_engine.py            (sklearn)
tests/test_backtest/test_walkforward.py       (sklearn)
tests/test_factors/test_expression_engine.py  (scipy)
tests/test_factors/test_processing.py         (sklearn)
tests/test_reporting/test_dashboard.py        (matplotlib)
tests/test_risk/test_monte_carlo.py           (scipy)
tests/test_risk/test_var.py                   (scipy)
```

**多出来的 15 个 failure** 之前卡在 fixture setup 从未运行。它们的构成见 2.2。

> **这意味着**：此前任何基于"126 errors"的判断都需要重新审视。所幸第一轮裁定中
> 没有依赖这个数字的结论（Q3 的建议是"不要升级"，而新证据反而**加强**了它）。

### 2.2 ⚠️ Q3 从"抽象担心"变成"可量化破坏"——而且影响面比观测到的更大

那 15 个新增 failure **全部**来自 `tests/test_operations/test_investor.py`，根因是：

```
ValueError: Invalid frequency: M. Failed to parse with error message:
ValueError("'M' is no longer supported for offsets. Please use 'ME' instead.")
```

本地 venv 装的是 **pandas 3.0.3**，而项目 pin 的是 **2.1.3**。pandas 2.2 弃用 `'M'` 频率别名，
3.0 移除。

**关键是 `'M'` 出现在 4 处生产代码，不只是测试：**

| 位置 | 代码 |
|---|---|
| `operations/investor.py:163` | `df["nav_per_unit"].resample("M").last()` |
| `reporting/performance.py:111` | `strategy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)` |
| `api/routes.py:747` | `sr.resample("M").apply(lambda x: (1 + x).prod() - 1)` |
| `api/routes.py:748` | `monthly_ret.index.to_period("M")` |

**后两处所在的测试文件（`tests/test_reporting/test_dashboard.py`）因为缺 matplotlib 处于
collection error 状态，根本没执行到** —— 所以升级 pandas 的真实影响面**大于**观测到的 15 个失败。

这正好为 Q3 的"不要顺手升级"提供了实证：**它不是一个抽象的风险，而是已经可以数出来的破坏。**

### 2.3 49 个 failure 的完整归因

| 原因 | 数量 | 归属测试文件 | 性质 |
|---|---|---|---|
| pandas 3.x 移除 `'M'` 频率 | **15** | `test_operations/test_investor.py` | **本地依赖版本错**（见 2.2） |
| 缺 `pytest-asyncio`（async 测试不支持） | **14** | `test_core/test_event_bus_v2.py`(7) `test_core/test_message_bus.py`(5) `test_risk/test_healthcheck.py`(2) | 本地缺包 |
| 缺 `sqlalchemy`（`AttributeError: create_engine`） | **10** | `test_data/test_postgres_provider.py` | 本地缺包 |
| 缺 `scipy` | **7** | `test_factors/test_ic_monitor.py`(6) `test_factors/test_evaluation.py`(1) | 本地缺包 |
| 缺 `cvxpy` | **3** | `test_portfolio/test_optimizers.py` | 本地缺包 |
| | **49** | | |

（数字直接来自 `--tb=line` 的逐行归因，按来源文件分组统计，合计精确等于 49。）

**结论（严谨表述）**：

> 当前 49 个 failure 已按 traceback 逐条定位到依赖/环境不一致：**15 个**由 pandas 3.x 行为变更
> 触发，**14 个**由缺失 `pytest-asyncio`，**10 个**由缺失 SQLAlchemy，**7 个**由缺失 SciPy，
> **3 个**由缺失 CVXPY；**未发现与本次审计改动相关的 failure**。

（不能写成"全部是环境问题所以不是代码问题"—— 那是一个未经证明的泛化。）

### 测试健康度的正确表述

不要把本机这组数字当作项目的测试健康度，否则后来者看到 `49 failed` 会误以为审计后的项目是红的。正确表述：

```
CI canonical environment    → green（与 requirements.txt 一致）
local 3.12 clean env        → pending revalidation（见 §6）
current legacy .venv        → invalid / non-canonical
```

本机 `1198 passed / 49 failed / 7 skipped / 7 errors` 只是
**"当前 legacy 3.14 venv 的诊断结果"**，不构成对代码库的结论。

### 2.4 ✅ Docker 端到端验证已完成（第一轮文档里标注为"未完成"）

镜像构建成功，**大小 1.71 GB**，容器启动后 **5 秒**通过健康检查：

| 验证项 | 结果 |
|---|---|
| ① 原始 CI 失败点：`import quant_core` | ✅ `/usr/local/lib/python3.12/site-packages/quant_core/__init__.py` |
| ② `core.store → Store` 转发 shim 链路 | ✅ `<class 'quant_core.store.Store'>` ← **CI 正是炸在这一行** |
| ③ `import xgboost` | ✅ `2.1.4`（换包后模块名不变） |
| ④ CUDA 残留 | ✅ `nvidia-nccl` 未安装 |
| ⑤ 前端 dist（多阶段构建产物） | ✅ 已被服务，返回 `<html lang="en">...` |
| ⑥ Docker `HEALTHCHECK` 指令 | ✅ `healthy` |
| ⑦ `/api/health` | ✅ `{"status":"ok","timestamp":"...","runs_completed":0,"runs_active":0}` |

### 2.5 新发现：xgboost 在拖 561 MB 无用 CUDA（已修）

审计中查看 Docker 构建日志时发现：

```
Collecting nvidia-nccl-cu12 (from xgboost==2.1.4->-r requirements.txt (line 30))
  Downloading nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64.whl (342.1 MB)
```

查 PyPI 元数据确认 `xgboost==2.1.4` 在 Linux x86_64 上**硬依赖** CUDA 库：

```
nvidia-nccl-cu12; platform_system == "Linux" and platform_machine != "aarch64"
```

而本平台是**纯 CPU 的**（无任何 GPU 代码路径）。改用官方同版本 CPU-only 发行版后：

| | `xgboost`（GPU） | `xgboost-cpu` |
|---|---|---|
| wheel 本身 | **223.6 MB** | **4.5 MB** |
| CUDA 依赖 | +342.1 MB | 0 |
| **合计** | **565.7 MB** | **4.5 MB** |

**每次安装省 561 MB**；CI 有 2 个 matrix leg，即 **~1.12 GB/次**——这大概率也是 test job
此前要跑 5 分钟的主因之一。

**安全性验证**：我没有凭印象替换，而是用 HTTP range 请求直接读取了 wheel 的文件清单，
确认 `xgboost-cpu` 提供的同样是 `xgboost/__init__.py`（导入名不变）：

```
xgboost-cpu 2.1.4: xgboost_cpu-2.1.4-...whl
    xgboost/
    xgboost/VERSION
    xgboost/__init__.py
```

并在容器内实测 `import xgboost → 2.1.4` 成功（见 2.4 ③）。

> 注意：这是**替换同版本的 CPU-only 发行版**，不是升级版本，因此不违反 Q3 的"不要升级"。

### 2.6 其他已修项（第一轮文档 §1 已列，此处不重复）

Dockerfile 多阶段化 / `.dockerignore`（构建上下文从 ~450 MB 降到 366.93 kB）/
docker-compose healthcheck 用 curl 而镜像无 curl / `POST /api/screen` 的 `_load_config`
NameError / `get_execution_params()` 必然 TypeError / 2 个 tools 脚本的 3.12+ 语法 / ruff 256 → 0。

### 2.7 ⚠️ 新发现：「研究基线」目前**并没有被冻结**——72 个传递依赖完全浮动

在为 QD 建立 canonical 环境时核对 `requirements.txt`，发现一个此前所有讨论都默认错误的前提：

| 项 | 数量 |
|---|---|
| `requirements.txt` 里的依赖声明 | 30 行 |
| 其中用 `==` 精确锁定 | **29** |
| 未锁定的 | 1（`adata>=1.0.0`） |
| **Docker 构建实际安装的包** | **101** |
| **因此完全未锁版本的传递依赖** | **约 72 个** |

这个文件**不是 `pip freeze` 的产物**，而是一份手写的顶层依赖清单。未被锁定的传递依赖
（示例，全部来自实测构建日志）：

```
llvmlite  urllib3  starlette  joblib  threadpoolctl  blosc2  pytz
tzdata    certifi  charset_normalizer  idna  anyio  jinja2  rich
markdown-it-py  curl_cffi  akracer  scs  osqp  clarabel  numexpr
greenlet  cffi  pyparsing  requests  python-dateutil  ...（约 72 个）
```

**为什么这件事重要**：

1. **第一轮和第二轮裁定都建立在"当前 baseline 是冻结的、升级才有风险"这个前提上。**
   实际上前提不成立 —— 那 72 个传递依赖**任何时候都可能解析出新版本**，
   在没有 diff 的情况下悄悄改变运行时行为。
2. 这**不是理论风险**：本次审计中 Docker 构建就解析出了
   `llvmlite 0.46.0` / `scs 3.3.1` / `curl_cffi 0.16.3` / `akracer 0.0.14` ——
   这些都不在 `requirements.txt` 里，版本完全由解析当天决定。
3. 它让 QC 的"**零差异门**"从"迁移时的好习惯"变成**必须**：
   因为现状下根本不存在的约束，正是 uv.lock 要补上的东西。
4. 也意味着 **CI 的"绿"本身不是完全可复现的** —— 同一个 commit 在不同时间跑，
   可能装到不同的传递依赖。

**这带来一个新的、可能需要第三轮裁定的问题**：见 §3-QG。

---

## §3 本轮需要裁定的问题

### QA. Q8a 我做的不是"字节级回退"，需要你确认这是否可接受

裁定说 Q8a "建议回退"。我的理解是"不要改变语义"，于是做法是：

```python
# 原始代码（我的理解：语义是"缺输入就崩"，但异常来自 numpy 内部，消息晦涩）
tradability = features.get("tradability")
transition = features.get("transition")
if not np.isfinite(tradability) or not np.isfinite(transition):
    return "All_other"
# 传入 None 时 → np.isfinite(None) → TypeError: ufunc 'isfinite' not supported...

# 我改成的版本
if tradability is None or transition is None:
    raise TypeError(
        "classify_profile() requires 'tradability' and 'transition'; got "
        f"tradability={tradability!r}, transition={transition!r}"
    )
if not np.isfinite(tradability) or not np.isfinite(transition):
    return "All_other"
```

**相同点**：异常类型仍是 `TypeError`；合法（数值）输入的行为逐字节不变；不吞调用方 bug。
**不同点**：异常抛出的位置和消息变了；并且它同时给了 mypy 类型收窄，所以不需要 `# type: ignore`。

**问题**：这算满足"回退"的要求吗？还是应该做**完全字节级的原始回退**、然后用
`# type: ignore[arg-type]`（或把那几行排除在 mypy 门槛外）来让 lint 通过？

倾向：我认为前者更好（保留了"失败要响"的意图且诊断信息更强），但这偏离了字面裁定，请你定。

### QB. Q7 分支保护的粒度

裁定说"应该现在就做"，但需要确定粒度。**关键工作流背景**：这个仓库最近的 20 个提交
**全部是直接推 `main`** 的（作者身份是本地 git config `Builder <builder@quant-platform.local>`），
没有走 PR。加上 "Require PR" 会实质改变日常流程。

选项：

- **(A) 全套**：Require PR + Require status checks（lint、test）+ 禁 force push + 禁删除分支。
  最稳，但以后每次改动都要开分支、推、开 PR、等 CI、合并。
- **(B) 中等**：不改 PR 流程，只加 **禁 force push / 禁删除 main** + 要求 CI 通过（若可配）。
  保留直推，但阻止"最后一次提交是红的"这种状态被推送上去。
- **(C) 只做最小**：仅禁 force push。基本无摩擦。
- **(D) 暂不设**。

倾向：还没想好。这正是当初 CI 变红却没人拦住的原因，所以 (D) 我不同意；但 (A) 对单人项目的
日常摩擦有多大，我没有经验，请你判断。

另外：**Docker job 是否应该 required**？裁定说"取决于它是不是部署产物"。事实是：这个镜像目前
只是本地/demo 用途（`docker-compose.yml` 起一个 API 服务），没有部署到任何地方。但它恰恰是
**唯一能发现 `quant_core` 打包缺失的检查**（§0.2）——如果当初它是 required，这次的事故就会被
CI 拦住。这个矛盾请你裁。

### QC. `uv.lock` 的例外情况

第一轮裁定说"若采用 uv，`uv.lock` 应提交"。方向我同意，但**当前工作区里的这个 `uv.lock`
不能直接提交**：

它是 2026-08-04 由本地 `.venv` 生成的，而那个 venv 是 **Python 3.14.6 + uv 的精简依赖子集**
（numpy 2.5.1 / pandas 3.0.3，且缺 scipy / scikit-learn / cvxpy / matplotlib / sqlalchemy /
pytest-asyncio）。它**不代表项目的 research baseline**，提交它等于把错误环境固化。

我建议的正确顺序是：
1. 先确定 canonical 依赖来源（`pyproject.toml → uv.lock` 或 `requirements.in → requirements.txt`，**二选一，不要并存两套**）
2. 在 Python 3.12 环境用完整依赖重新 `uv lock`
3. 验证 `uv sync` 后 `pytest` 结果与 CI 一致
4. 再提交

**问题**：这个顺序对吗？以及——**在 Python 3.12 环境重新 lock 时，能不能保证 lock 出来的版本
与 `requirements.txt` 里 pin 的完全一致？** 如果 uv 会解析出不同的传递依赖版本，那这个 lock
本身就成了一次隐性的依赖变更，需要按 Q3 的原则走 golden-result 差分。我不确定 uv 的行为，请你判断。

### QD. 是否应该现在建立 canonical 3.12 环境？优先级怎么排？

§2.1–2.3 揭示的结果是：**本地环境的信号目前完全没有参考价值**——49 个 failure 里 27 个是缺包、
15 个是 pandas 版本错、7 个是缺 pytest-asyncio，**零个反映真实代码问题**。

这带来一个优先级问题：第一轮裁定把"建立冻结的 Research Baseline"放在比较后的层次（后续独立工作）。
但现在有证据表明：

- 本地跑测试得到的任何结论都不可信（包括我第一轮报告给你的数字，见 2.1）
- 任何人（包括未来的我）在这台机器上做任何判断都会踩同一个坑
- 而"建立 canonical 环境"**本身就是 research baseline 工作的第一步**

**问题**：要不要把"建立 3.12 canonical 环境并让本地测试结果与 CI 一致"提到 Q1 之前？
还是仍然维持裁定里的顺序（先 Q1 issue、再 baseline）？

我的倾向：**应该提前**，因为它是其他所有判断的前提条件。但这也可能是我在给自己"扩大范围"找理由，
请你判断。

### QE. Q1 两个 issue 的草稿（请审阅措辞与 Severity 定级）

草稿已写入 `docs/AUDIT_OPEN_QUESTIONS.md` §4，要点：

**Issue A — `LiveEngine` loses expired-position proceeds**

- Location: `trading/live_engine.py::_execute_rebalance`
- `self.state.positions = [p for p in self.state.positions if p.exit_date > date_str]`
  —— 到期仓位被直接过滤掉，无卖出、无回款、无成本
- `_update_equity()` 只给仍在列表中的持仓估值 → 到期仓位市值从权益中凭空消失
- 全文件对 `cash` 的写入只有 2 处：买入扣减、kill-switch 紧急平仓回款。正常调仓路径**无卖出动作**
- 影响：资金不守恒 → 权益/收益/回撤/peak 全失真 → 后续仓位规模错误 → 回撤熔断失效
- 建议方向：短期按当日价格结算并扣成本；中期抽出统一 `_execute_sell()`，让到期/风控/策略/kill-switch
  四类退出汇聚到同一路径。**不采用**"降级成单日模拟器"

**Issue B — `daemon/runner.py` recreates `LiveEngine` every run**

- `self.engine = LiveEngine()` → `load_history()` → `run_once(last_date)`，只调用一次
- 与 `run_once()` 的契约 `"""单日运行 (每次调用 = 一个交易日)."""` 直接矛盾
- 后果：`positions` / `cash` / `equity_peak` 每次重置 → **15%/25% 回撤熔断永远无法累计触发**
- A 与 B 必须一起修：只修 A 修的是"当前调用方式碰不到、未来接持久 runtime 就爆炸"的 bug；
  只修 B 会让 A 立刻开始实际发生

**问题**：Severity 定级用 P0 还是 P1？措辞有没有夸大或遗漏？以及——**这两个 issue 应该公开建在
GitHub 上（仓库是 public）还是先记录在本地文档里？** 项目所有者对此尚未表态。

### QF. 小型问题

**(1) 损坏的 `tmp_path` 目录怎么清理？**
`C:\Users\admin\AppData\Local\Temp\pytest-of-admin` 的 ACL 损坏，`takeown /f ... /r /d y` 与
`icacls` 都被 `Access is denied` 拒绝，普通删除也失败。这台机器上是否有非提权的清理方式？
还是只能让用户用管理员权限手动删？

**(2) 我的 Q8b 处理是否也偏离了字面裁定？**
裁定说"建议回退当前'静默跳过'"，我改成了 `raise RuntimeError`（而不是恢复成原来的
`AttributeError`）。理由：该函数在压测前已 `get_or_create()` 全部 symbol，`None` 不可达，
显式抛错能表达"这是程序逻辑错误"而不是"数据缺失"。但严格说这也是一种改写。

**(3) 覆盖率 baseline 应该在哪里测？**
第一轮裁定说"先测可信 baseline 再决定阈值"。但现在已知**本地环境不可信**（§2.1）。
所以 baseline 是否应该**只从 CI 的 3.12 leg 取**（CI 环境与 `requirements.txt` 一致，
且 test job 一直是通过的）？我倾向于这样，但不确定覆盖率在不同 Python 版本上是否可比。

### QG.（新，由 §2.7 推出）既然 baseline 从未被冻结，是否该插一个"零风险补锁"步骤？

§2.7 证明：当前 29 个顶层依赖被锁定，但 **72 个传递依赖完全浮动**。这意味着
**在 uv.lock 迁移完成之前，这个项目的研究结果在严格意义上都不是可复现的** ——
而"可复现"恰恰是它自己宣称的核心价值（Truth First / Knowledge compounds）。

原执行顺序把锁迁移放在第 ⑨ 步（比较靠后）。是否需要在它之前**插一个零风险中间步骤**：

> **在当前 canonical 3.12 环境里 `pip freeze` 出完整的 101 个包清单，
> 另存为 `requirements.lock.txt`，完全不改 `requirements.txt`。**

三个候选：

- **(a) 不插入**，按原顺序走。代价：在第 ⑨ 步完成前，任何研究结论都带着"依赖浮动"的隐患，
  而且这个窗口可能是几周甚至几个月。
- **(b) 插入 `pip freeze` 补锁**（我倾向这个）。理由：
  1. **零版本变更** —— `pip freeze` 只是把已经解析出来的版本记下来，不改变任何东西，
     不违反 Q3 的冻结原则
  2. **成本极低** —— 一条命令，不需要写代码
  3. **它正好是 QC"零差异门"所需要的对照基准** —— 第 5 步要 diff，就必须有一个"旧版本清单"，
     而现在没有。没有它，第 5 步无从执行
  4. 立刻把"可复现"这个属性拿回来
- **(c) 直接跳到 uv.lock 迁移**，省掉中间产物。代价：把"补锁"和"换锁定机制"两件事
  绑在一次变更里，一旦出问题不好二分定位。

**我的不确定点**：

1. `pip freeze` 会包含一些**不该进 lock 的东西**吗？（例如在 Windows 上解析出的
   `py-mini-racer`、平台专用 wheel 的差异，或者本地 editable 安装的 `quant-core`）
   —— 我不确定 `pip freeze` 的输出是否适合作为跨平台的 lock，还是只适合作为
   "本机快照 / 对照基准"。**如果它只适合做对照基准而不能当 lock，那 (b) 的定位需要调整。**
2. 如果采用 (b)，这个文件要不要进 git？还是只作为一次性的对照物？
3. (b) 会不会反而制造出"第三套锁定来源"，正是 QC 想避免的局面？

请你裁定。这个问题的答案会直接改变第 ⑥–⑨ 步的顺序。

---

## §4 附：可复核的证据

### 4.1 本次改动规模

```
91 files changed, +688 / -415
新增 5 个文件：
  .dockerignore
  .github/workflows/codeql.yml
  .github/dependabot.yml
  tests/test_api/test_screen_endpoint.py        (回归测试，验证过能抓到原始 bug)
  docs/AUDIT_OPEN_QUESTIONS.md
```

### 4.2 最终验证输出

```
ruff check .                    → All checks passed!              (256 → 0)
ruff check . (仅 git 跟踪文件)   → All checks passed!              (等价于 CI 条件)
mypy packages/quant_core/src    → Success: no issues found in 6 source files
pytest tests/ --basetemp=<可写>  → 1198 passed, 49 failed, 7 skipped, 7 errors
main.py --help                  → 19 个命令全部完好
Docker                         → 镜像 1.71 GB，容器 healthy，/api/health 5s 响应
```

### 4.3 回归测试有效性验证（把 bug 改回去）

```
$ sed -i 's/load_config(req.config)/_load_config(req.config)/' api/routes.py
$ pytest tests/test_api/test_screen_endpoint.py
FAILED test_screen_reaches_the_data_load_guard
FAILED test_screen_passes_request_config_path_through
2 failed, 1 passed
```

### 4.4 当前状态

**所有改动仍在本地工作区，未 commit、未 push。** 数据库未做任何删除。

---

## §6 Canonical 3.12 环境：建立与复验（执行中）

按裁定 QD 前移。**严格只做"环境一致性"，不做任何依赖升级** —— 使用当前
`requirements.txt` 原样安装。

### 6.1 环境构成

| 项 | 值 |
|---|---|
| Python | **3.12.14**（由 `uv python install 3.12` 安装；本机原先只有 3.14.6 与 uv 管理的 3.11.15） |
| venv 路径 | `.venv312/`（与旧的 `.venv/` 并存，互不影响） |
| 依赖来源 | **当前 `requirements.txt`，一个版本都不改** |
| 安装器 | `uv pip install`（比 pip 更快、重试更稳） |

### 6.2 为什么必须有这个环境

本机原有 `.venv` 是 **Python 3.14.6 + uv 精简子集**，与 `requirements.txt` 完全是两套东西。
它产生的 49 个 failure 中 34 个缺包、15 个是 pandas 版本错 —— **这个环境的输出不能作为
任何判断的依据**（§2.1 里我自己的第一轮报告就是被它污染的）。

### 6.3 复验结果 ✅

- [x] `uv pip install -r requirements.txt` 完整装成功
- [x] `pip install -e ./packages/quant_core` + `pip install -e .`（`quant-core==0.1.0` / `quant-platform==0.1.0`）
- [x] `pytest tests/`（配合 `--basetemp` 绕过本机损坏的 ACL）
- [x] 与 CI 结果对照：**确认本次审计改动未引入真实 failure**
- [x] 记录该环境下的基线

**结果**：

```
1313 passed, 1 skipped, 0 failed, 0 collection errors   (3 分 27 秒)
```

**环境对齐核对**（严格等于 `requirements.txt` 的 pin，无任何升级）：

| 包 | 版本 | | 包 | 版本 |
|---|---|---|---|---|
| numpy | 1.26.2 | | pytest | 7.4.3 |
| pandas | 2.1.3 | | pytest-asyncio | 0.21.2 |
| scipy | 1.16.0 | | xgboost-cpu | 2.1.4 |
| scikit-learn | 1.7.2 | | lightgbm | 4.5.0 |
| cvxpy | 1.7.5 | | shap | 0.46.0 |
| matplotlib | 3.8.4 | | fastapi | 0.128.0 |
| sqlalchemy | 2.0.36 | | **发行包总数** | **106** |

### 6.4 这个结果证明了什么

| 结论 | 证据 |
|---|---|
| legacy venv 的 **49 个 failure 全部是环境问题** | 同一份代码在 canonical 环境里 0 failed |
| **本次审计改动未引入任何回归** | 0 failed / 0 errors |
| 此前的 `1247 collected` 是被 7 个 collection error **藏住了 66 个测试** | canonical 环境实际收集到 1313 个 |
| §2.7 的"传递依赖未锁"得到二次印证 | `requirements.txt` 声明 30 个，实际安装 **106 个发行包** |

> 注意最后一点：**106 − 30 = 76 个传递依赖没有版本约束**。这与 §2.7 的估算一致，
> 也就意味着"在这个环境里跑出的绿"仍然**不是完全可复现的** —— 换个时间重装可能装到
> 不同的 `llvmlite` / `starlette` / `blosc2`。这正是 QC 零差异门与 QG 要解决的问题。

---

## §7 待用户授权的三个动作

以下动作属于对外或仓库设置变更，**均已准备好但未执行**：

| # | 动作 | 前置约束 |
|---|---|---|
| 1 | `git commit` + `push` 到 `main` | — |
| 2 | 配置 `main` 分支保护（Require PR + lint/test/docker + 禁 force push + 禁删除分支，不要求 review） | **必须在 1 之后**：required checks 只能从"已在仓库上运行过的 check 名称"中选择，而新 CI 的 job 名尚未存在于 GitHub |
| 3 | 建 Q1-A / Q1-B 两个 public issue（P1，互相 `Related to`，措辞不得声称真实资金损失） | 可在 1 之前或之后 |
