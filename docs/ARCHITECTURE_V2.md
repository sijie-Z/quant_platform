# MiQi / quant_platform Architecture v2

> **STATUS: PROPOSAL — not implemented. See #73.**
> Only `main.py factor <name>` is currently true; `live/` and `prod/` do not
> exist. The enforced boundary is `tests/test_architecture/test_core_boundaries.py`,
> not import-linter. The ten-claim audit is in `docs/HANDOVER.md` §四.

> 这是本仓库的架构决策页，不是讨论日志。后续所有 PR、重构、拆分都以这一页为判断标准。

## 1. Vision

**一句话定义（可改，但必须先有）：**

> MiQi 是一个 AI 原生的 A 股量化研究操作系统：一个人从数据、因子、回测到模拟交易，一个命令跑完，一个浏览器看完。

**用户 use case（按动作，不按岗位）：**

- 研究者：`main.py factor <name>` 跑因子，`main.py run` 跑回测，产出报告
- 开发者：实现一个 Factor，注册到 core，跑一次评估
- 交易员：`main.py web` 打开 Dashboard，Paper Trading 跑通，监控风控

## 2. Capability Map

```text
AI（横切，不占单独一层）
  ├── Research 可被 AI 驱动（因子挖掘、归因解释）
  ├── Risk 可被 AI 驱动（告警解释、压力叙述）
  └── Trading 可被 AI 驱动（信号生成、执行建议）

Data -> Research -> Portfolio -> Backtest -> Execution -> Risk -> Monitoring
                     └─────────── Platform（CLI / API / Web / Agent / Scheduler）
```

每个 Capability 只暴露稳定接口，所有入口（CLI、API、Web、Agent）共享同一套接口，不复制逻辑。

## 3. Dependency Rules

- `core/`（研究内核）：数据、因子、评估、回测、报告。不依赖 `live/`、`lab/`。
- `live/`（交易平台）：依赖 `core/` 的公开接口，不反向依赖。
- `lab/`（研究笔记）：只依赖 `core/`。
- AI 是横切能力：允许调用各 Capability，但不允许被 Capability 反向耦合。
- 跨层依赖由 `tests/test_architecture/test_core_boundaries.py` 强制（CI 通过 `pytest tests/` 运行），不是 import-linter。

## 4. Runtime

- CLI：`main.py`（研究、回测、web、factor 统一入口）
- API：FastAPI，`/api/*`
- Web：Vue Dashboard
- 后台：Paper Trading 循环、Scheduler
- 存储：SQLite（研究 Registry + 交易 Store），PostgreSQL 为生产可选

## 5. Repository Layout

当前阶段：**一个仓库，三层边界，先不物理拆分**。

> **现实（2026-10）**：`live/` 和 `prod/` 不存在；仓库的 `core/` 是 EventBus/Store/StateMachine，
> 不是下面所说的“研究内核”；`data/` 是纳入 git 的源代码，不是运行时数据。下面是目标布局，不是现状。

```text
quant_platform/
├── core/      研究内核（未来可拆成 quant-core）
├── live/      交易平台（未来可拆成 quant-live）
├── lab/       研究笔记和运行记录（未来可拆成 quant-lab）
├── api/       归入 live 的 Web 服务
├── frontend/  归入 live 的 UI
├── data/      运行时数据，不进 git
├── docs/      文档
└── tests/
```

目录是能力的物理形式，不是设计的替代品。接口稳定后，目录只做机械移动。

## 6. Migration Plan

1. P0：跑通研究者 use case（已完成离线验证）
2. P2：本文件落地 + 修掉悬空文档引用
3. P3：统一 `main.py factor <name>` 入口
4. P4：`main.py web` 起服务，Paper Trading 跑通（已验证：`/api/health` 200、前端首页 200、模拟/Paper 双轨跑通）
5. core 包化：live/lab 改为依赖安装版 core
6. 全部绿后，再物理拆成 quant-core / quant-lab / quant-live

## 原则（设计意图；ADR 目录不存在，见 #73）

- Principle 1：Truth First。数据调整方式、PIT 状态、偏差必须机器可读记录。
- Principle 2：Point-in-time。任何回测数据必须声明可用日期，禁止静默用未来数据。
- Principle 3：Knowledge compounds。研究运行、参数、评估全部进 Registry，可跨 run 查询。
- Principle 4：Protocol over Implementation。接口先锁，实现可换。
- ADR-0003：Single Live Runner。OMS、风控、实盘引擎各只有一条主实现。**（当前不成立，见 #73/#74）**
- ADR-0004：Protocol Before Plugin。新实现必须实现已有协议，不新增平行接口。**（当前不成立，见 #73/#74）**
