# Ops Runbook — 系统运维手册

> 系统挂了怎么办、数据断了怎么办、调仓失败怎么办。
> 不做优化，只做"活下来"。

> 入口说明：日常交易与 Web 服务分别用 `python main.py trade` 和
> `python main.py web`。`trading/live_engine.py` 与
> `trading/reversal_paper_trader.py` 是库模块（没有 `__main__`），直接
> `python trading/live_engine.py` 只会导入类，不会启动任何东西。

---

## 1. 系统启动

```bash
# 实盘/纸面交易（单进程；券商连接失败会直接退出，不会回退模拟盘）
python main.py trade --broker paper --days 30        # Paper Trading
python main.py trade --broker qmt --days 1           # QMT 实盘（需 miniQMT + xtquant；真实券商每次 1 个交易所日）
python main.py trade --broker qmt_live --days 1      # 显式 live 模式

# Web/API（必须单 worker，见第 11 节）
python main.py web

# 每日监控检查（脚本自带 __main__）
python monitoring/run_monitor.py
```

QMT 密码优先读环境变量 `QMT_PASSWORD`，账户/服务器/模式见
`config/default.yaml` 的 `execution.qmt` 段。`--broker qmt` 不再静默等价于模拟盘。
真实券商只在**交易所交易日**执行（非交易日直接拒绝），每次运行只支持 **1 个交易所日**，
且必须提供真实行情 feed（拒绝合成价格）。

## 2. 系统停止

```bash
# Ctrl+C 正常停止；CLI LiveRunner 会结束会话并 checkpoint SQLite WAL；
# Web API 的 LiveTradingEngine 与 daemon 的 LiveEngine 还会落盘/恢复 engine_state
# （CLI LiveRunner 不落盘/不恢复）
# 如果进程卡死（POSIX）:
pkill -f "main.py trade"

# Docker（QUANT_API_TOKEN 需在 .env 中，否则 compose 插值会报错）
docker compose stop        # stop_grace_period: 30s
```

## 3. 故障处理

| 故障 | 现象 | 处理 |
|------|------|------|
| 数据源断开 | log 显示 connection error / `DataFetchIncomplete` | 检查网络/切换备用数据源；真实源默认 fail closed，不要用 `--allow-synthetic-fallback` 掩盖 |
| 行情快照陈旧 / QMT tick 无有效 `timetag` | log 显示 `Skipping stale market snapshot` 或 `Dropping stale/undated QMT tick` | 本周期不下单；检查实时行情与 miniQMT。旧缓存不会重新标记为新鲜，也不会继续写入信号历史 |
| QMT 下单超时 / `submission outcome UNKNOWN` | 订单保持 PENDING，提示 `reconcile before retry` | **不要手工重复下单**。先查券商委托；系统会用 `strategy_name=quant_platform:<本地订单号>` 在 resync 时找回原委托，找回前该股票不生成新信号 |
| 持仓/账户/订单簿查询失败 | `UNKNOWN` / `BrokerConnectionError` | 跳过 sizing 与新信号；缓存账本只用于展示，不能作为空仓或可用现金依据。EOD 对账应为 unknown，而不是假通过 |
| 调仓失败 | 部分股票未成交 | 回测中未成交目标会顺延到下一个可交易日（`tradable_mask`）；实盘 LiveRunner 每周期重算信号，未成交余量按券商/模拟盘撤单规则处理（SimulatedBroker 会撤销未成交余量），必要时手动补单并记录偏差 |
| 净值异常下跌 | 单日跌幅 > 10% | 触发 kill switch，检查行情数据 |
| 系统崩溃 | 进程退出 | 重启，检查 log；Web API 的 `LiveTradingEngine` / daemon 的 `LiveEngine` 从 `engine_state` 恢复持仓/现金/Kill Switch；CLI `main.py trade` 每次新建 broker 账本，**不会恢复上一次会话**，必须与券商/Store 实际持仓对账 |
| 对账不一致 | 期望持仓 ≠ 实际持仓 | 以实际持仓为准，记录偏差原因；Web API 的 `LiveTradingEngine.start()` 会运行 HealthCheck（含持仓同步）；daemon 只做 SafetySystem 检查（不对比持仓）；CLI runner 不运行 healthcheck |
| 券商连接失败 | `Broker ... failed to connect`，退出码 1 | 检查 miniQMT/xtquant/账户；系统不会自动转模拟盘 |
| API 401/403 | `Invalid or missing API token` / `loopback-only` | 设置 `QUANT_API_TOKEN` 并在 UI Settings 填入同一 token |

## 4. 重启流程

```
1. 检查上次运行 log → 确认最后状态
2. 检查当前持仓 (与实际券商对账；Web API 的 LiveTradingEngine 或 daemon 的
   LiveEngine 会从 data/trading.db 的 engine_state 恢复，CLI `main.py trade` 不恢复)
3. 检查现金余额
4. 启动 python main.py web（由 API/daemon 管理引擎）或 python main.py trade（单次会话）
5. 确认系统输出正常；/api/health 返回 200
```

## 5. 调仓日操作流程

```
1. 开盘前: 确认数据源正常（healthcheck）
2. 盘中: 确认信号计算无异常
3. 收盘后: 检查成交情况（涨跌停/停牌导致的未成交会顺延）
4. 对账: 期望 vs 实际
5. 记录: 调仓日志
```

## 6. 紧急处理

| 场景 | 动作 |
|------|------|
| 系统失控 | 手动 kill switch（`POST /api/risk/kill-switch` 或 Monitor 面板）→ 全部平仓 |
| 券商 API 异常 | 暂停交易，转为手动模式 |
| 重大市场事件 | 暂停系统，等待市场稳定 |

## 7. 日常检查清单

- [ ] 数据源连接正常
- [ ] 昨日对账一致
- [ ] 净值未异常下跌
- [ ] 下次调仓日在预期范围内
- [ ] 系统 log 无异常报错
- [ ] `GET /api/health` 返回 200；`GET /api/readiness` 返回 200（503 时按返回的 checks 排查）
- [ ] 备份任务成功（见第 8 节）

## 8. 备份与恢复

### 需要备份的状态

| 路径 | 内容 | 可否重建 |
|------|------|----------|
| `data/trading.db` | 订单/持仓/成交/P&L/信号/会话/事件/配置快照/NAV/引擎状态/EOD 标记（11 张表） | 否 |
| `data/reports/` | 研究 Registry 生成的报告 | 部分 |
| `results/` | 回测输出（HTML/CSV/PNG/metrics.json） | 可重跑，但历史 run 不保证复现 |
| `.quant_versions/` | 配置版本快照 | 否（若未纳入外部版本管理） |
| `config/` | 当前配置 | 否 |
| `.quant_cache/`、`.cache/`、`.agent_cache/`、`.sentiment_cache/` | 缓存 | 可重建，可不备份 |

### 热备份（服务运行中）

SQLite 处于 WAL 模式，直接拷贝主文件可能拿到不一致快照。先 checkpoint，再用
SQLite 的在线备份接口（或短暂停服拷贝）：

```bash
# 1) 尽量让 WAL 落盘（服务运行中也可执行）
python -c "import sqlite3; c=sqlite3.connect('data/trading.db'); c.execute('PRAGMA wal_checkpoint(TRUNCATE)'); c.close()"

# 2) 用 sqlite3 .backup 生成一致性副本（先确保目标目录存在；需安装 sqlite3 CLI）
mkdir -p backups
sqlite3 data/trading.db ".backup 'backups/trading-20261020.db'"
```

没有 `sqlite3` CLI 时，最稳妥的做法是**停止写入进程**后拷走
`data/trading.db` 连同 `data/trading.db-wal`、`data/trading.db-shm`。

### 完整备份示例

```bash
mkdir -p backups/20261020
sqlite3 data/trading.db ".backup 'backups/20261020/trading.db'"
cp -r .quant_versions config results data/reports backups/20261020/
```

### 恢复

```bash
# 1) 停止服务（确保没有进程持有数据库）
# 2) 删除旧的 -wal/-shm，避免与新副本混用
rm -f data/trading.db-wal data/trading.db-shm
# 3) 拷回备份
cp backups/20261020/trading.db data/trading.db
cp -r backups/20261020/.quant_versions .quant_versions
# 4) 启动并验证
python main.py web
curl http://127.0.0.1:8000/api/health
```

建议在每个交易日收盘后（EOD 之后）做一次备份；至少保留最近 5 个交易日与 1 个周备份。

## 9. 升级与迁移

```bash
# 1) 先备份（第 8 节）
# 2) 更新依赖并重装本地包
pip install -r requirements.txt
pip install -e ./packages/quant_core
pip install -e .
# 3) 前端产物不随仓库分发，升级后重新构建
cd frontend && npm ci && npm run build && cd ..
# 4) 启动并验证
python main.py web
```

- **缓存 schema**：`utils/cache.py` 的 `DATA_SCHEMA_VERSION` 当前为 **5**。版本变化时旧缓存条目会被自动淘汰；也可以直接删除 `.quant_cache/`。
- **数据库 schema**：Store 使用 `CREATE TABLE IF NOT EXISTS` 与增量迁移（例如新增 `tenant_id`、`engine_state`、`eod_reconciliation` 列/表）。升级前必须备份；没有自动降级路径。
- **配置版本**：`python main.py config list`（另有 `show` / `diff` / `rollback` 子命令）；版本快照在 `.quant_versions/`。
- **CLI/API 兼容**：升级后先跑 `python main.py --help` 与 `pytest tests/ -q`。

## 10. 数据来源与 provenance

`python main.py run` 会在结果目录写出 `metrics.json`（`schema_version: 1`），其中：

| 字段 | 含义 |
|------|------|
| `provenance.requested_provider` | 配置请求的数据源 |
| `provenance.effective_provider` | 实际使用的数据源（回退时不同） |
| `provenance.synthetic` | 是否合成数据 |
| `provenance.start_date` / `end_date` | 请求区间 |
| `provenance.data_date` | 数据截止交易日 |
| `provenance.fallback_allowed` | 是否允许显式回退合成数据 |
| `provenance.failures` | 数据源初始化/加载失败列表 |
| `performance` / `risk` / `factors` / `stress_tests` | 回测指标、风险、因子 IC、压力测试 |
| `daily_returns` / `daily_return_dates` | 日收益序列 |

同样的 provenance 会写入 HTML 报告；控制台也会打印一行
`Data provenance: requested=... effective=... synthetic=... data_date=...`。
研究 run 的 validity 用 `python main.py runs list`（另有 `affected` / `mark` 子命令）管理。

## 11. 部署约束：单 worker

运行注册表、pipeline 并发上限、实盘引擎句柄、Kill Switch、OMS 账本和 WebSocket
客户端集合都是**进程内状态**。必须单 uvicorn worker 运行：不要加 `--workers N`
或 `--reload`。多 worker 会导致 run 404、重复下单、Kill Switch 只对一个 worker
生效、WS 事件只推给部分客户端、多个进程写同一个 SQLite。横向扩展需要先把这些
状态外置到 Redis/数据库（未实现）。详见 README「部署约束：单 worker」。

## 12. 告警与数据保留

### 告警

- 日志 sink 常开；设置 `ALERT_WEBHOOK_URL` 后会额外把告警以 JSON POST 到该地址
  （`ALERT_WEBHOOK_TIMEOUT` 默认 3 秒）。相同告警 5 分钟内去重；投递失败只记日志
  并增加 `alerts_webhook_failures_total`，不会阻断引擎周期。
- 告警来源：盘前 healthcheck 失败、调度器 EOD 异常、引擎/数据源严重事件。
- Prometheus `/api/metrics` + Grafana 16 面板模板（`monitoring/grafana_dashboard.json`）。
- `monitoring/run_monitor.py` 有告警时非零退出，适合 cron / 任务计划调用。

### 健康与就绪探针

| 端点 | 用途 |
|------|------|
| `GET /api/health` | liveness；休市不会失败，Docker healthcheck 用它 |
| `GET /api/readiness` | dead-man switch；200 ready / 503 degraded |

`/api/readiness` 的判断逻辑：引擎预期运行但超过 max(2×rebalance interval, 60 秒) 没有
cycle、引擎在跑但没有活着的 scheduler（或盘中 scheduler tick 超过 90 秒）、或行情数据
超过 2 个交易日未成功更新时返回 503；休市期间不要求数据更新。研究-only 部署
（未启动引擎）不会被判为不健康。

### 数据保留（retention）

```bash
# 归档并删除超过保留期的行（删除前写 dated CSV 到 archive 目录）
python main.py data retention --archive-dir ./archive

# 按表覆盖保留期（0 = 禁用该表）；CLI 参数优先于内置默认值
python main.py data retention --events-days 90 --signals-days 365 --trades-days 1825
```

注意：环境变量 `QUANT_RETENTION_*_DAYS`（ORDERS/TRADES/PNL/SIGNALS/SESSIONS/EVENTS/NAV）
**只被 scheduler 的每日 retention hook 读取**（`RetentionPolicy.from_env()`）；
`main.py data retention` 使用 CLI 参数，不读取这些环境变量。未知/非法值回退默认。

默认保留期：orders/trades/pnl 5 年，signals/sessions 1 年，events 180 天，NAV 10 年。
删除 `orders` 时还会检查是否仍有被保留的 `trades` 引用该订单，避免留下孤儿成交。
删除后默认 `PRAGMA wal_checkpoint(TRUNCATE)` + `VACUUM`（`--no-vacuum` 跳过）。
建议在 EOD 之后、备份之前运行 retention。

---

## 13. Forward paper 跟踪（只读，不动钱）

**目的**：用真实行情按冻结策略逐日记录模拟盘，积累真实前向记录；不接券商、不下单。

**策略与目标**
- EP+换手：`docs/research/paper_track/forward_targets/ep_turnover/`
- 换手单因子：`docs/research/paper_track/forward_targets/turnover/`
- 首批目标 as_of `2026-10-23`，执行日 `2026-10-26`；此后每个**交易所月末最后交易日**收盘后自动生成下一期目标（`daily_signal --prev-weights` 链接上一期）。

**每日流程**（工作日收盘后）
```bash
# 抓新浪数据到当天 → 重建 merged panel → 月末生成新目标 → 重建两个滚动台账
python tools/forward_update.py
```
- 数据：`.scratch/wide_sina_2026/stocks/`（300 只、后复权、含 amount/turnover）；
- 合并面板：`.scratch/orchestrate/merged_wide.parquet`（估值按日 ffill，无 bfill）；
- 台账：`docs/research/paper_track/forward_2026/`（EP+换手）与 `forward_2026_turnover/`（换手）；
- 成本：佣金 0.03%（最低 5 元）、印花税按日期、平方根冲击；原始价由 `paper_track --raw-price-mode auto` 自动识别（新浪面板用 `amount/volume`）。

**自动化**：桌面端 Scheduled 任务 `quant-forward-paper-daily`，工作日 16:20 运行；失败写入 `.scratch/orchestrate/forward_update_failures.log`。月末目标生成前会抓取当日 ST 名单并强制排除；快照或目标生成失败、任一台账子进程失败时，主脚本以非零状态退出（fail closed），不得把旧台账当成当日更新成功。若日志出现 `merged cache ends <旧日期> ... provider has not published the current close yet`，说明数据源尚未发布当日收盘数据（2026-10-26 15:20 的新浪日线即仍停在 10-23），等待下一次运行即可，不要手工伪造当日行或放宽检查。周末/节假日按最近交易日校验，月末 Friday 不会因周末运行而被跳过。个股与合并面板均先写临时文件再原子替换；**非空但落后于当日的响应同样视为失败并重试**，不得覆盖旧的好数据。目标/当前持仓资产刷新失败、月末任一资产刷新失败、或超过 5 只刷新失败时退出 2，不把混合日期缓存用于调仓、基准或台账。补生成历史月末时只使用该月末之前的权重，拒绝用未来权重链接。

**v3.1 月末加固（2026-10-28）**：
- **月末交易日开跑就先抓当日 ST 名单并落盘（在任何数据门禁之前）**。即使当天门禁失败退出 2，之后运行仍可用 `st_codes_<月末>.txt` 补生成目标；该文件只会由全量覆盖校验通过的抓取写入，不要删除 `.scratch/orchestrate/` 下的 `st_codes_*.txt`。
- ST 快照抓取每次 90 秒超时、至多 2 次尝试，失败在进程内记忆 10 分钟；每个进程至多一次成功全市场分页。全部失败但当日已有同日成功快照时复用之（仍为当日 PIT，复用前校验文件形状：六位代码且数量 ≥ max(10, 股票池 10%)），否则 fail closed。名单数量下限为覆盖资产的 0.5%。
- `refetch` 的失败资产会再用实时快照复核停牌：快照必须是全市场帧（≥100 个代码）、≥50% 的代码当日有成交、该代码所有行（含重复行、无缺失值）成交量为 0、且重分类数量 ≤ 上限时，才保留缓存最后收盘价、不计失败；任何一项不满足仍计失败。零成交量或部分陈旧的降级快照无法绕过月末门禁。
- 目标文件只认可解析、`as_of` 与文件名一致、权重非空的版本；损坏文件会被重新生成，month-end 扫描忽略损坏文件。`daily_signal` 输出为原子替换，不会留下半截目标。
- 月度目标被拒时台账照常重建（按现有持仓估值），不会因换仓失败丢掉当日 NAV。

**盘中实时监控（仅参考，不是正式 NAV）**：
```bash
python tools/forward_realtime.py                         # 单快照
python tools/forward_realtime.py --interval 300 --count 12  # 每 5 分钟，共 12 次
```
该工具用新浪实时 `最新价/昨收` 计算盘中参考值，输出仅写入 `.scratch/forward_realtime/`。正式 `summary.json` 含 `book` 后，它按**实际持仓、参考价和现金**相对正式收盘 NAV 重估（`basis=official_book`）；没有该字段时才回退到目标权重参考（`basis=target_weights`）。它不包含新的执行成本、不改变“次日收盘执行”的冻结规则，也不会写入或覆盖 `forward_2026*` 台账。单次抓取失败只记录 `errors.log` 并等待下一周期，不发布旧快照；连续 3 次失败才停止。

**首次运行前**：在 Scheduled 页点一次 “Run now” 预授权 Bash 等工具，否则可能停在权限确认。

**已知限制**：宽池只含当前上市股票（幸存者偏差）；PE/PB 为半月频 ffill（≤3 周旧值）；行业映射静态；权重层面不取整手；这是模拟跟踪，不是实盘成交。
