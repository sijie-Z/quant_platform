# QMT / Forward 安全演练与明日运行手册（2026-10-27）

## 当前结论

- **模拟故障注入演练：7/7 通过**。报告：`docs/research/QMT_SAFETY_DRILL_LATEST.json`。
- QMT 专项测试：**118 passed**；全工具回归：**149 passed**。
- 本机当前 `xtquant` 未安装，也没有运行中的 miniQMT，因此**尚未完成真实券商认证演练**，不能把下面的模拟通过理解为实盘放行。
- 明日默认“正式跑”指 **正式 forward paper：真实行情、真实目标、模拟成交、逐日台账**，不连接券商、不下真实订单。

## 今日已通过的演练

| 场景 | 结果 |
|---|---|
| 陈旧/缺失/未来行情与 QMT 上海时区时间戳 | 旧 tick 拒绝，UTC 主机不误判上海时间 |
| XtOrder 累计快照 + XtTrade 去重与 VWAP | 200 股不会被算成 300 股；两笔 10/11 元成交 VWAP=10.5 |
| 部分撤单与迟到成交 | 模糊单笔回报不猜增量；更大的累计快照可单调补记，订单仍保持 cancelled |
| 下单超时 + 并发 resync | 只保留一个本地意图，恢复原 broker_order_id，无重复 PENDING |
| Kill Switch 延迟卖出 | 连续两轮只发一笔卖出，不全量重复卖出 |
| 持仓/账户/订单簿查询失败 | 返回 unknown/抛错，不当作空仓或可交易现金 |
| 无效 ADV 与费用现金约束 | 零 ADV 固定滑点回退；费用纳入买入资金，NAV/现金只扣一次 |

运行：

```bash
python tools/qmt_safety_drill.py
```

输出：`docs/research/QMT_SAFETY_DRILL_LATEST.json` 中的 `all_passed=true`、`real_miniqmt_certified=false`。

## 明日正式 forward 怎么跑

### 1. 开盘前：代码与回归检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="packages/quant_core/src" python -m pytest tests/test_trading tests/test_live tests/test_tools -p no:codspeed --basetemp=.scratch/pytest/preopen -q
```

然后执行模拟演练，必须 7/7 通过：

```bash
python tools/qmt_safety_drill.py
```

若任一失败：不要改参数掩盖，不要生成新的正式台账，先定位原因。

### 2. 盘中：可选的实时持仓参考

```bash
python tools/forward_realtime.py --interval 300 --count 66
```

- 每 5 分钟一个快照，使用 `summary.book` 的实际持仓与现金。
- 输出只在 `.scratch/forward_realtime/`，**不是正式 NAV，不扣新的执行成本，不改变正式收盘规则**。

### 3. 收盘后：生成正式台账

```bash
python tools/forward_update.py
```

脚本会刷新 300 只股票、检查最近交易日缓存、在月末生成目标并更新两个台账。数据源未发布、目标/持仓资产刷新失败、ST 快照不完整时都会失败关闭，不会假装成功。

### 4. 独立读回

```bash
python .scratch/orchestrate/read_forward_results.py
```

读回会检查首日、最新台账日与缓存日、持仓数、成本、目标哈希。输出在 `.scratch/orchestrate/forward_readback_<日期>.json`。

正式文件：

- `docs/research/paper_track/forward_2026_turnover/{nav.csv,summary.json}`
- `docs/research/paper_track/forward_2026/{nav.csv,summary.json}`

首日之后才会有持仓期基准比较；不要用盘中 indicative NAV 替代它。

## 如果“正式跑”指真实券商下单

**明日不按这个含义放行。** 真实 miniQMT 还需要一套独立验收：

1. 安装并登录券商 miniQMT，确认 `import xtquant` 可用；账户与服务器配置只走环境变量/本地配置，密码不进仓库。
2. 先做只读连通性检查：真实持仓、现金、当日委托/成交可查询；接口失败时必须是 unknown，而不是空账户。
3. 在券商仿真/极小范围环境验证延迟成交、重复回调、断线重连、超时恢复、部分撤单和 EOD 对账；必须无重复委托、无重复成交、VWAP 正确。
4. 重启进程，确认原委托能够按本地意图 ID 找回；无法找回时禁止重试同一意图。
5. 真实 forward 达到 `FREEZE_MANIFEST_2026-10-25.md` 的评估门槛之前，仍不投真钱；当前研究证据没有证明可部署 alpha。

只有用户明确指定“真实账户、允许下单”后，才应另写一份带资金上限、标的范围、单日限制和人工 Kill Switch 的实盘执行单。**不要直接运行 `main.py trade --broker qmt` 作为明日默认动作。**
