# QMT Broker 接入 — 完成状态

**日期**: 2026-05-14（历史快照；2026-10-20 更新 fail-closed 行为）  
**测试**: 当时 1,077 passed；当前套件见 `docs/HANDOVER.md` 第十轮。

> ⚠️ 注意：早期版本在 QMT 连接失败时会回退到 `SimulatedBroker`。**当前代码不会**
> ——`LiveRunner` 在 `broker.connect()` 返回 False 时抛 `RuntimeError` 并退出
> （fail closed），避免把实盘故障伪装成一次成功的模拟盘会话。

---

## 架构

```
python main.py trade --broker qmt --days 1
         │
         ▼
    LiveRunner
    ├── 主 Broker: QMTBroker ──TCP── miniQMT ── 国金券商
    │   ├── order_stock / cancel / query
    │   ├── Callback → EventBus (qmt.fill, order.filled)
    │   └── 连接/下单失败 → 中止并报错（fail closed，不回退模拟盘）
    │
    ├── PaperBroker (并行, TCA 对比)
    │
    └── 每日: DailyReport → 结束: SessionReport (Sharpe/MaxDD/年化)
```

## 新增/修改文件

| 文件 | 说明 |
|------|------|
| `trading/qmt_utils.py` | Symbol 映射、错误码、状态转换 |
| `trading/broker.py` | QMTBroker 重写 + BrokerRegistry 扩展 |
| `trading/live_runner.py` | QMT fail-closed 连接校验 + broker_type |
| `config/schema.py` | QMTConfig + ExecutionConfig |
| `config/default.yaml` | execution.qmt 段 |
| `main.py` | `trade` 子命令 |
| `tests/test_trading/test_qmt_broker.py` | 46 tests |

## 使用

```bash
# 需要先安装: pip install xtquant
# 需要先启动: miniQMT 客户端 + 登录

export QMT_PASSWORD="your_password"
python main.py trade --broker qmt --days 1   # 真实券商每次运行只支持 1 个交易所日
```

## 下一步

1. 安装 miniQMT + xtquant
2. `python main.py trade --broker qmt --days 1` 试跑（需真实行情 feed，不能喂合成价格）
3. 跑 2-4 周后对比 QMT 真实成交 vs PaperBroker 仿真，校准 TCA 参数
