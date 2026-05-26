# QMT Broker 接入 — 完成状态

**日期**: 2026-05-14  
**测试**: 1077 passed, 0 failed

---

## 架构

```
python main.py trade --broker qmt
         │
         ▼
    LiveRunner
    ├── 主 Broker: QMTBroker ──TCP── miniQMT ── 国金券商
    │   ├── order_stock / cancel / query
    │   ├── Callback → EventBus (qmt.fill, order.filled)
    │   └── 失败 → 自动回退 SimulatedBroker
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
| `trading/live_runner.py` | QMT 回退 + broker_type |
| `config/schema.py` | QMTConfig + ExecutionConfig |
| `config/default.yaml` | execution.qmt 段 |
| `main.py` | `trade` 子命令 |
| `tests/test_trading/test_qmt_broker.py` | 46 tests |

## 使用

```bash
# 需要先安装: pip install xtquant
# 需要先启动: miniQMT 客户端 + 登录

export QMT_PASSWORD="your_password"
python main.py trade --broker qmt --days 30
```

## 下一步

1. 安装 miniQMT + xtquant
2. `python main.py trade --broker qmt --days 5` 试跑
3. 跑 2-4 周后对比 QMT 真实成交 vs PaperBroker 仿真，校准 TCA 参数
