# Trading System Runtime Kernel — alpha-v1.0 控制平面

> 把零件装成机器。从 "collection of components" 变成 "autonomous trading organism"。

---

## 1. 问题

当前系统状态：

```
signal.py → live_engine.py → monitoring/
                                    ↑ 各自独立
risk/safety.py → operations/reconciliation.py
                                    ↑ 没有联动
daemon/runner.py                    ↑ 只是定时器
```

问题：没有 control plane。没有单一主循环。模块各自为政。

---

## 2. 架构

```
┌─────────────────────────────────────────────────────────┐
│                     KERNEL                               │
│  (唯一主循环, 控制平面, 状态管理)                          │
│                                                         │
│  while market_is_open:                                   │
│    data = ingest()              # 数据层                │
│    check_consistency(data)      # 校验                  │
│    signal = alpha(data)         # 策略                  │
│    risk_check(signal)           # 风控                  │
│    orders = execute(signal)     # 执行                  │
│    reconcile(orders)            # 对账                  │
│    log_state()                  # 日志                  │
│    sleep(interval)                                       │
└─────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
    data/reliability   strategy/alpha   broker API
    operations/risk    monitoring       reconciliation
```

## 3. Kernel 定义

```python
class TradingKernel:
    """系统唯一主循环. 控制平面."""

    def __init__(self):
        self.state = SystemState()
        self.data = DataReliability()
        self.safety = SafetySystem()
        self.engine = LiveEngine()
        self.recon = Reconciliation()

    def cycle(self):
        """一次完整运行周期."""

        # 1. 数据
        data = self.data.connect()
        if not self.data.check_latency(data):
            self._failover_or_halt("data_lag")

        # 2. 策略
        self.engine.update(data)
        signal = self.engine.compute_signal()

        # 3. 风控
        checks = self.safety.check_all(...)
        if not self.safety.can_trade():
            self._halt("safety_limits_exceeded")
            return

        # 4. 执行
        orders = self.engine.generate_orders(signal)
        fills = self.broker.execute(orders)

        # 5. 对账
        report = self.recon.reconcile(
            expected=orders,
            actual=fills,
        )
        if not report.passes:
            self._alert("reconciliation_failed", report)

        # 6. 日志
        self.state.log_cycle(...)

    def run_forever(self):
        """永不结束的主循环."""
        while True:
            try:
                if self._should_cycle():
                    self.cycle()
            except DataFailure:
                self._recover_data()
            except BrokerFailure:
                self._recover_broker()
            except CriticalFailure:
                self._emergency_stop()
                break
            time.sleep(self._next_interval())
```

## 4. 缺失的总控

| 当前 | Kernel |
|------|--------|
| 模块各自独立 | 单一控制平面 |
| 没有全局状态 | SystemState (单一事实来源) |
| 失败无人处理 | 自动检测 + 分类 + 恢复 |
| 时间不一致 | 统一时钟 (market_time, system_time, broker_time) |
| 手动协调 | 自动编排 |

## 5. 实现计划

```
P0: Kernel core (cycle + state + run_forever)
P1: 整合现有的 data/reliability/safety/reconciliation
P2: Failure handling (detect → classify → recover)
P3: 替换 daemon/runner.py 为 kernel
```

## 6. 原则

- 一个主循环, 不分散
- 一个状态对象, 不复制
- 失败自动处理, 不等人
- 时间唯一来源, 不冲突
