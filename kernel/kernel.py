"""TradingKernel — alpha-v1.0 系统控制平面.

唯一主循环. 单一事实来源. 失败自治.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 系统状态 ──

class SystemMode(Enum):
    NORMAL = "normal"
    SAFE = "safe"       # 降级运行 (只监控, 不交易)
    HALT = "halt"       # 停止交易, 等待恢复
    RECOVERY = "recovery"  # 正在恢复

class SystemStatus(Enum):
    INIT = "init"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class SystemState:
    """系统单一事实来源."""

    # 时间
    market_time: datetime | None = None
    system_time: datetime | None = None
    broker_time: datetime | None = None

    # 资金
    cash: float = 10_000_000
    equity: float = 10_000_000
    equity_peak: float = 10_000_000
    daily_pnl: float = 0.0

    # 持仓
    n_positions: int = 0
    gross_exposure: float = 0.0

    # 状态
    mode: SystemMode = SystemMode.NORMAL
    status: SystemStatus = SystemStatus.INIT
    last_cycle: str = ""
    cycle_count: int = 0
    errors: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)

    # 调仓
    last_rebalance: str = ""
    next_rebalance: str = ""
    days_to_rebalance: int = 0

    def update_equity(self, total: float):
        self.equity = total
        if total > self.equity_peak:
            self.equity_peak = total

    @property
    def drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return (self.equity_peak - self.equity) / self.equity_peak

    def summary(self) -> str:
        return (
            f"State | mode={self.mode.value} status={self.status.value} "
            f"equity={self.equity:>.0f} dd={self.drawdown:.2%} "
            f"cycle={self.cycle_count} errors={len(self.errors)}"
        )


# ── 时钟 ──

@dataclass
class SystemClock:
    """系统时间一致性管理器."""

    market_offset: float = 0.0  # market_time - system_time
    broker_offset: float = 0.0  # broker_time - system_time

    def sync(self, market_time: datetime | None = None, broker_time: datetime | None = None):
        now = datetime.now()
        if market_time:
            self.market_offset = (market_time - now).total_seconds()
        if broker_time:
            self.broker_offset = (broker_time - now).total_seconds()

    @property
    def market_now(self) -> datetime:
        return datetime.now() + timedelta(seconds=self.market_offset)

    @property
    def broker_now(self) -> datetime:
        return datetime.now() + timedelta(seconds=self.broker_offset)


# ── 内核 ──

class TradingKernel:
    """系统控制平面. 唯一主循环. 决策中枢."""

    def __init__(self, initial_capital: float = 10_000_000):
        self.state = SystemState(cash=initial_capital, equity=initial_capital)
        self.clock = SystemClock()
        self.components: dict[str, Any] = {}
        self._running = False
        self._cycle_interval = 60  # seconds
        self._last_cycle_time: float = 0.0

    # ── 组件注册 ──

    def register(self, name: str, component: Any):
        """注册系统组件."""
        self.components[name] = component
        logger.info("Component registered: %s", name)

    # ── 主循环 ──

    def cycle(self) -> dict[str, Any]:
        """一次完整运行周期."""
        result: dict[str, Any] = {"status": "ok", "alerts": []}

        try:
            self.state.system_time = datetime.now()
            self.state.cycle_count += 1
            self.state.last_cycle = self.state.system_time.strftime("%Y-%m-%d %H:%M:%S")

            # 1. 数据
            data = self._get_data()
            if data is None:
                self._transition(SystemMode.SAFE, "data_unavailable")
                result["alerts"].append("data_unavailable")
                return result

            # 2. 策略
            engine = self.components.get("engine")
            if engine:
                engine.run_once(data.get("last_date"))
                self.state.n_positions = len(engine.state.positions)
                self.state.cash = engine.state.cash
                self.state.update_equity(engine.state.current_equity)

            # 3. 风控
            safety = self.components.get("safety")
            if safety:
                checks = safety.check_all(
                    equity=self.state.equity,
                    peak_equity=self.state.equity_peak,
                    cash=self.state.cash,
                    daily_pnl=self.state.daily_pnl,
                    positions={},
                )
                failed = [c for c in checks if not c.passed]
                if failed:
                    result["alerts"].extend(f"{c.name}: {c.message}" for c in failed)
                    if safety.kill_switch_triggered:
                        self._transition(SystemMode.HALT, "kill_switch")
                        result["status"] = "killed"
                        return result

            # 4. 对账
            recon = self.components.get("reconciliation")
            if recon:
                pass  # 需要 broker 接入后启用

            # 5. 状态检查
            dd = self.state.drawdown
            if dd > 0.25:
                self._transition(SystemMode.SAFE, f"drawdown_warning:{dd:.2%}")
                result["alerts"].append(f"drawdown={dd:.2%}")

            self.state.status = SystemStatus.RUNNING
            self.state.mode = SystemMode.NORMAL

        except Exception as e:
            logger.error("Cycle failed: %s\n%s", e, traceback.format_exc())
            self.state.errors.append({
                "time": str(datetime.now()),
                "error": str(e),
                "trace": traceback.format_exc(),
            })
            self._transition(SystemMode.RECOVERY, str(e)[:100])
            result["status"] = "error"

        return result

    def run_forever(self, interval: int = 3600):
        """永不结束的主循环."""
        self._running = True
        self._cycle_interval = interval
        self.state.status = SystemStatus.RUNNING

        logger.info("Kernel started. Interval: %ds", interval)

        while self._running:
            try:
                self.cycle()
                time.sleep(self._cycle_interval)
            except KeyboardInterrupt:
                logger.info("Kernel stopped by user")
                self.stop()
                break
            except Exception:
                logger.error("Kernel crashed: %s", traceback.format_exc())
                self._recover()

    def stop(self):
        """停止系统."""
        self._running = False
        self.state.status = SystemStatus.STOPPED
        logger.info("Kernel stopped. Cycles: %d, Errors: %d",
                     self.state.cycle_count, len(self.state.errors))

    # ── 内部 ──

    def _get_data(self) -> dict | None:
        """获取最新数据."""
        data_rel = self.components.get("data")
        if data_rel:
            try:
                result = data_rel.connect()
                if result is not None:
                    return {"last_date": datetime.now()}
            except Exception as e:
                logger.warning("Data unavailable: %s", e)
                return None

        # 从 engine 获取数据
        engine = self.components.get("engine")
        if engine and engine._returns is not None:
            return {"last_date": engine._returns.index[-1]}
        return None

    def _transition(self, mode: SystemMode, reason: str):
        """模式切换."""
        old = self.state.mode
        self.state.mode = mode
        logger.info("Mode transition: %s -> %s (%s)", old.value, mode.value, reason)
        self.state.alerts.append({
            "time": str(datetime.now()),
            "from": old.value,
            "to": mode.value,
            "reason": reason,
        })

    def _recover(self):
        """崩溃恢复."""
        self._transition(SystemMode.RECOVERY, "auto_recovery")
        logger.info("Recovery attempt...")
        time.sleep(5)
        self.state.errors = []
        self._transition(SystemMode.NORMAL, "recovered")

    # ── 报告 ──

    def report(self) -> str:
        dd = self.state.drawdown
        return (
            f"\n{'=' * 60}\n"
            f"  TradingKernel — alpha-v1.0\n"
            f"{'=' * 60}\n"
            f"  Status:      {self.state.status.value}\n"
            f"  Mode:        {self.state.mode.value}\n"
            f"  Equity:      {self.state.equity:>12,.2f}\n"
            f"  Drawdown:    {dd*100:>10.2f}%\n"
            f"  Positions:   {self.state.n_positions:>6d}\n"
            f"  Cycles:      {self.state.cycle_count:>6d}\n"
            f"  Alerts:      {len(self.state.alerts):>6d}\n"
            f"  Errors:      {len(self.state.errors):>6d}\n"
            f"  Next rebal:  {self.state.next_rebalance}\n"
            f"  Mode hist:   {','.join(a['to'] for a in self.state.alerts[-5:])}\n"
            f"{'=' * 60}"
        )
