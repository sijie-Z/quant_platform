"""TradingKernel — alpha-v1.0 系统控制平面.

Single control plane. 唯一主循环. 失败自治.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("kernel")


@dataclass
class KernelState:
    """系统运行时状态."""
    running: bool = True
    mode: str = "LIVE"  # LIVE / SAFE / HALT
    last_tick: float = 0.0
    cycle_count: int = 0
    errors: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    equity: float = 10_000_000
    equity_peak: float = 10_000_000
    cash: float = 10_000_000
    n_positions: int = 0

    @property
    def drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return (self.equity_peak - self.equity) / self.equity_peak


class TradingKernel:
    """系统控制平面. 唯一主循环. 全系统串联."""

    def __init__(self, data_engine=None, signal_engine=None, risk_engine=None,
                 execution_engine=None, monitor=None, reconciliation=None):
        self.data = data_engine
        self.signal = signal_engine
        self.risk = risk_engine
        self.exec = execution_engine
        self.monitor = monitor
        self.recon = reconciliation
        self.state = KernelState()

    # ── 主循环 ──

    def run(self):
        """永不结束的主循环."""
        logger.info("TradingKernel started")
        while self.state.running:
            try:
                self._tick()
            except Exception as e:
                logger.exception("Kernel crash: %s", e)
                self.state.errors.append(str(e))
                self._enter_safe_mode()

    def _tick(self):
        """单次运行周期."""
        now = time.time()
        self.state.last_tick = now
        self.state.cycle_count += 1

        # 1. Data
        market_data = self.data.get_latest() if self.data else None

        # 2. Signal
        signal = None
        if self.signal and market_data is not None:
            signal = self.signal.generate(market_data)

        # 3. Risk
        risk_ok = True
        if self.risk and signal is not None:
            risk_ok = self.risk.check(signal, market_data)
        if not risk_ok:
            logger.warning("Risk block — skipping execution")
            if self.monitor:
                self.monitor.log_event("RISK_BLOCK")
            return

        # 4. Execution
        fills = None
        if self.exec and signal is not None:
            orders = self.exec.generate_orders(signal)
            fills = self.exec.execute(orders)

        # 5. Reconciliation
        if self.recon and fills:
            self.recon.update(fills)

        # 6. Monitoring
        if self.monitor:
            self.monitor.update(market_data=market_data, signal=signal,
                                fills=fills, state=self.state)

        # 7. Safety
        self._post_tick_safety()

    # ── 安全 ──

    def _post_tick_safety(self):
        if self.risk and self.risk.drawdown_exceeded():
            self._enter_safe_mode()
        if self.risk and self.risk.daily_loss_limit_hit():
            self._enter_halt()

    def _enter_safe_mode(self):
        logger.critical("SAFE MODE")
        self.state.mode = "SAFE"
        self.state.alerts.append("safe_mode")
        if self.exec:
            self.exec.cancel_all()

    def _enter_halt(self):
        logger.critical("HALT MODE")
        self.state.mode = "HALT"
        self.state.running = False
        self.state.alerts.append("halt")
        if self.exec:
            self.exec.cancel_all()

    def stop(self):
        """手动停止."""
        logger.info("Kernel stopping. Cycles: %d, Errors: %d",
                     self.state.cycle_count, len(self.state.errors))
        self.state.running = False

    def report(self) -> str:
        dd = self.state.drawdown
        return (
            f"\n{'=' * 60}\n"
            f"  TradingKernel — alpha-v1.0\n"
            f"{'=' * 60}\n"
            f"  Mode:      {self.state.mode}\n"
            f"  Cycles:    {self.state.cycle_count}\n"
            f"  Equity:    {self.state.equity:>12,.2f}\n"
            f"  Drawdown:  {dd*100:>10.2f}%\n"
            f"  Positions: {self.state.n_positions:>6d}\n"
            f"  Alerts:    {len(self.state.alerts)}\n"
            f"  Errors:    {len(self.state.errors)}\n"
            f"{'=' * 60}"
        )
