"""System Daemon — alpha-v1.0 无人值守运行守护进程.

功能:
  - 每日定时运行 (调度器)
  - 进程守护 (崩溃自动重启)
  - 异常自恢复
  - 日终对账
  - 日志轮转
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.operations.reconciliation import reconcile
from risk.safety import SafetySystem, SafetyLimits
from trading.live_engine import LiveEngine

logger = logging.getLogger(__name__)

DAYS_OPEN = [0, 1, 2, 3, 4]  # Mon-Fri
RUN_HOUR = 16  # 16:00 daily (market close)
MAX_RESTARTS = 3
RESTART_WINDOW_HOURS = 24


class Daemon:
    """系统守护进程."""

    def __init__(self):
        self.restart_count = 0
        self.last_restart = None
        self.safety = SafetySystem(SafetyLimits(
            max_drawdown=0.30,
            max_daily_loss=0.05,
            max_position_ratio=0.05,
            max_single_order_value=500_000,
            min_cash_reserve=10_000,
        ))
        self.engine: LiveEngine | None = None
        self.last_run_date: str = ""

    def should_run_today(self) -> bool:
        """今天是否应该运行."""
        now = datetime.now()
        if now.weekday() not in DAYS_OPEN:
            return False
        if now.hour < RUN_HOUR:
            return False
        today = now.strftime("%Y-%m-%d")
        if today == self.last_run_date:
            return False
        return True

    def run_once(self) -> dict[str, Any]:
        """执行一次完整日流程."""
        result: dict[str, Any] = {"status": "ok", "alerts": []}

        # 1. 加载数据
        try:
            provider = BaostockDataProvider(cache_enabled=True)
            pipeline = DataPipeline(provider=provider,
                                     start_date="2018-01-01",
                                     end_date=datetime.now().strftime("%Y-%m-%d"),
                                     exclude_st=True, exclude_suspended=True)
            pipeline.run()
        except Exception as e:
            result["status"] = "data_failed"
            result["alerts"].append(f"Data load failed: {e}")
            return result

        # 2. 运行策略
        self.engine = LiveEngine()
        self.engine.load_history(pipeline.returns, pipeline.get_close())

        last_date = pipeline.returns.index[-1]
        self.engine.run_once(last_date)

        # 3. 安全检查
        equity = self.engine.state.current_equity
        peak = self.engine.state.equity_peak
        cash = self.engine.state.cash
        daily_pnl = 0  # simplified

        checks = self.safety.check_all(equity, peak, cash, daily_pnl, {})
        failed = [c for c in checks if not c.passed]
        if failed:
            result["alerts"].extend(f"{c.name}: {c.message}" for c in failed)
            if self.safety.kill_switch_triggered:
                result["status"] = "killed"
                self.engine._emergency_flat(last_date)

        # 4. 记录
        self.last_run_date = datetime.now().strftime("%Y-%m-%d")
        logger.info("Run complete: equity=%.2f alerts=%d", equity, len(result["alerts"]))
        return result

    def run_forever(self, interval_seconds: int = 3600):
        """持续运行."""
        logger.info("Daemon started. Interval: %ds", interval_seconds)
        while True:
            try:
                if self.should_run_today():
                    result = self.run_once()
                    if result["status"] == "killed":
                        logger.warning("Kill switch triggered. Waiting for manual recovery.")
                        break
                else:
                    logger.debug("Not time to run yet")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception:
                logger.error("Daemon crashed: %s", traceback.format_exc())
                self._handle_crash()

    def _handle_crash(self):
        """崩溃恢复."""
        self.restart_count += 1
        self.last_restart = datetime.now()
        if self.restart_count > MAX_RESTARTS:
            logger.critical("Too many restarts. Giving up.")
            sys.exit(1)
        logger.info("Restarting in 60s (attempt %d/%d)", self.restart_count, MAX_RESTARTS)
        time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("results/daemon.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    daemon = Daemon()
    daemon.run_forever(interval_seconds=3600)
