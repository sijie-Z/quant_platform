"""System Daemon — alpha-v1.0 无人值守运行守护进程.

功能:
  - 每日定时运行 (调度器)
  - 进程守护 (崩溃自动重启)
  - 异常自恢复
  - 日终对账
  - 日志轮转

State handling (issue S18):
  - ``last_run_date`` is persisted next to the Store database in
    ``daemon_state.json``, so a process restart does not re-pull the same
    exchange day.
  - Every exchange day after the engine snapshot's ``last_processed_date`` is
    replayed (bounded by ``MAX_CATCHUP_DAYS``) so downtime does not leave
    silent gaps.
  - All scheduling uses the Asia/Shanghai exchange clock, independent of the
    host time zone; ``QUANT_DAILY_RUN_HOUR`` overrides the 16:00 run hour.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_platform.core.events import get_event_bus
from quant_platform.core.exchange_calendar import is_trading_day
from quant_platform.core.store import Store
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.risk.healthcheck import emit_alert, install_alert_handler
from quant_platform.risk.safety import SafetyLimits, SafetySystem
from quant_platform.trading.live_engine import LiveEngine

logger = logging.getLogger(__name__)

RUN_HOUR = max(0, min(23, int(os.environ.get("QUANT_DAILY_RUN_HOUR", "16"))))
MAX_RESTARTS = 3
RESTART_WINDOW_HOURS = 24
#: Upper bound on replayed exchange days after downtime.
MAX_CATCHUP_DAYS = max(1, int(os.environ.get("QUANT_DAEMON_MAX_CATCHUP_DAYS", "10")))

_STATE_FILENAME = "daemon_state.json"


class Daemon:
    """系统守护进程."""

    def __init__(self, store: Store | None = None,
                 strategy_id: str = "live_engine_alpha_v1",
                 state_path: str | Path | None = None,
                 run_hour: int | None = None):
        self.restart_count = 0
        self.last_restart: datetime | None = None
        self.safety = SafetySystem(SafetyLimits(
            max_drawdown=0.30,
            max_daily_loss=0.05,
            max_position_ratio=0.05,
            max_single_order_value=500_000,
            min_cash_reserve=10_000,
        ))
        self.engine: LiveEngine | None = None
        # Runtime state lives in the shared Store, so a restart (and a
        # separate process invocation) continues from the last committed day
        # instead of resetting cash to 10M and positions to empty.
        self.store = store or Store()
        self.strategy_id = strategy_id
        # The daemon runs in a container that is usually UTC; 16:00 must be
        # 16:00 in Shanghai.
        self._tz = ZoneInfo("Asia/Shanghai")
        self._run_hour = RUN_HOUR if run_hour is None else int(run_hour)
        if state_path is None:
            db_path = Path(getattr(self.store, "_db_path", "data/trading.db"))
            state_path = db_path.parent / _STATE_FILENAME
        self._state_path = Path(state_path)
        # The watermark survives restarts; the old in-memory value re-ran the
        # same day's data pull after every crash.
        self.last_run_date: str = str(
            self._load_state().get("last_run_date", "") or ""
        )
        # S4: an unattended daemon must have the production alert handler
        # wired, so health-check/system failures reach the sink.
        try:
            install_alert_handler(get_event_bus())
        except Exception:
            logger.warning("Could not install the alert handler", exc_info=True)

    # ── Persisted watermark ──

    def _load_state(self) -> dict:
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Daemon state file unreadable (%s); starting empty", exc)
            return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_run_date": self.last_run_date,
            "saved_at": datetime.now(self._tz).isoformat(),
        }
        tmp = self._state_path.with_name(f"{self._state_path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_path)
        except OSError as exc:
            logger.critical("ALERT: failed to persist daemon state: %s", exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def should_run_today(self) -> bool:
        """今天是否应该运行."""
        now = datetime.now(self._tz)
        if not is_trading_day(now.date()):
            return False
        if now.hour < self._run_hour:
            return False
        today = now.strftime("%Y-%m-%d")
        return today != self.last_run_date

    def run_once(self) -> dict[str, Any]:
        """执行一次完整日流程 (含停机期间的补跑)."""
        result: dict[str, Any] = {"status": "ok", "alerts": [], "days_processed": 0}

        # 1. 加载数据
        try:
            provider = BaostockDataProvider(cache_enabled=True)
            pipeline = DataPipeline(provider=provider,
                                     start_date="2018-01-01",
                                     end_date=datetime.now(self._tz).strftime("%Y-%m-%d"),
                                     exclude_st=True, exclude_suspended=True)
            pipeline.run()
        except Exception as e:
            # A daily data failure is an operational alert, not a debug line.
            logger.error("ALERT: daily data load failed: %s", e)
            emit_alert(
                "critical", f"daily data load failed: {e}", source="daemon",
                details={"error": type(e).__name__},
            )
            result["status"] = "data_failed"
            result["alerts"].append(f"Data load failed: {e}")
            return result

        if pipeline.returns is None or len(pipeline.returns.index) == 0:
            logger.error("ALERT: data pipeline returned an empty panel")
            emit_alert(
                "critical", "data pipeline returned an empty panel",
                source="daemon",
            )
            result["status"] = "data_failed"
            result["alerts"].append("Data pipeline returned an empty panel")
            return result

        # 2. 恢复持久化状态并运行策略
        self.engine = LiveEngine()
        # Money marks require the raw (traded CNY) panel; get_close() is the
        # adjusted research series and LiveEngine now refuses it (D15). A
        # legacy/test pipeline double without get_raw_close() falls back to
        # its close panel (the engine warns when the basis is unlabeled).
        try:
            history_prices = pipeline.get_raw_close()
        except AttributeError:
            history_prices = pipeline.get_close()
        except Exception as e:
            logger.error("ALERT: raw price panel unavailable: %s", e)
            emit_alert(
                "critical", f"raw price panel unavailable: {e}", source="daemon",
                details={"error": type(e).__name__},
            )
            result["status"] = "data_failed"
            result["alerts"].append(f"Raw price panel unavailable: {e}")
            return result
        self.engine.load_history(pipeline.returns, history_prices)

        previous_equity: float | None = None
        try:
            snapshot = self.store.get_engine_state(self.strategy_id)
        except (ValueError, TypeError, AttributeError) as exc:
            # `get_engine_state` decodes JSON itself; a corrupt row raised
            # before the guarded restore below and sent the daemon into a
            # crash/restart loop instead of reporting state_corrupt.
            logger.critical("Persisted engine state is unreadable: %s", exc)
            result["status"] = "state_corrupt"
            result["alerts"].append(f"Engine state unreadable: {exc}")
            return result
        if snapshot is not None:
            try:
                self.engine.restore_snapshot(snapshot)
                self.safety.kill_switch_triggered = bool(
                    snapshot.get("safety", {}).get("kill_switch_triggered", False)
                )
                previous_equity = float(snapshot.get("current_equity", 0.0))
            except (ValueError, TypeError, KeyError) as exc:
                # Do NOT overwrite an unreadable snapshot: it may be the only
                # record of the real book. Fail loudly and let an operator
                # inspect it instead of silently starting from cash.
                logger.critical(
                    "Persisted engine state is unreadable; refusing to overwrite it: %s", exc,
                )
                result["status"] = "state_corrupt"
                result["alerts"].append(f"Engine state unreadable: {exc}")
                return result

        if self.engine.state.status == "killed" or self.safety.kill_switch_triggered:
            result["status"] = "killed"
            return result

        # 3. Replay every exchange day since the persisted watermark.
        pending = self._pending_days(pipeline.returns.index, snapshot)
        if len(pending) == 0:
            # Nothing new since the watermark: a successful no-op invocation.
            # Still consume the day so run_forever does not re-pull the data
            # every interval when only the engine snapshot knows the date.
            logger.info("No unprocessed exchange days since the watermark")
            self.last_run_date = datetime.now(self._tz).strftime("%Y-%m-%d")
            self._save_state()
            return result
        if len(pending) > MAX_CATCHUP_DAYS:
            logger.warning(
                "ALERT: %d unprocessed exchange days; catching up the last %d only",
                len(pending), MAX_CATCHUP_DAYS,
            )
            result["alerts"].append(
                f"Catch-up bounded to {MAX_CATCHUP_DAYS} of {len(pending)} pending day(s)"
            )
            pending = pending[-MAX_CATCHUP_DAYS:]

        try:
            for day in pending:
                before = (
                    previous_equity
                    if previous_equity is not None
                    else self.engine.state.current_equity
                )
                self.engine.run_once(day)
                result["days_processed"] += 1

                # 4. 安全检查 (per replayed day)
                equity = self.engine.state.current_equity
                peak = self.engine.state.equity_peak
                cash = self.engine.state.cash
                daily_pnl = equity - before if before else 0.0
                previous_equity = equity

                checks = self.safety.check_all(equity, peak, cash, daily_pnl, {})
                failed = [c for c in checks if not c.passed]
                if failed:
                    for check in failed:
                        logger.error(
                            "ALERT: safety check failed [%s]: %s",
                            check.name, check.message,
                        )
                        emit_alert(
                            "error",
                            f"safety check failed [{check.name}]: {check.message}",
                            source="daemon",
                        )
                    result["alerts"].extend(
                        f"{c.name}: {c.message}" for c in failed
                    )
                    if self.safety.kill_switch_triggered:
                        result["status"] = "killed"
                        self.engine._emergency_flat(day)
                        break
        finally:
            # 5. 持久化：无论成功、异常还是熔断都要落盘。快照包含
            #    last_processed_date，因此重启后不会重复执行同一天。
            snapshot = self.engine.to_snapshot()
            snapshot["safety"] = {
                "kill_switch_triggered": self.safety.kill_switch_triggered,
            }
            try:
                self.store.save_engine_state(snapshot, self.strategy_id)
            except Exception:
                logger.critical(
                    "Failed to persist engine state: %s", traceback.format_exc(),
                )
                # Reporting "ok" with a stale snapshot loses a day after the
                # next restart (last_processed_date did not advance).
                result["status"] = "state_save_failed"
                result["alerts"].append("Failed to persist engine state")

        self.last_run_date = datetime.now(self._tz).strftime("%Y-%m-%d")
        self._save_state()
        logger.info(
            "Run complete: days=%d equity=%.2f alerts=%d",
            result["days_processed"], self.engine.state.current_equity,
            len(result["alerts"]),
        )
        return result

    def _pending_days(self, index, snapshot: dict | None) -> list:
        """Exchange days after the engine's last processed date.

        With no snapshot (first run) only the latest session is processed:
        replaying the full 2018+ history on first boot would be a very
        expensive surprise.
        """
        last_processed = ""
        if snapshot is not None:
            last_processed = str(snapshot.get("last_processed_date") or "")[:10]
        if not last_processed:
            return list(index[-1:])
        return [day for day in index if str(day)[:10] > last_processed]

    def close(self) -> None:
        """Persist state, flush the WAL and close the store (S16).

        Best-effort and idempotent: called from ``run_forever``'s finally and
        from the CLI shutdown path, so a SIGTERM leaves a complete snapshot
        and a checkpointed database.
        """
        if self.engine is not None:
            try:
                snapshot = self.engine.to_snapshot()
                snapshot["safety"] = {
                    "kill_switch_triggered": self.safety.kill_switch_triggered,
                }
                self.store.save_engine_state(snapshot, self.strategy_id)
            except Exception:
                logger.error(
                    "Shutdown engine snapshot failed: %s", traceback.format_exc(),
                )
        try:
            self._save_state()
        except Exception:
            logger.error("Shutdown daemon state save failed: %s", traceback.format_exc())
        try:
            self.store.checkpoint()
        except Exception as exc:
            logger.error("Shutdown WAL checkpoint failed: %s", exc)
        try:
            self.store.close()
        except Exception as exc:
            logger.error("Shutdown store close failed: %s", exc)

    def run_forever(self, interval_seconds: int = 3600):
        """持续运行."""
        logger.info("Daemon started. Interval: %ds", interval_seconds)
        try:
            while True:
                try:
                    if self.should_run_today():
                        result = self.run_once()
                        if result["status"] == "killed":
                            logger.warning("Kill switch triggered. Waiting for manual recovery.")
                            break
                        if result["status"] == "ok":
                            # A successful run resets the crash budget; otherwise
                            # three crashes spread over months would still exit.
                            self.restart_count = 0
                    else:
                        logger.debug("Not time to run yet")
                    time.sleep(interval_seconds)
                except KeyboardInterrupt:
                    logger.info("Daemon stopped by user")
                    break
                except Exception:
                    logger.error("Daemon crashed: %s", traceback.format_exc())
                    self._handle_crash()
        finally:
            # S16: SIGTERM/SIGINT or a normal exit must not abandon the
            # snapshot and the WAL.
            self.close()

    def _handle_crash(self):
        """崩溃恢复 (restart budget scoped to RESTART_WINDOW_HOURS)."""
        now = datetime.now(self._tz)
        if (
            self.last_restart is not None
            and now - self.last_restart > timedelta(hours=RESTART_WINDOW_HOURS)
        ):
            # The old count was never reset by time, so three crashes spread
            # over months still terminated the daemon.
            self.restart_count = 0
        self.restart_count += 1
        self.last_restart = now
        if self.restart_count > MAX_RESTARTS:
            logger.critical("Too many restarts in %dh. Giving up.", RESTART_WINDOW_HOURS)
            sys.exit(1)
        logger.info("Restarting in 60s (attempt %d/%d)", self.restart_count, MAX_RESTARTS)
        time.sleep(60)


if __name__ == "__main__":
    from quant_platform.utils.logging import setup_logging

    # S16: docker stop sends SIGTERM. Convert it into KeyboardInterrupt so
    # the run loop unwinds and run_forever's finally persists and closes.
    def _handle_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Rotating file handler: an unattended daemon must not grow one log file
    # without bound.
    setup_logging(log_file="results/daemon.log")
    daemon = Daemon()
    daemon.run_forever(interval_seconds=3600)
