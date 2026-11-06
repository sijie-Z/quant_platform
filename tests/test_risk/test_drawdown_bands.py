"""Drawdown bands: 15%-25% blocks with RED, only >25% arms the kill switch.

Regression for the band between ``max_drawdown_pct`` and
``kill_drawdown_pct``. The kill branch read ``max_drawdown_pct`` instead of
``kill_drawdown_pct``, so a 16% drawdown liquidated the whole book at the
warning threshold while the log message still claimed a 25% threshold.
"""

import logging
from contextlib import contextmanager

import pytest
from quant_platform.risk.circuit_breaker import (
    BreachType,
    RiskLevel,
    RiskLimits,
    RiskMonitor,
)

LOGGER_NAME = "quant_platform.risk.circuit_breaker"
PEAK = 1_000_000.0


@contextmanager
def _capture_log(caplog):
    """Capture this logger even if ``quant_platform`` has propagate=False.

    A full-suite run can leave the ``quant_platform`` logger non-propagating
    (see tests/test_daemon/test_runner_watermark.py), hiding records from
    caplog's root handler; attach the handler to the module logger directly.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            yield
    finally:
        logger.removeHandler(caplog.handler)


def _monitor_at_peak() -> RiskMonitor:
    monitor = RiskMonitor(RiskLimits(max_drawdown_pct=0.15, kill_drawdown_pct=0.25))
    monitor.update_portfolio_state(PEAK, 0.0, {}, {})
    assert monitor.kill_switch_active is False
    return monitor


def _drawdown_breaches(monitor: RiskMonitor) -> list:
    return [b for b in monitor.breaches if b.breach_type == BreachType.DRAWDOWN]


class TestRedBandBetweenTheTwoLimits:
    def test_16_percent_drawdown_blocks_without_killing(self, caplog):
        monitor = _monitor_at_peak()

        with _capture_log(caplog):
            monitor.update_portfolio_state(840_000.0, 0.0, {}, {})

        assert monitor.kill_switch_active is False
        assert monitor.risk_level == RiskLevel.RED
        breaches = _drawdown_breaches(monitor)
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.severity == RiskLevel.RED
        assert breach.auto_action == "block"
        assert breach.current_value == pytest.approx(0.16, rel=1e-12)
        assert breach.limit_value == pytest.approx(0.15, rel=1e-12)
        assert "15.00%" in breach.message

        # The logged threshold must be the one that actually fired: the
        # mutant logged "KILL SWITCH ... kill threshold 25.00%" at 16%.
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == LOGGER_NAME
        ]
        assert any("exceeds limit 15.00%" in message for message in messages)
        assert not any("KILL SWITCH" in message for message in messages)

    def test_exactly_at_the_warning_threshold_is_not_a_breach(self):
        """The comparison is strict: a 15.00% drawdown is still tolerated."""
        monitor = _monitor_at_peak()

        monitor.update_portfolio_state(850_000.0, 0.0, {}, {})

        assert monitor.kill_switch_active is False
        assert monitor.risk_level == RiskLevel.GREEN
        assert _drawdown_breaches(monitor) == []

    def test_just_past_the_warning_threshold_is_red(self):
        monitor = _monitor_at_peak()

        monitor.update_portfolio_state(849_990.0, 0.0, {}, {})

        assert monitor.kill_switch_active is False
        assert monitor.risk_level == RiskLevel.RED
        assert _drawdown_breaches(monitor)[0].limit_value == pytest.approx(0.15)

    def test_exactly_at_the_kill_threshold_still_only_blocks(self):
        monitor = _monitor_at_peak()

        monitor.update_portfolio_state(750_000.0, 0.0, {}, {})

        assert monitor.kill_switch_active is False
        assert monitor.risk_level == RiskLevel.RED
        breach = _drawdown_breaches(monitor)[0]
        assert breach.severity == RiskLevel.RED
        assert breach.auto_action == "block"
        assert breach.limit_value == pytest.approx(0.15)


class TestKillBandAboveTwentyFivePercent:
    def test_26_percent_drawdown_arms_the_kill_switch(self, caplog):
        monitor = _monitor_at_peak()

        with _capture_log(caplog):
            monitor.update_portfolio_state(740_000.0, 0.0, {}, {})

        assert monitor.kill_switch_active is True
        assert monitor.risk_level == RiskLevel.KILL
        breaches = _drawdown_breaches(monitor)
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.severity == RiskLevel.KILL
        assert breach.auto_action == "flatten"
        assert breach.current_value == pytest.approx(0.26, rel=1e-12)
        assert breach.limit_value == pytest.approx(0.25, rel=1e-12)
        assert "kill threshold 25.00%" in breach.message

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == LOGGER_NAME
        ]
        assert any("KILL SWITCH" in message for message in messages)

    def test_kill_threshold_is_not_the_warning_threshold(self):
        """A limits object with a wide warning band must not kill at 16%."""
        monitor = RiskMonitor(RiskLimits(max_drawdown_pct=0.15, kill_drawdown_pct=0.25))
        monitor.update_portfolio_state(PEAK, 0.0, {}, {})
        monitor.update_portfolio_state(840_000.0, 0.0, {}, {})

        assert monitor.kill_switch_active is False
        assert monitor.risk_level == RiskLevel.RED
        assert all(b.severity != RiskLevel.KILL for b in monitor.breaches)
