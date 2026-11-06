"""Safety checks must treat gains as gains, not as losses."""

import pytest
from quant_platform.risk.safety import SafetyLimits, SafetySystem


def _system() -> SafetySystem:
    return SafetySystem(SafetyLimits(
        max_drawdown=0.30,
        max_daily_loss=0.05,
        max_position_ratio=0.05,
        max_single_order_value=500_000,
        min_cash_reserve=10_000,
    ))


def _daily_loss_check(daily_pnl: float):
    checks = _system().check_all(
        equity=1_000_000,
        peak_equity=1_000_000,
        cash=500_000,
        daily_pnl=daily_pnl,
        positions={},
    )
    return next(c for c in checks if c.name == "max_daily_loss")


def test_a_profitable_day_does_not_trip_the_daily_loss_check():
    # +6% used to be treated as a 6% loss because the ratio used abs().
    assert _daily_loss_check(60_000).passed is True


@pytest.mark.parametrize("pnl", [-60_000, -100_000])
def test_a_losing_day_beyond_the_limit_fails(pnl):
    assert _daily_loss_check(pnl).passed is False


def test_a_small_loss_within_the_limit_passes():
    assert _daily_loss_check(-10_000).passed is True
