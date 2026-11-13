"""Regression tests for Black-Litterman unit conventions.

The old model estimated risk aversion as ``0.06 / market_variance`` from the
*daily* covariance matrix -- a 252x too-large, annualised-premium-over-daily-
variance number -- then multiplied it into the same daily covariance and
added the *annual* risk-free rate on top. The implied prior averaged ~5.7%
per day (>1000% annualised) instead of ~6% per year.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.portfolio.black_litterman import BlackLittermanModel

ASSETS = ["A", "B", "C"]
CAPS = pd.Series({"A": 300.0, "B": 500.0, "C": 200.0})


def daily_cov() -> pd.DataFrame:
    """Daily covariance from daily vols 1.0%, 1.2%, 0.8%."""
    vols = np.array([0.010, 0.012, 0.008])
    corr = np.array([
        [1.00, 0.30, 0.20],
        [0.30, 1.00, 0.25],
        [0.20, 0.25, 1.00],
    ])
    return pd.DataFrame(corr * np.outer(vols, vols), index=ASSETS, columns=ASSETS)


class TestUnitConsistency:
    def test_prior_annual_excess_return_is_about_six_percent(self):
        model = BlackLittermanModel(daily_cov(), CAPS, risk_free_rate=0.02)

        rf_daily = 0.02 / 252
        annual_excess = float(model.market_weights @ (model.prior - rf_daily)) * 252

        assert annual_excess == pytest.approx(0.06, abs=0.01)

    def test_prior_total_return_includes_the_daily_risk_free_rate(self):
        model = BlackLittermanModel(daily_cov(), CAPS, risk_free_rate=0.03)

        annual_total = float(model.market_weights @ model.prior) * 252

        assert annual_total == pytest.approx(0.03 + 0.06, abs=0.01)

    def test_risk_aversion_is_annual_premium_over_annual_variance(self):
        cov = daily_cov()
        model = BlackLittermanModel(cov, CAPS, risk_free_rate=0.02)

        w = model.market_weights.to_numpy()
        annual_var = float(w @ cov.to_numpy() @ w) * 252

        assert model.risk_aversion == pytest.approx(0.06 / annual_var, rel=1e-6)

    def test_market_implied_weights_round_trip(self):
        """Pins the unit convention: the prior and the reverse optimisation
        must use the same (daily) risk-free rate.

        This round trip also held before the fix, because the prior added the
        annual rf and the reverse optimisation subtracted the same annual
        figure. It breaks as soon as one side is converted and the other is
        not, so it guards the convention rather than a past failure.
        """
        model = BlackLittermanModel(daily_cov(), CAPS, risk_free_rate=0.02)

        implied = model.market_implied_weights()

        assert np.allclose(
            implied.to_numpy(), model.market_weights.to_numpy(), atol=1e-8,
        )
