"""Level factors must survive a suspension in the price panel."""

import numpy as np
import pandas as pd
from quant_platform.factors.technical import Momentum12M, Volatility60D


def _prices_with_one_suspension(periods: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=periods)
    assets = [f"S{i:02d}" for i in range(12)]
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, (periods, len(assets))),
        index=dates, columns=assets,
    )
    prices = 10 * (1 + returns).cumprod()
    prices.iloc[100, 3] = np.nan  # one suspended stock-day
    return prices


def test_momentum_is_defined_after_a_suspension():
    prices = _prices_with_one_suspension()

    values = Momentum12M().run(prices).values

    assert pd.notna(values.iloc[-1, 3]), "one suspension blanked 252 days of momentum"


def test_volatility_is_defined_after_a_suspension():
    prices = _prices_with_one_suspension()

    values = Volatility60D().run(prices).values

    assert pd.notna(values.iloc[-1, 3])


def test_ffill_matches_the_reference_level_series():
    prices = _prices_with_one_suspension()
    filled = prices.ffill()

    values = Momentum12M().run(prices).values
    # momentum_12m skips the most recent 21 days: period 252, skip 21.
    reference = (
        filled.iloc[-1 - 21] / filled.iloc[-1 - 21 - 252] - 1
    )

    np.testing.assert_allclose(values.iloc[-1], reference, atol=1e-10)
