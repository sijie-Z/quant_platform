"""Edge cases in technical factors found by the audit."""

import numpy as np
import pandas as pd
from quant_platform.factors.network import NetworkFactor
from quant_platform.factors.technical import RSIFactor


def test_rsi_of_a_monotonically_rising_series_is_100():
    """No losing days means RSI 100, not NaN."""
    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame({"A": np.arange(1.0, 41.0)}, index=dates)

    values = RSIFactor(period=14).run(prices).values

    assert values.iloc[-1, 0] == 100.0


def test_network_factor_runs_after_initializing_its_base_class():
    """`NetworkFactor.__init__` used to skip `BaseFactor.__init__`, so run()
    raised AttributeError on `self.params`."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=90)
    assets = [f"S{i:02d}" for i in range(8)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (90, 8)), axis=0)),
        index=dates, columns=assets,
    )

    values = NetworkFactor(
        centrality_type="degree", window=20, threshold=0.3,
    ).run(prices).values

    assert values.shape == prices.shape
    assert values.notna().any().any()
