"""Factor parameters from YAML must reach the factor constructors."""

import numpy as np
import pandas as pd
from quant_platform.main import _compute_factors
from quant_platform.utils.config import load_config


def _panel(n_dates=80, n_assets=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    assets = [f"S{i:03d}" for i in range(n_assets)]
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_dates, n_assets)), axis=0)),
        index=dates, columns=assets,
    )
    returns = prices.pct_change(fill_method=None).shift(-1)
    metadata = {"sector": pd.Series("Tech", index=assets)}
    return prices, returns, metadata


def _config(period: int):
    return load_config(raw={
        "factors": {
            "technical": {
                "momentum_1m": {"enabled": True, "params": {"period": period}},
            },
            "fundamental": {},
            "processing": {"neutralize": {"enabled": False}},
        },
    })


def test_the_configured_period_changes_the_factor():
    prices, returns, metadata = _panel()

    fast, _, _, _ = _compute_factors(
        prices, returns, None, metadata, config=_config(10),
    )
    slow, _, _, _ = _compute_factors(
        prices, returns, None, metadata, config=_config(21),
    )

    assert not fast["momentum_1m"].equals(slow["momentum_1m"]), (
        "the configured period was ignored"
    )
