"""Neutralization must survive partial market-cap coverage (issue found in audit)."""

import numpy as np
import pandas as pd
from quant_platform.factors.processing import neutralize


def test_neutralize_tolerates_partial_market_cap():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=3)
    assets = [f"S{i:02d}" for i in range(30)]

    factor = pd.DataFrame(rng.normal(size=(3, 30)), index=dates, columns=assets)
    market_cap = pd.DataFrame(
        rng.uniform(1e9, 1e11, (3, 30)), index=dates, columns=assets,
    )
    market_cap.iloc[1, 5] = np.nan
    sector_map = pd.Series("Tech", index=assets)

    result = neutralize(factor, sector_map=sector_map, market_cap=market_cap)

    assert result.shape == factor.shape
    # The row without a market cap stays NaN instead of raising and aborting
    # the whole factor pipeline.
    assert np.isnan(result.iloc[1, 5])
    assert np.isfinite(result.to_numpy()).sum() >= 3 * 30 - 1


def test_vectorized_neutralize_matches_the_sklearn_reference():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-02", periods=8)
    assets = [f"S{i:02d}" for i in range(60)]
    sectors = pd.Series(
        np.array(["Tech", "Finance", "Energy", "Health"])[np.arange(60) % 4],
        index=assets,
    )
    factor = pd.DataFrame(rng.normal(size=(8, 60)), index=dates, columns=assets)
    market_cap = pd.DataFrame(
        rng.uniform(1e9, 1e11, (8, 60)), index=dates, columns=assets,
    )

    result = neutralize(factor, sector_map=sectors, market_cap=market_cap)

    for date in dates:
        row = factor.loc[date]
        dummy = pd.get_dummies(sectors, drop_first=True).astype(float)
        X = dummy.copy()
        X["mcap"] = np.log(market_cap.loc[date].clip(lower=1e8))
        model = LinearRegression().fit(X, row)
        expected = row - model.predict(X)
        np.testing.assert_allclose(result.loc[date], expected, atol=1e-8)
