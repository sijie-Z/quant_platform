"""Tests for technical factors."""

import pandas as pd
import pytest
from quant_platform.factors.technical import (
    MACDFactor,
    MAConvergenceFactor,
    Momentum1M,
    Momentum12M,
    RSIFactor,
    Volatility20D,
)


def test_momentum_1m(prices):
    factor = Momentum1M()
    result = factor.compute(prices)
    assert result.shape == prices.shape
    # Should have valid values after the lookback period
    assert result.iloc[30:].dropna(how="all").shape[0] > 0


def test_momentum_12m_skip(prices):
    factor = Momentum12M()
    result = factor.compute(prices)
    assert result.shape == prices.shape


def test_volatility_positive(prices):
    factor = Volatility20D()
    result = factor.compute(prices)
    valid = result.dropna()
    if valid.size > 0:
        # Volatility should be non-negative
        assert (valid >= 0).all().all()


def test_rsi_range(prices):
    factor = RSIFactor()
    result = factor.compute(prices)
    valid = result.dropna()
    if valid.size > 0:
        assert (valid >= 0).all().all()
        assert (valid <= 100).all().all()


def test_macd(prices):
    factor = MACDFactor()
    result = factor.compute(prices)
    assert result.shape == prices.shape


def test_factor_run_method(prices):
    factor = Momentum1M()
    result = factor.run(prices)
    assert result.name == "momentum_1m"
    assert result.category.value == "technical"
    assert result.values.shape == prices.shape


def test_ma_convergence_scores_each_asset_against_its_own_moving_averages():
    """The score is per asset: 1.0 while an asset's MA5/MA10/MA20 sit on top of
    each other, 0.0 once the widest of the two gaps reaches 5%.

    `max(axis=1)` over the concatenated gap panels scored the cross-section as
    a whole instead, which is why this pins one asset against a second one that
    is doing something else.
    """
    dates = pd.bdate_range("2023-01-02", periods=30)
    flat = pd.Series(10.0, index=dates)
    prices = pd.DataFrame({"FLAT": flat, "JUMP": flat})
    prices.loc[dates[25]:, "JUMP"] = 20.0

    score = MAConvergenceFactor().compute(prices)

    assert list(score.columns) == ["FLAT", "JUMP"]
    assert score.loc[dates[24], "FLAT"] == pytest.approx(1.0)
    assert score.loc[dates[24], "JUMP"] == pytest.approx(1.0)
    # FLAT is untouched by JUMP's move: its own three MAs are still equal.
    assert score.loc[dates[-1], "FLAT"] == pytest.approx(1.0)
    # JUMP: MA5 = 20, MA10 = MA20 = 15 -> a 33% gap, past the 5% floor.
    assert score.loc[dates[-1], "JUMP"] == pytest.approx(0.0)
