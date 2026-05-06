"""Tests for technical factors."""

import numpy as np

from quant_platform.factors.technical import (
    Momentum1M,
    Momentum12M,
    Volatility20D,
    RSIFactor,
    MACDFactor,
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
