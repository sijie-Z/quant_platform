"""Tests for fundamental factors."""

import numpy as np
import pandas as pd

from quant_platform.factors.fundamental import (
    LogMarketCap,
    PbRatio,
    ROE,
    AssetGrowth,
)


def test_log_market_cap(prices, financials):
    factor = LogMarketCap()
    result = factor.compute(prices, financials)
    assert result.shape == financials["market_cap"].shape
    valid = result.dropna()
    if valid.size > 0:
        assert (valid > 0).all().all()  # Market cap log should be positive


def test_pb_ratio(prices, financials):
    factor = PbRatio()
    result = factor.compute(prices, financials)
    assert result.shape == financials["pb_ratio"].shape


def test_roe(prices, financials):
    factor = ROE()
    result = factor.compute(prices, financials)
    assert result.shape == financials["roe"].shape


def test_asset_growth(prices, financials):
    factor = AssetGrowth()
    result = factor.compute(prices, financials)
    assert result.shape == financials["asset_growth"].shape


def test_requires_financials(prices):
    """Fundamental factors should raise without financial data."""
    factor = LogMarketCap()
    try:
        factor.compute(prices, None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
