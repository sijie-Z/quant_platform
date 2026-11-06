"""Tests for fundamental factors."""


from quant_platform.factors.fundamental import (
    ROE,
    AssetGrowth,
    LogMarketCap,
    PbRatio,
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
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_pe_and_pb_exclude_non_positive_denominators():
    """Loss-makers and negative-book names must not rank as the cheapest."""
    import pandas as pd
    from quant_platform.factors.fundamental import PeRatio

    dates = pd.bdate_range("2024-01-02", periods=1)
    assets = ["A", "B", "C"]
    pe = pd.DataFrame([[5.0, 40.0, -20.0]], index=dates, columns=assets)
    pb = pd.DataFrame([[1.0, 8.0, -2.0]], index=dates, columns=assets)
    prices = pd.DataFrame([[10.0, 10.0, 10.0]], index=dates, columns=assets)

    pe_values = PeRatio().compute(prices, {"pe_ratio": pe})
    pb_values = PbRatio().compute(prices, {"pb_ratio": pb})

    assert pe_values.iloc[0, 0] > pe_values.iloc[0, 1]  # 5x is cheaper than 40x
    assert pd.isna(pe_values.iloc[0, 2])                 # loss-maker excluded
    assert pb_values.iloc[0, 0] > pb_values.iloc[0, 1]  # 1x cheaper than 8x
    assert pd.isna(pb_values.iloc[0, 2])                 # negative book excluded
