"""Explicit price-basis contract (D15/D18/D19/D41).

Controlled split/dividend scenario: a stock trades at raw 10 CNY, then a
2-for-1 split halves the traded price to 5 while the adjusted path stays at
10.  Research must see the continuous adjusted series; every paper/live mark
must use the raw level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.base import DataProvider
from quant_platform.data.schema import DataQualityError

DATES = pd.bdate_range("2024-01-02", periods=50, name="date")
SPLIT_DAY = 43
ASSET = "600519"          # deliberately bare: the pipeline canonicalizes it
CANONICAL = "600519.SH"


def _price_frame(
    *,
    asset: str = ASSET,
    split: bool = True,
    already_adjusted: bool = False,
) -> pd.DataFrame:
    n = len(DATES)
    raw_close = np.full(n, 10.0)
    divisor = np.ones(n)
    if split:
        raw_close[SPLIT_DAY:] = 5.0
        # The provider contract is a *divisor*: raw / 0.5 = 10 (continuous).
        divisor[SPLIT_DAY:] = 0.5
    if already_adjusted:
        # Legacy provider: close is already the adjusted level and the
        # factor is a placeholder.  There is no raw price in this frame.
        close = np.full(n, 10.0)
        divisor = np.ones(n)
    else:
        close = raw_close

    index = pd.MultiIndex.from_product([DATES, [asset]], names=["date", "asset"])
    frame = pd.DataFrame(index=index)
    frame["open"] = np.repeat(close, 1)
    frame["high"] = np.repeat(close, 1)
    frame["low"] = np.repeat(close, 1)
    frame["close"] = np.repeat(close, 1)
    # Volume must vary: an unchanged close *and* unchanged volume is the
    # pipeline's stale-row (suspension) signal, which would blank returns.
    frame["volume"] = np.repeat(1_000_000.0 + np.arange(n), 1)
    frame["turnover"] = 0.01
    frame["adj_factor"] = np.repeat(divisor, 1)
    return frame


class SplitProvider(DataProvider):
    """One stock: raw 10 -> 5 on a 2-for-1 split; divisor 1 -> 0.5."""

    price_adjustment = "raw"
    adjusted_price_basis = "hfq"
    adjustment_factor_available = True

    def __init__(self, **frame_kwargs):
        self._prices = _price_frame(**frame_kwargs)
        self._prices.attrs.update({
            "price_basis": "raw",
            "adjusted_price_basis": "hfq",
            "adjustment_factor_available": True,
        })
        self._metadata = pd.DataFrame(
            {
                "sector": ["食品饮料"],
                "market_cap_group": ["large"],
                "is_st": [False],
                "listing_date": [pd.Timestamp("2020-01-01")],
                "delisting_date": [pd.NaT],
            },
            index=pd.Index([ASSET], name="asset"),
        )

    def get_prices(self, start_date, end_date, fields=None):
        df = self._prices.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
        if fields:
            df = df[list(fields)]
        return df

    def get_financials(self, start_date, end_date):
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
            columns=[
                "market_cap", "total_assets", "net_assets", "revenue",
                "net_profit", "roe", "pb_ratio", "pe_ratio", "asset_growth",
            ],
        )

    def get_benchmark(self, start_date, end_date):
        return pd.Series(
            np.zeros(len(DATES)), index=DATES, name="benchmark",
        )

    def get_metadata(self):
        return self._metadata


class AdjustedOnlyProvider(SplitProvider):
    """Provider that only ships an adjusted (hfq) close with factor 1."""

    price_adjustment = "hfq"
    adjusted_price_basis = "hfq"
    adjustment_factor_available = False

    def __init__(self):
        super().__init__(already_adjusted=True, split=False)
        self._prices.attrs.update({
            "price_basis": "hfq",
            "adjusted_price_basis": "hfq",
            "adjustment_factor_available": False,
        })


def _run(provider, start=None, end=None) -> DataPipeline:
    pipe = DataPipeline(
        provider,
        str((start or DATES[0]).date()),
        str((end or DATES[-1]).date()),
        exclude_st=False,
        exclude_suspended=False,
    )
    pipe.run()
    return pipe


# ----------------------------------------------------------------------
# Basis contract on the pipeline
# ----------------------------------------------------------------------

def test_split_pipeline_exposes_raw_and_adjusted_bases():
    pipe = _run(SplitProvider())

    # One canonical asset label, even though the provider used a bare code.
    assert list(pipe.prices.index.get_level_values("asset").unique()) == [
        CANONICAL
    ]

    raw = pipe.get_raw_close()
    adjusted = pipe.get_adjusted_close()
    legacy = pipe.get_close()

    assert raw.attrs["price_basis"] == "raw"
    assert raw.attrs["adjustment_factor_available"] is True
    assert adjusted.attrs["price_basis"] == "hfq"
    # get_close() is the deprecated research alias; it must not change the
    # series research used before this wave.
    pd.testing.assert_frame_equal(legacy, adjusted)
    assert legacy.attrs["price_basis"] == "hfq"

    # Raw price halves on the split date; the adjusted series is continuous.
    assert raw.loc[DATES[SPLIT_DAY - 1], CANONICAL] == pytest.approx(10.0)
    assert raw.loc[DATES[SPLIT_DAY], CANONICAL] == pytest.approx(5.0)
    assert adjusted.loc[DATES[SPLIT_DAY - 1], CANONICAL] == pytest.approx(10.0)
    assert adjusted.loc[DATES[SPLIT_DAY], CANONICAL] == pytest.approx(10.0)

    # Research returns are continuous: no fabricated split-day move.
    assert pipe.returns[CANONICAL].abs().max() < 1e-12


def test_provider_declaring_adjusted_close_with_varying_factor_is_rejected():
    provider = SplitProvider()
    # Double-adjustment risk: adjusted close *and* a non-constant divisor.
    provider._prices = provider._prices.copy()
    provider._prices.attrs["price_basis"] = "hfq"
    provider.price_adjustment = "hfq"

    with pytest.raises(DataQualityError, match="double-adjusted"):
        _run(provider)


def test_adjusted_only_provider_fails_closed_for_raw_prices():
    pipe = _run(AdjustedOnlyProvider())

    assert pipe.price_basis == "hfq"
    assert pipe.get_close().attrs["price_basis"] == "hfq"
    with pytest.raises(DataQualityError, match="no raw close"):
        pipe.get_raw_close()


# ----------------------------------------------------------------------
# Provider frames carry the basis declaration
# ----------------------------------------------------------------------

def test_synthetic_provider_declares_raw_basis():
    from quant_platform.data.providers.base import provider_identity
    from quant_platform.data.providers.synthetic import SyntheticDataProvider

    provider = SyntheticDataProvider(
        n_stocks=5, start_date="2024-01-01", end_date="2024-01-31",
        embedded_alpha=False,
    )
    prices = provider.get_prices("2024-01-01", "2024-01-31")
    assert prices.attrs["price_basis"] == "raw"
    assert prices.attrs["adjusted_price_basis"] == "qfq"
    assert prices.attrs["adjustment_factor_available"] is True
    # The pipeline cache key must change with the new price semantics so an
    # old pickle produced under `synthetic_adj_factor` is never served.
    assert provider_identity(provider)["adjustment"] == "raw"


def test_baostock_derives_divisor_from_raw_and_hfq():
    from quant_platform.data.providers.baostock_provider import (
        BaostockDataProvider,
    )

    raw = _price_frame(asset="600519", split=True).drop(columns=["adj_factor"])
    hfq = _price_frame(asset="600519", split=True).drop(columns=["adj_factor"])
    hfq_hfq = hfq.copy()
    # hfq = raw / divisor -> raw * 2 after the split; continuous at 10.
    hfq_hfq["close"] = 10.0
    hfq_hfq["open"] = hfq_hfq["high"] = hfq_hfq["low"] = hfq_hfq["close"]

    provider = BaostockDataProvider(cache_enabled=False)
    provider._get_major_stocks_list = lambda **kwargs: ["600519"]
    provider._fetch_daily_data = lambda codes, start, end, adjustflag="3": (
        raw.copy() if adjustflag == "3" else hfq_hfq.copy()
    )
    prices = provider.get_prices(str(DATES[0].date()), str(DATES[-1].date()))

    assert prices.attrs["price_basis"] == "raw"
    assert prices.attrs["adjusted_price_basis"] == "hfq"
    assert prices.attrs["adjustment_factor_available"] is True
    assert prices.loc[(DATES[SPLIT_DAY], "600519"), "close"] == pytest.approx(5.0)
    assert prices.loc[(DATES[SPLIT_DAY], "600519"), "adj_factor"] == pytest.approx(0.5)
    adjusted = (
        prices["close"] / prices["adj_factor"]
    ).unstack("asset")
    assert adjusted["600519"].abs().diff().dropna().max() == pytest.approx(0.0)

    raw_only = BaostockDataProvider(cache_enabled=False, adjustment="raw")
    raw_only._get_major_stocks_list = lambda **kwargs: ["600519"]
    raw_only._fetch_daily_data = (
        lambda codes, start, end, adjustflag="3": raw.copy()
    )
    prices_raw = raw_only.get_prices(
        str(DATES[0].date()), str(DATES[-1].date()),
    )
    assert prices_raw.attrs["adjusted_price_basis"] == "raw"
    assert prices_raw.attrs["adjustment_factor_available"] is False
    assert set(prices_raw["adj_factor"].unique()) == {1.0}
