"""Regression tests for the Postgres provider hardening wave.

Covers D09 (delisting_date in metadata, NULL is_st), D10 (financials index
level ``asset`` and the full FINANCIAL_COLUMNS set), D20 (empty prices keep
the MultiIndex contract, adj_factor validation) and the ``DataPipeline.run()``
contract test with a mocked connection.

No database/network access: ``pd.read_sql`` is replaced with a dispatcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers import postgres_provider as pg
from quant_platform.data.providers.postgres_provider import PostgresDataProvider
from quant_platform.data.schema import FINANCIAL_COLUMNS

DATES = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
CODES = ["600001.SH", "000001.SZ"]


def price_rows():
    return pd.DataFrame([
        {"date": d, "code": c,
         "open": 10.0 + k, "high": 10.3 + k, "low": 9.9 + k, "close": 10.1 + k,
         "volume": 100_000, "turnover": 0.015, "adj_factor": 1.0}
        for k, d in enumerate(DATES) for c in CODES
    ])


def financial_rows():
    return pd.DataFrame([
        {"date": d, "code": c,
         "market_cap": 1e10, "total_assets": 2e10, "net_assets": 1e10,
         "revenue": 1e9, "net_profit": 1e8, "roe": 0.1,
         "pb_ratio": 2.0, "pe_ratio": 20.0, "asset_growth": 0.1,
         "publish_date": d}
        for d in DATES for c in CODES
    ])


def benchmark_rows():
    return pd.DataFrame([{"date": d, "return_value": 0.01} for d in DATES])


def metadata_rows(nullable_is_st=True):
    return pd.DataFrame([
        {"code": "600001.SH", "sector": "银行", "market_cap_group": "large",
         "is_st": None if nullable_is_st else False,
         "listing_date": pd.Timestamp("2020-01-01"), "delisting_date": pd.NaT},
        {"code": "000001.SZ", "sector": "医药生物", "market_cap_group": "mid",
         "is_st": None if nullable_is_st else False,
         "listing_date": pd.Timestamp("2020-01-01"), "delisting_date": pd.NaT},
    ])


def install_fake_read_sql(monkeypatch, frames):
    """Route each provider query to the matching mocked frame."""
    def fake_read_sql(query, engine, params=None):
        q = " ".join(str(query).split())
        if "FROM daily_prices" in q:
            return frames["prices"].copy()
        if "FROM financials" in q:
            return frames["financials"].copy()
        if "FROM benchmark_returns" in q:
            return frames["benchmark"].copy()
        if "FROM stock_metadata" in q:
            return frames["metadata"].copy()
        raise AssertionError(f"unexpected query: {q}")

    monkeypatch.setattr(pg.pd, "read_sql", fake_read_sql)


def make_provider(monkeypatch, frames):
    install_fake_read_sql(monkeypatch, frames)
    provider = object.__new__(PostgresDataProvider)
    provider._connected = True
    provider._engine = object()
    provider.st_available = False
    return provider


def base_frames():
    return {
        "prices": price_rows(),
        "financials": financial_rows(),
        "benchmark": benchmark_rows(),
        "metadata": metadata_rows(),
    }


# ----------------------------------------------------------------------
# D09 / D20: metadata contract
# ----------------------------------------------------------------------

def test_metadata_has_delisting_date_and_coalesces_null_is_st(monkeypatch):
    provider = make_provider(monkeypatch, base_frames())

    meta = provider.get_metadata()

    assert meta.index.name == "asset"
    assert "delisting_date" in meta.columns
    assert meta["delisting_date"].isna().all()
    # SQL NULL must not make `~meta['is_st']` raise TypeError downstream.
    assert meta["is_st"].dtype == bool
    assert (~meta["is_st"]).all()
    assert meta.attrs["st_available"] is False


def test_empty_metadata_keeps_the_contract(monkeypatch):
    frames = base_frames()
    frames["metadata"] = frames["metadata"].iloc[0:0]
    provider = make_provider(monkeypatch, frames)

    meta = provider.get_metadata()

    assert isinstance(meta.index, pd.Index)
    assert meta.index.name == "asset"
    assert {"sector", "market_cap_group", "is_st",
            "listing_date", "delisting_date"} <= set(meta.columns)


# ----------------------------------------------------------------------
# D10: financials contract
# ----------------------------------------------------------------------

def test_financials_index_named_asset_and_all_columns(monkeypatch):
    provider = make_provider(monkeypatch, base_frames())

    fin = provider.get_financials("2024-01-01", "2024-01-04")

    assert list(fin.index.names) == ["date", "asset"]
    assert set(FINANCIAL_COLUMNS) <= set(fin.columns)
    assert "publish_date" in fin.columns
    assert fin.loc[(pd.Timestamp("2024-01-02"), "600001.SH"), "total_assets"] == 2e10


def test_financials_missing_column_raises_actionable_error(monkeypatch):
    frames = base_frames()
    frames["financials"] = frames["financials"].drop(columns=["total_assets"])
    provider = make_provider(monkeypatch, frames)

    with pytest.raises(ValueError, match="total_assets"):
        provider.get_financials("2024-01-01", "2024-01-04")


def test_financials_without_publish_date_exposes_nat(monkeypatch):
    frames = base_frames()
    frames["financials"] = frames["financials"].drop(columns=["publish_date"])
    provider = make_provider(monkeypatch, frames)

    fin = provider.get_financials("2024-01-01", "2024-01-04")

    assert "publish_date" in fin.columns
    assert pd.to_datetime(fin["publish_date"]).isna().all()


# ----------------------------------------------------------------------
# D20: empty prices and adj_factor validation
# ----------------------------------------------------------------------

def test_empty_prices_keep_multiindex_and_columns(monkeypatch):
    frames = base_frames()
    frames["prices"] = frames["prices"].iloc[0:0]
    provider = make_provider(monkeypatch, frames)

    prices = provider.get_prices("2024-01-01", "2024-01-04")

    assert isinstance(prices.index, pd.MultiIndex)
    assert list(prices.index.names) == ["date", "asset"]
    assert "close" in prices.columns
    assert prices.empty


@pytest.mark.parametrize("bad_factor", [0.0, -1.0, np.nan, np.inf])
def test_invalid_adj_factor_is_rejected(monkeypatch, bad_factor):
    frames = base_frames()
    frames["prices"].loc[0, "adj_factor"] = bad_factor
    provider = make_provider(monkeypatch, frames)

    with pytest.raises(ValueError, match="adj_factor"):
        provider.get_prices("2024-01-01", "2024-01-04")


# ----------------------------------------------------------------------
# Contract: DataPipeline.run() accepts the provider's frames
# ----------------------------------------------------------------------

def test_pipeline_run_accepts_postgres_frames(monkeypatch):
    provider = make_provider(monkeypatch, base_frames())

    pipe = DataPipeline(
        provider, "2024-01-01", "2024-01-04",
        exclude_st=True, exclude_suspended=False,
    )
    pipe.run()

    assert pipe.valid_assets is not None
    assert len(pipe.valid_assets) == 2
    assert pipe.prices is not None and not pipe.prices.empty
    assert "close_adj" in pipe.prices.columns
    assert pipe.financials is not None
    assert list(pipe.financials.index.names) == ["date", "asset"]

    macro = pipe.get_market_cap()
    assert isinstance(macro, pd.DataFrame)
    assert not macro.empty
