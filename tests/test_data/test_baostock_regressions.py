"""Regression tests for the Baostock provider hardening wave.

Covers D03 (documented (date, asset) financial panel over the requested
range, PIT publish_date, pipeline market-cap), D04 (per-symbol failure is
never cached; retry/refetch), D05/D06 (no fabricated is_st, delisted names
and an expiring universe cache), D19 (asset index name, fields=, exchange
mapping, versioned cache keys) and D22 (fields contract).

No network access: the baostock module is replaced with a fake.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers import baostock_provider as mod
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.data.providers.base import DataFetchIncompleteError

BASIC_FIELDS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]
PROFIT_FIELDS = ["code", "pubDate", "statDate", "roeAvg", "npMargin",
                 "gpMargin", "netProfit", "epsTTM", "MBRevenue",
                 "totalShare", "liqaShare"]
BALANCE_FIELDS = ["code", "pubDate", "statDate", "liabilityToAsset",
                  "assetToEquity"]


class FakeResultSet:
    def __init__(self, rows, fields, error_code="0", error_msg=""):
        self._rows = list(rows)
        self.fields = list(fields)
        self.error_code = error_code
        self.error_msg = error_msg
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


def _quarter_end(year: int, quarter: int) -> pd.Timestamp:
    return pd.Timestamp(
        year=year,
        month={1: 3, 2: 6, 3: 9, 4: 12}[quarter],
        day={1: 31, 2: 30, 3: 30, 4: 31}[quarter],
    )


class FakeBaostock:
    def __init__(
        self,
        basic_rows=None,
        fail_codes=(),
        transient_fail_once=(),
        empty_financials=False,
        only_quarters=None,
    ):
        self.basic_rows = list(basic_rows or [])
        self.fail_codes = set(fail_codes)
        self.transient_fail_once = set(transient_fail_once)
        self.empty_financials = empty_financials
        self.only_quarters = set(only_quarters) if only_quarters else None
        self.basic_query_count = 0
        self.history_calls = []
        self.profit_calls = []

    def login(self):
        return types.SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        return None

    def query_stock_basic(self, code="", code_name=""):
        self.basic_query_count += 1
        rows = list(self.basic_rows)
        if code:
            rows = [r for r in rows if r[0] == code]
        return FakeResultSet(rows, fields=BASIC_FIELDS)

    def query_history_k_data_plus(self, code, fields, start_date="", end_date="",
                                  frequency="d", adjustflag="3"):
        self.history_calls.append({"code": code, "start": start_date,
                                   "end": end_date, "adjustflag": adjustflag})
        if code in self.fail_codes:
            raise RuntimeError("simulated history failure for " + code)
        if code in self.transient_fail_once:
            self.transient_fail_once.discard(code)
            raise RuntimeError("simulated transient failure for " + code)

        dates = pd.bdate_range(start_date, end_date)
        if code == "sh.000300":
            rows = [
                [str(d.date()), "1000.0", "990.0", "1.0"] for d in dates
            ]
            return FakeResultSet(rows, fields=["date", "close", "preclose", "pctChg"])

        rows = []
        for k, d in enumerate(dates):
            px = 10.0 + 0.1 * (k % 5)
            prev = 10.0 + 0.1 * (max(k - 1, 0) % 5)
            rows.append([
                str(d.date()), code,
                f"{px - 0.05:.2f}", f"{px + 0.2:.2f}", f"{px - 0.2:.2f}",
                f"{px:.2f}", f"{prev:.2f}",
                "10000", "100000", "2.5", "1.0", "15.0", "2.0",
            ])
        return FakeResultSet(rows, fields=fields.split(","))

    def query_profit_data(self, code="", year=2024, quarter=4):
        self.profit_calls.append((code, year, quarter))
        if self.empty_financials:
            return FakeResultSet([], fields=PROFIT_FIELDS)
        if self.only_quarters is not None and (year, quarter) not in self.only_quarters:
            return FakeResultSet([], fields=PROFIT_FIELDS)
        qend = _quarter_end(year, quarter)
        pub = qend + pd.Timedelta(days=25)
        return FakeResultSet([[
            code, pub.strftime("%Y-%m-%d"), qend.strftime("%Y-%m-%d"),
            "10.0", "30.0", "40.0", "100000000", "1.5",
            "1000000000", "100000000", "50000000",
        ]], fields=PROFIT_FIELDS)

    def query_balance_data(self, code="", year=2024, quarter=4):
        qend = _quarter_end(year, quarter)
        pub = qend + pd.Timedelta(days=25)
        return FakeResultSet([[
            code, pub.strftime("%Y-%m-%d"), qend.strftime("%Y-%m-%d"),
            "50.0", "2.0",
        ]], fields=BALANCE_FIELDS)

    def query_stock_industry(self):
        rows = [[r[0], r[1], "J66 banking", "J66"] for r in self.basic_rows]
        return FakeResultSet(rows, fields=["code", "code_name", "industry",
                                           "industryClassification"])


def universe_rows():
    return [
        ["sh.600001", "AAA", "2020-01-01", "", "1", "1"],
        ["sh.600002", "BBB", "2020-01-01", "", "1", "1"],
        ["sz.000001", "CCC", "2020-01-01", "", "1", "1"],
        ["sz.000002", "DDD", "2020-01-01", "", "1", "1"],
        # Delisted name: must stay available for historical ranges.
        ["sh.600999", "OLD", "2010-01-01", "2022-01-01", "1", "0"],
        # Index: type=2, must be excluded from the stock universe.
        ["sh.000300", "INDEX", "2005-01-01", "", "2", "1"],
    ]


def make_provider(fake, tmp_path, monkeypatch, cache_enabled=False, **kwargs):
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    provider = BaostockDataProvider(
        cache_enabled=cache_enabled, retry_backoff=0.0, **kwargs,
    )
    provider._bs = fake
    return provider


# ----------------------------------------------------------------------
# D04: per-symbol failures are retried, never cached, and refetched
# ----------------------------------------------------------------------

def test_partial_failure_raises_and_second_call_refetches(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows(), fail_codes={"sh.600001"})
    provider = make_provider(fake, tmp_path, monkeypatch, cache_enabled=True)

    codes = ["600001", "600002", "000001"]
    with pytest.raises(DataFetchIncompleteError):
        provider._fetch_daily_data(codes, "2024-01-01", "2024-01-04")

    # Nothing was cached: the truncated panel cannot be replayed.
    assert list(tmp_path.glob("daily_*")) == []

    fake.fail_codes.clear()
    before = len(fake.history_calls)
    df = provider._fetch_daily_data(codes, "2024-01-01", "2024-01-04")

    assert len(fake.history_calls) > before  # network was hit again
    assert {"600001", "600002", "000001"} <= set(
        df.index.get_level_values("asset")
    )
    assert list(tmp_path.glob("daily_*"))  # complete panel is cached


def test_transient_failure_is_retried(tmp_path, monkeypatch):
    fake = FakeBaostock(
        basic_rows=universe_rows(), transient_fail_once={"sh.600001"},
    )
    provider = make_provider(fake, tmp_path, monkeypatch, max_retries=2)

    df = provider._fetch_daily_data(["600001"], "2024-01-01", "2024-01-04")

    calls = [c for c in fake.history_calls if c["code"] == "sh.600001"]
    assert len(calls) == 2
    assert "600001" in set(df.index.get_level_values("asset"))


# ----------------------------------------------------------------------
# D18/D19: units, index name, fields, exchange mapping
# ----------------------------------------------------------------------

def test_price_units_and_contract(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch)

    prices = provider.get_prices("2024-01-02", "2024-01-04")

    assert list(prices.index.names) == ["date", "asset"]
    assert set(prices["adj_factor"].unique()) == {1.0}
    row = prices.iloc[0]
    assert row["volume"] == pytest.approx(10_000)       # shares
    assert row["turnover"] == pytest.approx(0.025)      # 2.5% -> fraction
    assert "vwap" in prices.columns


def test_fields_contract(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch)

    sub = provider.get_prices(
        "2024-01-02", "2024-01-04", fields=["close", "volume"],
    )
    assert list(sub.columns) == ["close", "volume"]

    with pytest.raises(ValueError, match="Unknown price fields"):
        provider.get_prices("2024-01-02", "2024-01-04", fields=["nope"])


def test_exchange_mapping_including_beijing_and_suffix_forms():
    assert mod._to_bs_code("600000") == "sh.600000"
    assert mod._to_bs_code("000001") == "sz.000001"
    assert mod._to_bs_code("300750") == "sz.300750"
    assert mod._to_bs_code("830001") == "bj.830001"
    assert mod._to_bs_code("430001") == "bj.430001"
    assert mod._to_bs_code("920001") == "bj.920001"
    assert mod._to_bs_code("600000.SH") == "sh.600000"
    assert mod._to_bs_code("000001.SZ") == "sz.000001"
    assert mod._to_bs_code("bj.830001") == "bj.830001"
    assert mod._from_bs_code("bj.830001") == "830001"


# ----------------------------------------------------------------------
# D05/D06: no fabricated is_st, delisted names, versioned/expiring cache
# ----------------------------------------------------------------------

def test_metadata_is_asset_indexed_with_delisting_and_no_fake_st(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch)

    meta = provider.get_metadata()

    assert meta.index.name == "asset"
    assert "600999" in meta.index
    assert pd.notna(meta.loc["600999", "delisting_date"])
    assert "is_st" not in meta.columns
    assert provider.st_available is False
    assert meta.attrs["st_available"] is False

    st = provider.get_st_timeseries()
    assert st.empty and "end_date" in st.columns


def test_universe_cache_is_versioned(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch, cache_enabled=True)

    provider._get_major_stocks_list()
    provider._get_major_stocks_list()

    assert fake.basic_query_count == 1
    assert (tmp_path / mod._STOCK_BASIC_CACHE_FILE).exists()
    assert "v2" in mod._STOCK_BASIC_CACHE_FILE


# ----------------------------------------------------------------------
# D03: financial panel contract, PIT, requested range, empty results
# ----------------------------------------------------------------------

def test_financials_are_a_date_asset_panel_with_publish_date(tmp_path, monkeypatch):
    fake = FakeBaostock(
        basic_rows=universe_rows(), only_quarters=[(2023, 4)],
    )
    provider = make_provider(fake, tmp_path, monkeypatch)

    fin = provider.get_financials("2024-01-01", "2024-01-31")

    assert list(fin.index.names) == ["date", "asset"]
    for col in ("market_cap", "total_assets", "net_assets", "revenue",
                "net_profit", "roe", "pb_ratio", "pe_ratio", "asset_growth",
                "publish_date"):
        assert col in fin.columns
    assert fin["market_cap"].notna().any()

    # PIT: the Q4-2023 report was published 2024-01-25; nothing before it.
    visible = fin["market_cap"].dropna().index.get_level_values("date").min()
    assert visible == pd.Timestamp("2024-01-25")
    assert fin["market_cap"].loc[:pd.Timestamp("2024-01-24")].isna().all().all()
    # publish_date is retained and never exceeds the visibility date.
    published = pd.to_datetime(fin["publish_date"]).dropna()
    assert (published <= visible).all()


def test_financials_query_the_requested_range_not_a_fixed_quarter(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch)

    provider.get_financials("2022-01-01", "2022-12-31")

    assert fake.profit_calls
    assert all(year == 2022 or (year == 2021) for _, year, _ in fake.profit_calls)
    assert not all((year, quarter) == (2024, 4) for _, year, quarter in fake.profit_calls)


def test_empty_financials_keep_the_multiindex_contract(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows(), empty_financials=True)
    provider = make_provider(fake, tmp_path, monkeypatch)

    fin = provider.get_financials("2024-01-01", "2024-01-31")

    assert isinstance(fin.index, pd.MultiIndex)
    assert list(fin.index.names) == ["date", "asset"]
    assert fin.empty
    assert "market_cap" in fin.columns


def test_pipeline_runs_and_market_cap_accessor_works(tmp_path, monkeypatch):
    fake = FakeBaostock(basic_rows=universe_rows())
    provider = make_provider(fake, tmp_path, monkeypatch)

    pipe = DataPipeline(
        provider, "2024-01-01", "2024-01-31",
        exclude_st=True, exclude_suspended=False,
    )
    pipe.run()

    macro = pipe.get_market_cap()
    assert isinstance(macro, pd.DataFrame)
    assert not macro.empty
    assert macro.notna().any().any()
    assert set(pipe.financials.columns) >= {
        "market_cap", "total_assets", "net_assets", "revenue",
        "net_profit", "roe", "pb_ratio", "pe_ratio", "asset_growth",
    }
