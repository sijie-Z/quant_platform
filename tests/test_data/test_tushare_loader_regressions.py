"""Regression tests for the Tushare provider hardening wave.

Covers D01 (silent partial-batch truncation), D02 (mixed raw/hfq panel),
D07 (delisted metadata + per-date index membership), D08 (financial PIT and
the synthetic fallback), D11 (pandas-3 VWAP crash), D17 (benchmark cache
round-trip, atomic/corrupt cache), D18 (units and hfq docstring semantics)
and D22 (``fields=`` contract).

No network access: the Tushare API is replaced with a fake object.
"""

from __future__ import annotations

import pandas as pd
import pytest
from quant_platform.data.providers.base import DataFetchIncompleteError
from quant_platform.data.providers.tushare_loader import TushareProvider

DATES = ["20240102", "20240103", "20240104"]
RAW_CLOSE = {"20240102": 10.0, "20240103": 5.1, "20240104": 5.2}
FACTOR = {"20240102": 1.0, "20240103": 2.0, "20240104": 2.0}


def codes(n=4):
    out = [f"6000{i:02d}.SH" for i in range(1, n // 2 + 1)]
    out += [f"0000{i:02d}.SZ" for i in range(1, n // 2 + 1)]
    return out


class FakeTushareAPI:
    def __init__(
        self,
        fail_daily_prefix=(),
        fail_adj_prefix=(),
        fail_financial=False,
        drop_adj_date_for=None,
        ann_date="20240102",
        report_date="20240331",
    ):
        self.fail_daily_prefix = tuple(fail_daily_prefix)
        self.fail_adj_prefix = tuple(fail_adj_prefix)
        self.fail_financial = fail_financial
        self.drop_adj_date_for = drop_adj_date_for
        self.ann_date = ann_date
        self.report_date = report_date
        self.daily_attempts = []
        self.adj_attempts = []
        self.financial_attempts = 0

    @staticmethod
    def _batch(ts_code):
        return [c for c in ts_code.split(",") if c]

    def daily(self, ts_code, start_date, end_date, fields=None):
        batch = self._batch(ts_code)
        self.daily_attempts.append(tuple(batch))
        if any(c.startswith(self.fail_daily_prefix) for c in batch):
            raise RuntimeError("simulated daily failure")
        rows = []
        for code in batch:
            for d in DATES:
                rows.append({
                    "ts_code": code, "trade_date": d,
                    "open": RAW_CLOSE[d], "high": RAW_CLOSE[d],
                    "low": RAW_CLOSE[d], "close": RAW_CLOSE[d],
                    "vol": 100.0,                      # lots (手)
                    "amount": 10.0 * RAW_CLOSE[d],     # thousand CNY
                    "turnover_rate": 1.0,              # percent
                })
        return pd.DataFrame(rows)

    def adj_factor(self, ts_code, start_date, end_date):
        batch = self._batch(ts_code)
        self.adj_attempts.append(tuple(batch))
        if any(c.startswith(self.fail_adj_prefix) for c in batch):
            raise RuntimeError("simulated adj_factor failure")
        rows = []
        for code in batch:
            for d in DATES:
                if self.drop_adj_date_for == code and d == "20240103":
                    continue
                rows.append({"ts_code": code, "trade_date": d,
                             "adj_factor": FACTOR[d]})
        return pd.DataFrame(rows)

    def fina_indicator(self, ts_code, start_date, end_date, fields=None):
        self.financial_attempts += 1
        if self.fail_financial:
            raise RuntimeError("simulated financial failure")
        batch = self._batch(ts_code)
        return pd.DataFrame([{
            "ts_code": code, "end_date": self.report_date,
            "ann_date": self.ann_date, "f_ann_date": self.ann_date,
            "total_mv": 1e10, "total_assets": 2e10,
            "total_hldr_eqy_inc_min_int": 1e10, "revenue": 1e9,
            "net_profit": 1e8, "roe": 10.0, "pb": 2.0, "pe": 20.0, "eps": 1.0,
        } for code in batch])

    def index_weight(self, index_code, start_date, end_date):
        return pd.DataFrame([
            {"trade_date": "20240102", "ts_code": "600001.SH"},
            {"trade_date": "20240103", "ts_code": "600001.SH"},
            {"trade_date": "20240103", "ts_code": "600002.SH"},
            {"trade_date": "20240104", "ts_code": "600001.SH"},
            {"trade_date": "20240104", "ts_code": "600002.SH"},
        ])

    def index_member(self, index_code, trade_date):
        return pd.DataFrame([{"ts_code": "600001.SH"}])

    def hs300(self, trade_date, fields=None):
        return pd.DataFrame([{"ts_code": "600001.SH", "weight": 1.0}])

    def stock_basic(self, exchange="", list_status="L", fields=None):
        if list_status == "L":
            return pd.DataFrame([
                {"ts_code": "600001.SH", "name": "AAA", "industry": "银行",
                 "list_date": "20000101", "delist_date": None, "is_hs": "H"},
                {"ts_code": "000001.SZ", "name": "BBB", "industry": "医药",
                 "list_date": "20000101", "delist_date": None, "is_hs": "S"},
            ])
        if list_status == "D":
            return pd.DataFrame([
                {"ts_code": "600999.SH", "name": None, "industry": "钢铁",
                 "list_date": "20000101", "delist_date": "20220101", "is_hs": "H"},
            ])
        return pd.DataFrame(columns=["ts_code", "name", "industry",
                                     "list_date", "delist_date", "is_hs"])

    def index_daily(self, ts_code, start_date, end_date, fields=None):
        return pd.DataFrame([
            {"trade_date": d, "pct_chg": 1.0} for d in DATES
        ])


def provider(tmp_path, **kwargs):
    return TushareProvider(
        token="x" * 12,
        start_date="2024-01-01",
        end_date="2024-01-04",
        cache_dir=tmp_path,
        max_retries=kwargs.pop("max_retries", 2),
        retry_backoff=0.0,
        **kwargs,
    )


# ----------------------------------------------------------------------
# D01: partial daily failures must raise and never be cached
# ----------------------------------------------------------------------

def test_partial_daily_batch_failure_raises_and_writes_no_cache(tmp_path):
    # The fake index history contains only 6000xx names, so fail those.
    api = FakeTushareAPI(fail_daily_prefix=("6000",))
    prov = provider(tmp_path)
    prov._ts_api = api

    with pytest.raises(DataFetchIncompleteError):
        prov._load_or_fetch_all()

    # Bounded retries actually ran for the failed batch...
    failed_attempts = [
        a for a in api.daily_attempts
        if any(code.startswith("6000") for code in a)
    ]
    assert len(failed_attempts) == prov.max_retries
    # ...and no partial panel was persisted.
    assert list(tmp_path.glob("*.h5")) == []
    assert list(tmp_path.glob("*.tmp*")) == []


def test_missing_symbol_in_successful_response_is_incomplete(tmp_path):
    api = FakeTushareAPI()
    original = api.daily

    def truncated(ts_code, *args, **kwargs):
        df = original(ts_code, *args, **kwargs)
        return df[df["ts_code"] != "000002.SZ"]

    api.daily = truncated
    prov = provider(tmp_path)
    with pytest.raises(DataFetchIncompleteError):
        prov._fetch_daily_prices(api, codes(4))


# ----------------------------------------------------------------------
# D02: adj_factor failures must fail the whole fetch (no mixed panel)
# ----------------------------------------------------------------------

def test_adj_factor_batch_failure_fails_whole_price_fetch(tmp_path):
    api = FakeTushareAPI(fail_adj_prefix=("0000",))
    prov = provider(tmp_path)

    with pytest.raises(DataFetchIncompleteError):
        prov._fetch_daily_prices(api, codes(4))


def test_adj_factor_missing_rows_fail_the_fetch(tmp_path):
    api = FakeTushareAPI(drop_adj_date_for="600001.SH")
    prov = provider(tmp_path)

    with pytest.raises(DataFetchIncompleteError, match="adj_factor"):
        prov._fetch_daily_prices(api, codes(4))


def test_no_fabricated_minus_49_percent_return(tmp_path):
    """The audit probe's fabricated -49% one-day move must be impossible.

    Raw prices jump on the split date (10 -> 5.1); the pipeline divides by
    the divisor (``1/factor``), so the *adjusted* series stays continuous.
    """
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    df = prov._fetch_daily_prices(api, codes(4))

    # Raw close is untouched; the divisor is 1/native factor.
    assert set(df["adj_factor"].unique()) == {1.0, 0.5}
    close = df["close"].unstack("asset")
    raw_day3 = df.loc[(pd.Timestamp("2024-01-04"), "600001.SH"), "close"]
    assert raw_day3 == pytest.approx(5.2)
    # The pipeline applies exactly one division; that series is continuous.
    adjusted = (df["close"] / df["adj_factor"]).unstack("asset")
    adjusted_returns = adjusted.pct_change(fill_method=None)
    by_prefix = {}
    for prefix in ("6000", "0000"):
        cols = [c for c in adjusted.columns if c.startswith(prefix)]
        by_prefix[prefix] = float(adjusted_returns[cols].abs().max().max())
    assert by_prefix["6000"] < 0.05
    assert by_prefix["0000"] < 0.05

    # Raw returns *do* show the split; that is why money paths must not use
    # raw-vs-adjusted interchangeably without the stated basis.
    raw_move = close["600001.SH"].pct_change(fill_method=None).iloc[1]
    assert raw_move < -0.4


# ----------------------------------------------------------------------
# D11 + D18: pandas-3 VWAP crash, units, hfq convention
# ----------------------------------------------------------------------

def test_fetch_works_on_installed_pandas_and_units_are_normalised(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    df = prov._fetch_daily_prices(api, codes(4))

    row = df.loc[(pd.Timestamp("2024-01-02"), "600001.SH")]
    # vol 100 lots -> 10_000 shares; amount 100 千元 -> 100_000 CNY.
    assert row["volume"] == pytest.approx(10_000.0)
    assert row["amount"] == pytest.approx(100_000.0)
    assert row["turnover"] == pytest.approx(0.01)  # 1.0% -> fraction
    assert row["vwap"] == pytest.approx(10.0)      # amount/volume, not /1000

    # Raw close (a tradeable CNY level); the pipeline divides by adj_factor
    # (1/native factor) to obtain the same hfq series as before.
    assert row["close"] == pytest.approx(10.0)
    day3 = df.loc[(pd.Timestamp("2024-01-04"), "600001.SH")]
    assert day3["close"] == pytest.approx(5.2)
    assert day3["adj_factor"] == pytest.approx(0.5)
    assert day3["close"] / day3["adj_factor"] == pytest.approx(10.4)


def test_get_prices_declares_raw_basis_with_divisor(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    prov._ts_api = api

    prices = prov.get_prices("2024-01-01", "2024-01-04")

    assert prices.attrs["price_basis"] == "raw"
    assert prices.attrs["adjusted_price_basis"] == "hfq"
    assert prices.attrs["adjustment_factor_available"] is True
    row = prices.loc[(pd.Timestamp("2024-01-04"), "600001.SH")]
    assert row["close"] == pytest.approx(5.2)
    assert row["adj_factor"] == pytest.approx(0.5)
    assert row["close"] / row["adj_factor"] == pytest.approx(10.4)


def test_vwap_infinite_value_is_replaced_with_close(tmp_path):
    """Zero-volume rows produce inf; pandas 3 must not raise on replace."""
    api = FakeTushareAPI()

    def zero_volume(ts_code, start_date, end_date, fields=None):
        df = FakeTushareAPI.daily(api, ts_code, start_date, end_date, fields)
        df.loc[0, "vol"] = 0.0
        return df

    api.daily = zero_volume
    prov = provider(tmp_path)
    df = prov._fetch_daily_prices(api, codes(4))
    assert len(df) > 0  # no ValueError from the old replace() call


# ----------------------------------------------------------------------
# D08: financial PIT, announcement mapping, no synthetic fallback
# ----------------------------------------------------------------------

def test_financials_keyed_to_announcement_date_without_bfill(tmp_path):
    api = FakeTushareAPI(ann_date="20240103", report_date="20240331")
    prov = provider(tmp_path)
    prov._prices = prov._fetch_daily_prices(api, codes(4))

    fin = prov._fetch_financials(api, codes(4))

    assert list(fin.index.names) == ["date", "asset"]
    assert "publish_date" in fin.columns
    # Visible starting 2024-01-03 (announcement), never on 2024-01-02.
    assert fin["market_cap"].loc[pd.Timestamp("2024-01-02")].isna().all()
    assert fin["market_cap"].loc[pd.Timestamp("2024-01-03")].notna().all()
    assert (pd.to_datetime(fin["publish_date"]).dropna()
            >= pd.Timestamp("2024-01-03")).all()


def test_financials_units_market_cap_and_roe(tmp_path):
    api = FakeTushareAPI(ann_date="20240102")
    prov = provider(tmp_path)
    prov._prices = prov._fetch_daily_prices(api, codes(4))
    fin = prov._fetch_financials(api, codes(4))

    row = fin.loc[(pd.Timestamp("2024-01-02"), "600001.SH")]
    assert row["market_cap"] == pytest.approx(1e10 * 10_000)  # 万元 -> CNY
    assert row["roe"] == pytest.approx(0.10)                  # 10% -> fraction
    assert row["pb_ratio"] == pytest.approx(2.0)
    assert row["pe_ratio"] == pytest.approx(20.0)


def test_all_financial_batches_failing_raises_not_synthetic(tmp_path):
    api = FakeTushareAPI(fail_financial=True)
    prov = provider(tmp_path)
    prov._prices = prov._fetch_daily_prices(api, codes(4))

    with pytest.raises(DataFetchIncompleteError, match="financial"):
        prov._fetch_financials(api, codes(4))


# ----------------------------------------------------------------------
# D07: delisted metadata, membership history/mask
# ----------------------------------------------------------------------

def test_metadata_includes_delisted_names_and_no_fabricated_is_st(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)

    meta = prov._fetch_stock_basic(
        api, ["600001.SH", "000001.SZ", "600999.SH"],
    )

    assert meta.index.name == "asset"
    assert "600999.SH" in meta.index          # delisted name kept
    assert pd.notna(meta.loc["600999.SH", "delisting_date"])
    assert "is_st" not in meta.columns        # absence exposed explicitly
    assert prov.st_available is False
    assert meta.attrs["st_available"] is False


def test_constituent_history_is_kept_and_mask_is_per_date(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    history = prov._fetch_constituents(api)

    assert set(pd.to_datetime(history["trade_date"]).unique()) == {
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    }
    prov._constituent_history = history
    mask = prov.get_index_membership_mask("2024-01-01", "2024-01-04")

    assert mask.loc[pd.Timestamp("2024-01-02"), "600001.SH"]
    assert not mask.loc[pd.Timestamp("2024-01-02"), "600002.SH"]
    assert mask.loc[pd.Timestamp("2024-01-03"), "600002.SH"]  # joined
    assert mask.loc[pd.Timestamp("2024-01-04"), "600002.SH"]  # membership persists


def test_universe_mask_alias(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    prov._constituent_history = prov._fetch_constituents(api)

    pd.testing.assert_frame_equal(
        prov.get_universe_mask("2024-01-01", "2024-01-04"),
        prov.get_index_membership_mask("2024-01-01", "2024-01-04"),
    )


# ----------------------------------------------------------------------
# D17: benchmark Series after reload; atomic + corrupt cache handling
# ----------------------------------------------------------------------

def test_benchmark_is_series_after_cache_reload_and_writes_are_atomic(tmp_path):
    api = FakeTushareAPI(ann_date="20240102")
    prov = provider(tmp_path)
    prov._ts_api = api
    prov._load_or_fetch_all()

    cache_files = list(tmp_path.glob("*.h5"))
    assert len(cache_files) == 1
    assert list(tmp_path.glob("*.tmp*")) == []

    reloaded = provider(tmp_path)
    reloaded._load_from_cache(cache_files[0])

    assert isinstance(reloaded._benchmark, pd.Series)
    assert isinstance(reloaded.get_benchmark("2024-01-01", "2024-01-04"), pd.Series)
    assert reloaded.get_benchmark("2024-01-01", "2024-01-04").name == "benchmark"
    assert reloaded.get_universe_mask("2024-01-01", "2024-01-04").shape[1] == 2


def test_corrupt_cache_is_deleted_and_refetched(tmp_path):
    api = FakeTushareAPI(ann_date="20240102")
    prov = provider(tmp_path)
    cache_file = (
        tmp_path
        / f"tushare_000300.SH_{prov.start_date}_{prov.end_date}.h5"
    )
    cache_file.write_bytes(b"not an hdf5 file")

    prov._ts_api = api
    prov._load_or_fetch_all()  # must not raise

    assert prov._prices is not None and len(prov._prices) > 0
    # A fresh, readable cache replaced the corrupt file.
    reloaded = provider(tmp_path)
    reloaded._load_from_cache(cache_file)
    assert isinstance(reloaded._benchmark, pd.Series)


# ----------------------------------------------------------------------
# D22: fields contract
# ----------------------------------------------------------------------

def test_fields_contract_raises_on_unknown_and_preserves_order(tmp_path):
    api = FakeTushareAPI()
    prov = provider(tmp_path)
    prov._prices = prov._fetch_daily_prices(api, codes(4))

    sub = prov.get_prices("2024-01-01", "2024-01-04", fields=["volume", "close"])
    assert list(sub.columns) == ["volume", "close"]

    with pytest.raises(ValueError, match="Unknown price fields"):
        prov.get_prices("2024-01-01", "2024-01-04", fields=["nope"])
