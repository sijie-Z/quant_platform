"""Corporate-action provider contract, synthetic feed and pipeline accessor.

Covers:
- the frame schema / empty contract (never ``None``);
- synthetic splits + cash dividends staying consistent with ``adj_factor``
  (the adjusted research path is continuous and unchanged);
- ``DataPipeline.get_corporate_actions()`` alignment and availability flag;
- best-effort Tushare ``dividend`` / Baostock ``query_dividend_data`` parsing
  behind fakes, including the unavailable-API path (no fabricated amounts).
"""

from __future__ import annotations

import types

import pandas as pd
import pytest
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.data.providers.base import (
    CORPORATE_ACTION_COLUMNS,
    DataProvider,
    empty_corporate_actions,
    normalize_corporate_actions,
)
from quant_platform.data.providers.synthetic import SyntheticDataProvider
from quant_platform.data.providers.tushare_loader import TushareProvider

START, END = "2023-01-01", "2024-12-31"


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_empty_frame_has_the_documented_schema():
    frame = empty_corporate_actions()
    assert list(frame.columns) == list(CORPORATE_ACTION_COLUMNS)
    assert frame.empty
    assert str(frame["ex_date"].dtype) == "datetime64[ns]"
    assert frame["cash_per_share"].dtype == "float64"
    assert frame["split_ratio"].dtype == "float64"


def test_combined_action_row_is_split_into_cash_and_split():
    frame = normalize_corporate_actions(pd.DataFrame([{
        "ex_date": "2024-06-20",
        "asset": "600519",
        "kind": "cash",
        "cash_per_share": 0.5,
        "split_ratio": 1.5,
    }]))
    assert len(frame) == 2
    cash = frame[frame["kind"] == "cash"].iloc[0]
    split = frame[frame["kind"] == "split"].iloc[0]
    assert cash["cash_per_share"] == pytest.approx(0.5)
    assert cash["split_ratio"] == 1.0
    assert split["split_ratio"] == pytest.approx(1.5)
    assert split["cash_per_share"] == 0.0
    assert cash["asset"] == "600519.SH"  # canonical form


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        normalize_corporate_actions(pd.DataFrame({"ex_date": ["2024-01-01"]}))


class _NoFeedProvider(SyntheticDataProvider):
    """Synthetic data, but no corporate-action capability (base default)."""

    corporate_actions_available = False

    def get_corporate_actions(self, start_date, end_date):
        return DataProvider.get_corporate_actions(self, start_date, end_date)


# ---------------------------------------------------------------------------
# Synthetic feed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def provider():
    prov = SyntheticDataProvider(n_stocks=100, start_date=START, end_date=END, seed=123)
    prov._generate_all()
    return prov


@pytest.fixture(scope="module")
def actions(provider):
    return provider.get_corporate_actions(START, END)


def test_synthetic_generates_both_splits_and_cash(actions):
    assert not actions.empty
    assert {"cash", "split"} <= set(actions["kind"])
    assert list(actions.columns) == list(CORPORATE_ACTION_COLUMNS)
    assert (actions.loc[actions["kind"] == "cash", "cash_per_share"] > 0).all()
    assert (actions.loc[actions["kind"] == "split", "split_ratio"] > 1).all()


def test_synthetic_actions_are_deterministic_by_seed():
    other = SyntheticDataProvider(
        n_stocks=100, start_date=START, end_date=END, seed=123,
    )
    pd.testing.assert_frame_equal(
        other.get_corporate_actions(START, END),
        SyntheticDataProvider(
            n_stocks=100, start_date=START, end_date=END, seed=123,
        ).get_corporate_actions(START, END),
    )


def test_synthetic_adjusted_path_is_the_generated_continuous_path(provider):
    prices = provider.get_prices(START, END)
    close_adj = (prices["close"] / prices["adj_factor"]).unstack("asset")
    continuous = provider._continuous_close
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    comparable = close_adj.notna() & continuous.notna() & ~suspended
    diff = (close_adj - continuous).abs().where(comparable)
    assert float(diff.max().max()) < 1e-9


def test_synthetic_split_is_absorbed_by_the_adjustment_factor(provider, actions):
    prices = provider.get_prices(START, END)
    factor = prices["adj_factor"].unstack("asset")
    suspended = prices["is_suspended"].unstack("asset").fillna(False).astype(bool)
    splits = actions[actions["kind"] == "split"]
    checked = 0
    for _, action in splits.iterrows():
        asset, ex_date = action["asset"], action["ex_date"]
        if asset not in factor.columns or ex_date not in factor.index:
            continue
        if bool(suspended.loc[ex_date, asset]):
            continue
        position = factor.index.get_loc(ex_date)
        previous = factor[asset].iloc[:position][~suspended[asset].iloc[:position]]
        if previous.empty or pd.isna(factor.loc[ex_date, asset]):
            continue
        ratio = previous.iloc[-1] / factor.loc[ex_date, asset]
        assert ratio == pytest.approx(action["split_ratio"], rel=1e-9)
        checked += 1
    assert checked >= 1, "fixture must contain a non-suspended split"


def test_research_returns_are_unchanged_by_corporate_actions(provider):
    """The pipeline's return series is the generated path, action days included."""
    pipe = DataPipeline(
        provider, START, END, exclude_st=False, exclude_suspended=False,
    )
    pipe.run()
    returns = pipe.returns
    action_dates = provider.get_corporate_actions(START, END)["ex_date"]
    for ex_date in action_dates.unique():
        if ex_date not in returns.index:
            continue
        row = returns.loc[ex_date].dropna()
        assert (row.abs() <= 0.11).all()
    # No fabricated corporate-action jump anywhere in the adjusted series.
    adjusted = pipe.get_adjusted_close()
    assert float(adjusted.pct_change(fill_method=None).abs().max().max()) < 0.12


# ---------------------------------------------------------------------------
# Pipeline accessor
# ---------------------------------------------------------------------------


def test_pipeline_accessor_is_aligned_to_the_sample(provider):
    pipe = DataPipeline(
        provider, START, END, exclude_st=False, exclude_suspended=False,
    )
    pipe.run()
    frame = pipe.get_corporate_actions()
    assert pipe.corporate_actions_available is True
    assert list(frame.columns) == list(CORPORATE_ACTION_COLUMNS)
    assets = set(pipe.prices.index.get_level_values("asset").unique())
    assert set(frame["asset"]) <= assets
    assert frame["ex_date"].min() >= pipe.start_date
    assert frame["ex_date"].max() <= pipe.end_date


def test_pipeline_without_a_feed_returns_the_empty_contract():
    provider = _NoFeedProvider(
        n_stocks=30, start_date=START, end_date=END, seed=5,
    )
    pipe = DataPipeline(provider, START, END)
    pipe.run()
    frame = pipe.get_corporate_actions()
    assert frame.empty
    assert list(frame.columns) == list(CORPORATE_ACTION_COLUMNS)
    assert pipe.corporate_actions_available is False


# ---------------------------------------------------------------------------
# Tushare (mock)
# ---------------------------------------------------------------------------


class _FakeDividendAPI:
    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error
        self.calls: list[dict] = []

    def dividend(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return pd.DataFrame(self.rows)


def _tushare_provider(tmp_path, api, max_retries=1):
    provider = TushareProvider(
        token="x" * 20, cache_dir=tmp_path / "ts", max_retries=max_retries,
    )
    provider._ts_api = api
    return provider


def test_tushare_dividend_fetch_maps_cash_and_split(tmp_path):
    api = _FakeDividendAPI(rows=[
        {"ts_code": "600519.SH", "ex_date": "20240115", "cash_div": 0.4,
         "cash_div_tax": 0.5, "stk_div": 0.0, "div_proc": "实施"},
        {"ts_code": "000001.SZ", "ex_date": "20240620", "cash_div": 0.0,
         "cash_div_tax": 0.0, "stk_div": 0.5, "div_proc": "实施"},
        {"ts_code": "600000.SH", "ex_date": "20240701", "cash_div": 1.0,
         "cash_div_tax": 1.0, "stk_div": 0.0, "div_proc": "预案"},
        {"ts_code": "600001.SH", "ex_date": "20221201", "cash_div": 1.0,
         "cash_div_tax": 1.0, "stk_div": 0.0, "div_proc": "实施"},
    ])
    provider = _tushare_provider(tmp_path, api)

    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert provider.corporate_actions_available is True
    assert len(frame) == 2  # 预案 and out-of-range rows dropped
    cash = frame[frame["kind"] == "cash"].iloc[0]
    split = frame[frame["kind"] == "split"].iloc[0]
    assert cash["asset"] == "600519.SH"
    assert cash["cash_per_share"] == pytest.approx(0.5)  # pre-tax amount
    assert split["asset"] == "000001.SZ"
    assert split["split_ratio"] == pytest.approx(1.5)


def test_tushare_unavailable_api_returns_empty_and_marks_unavailable(tmp_path):
    provider = _tushare_provider(
        tmp_path, _FakeDividendAPI(error=RuntimeError("api down")),
    )
    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")
    assert frame.empty
    assert list(frame.columns) == list(CORPORATE_ACTION_COLUMNS)
    assert provider.corporate_actions_available is False


# ---------------------------------------------------------------------------
# Baostock (mock)
# ---------------------------------------------------------------------------


class _FakeResultSet:
    def __init__(self, rows, fields, error_code="0"):
        self._rows = list(rows)
        self.fields = list(fields)
        self.error_code = error_code
        self.error_msg = ""
        self._index = -1

    def next(self):
        self._index += 1
        return self._index < len(self._rows)

    def get_row_data(self):
        return self._rows[self._index]


_BS_FIELDS = [
    "code", "dividOperateDate", "dividCashPsBeforeTax", "dividCashPsAfterTax",
    "dividStocksPs", "dividReserveToStockPs",
]


class _FakeBaostock:
    def __init__(self, rows=None, error_code="0", raise_exc=None):
        self.rows = rows or []
        self.error_code = error_code
        self.raise_exc = raise_exc
        self.queries: list[tuple] = []

    def query_dividend_data(self, code, year, yearType):
        self.queries.append((code, year, yearType))
        if self.raise_exc is not None:
            raise self.raise_exc
        return _FakeResultSet(self.rows, _BS_FIELDS, self.error_code)


def _baostock_provider(fake):
    provider = BaostockDataProvider(cache_enabled=False)
    provider._bs = fake
    provider.corporate_action_codes = ["600519"]
    return provider


def test_baostock_dividend_fetch_maps_cash_and_bonus(monkeypatch):
    fake = _FakeBaostock(rows=[
        ["sh.600519", "2024-06-20", "0.50", "0.45", "0.00", "0.00"],
        ["sh.600519", "2024-09-10", "0.00", "0.00", "0.30", "0.20"],
    ])
    monkeypatch.setitem(
        __import__("sys").modules, "baostock", types.SimpleNamespace(),
    )
    provider = _baostock_provider(fake)

    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")

    assert provider.corporate_actions_available is True
    assert len(frame) == 2
    cash = frame[frame["kind"] == "cash"].iloc[0]
    split = frame[frame["kind"] == "split"].iloc[0]
    assert cash["asset"] == "600519.SH"
    assert cash["cash_per_share"] == pytest.approx(0.5)
    assert split["split_ratio"] == pytest.approx(1.5)  # 1 + 0.3 + 0.2
    assert all(query[2] == "operate" for query in fake.queries)


def test_baostock_unavailable_api_returns_empty_and_marks_unavailable(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "baostock", types.SimpleNamespace(),
    )
    provider = _baostock_provider(
        _FakeBaostock(error_code="1"),
    )
    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")
    assert frame.empty
    assert list(frame.columns) == list(CORPORATE_ACTION_COLUMNS)
    assert provider.corporate_actions_available is False


def test_baostock_missing_module_returns_empty(monkeypatch):
    """No baostock install: import fails inside ``_login`` -> unavailable."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "baostock":
            raise ImportError("baostock not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    provider = BaostockDataProvider(cache_enabled=False)
    provider.corporate_action_codes = ["600519"]
    frame = provider.get_corporate_actions("2024-01-01", "2024-12-31")
    assert frame.empty
    assert provider.corporate_actions_available is False
