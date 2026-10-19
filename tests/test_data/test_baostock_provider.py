"""Regression tests for BaostockDataProvider universe selection and caching.

Two bugs are pinned here:

1. ``_get_major_stocks_list`` returned ``sorted(self._get_major_stocks())[:500]``
   which is 000001-000500 only -- the whole cached panel held 101 Shenzhen
   main-board codes and no Shanghai/ChiNext/STAR names. The list must be a
   deterministic stratified sample across the 60/68/00/30 prefixes, preferably
   from baostock's ``query_stock_basic`` (type=1, status=1), falling back to a
   cross-exchange candidate pool when that call is unavailable.

2. ``_fetch_daily_data`` keyed its cache on ``len(codes)`` only, so any two
   universes of the same size shared one cache entry. The key must include a
   stable hash of the full sorted code list.

No network access: baostock is replaced with a fake module.
"""

from __future__ import annotations

import sys
import types

import pytest
from quant_platform.data.providers import baostock_provider as mod
from quant_platform.data.providers.baostock_provider import BaostockDataProvider

FIELDS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]


class FakeResultSet:
    def __init__(self, rows, fields=None, error_code="0"):
        self._rows = list(rows)
        self.fields = list(fields or FIELDS)
        self.error_code = error_code
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class FakeBaostock:
    def __init__(self, basic_rows=None, basic_error=None):
        self.basic_rows = list(basic_rows or [])
        self.basic_error = basic_error
        self.basic_query_count = 0
        self.history_queries = []

    def login(self):
        return types.SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        return None

    def query_stock_basic(self, code="", code_name=""):
        self.basic_query_count += 1
        if self.basic_error is not None:
            raise self.basic_error
        return FakeResultSet(self.basic_rows)

    def query_history_k_data_plus(self, *args, **kwargs):
        self.history_queries.append((args, kwargs))
        return FakeResultSet([], error_code="0")


def _basic_row(code, security_type="1", status="1"):
    exchange = "sh" if code.startswith(("6", "9")) else "sz"
    return [f"{exchange}.{code}", f"name_{code}", "2020-01-01", "", security_type, status]


def _universe():
    rows = []
    for i in range(200):
        rows.append(_basic_row(f"{600000 + i:06d}"))
    for i in range(100):
        rows.append(_basic_row(f"{688001 + i:06d}"))
    for i in range(299):
        rows.append(_basic_row(f"{i + 1:06d}"))
    for i in range(150):
        rows.append(_basic_row(f"{300001 + i:06d}"))
    # Noise that must be filtered out: an index and a delisted stock.
    rows.append(_basic_row("000300", security_type="2", status="1"))
    rows.append(_basic_row("600999", security_type="1", status="0"))
    return rows


def _provider_with(fake, monkeypatch, cache_enabled=False):
    monkeypatch.setitem(sys.modules, "baostock", fake)
    provider = BaostockDataProvider(cache_enabled=cache_enabled)
    provider._bs = fake
    return provider


def test_major_stock_list_spans_exchanges_and_boards(monkeypatch):
    fake = FakeBaostock(basic_rows=_universe())
    provider = _provider_with(fake, monkeypatch)

    codes = provider._get_major_stocks_list()

    assert len(codes) == 500
    prefixes = {c[:2] for c in codes}
    assert {"60", "68", "00", "30"} <= prefixes
    assert "000300" not in codes  # index filtered out by type=1
    assert "600999" not in codes  # delisted filtered out by status=1
    assert all(len(c) == 6 and c.isdigit() for c in codes)


def test_major_stock_list_is_deterministic(monkeypatch):
    fake = FakeBaostock(basic_rows=_universe())
    provider = _provider_with(fake, monkeypatch)

    assert provider._get_major_stocks_list() == provider._get_major_stocks_list()


def test_major_stock_list_size_matches_requested_n(monkeypatch):
    fake = FakeBaostock(basic_rows=_universe())
    provider = _provider_with(fake, monkeypatch)

    codes = provider._get_major_stocks_list(n=120)

    assert len(codes) == 120
    assert {"60", "00"} <= {c[:2] for c in codes}


def test_tiny_sample_still_covers_both_exchanges(monkeypatch):
    rows = [_basic_row(f"{600000 + i:06d}") for i in range(1000)]
    rows += [_basic_row("688001"), _basic_row("000001"), _basic_row("300001")]
    fake = FakeBaostock(basic_rows=rows)
    provider = _provider_with(fake, monkeypatch)

    codes = provider._get_major_stocks_list(n=2)

    assert len(codes) == 2
    assert any(c.startswith(("60", "68")) for c in codes)
    assert any(c.startswith(("00", "30")) for c in codes)


def test_fallback_pool_spans_exchanges_when_query_stock_basic_fails(monkeypatch):
    fake = FakeBaostock(basic_error=RuntimeError("offline"))
    provider = _provider_with(fake, monkeypatch)

    codes = provider._get_major_stocks_list()

    assert len(codes) == 500
    assert any(c.startswith(("60", "68")) for c in codes)
    assert any(c.startswith(("00", "30")) for c in codes)
    # The old bug: sorted(candidates)[:500] was 000001-000500 only.
    assert codes != sorted(provider._get_major_stocks())[:500]


def test_stock_basic_universe_is_cached(monkeypatch, tmp_path):
    fake = FakeBaostock(basic_rows=_universe())
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    provider = _provider_with(fake, monkeypatch, cache_enabled=True)

    provider._get_major_stocks_list()
    provider._get_major_stocks_list()

    assert fake.basic_query_count == 1
    assert (tmp_path / "stock_basic_universe.json").exists()


def test_stock_basic_universe_cache_does_not_depend_on_parquet(monkeypatch, tmp_path):
    """Clean installs may lack pyarrow; the universe cache must not need it."""
    fake = FakeBaostock(basic_rows=_universe())
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    provider = _provider_with(fake, monkeypatch, cache_enabled=True)
    monkeypatch.setattr(provider, "_load_cache", lambda key: pytest.fail("parquet load used"))
    monkeypatch.setattr(provider, "_save_cache", lambda key, df: pytest.fail("parquet save used"))

    provider._get_major_stocks_list()
    provider._get_major_stocks_list()

    assert fake.basic_query_count == 1


def test_daily_cache_key_includes_full_code_list(monkeypatch):
    fake = FakeBaostock()
    provider = _provider_with(fake, monkeypatch, cache_enabled=True)
    keys = []
    monkeypatch.setattr(provider, "_load_cache", lambda key: keys.append(key) or None)
    monkeypatch.setattr(provider, "_save_cache", lambda key, df: None)

    provider._fetch_daily_data(["000001", "600000"], "2024-01-02", "2024-01-03")
    provider._fetch_daily_data(["000001", "300750"], "2024-01-02", "2024-01-03")
    provider._fetch_daily_data(["600000", "000001"], "2024-01-02", "2024-01-03")

    assert keys[0] != keys[1], "different universes of the same size share a cache key"
    assert keys[0] == keys[2], "the cache key must not depend on code order"
