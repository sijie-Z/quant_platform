"""Incremental data updates (D32) — provider boundary and pipeline equality.

Network-free: the Tushare API and baostock module are replaced with fakes.
Pins the required behaviours:

* extending a cached 5-session panel to 8 sessions appends exactly 3 and the
  resulting provider/pipeline panels are byte-identical to a full fetch;
* a partial provider response marks the update incomplete, does not advance
  the completion marker and keeps the old cache;
* a split after the cached range leaves the adjusted series continuous while
  raw prices stay untouched (adjustment refresh);
* a changed constituent universe forces a full refetch;
* a write failure leaves the old cache and marker intact.
"""

from __future__ import annotations

import pickle
import sys
import types

import pandas as pd
import pytest
import quant_platform.data.providers.baostock_provider as bs_mod
import quant_platform.data.providers.tushare_loader as ts_mod
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider
from quant_platform.data.providers.base import (
    DataFetchIncompleteError,
    expected_data_end,
    incremental_fetch_start,
    incremental_marker_extra,
    sessions_after,
)
from quant_platform.data.providers.tushare_loader import TushareProvider

SESSIONS = [
    "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
    "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
]
CODES = ["600001.SH", "600002.SH", "600003.SH", "000001.SZ"]
CACHED_END = "2024-01-08"   # SESSIONS[4]
REQUESTED_END = "2024-01-11"


def _compact(day: str) -> str:
    return day.replace("-", "")


class RangeTushareAPI:
    """Deterministic Tushare fake that honours requested date windows."""

    def __init__(
        self,
        *,
        raw_close: dict | None = None,
        native_factor: dict | None = None,
        codes: list[str] | None = None,
        omit: set[tuple[str, str]] | None = None,
        omit_after: str | None = None,
        omit_last_day_for: set[str] | None = None,
        adj_full_omit_codes: set[str] | None = None,
        universe_extra_from: tuple[str, str] | None = None,
    ):
        self.codes = list(codes or CODES)
        self.raw_close = dict(raw_close or {
            day: 10.0 + index for index, day in enumerate(SESSIONS)
        })
        self.native_factor = dict(native_factor or {
            day: 1.0 for day in SESSIONS
        })
        self.omit = set(omit or ())
        self.omit_after = omit_after
        self.omit_last_day_for = set(omit_last_day_for or ())
        self.adj_full_omit_codes = set(adj_full_omit_codes or ())
        self.universe_extra_from = universe_extra_from
        self.daily_calls = 0

    def _window(self, start_date: str, end_date: str) -> list[str]:
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        return [day for day in SESSIONS if start <= day <= end]

    def _members(self, day: str) -> set[str]:
        members = set(self.codes)
        if self.universe_extra_from:
            extra_day, extra_code = self.universe_extra_from
            if day >= extra_day:
                members.add(extra_code)
        return members

    def _omitted(self, day: str, code: str, window: list[str]) -> bool:
        if (day, code) in self.omit:
            return True
        if self.omit_after is not None and day > self.omit_after:
            return True
        return bool(
            window and day == window[-1] and code in self.omit_last_day_for
        )

    def daily(self, ts_code, start_date, end_date, fields=None):
        self.daily_calls += 1
        batch = [code for code in ts_code.split(",") if code]
        window = self._window(start_date, end_date)
        rows = []
        for code in batch:
            for day in window:
                if code not in self._members(day):
                    continue
                if self._omitted(day, code, window):
                    continue
                price = self.raw_close[day]
                rows.append({
                    "ts_code": code, "trade_date": _compact(day),
                    "open": price, "high": price, "low": price,
                    "close": price,
                    "vol": 100.0,                  # lots
                    "amount": 10.0 * price,        # thousand CNY
                    "turnover_rate": 1.0,          # percent
                })
        return pd.DataFrame(rows)

    def adj_factor(self, ts_code, start_date, end_date):
        batch = [code for code in ts_code.split(",") if code]
        window = self._window(start_date, end_date)
        full_range = (
            bool(window) and window[0] == SESSIONS[0]
        )
        rows = []
        for code in batch:
            if full_range and code in self.adj_full_omit_codes:
                continue
            for day in window:
                if code not in self._members(day):
                    continue
                if self._omitted(day, code, window):
                    continue
                rows.append({
                    "ts_code": code, "trade_date": _compact(day),
                    "adj_factor": self.native_factor[day],
                })
        return pd.DataFrame(rows)

    def index_weight(self, index_code, start_date, end_date):
        rows = []
        for day in self._window(start_date, end_date):
            for code in self._members(day):
                rows.append({
                    "trade_date": _compact(day), "ts_code": code,
                })
        return pd.DataFrame(rows)

    def index_member(self, index_code, trade_date):
        return pd.DataFrame([{"ts_code": code} for code in self.codes])

    def hs300(self, trade_date, fields=None):
        return pd.DataFrame(
            [{"ts_code": code, "weight": 1.0} for code in self.codes]
        )

    def stock_basic(self, exchange="", list_status="L", fields=None):
        if list_status != "L":
            return pd.DataFrame(
                columns=["ts_code", "name", "industry", "list_date",
                         "delist_date", "is_hs"]
            )
        codes = set(self.codes)
        if self.universe_extra_from:
            codes.add(self.universe_extra_from[1])
        return pd.DataFrame([
            {
                "ts_code": code, "name": code, "industry": "银行",
                "list_date": "20000101", "delist_date": None, "is_hs": "H",
            }
            for code in sorted(codes)
        ])

    def fina_indicator(self, ts_code, start_date, end_date, fields=None):
        batch = [code for code in ts_code.split(",") if code]
        return pd.DataFrame([
            {
                "ts_code": code, "end_date": "20240331",
                "ann_date": _compact(SESSIONS[0]),
                "f_ann_date": _compact(SESSIONS[0]),
                "total_mv": 1e10, "total_assets": 2e10,
                "total_hldr_eqy_inc_min_int": 1e10, "revenue": 1e9,
                "net_profit": 1e8, "roe": 10.0, "pb": 2.0, "pe": 20.0,
                "eps": 1.0,
            }
            for code in batch
        ])

    def index_daily(self, ts_code, start_date, end_date, fields=None):
        return pd.DataFrame([
            {"trade_date": _compact(day), "pct_chg": 0.5}
            for day in self._window(start_date, end_date)
        ])


def tushare_provider(cache_dir, end_date=REQUESTED_END, api=None):
    provider = TushareProvider(
        token="x" * 12,
        start_date="2024-01-01",
        end_date=end_date,
        cache_dir=cache_dir,
        max_retries=1,
        retry_backoff=0.0,
    )
    provider._ts_api = api or RangeTushareAPI()
    return provider


def _seed_cache(tmp_path, api=None, end_date=CACHED_END):
    """Full fetch a panel into ``tmp_path``."""
    provider = tushare_provider(tmp_path, end_date=end_date, api=api)
    provider._load_or_fetch_all()
    return provider


# ----------------------------------------------------------------------
# Append, boundary dedup and full-fetch equality
# ----------------------------------------------------------------------


def test_append_three_sessions_equals_full_fetch(tmp_path):
    new_dir = tmp_path / "new"
    fresh_dir = tmp_path / "fresh"
    _seed_cache(new_dir)

    updated = tushare_provider(new_dir, api=RangeTushareAPI())
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["performed"] is True
    assert status["incomplete"] is False
    assert status["full_refetch"] is False
    assert status["new_sessions"] == 3
    assert status["last_complete_date"] == REQUESTED_END

    dates = pd.DatetimeIndex(
        updated._prices.index.get_level_values("date").unique()
    )
    assert len(dates) == 8
    assert not updated._prices.index.duplicated().any()
    # Exactly the three sessions after the marker were appended.
    new = sorted(day.date().isoformat() for day in dates)[5:]
    assert new == ["2024-01-09", "2024-01-10", "2024-01-11"]

    # The merged panel is byte-identical to a from-scratch full fetch.
    fresh = tushare_provider(fresh_dir, api=RangeTushareAPI())
    fresh._load_or_fetch_all()
    pd.testing.assert_frame_equal(
        updated._prices, fresh._prices, check_exact=True,
    )
    assert updated._prices.to_csv() == fresh._prices.to_csv()

    # ... and the incremental cache file reloads to the same panel.
    reloaded = tushare_provider(new_dir, api=RangeTushareAPI())
    reloaded._load_or_fetch_all()
    pd.testing.assert_frame_equal(reloaded._prices, fresh._prices)
    assert reloaded.data_status()["last_complete_date"] == REQUESTED_END

    # The pipeline output is identical too (the task's byte-for-byte rule).
    def run(provider):
        pipeline = DataPipeline(
            provider=provider,
            start_date="2024-01-01",
            end_date=REQUESTED_END,
        )
        pipeline.run()
        return pipeline

    incremental_pipeline = run(updated)
    fresh_pipeline = run(fresh)
    pd.testing.assert_frame_equal(
        incremental_pipeline.prices, fresh_pipeline.prices,
    )
    pd.testing.assert_frame_equal(
        incremental_pipeline.returns, fresh_pipeline.returns,
    )
    pd.testing.assert_frame_equal(
        incremental_pipeline.financials, fresh_pipeline.financials,
    )
    pd.testing.assert_series_equal(
        incremental_pipeline.benchmark, fresh_pipeline.benchmark,
    )
    assert incremental_pipeline.data_freshness["last_complete_date"] == (
        REQUESTED_END
    )


def test_boundary_overlap_is_refetched_and_deduplicated(tmp_path):
    """The overlap session is re-read; fresh rows win and duplicates vanish."""
    api = RangeTushareAPI()
    _seed_cache(tmp_path)

    # Revise the boundary session's close in the provider's history.
    api.raw_close[CACHED_END] = 10.9
    updated = tushare_provider(tmp_path, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["new_sessions"] == 3
    assert not updated._prices.index.duplicated().any()
    boundary = updated._prices.loc[
        (pd.Timestamp(CACHED_END), "600001.SH"), "close"
    ]
    assert boundary == pytest.approx(10.9)


# ----------------------------------------------------------------------
# Adjustment refresh across a split
# ----------------------------------------------------------------------


def test_split_after_cached_range_keeps_adjusted_series_continuous(tmp_path):
    raw = {
        "2024-01-02": 10.0, "2024-01-03": 10.1, "2024-01-04": 10.2,
        "2024-01-05": 10.3, "2024-01-08": 10.4,
        # 2:1 split on 2024-01-09: raw halves, native factor doubles.
        "2024-01-09": 5.25, "2024-01-10": 5.30, "2024-01-11": 5.35,
    }
    factor = {day: (1.0 if day < "2024-01-09" else 2.0) for day in SESSIONS}
    _seed_cache(tmp_path, api=RangeTushareAPI(
        raw_close=raw, native_factor=factor,
    ))

    updated = tushare_provider(tmp_path, api=RangeTushareAPI(
        raw_close=raw, native_factor=factor,
    ))
    status = updated.update("2024-01-01", REQUESTED_END)
    assert status["incomplete"] is False
    assert status["new_sessions"] == 3

    prices = updated._prices
    # Raw prices are untouched by the append (the 10.4 -> 5.25 drop is real).
    assert prices.loc[(pd.Timestamp(CACHED_END), "600001.SH"), "close"] == (
        pytest.approx(10.4)
    )
    assert prices.loc[(pd.Timestamp("2024-01-09"), "600001.SH"), "close"] == (
        pytest.approx(5.25)
    )
    assert prices.loc[(pd.Timestamp("2024-01-09"), "600001.SH"), "adj_factor"] == (
        pytest.approx(0.5)
    )

    adjusted = (prices["close"] / prices["adj_factor"]).unstack("asset")
    adjusted_returns = adjusted.pct_change(fill_method=None).abs()
    # No fabricated -50% return anywhere in the adjusted series.
    assert float(adjusted_returns.max().max()) < 0.05
    assert adjusted.loc[pd.Timestamp("2024-01-09"), "600001.SH"] == (
        pytest.approx(10.5)
    )
    # Raw prices *do* show the split.
    raw_returns = prices["close"].unstack("asset").pct_change(fill_method=None)
    assert float(raw_returns.loc[
        pd.Timestamp("2024-01-09"), "600001.SH"
    ]) < -0.4


# ----------------------------------------------------------------------
# Partial responses and the completion marker
# ----------------------------------------------------------------------


def _cache_marker(cache_dir):
    from quant_platform.data.providers.base import (
        cache_manifest_path,
        read_cache_manifest,
    )

    path = cache_dir / (
        f"tushare_000300.SH_2024-01-01_{CACHED_END}.h5"
    )
    assert cache_manifest_path(path).exists()
    return read_cache_manifest(path)


def _cache_marker_for(provider):
    from quant_platform.data.providers.base import read_cache_manifest

    return read_cache_manifest(provider._cache_file())


def test_partial_response_marks_incomplete_and_keeps_old_cache(tmp_path):
    _seed_cache(tmp_path)
    old_bytes = (
        tmp_path / f"tushare_000300.SH_2024-01-01_{CACHED_END}.h5"
    ).read_bytes()

    # 000001.SZ never appears in the incremental window.
    api = RangeTushareAPI(omit={
        (day, "000001.SZ") for day in SESSIONS if day > CACHED_END
    })
    updated = tushare_provider(tmp_path, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["performed"] is False
    assert status["new_sessions"] == 0
    assert "000001.SZ" in status["missing_symbols"]
    assert status["last_complete_date"] == CACHED_END

    # Old cache byte-for-byte intact, marker not advanced, no new data file.
    assert (
        tmp_path / f"tushare_000300.SH_2024-01-01_{CACHED_END}.h5"
    ).read_bytes() == old_bytes
    assert _cache_marker(tmp_path)["last_complete_date"] == CACHED_END
    assert not (
        tmp_path / f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    ).exists()

    # The last complete panel keeps being served (flagged as incomplete).
    assert len(
        updated._prices.index.get_level_values("date").unique()
    ) == 5
    assert updated.data_status()["incomplete"] is True


def test_truncated_last_day_does_not_advance_marker(tmp_path):
    _seed_cache(tmp_path)
    # One symbol appears earlier in the window but is missing on the last day.
    api = RangeTushareAPI(omit={
        ("2024-01-11", "600003.SH"),
    })
    updated = tushare_provider(tmp_path, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["last_complete_date"] == CACHED_END
    assert _cache_marker(tmp_path)["last_complete_date"] == CACHED_END


# ----------------------------------------------------------------------
# Universe change -> full refetch
# ----------------------------------------------------------------------


def test_universe_change_triggers_full_refetch(tmp_path):
    new_dir = tmp_path / "new"
    fresh_dir = tmp_path / "fresh"
    _seed_cache(new_dir)

    api = RangeTushareAPI(
        universe_extra_from=("2024-01-10", "600009.SH"),
    )
    updated = tushare_provider(new_dir, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["full_refetch"] is True
    assert status["performed"] is False
    assert "universe" in status["reason"]
    assets = set(updated._prices.index.get_level_values("asset"))
    assert "600009.SH" in assets

    fresh = tushare_provider(fresh_dir, api=RangeTushareAPI(
        universe_extra_from=("2024-01-10", "600009.SH"),
    ))
    fresh._load_or_fetch_all()
    pd.testing.assert_frame_equal(updated._prices, fresh._prices)


# ----------------------------------------------------------------------
# Atomicity: a failed publication leaves the old cache + marker intact
# ----------------------------------------------------------------------


def test_write_failure_preserves_old_cache_and_marker(tmp_path, monkeypatch):
    _seed_cache(tmp_path)
    old_file = tmp_path / f"tushare_000300.SH_2024-01-01_{CACHED_END}.h5"
    old_bytes = old_file.read_bytes()

    real_replace = ts_mod.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        if str(dst).endswith(f"_{REQUESTED_END}.h5"):
            raise OSError("simulated crash between write and publish")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(ts_mod.os, "replace", failing_replace)

    updated = tushare_provider(tmp_path, api=RangeTushareAPI())
    with pytest.raises(OSError):
        updated.update("2024-01-01", REQUESTED_END)

    assert old_file.read_bytes() == old_bytes
    assert _cache_marker(tmp_path)["last_complete_date"] == CACHED_END
    assert not (
        tmp_path / f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    ).exists()
    assert list(tmp_path.glob("*.tmp*")) == []


def test_legacy_cache_without_marker_is_refetched_not_appended(tmp_path):
    """Old caches may be read, but an append with an unknown boundary is
    rejected and refetched (never misread) -- D32 requirement 8."""
    import json

    from quant_platform.data.providers.base import (
        cache_manifest_path,
        read_cache_manifest,
    )

    _seed_cache(tmp_path)
    cache_file = tmp_path / f"tushare_000300.SH_2024-01-01_{CACHED_END}.h5"
    manifest = read_cache_manifest(cache_file)
    for key in (
        "last_complete_date", "previous_last_complete_date", "requested_end",
        "updated_at", "data_age_days", "incomplete", "new_sessions",
        "incremental_source",
    ):
        manifest.pop(key, None)
    cache_manifest_path(cache_file).write_text(
        json.dumps(manifest), encoding="utf-8",
    )

    updated = tushare_provider(tmp_path, api=RangeTushareAPI())
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["full_refetch"] is True
    assert status["incomplete"] is False
    assert status["last_complete_date"] == REQUESTED_END
    # The legacy file was not rewritten; the full refetch published a new
    # end-dated cache with a completion marker.
    assert "last_complete_date" not in read_cache_manifest(cache_file)
    new_file = tmp_path / f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    assert new_file.exists()
    assert read_cache_manifest(new_file)["last_complete_date"] == REQUESTED_END


def test_synthetic_advertises_no_incremental_support():
    from quant_platform.data.providers.synthetic import SyntheticDataProvider

    provider = SyntheticDataProvider(n_stocks=2)
    assert provider.supports_incremental is False
    with pytest.raises(NotImplementedError):
        provider.update("2024-01-01", "2024-01-02")


# ----------------------------------------------------------------------
# Completion-marker edge cases found in adversarial review
# ----------------------------------------------------------------------


def test_longer_cache_with_marker_behind_still_appends(tmp_path):
    """Coverage must be judged by the marker, never by the cache file name.

    A full fetch requested through 2024-01-11 may only contain data through
    2024-01-08 (provider lag) and therefore carry a marker of 2024-01-08.
    Updating to 2024-01-10 must fetch, not report "already covered".
    """
    from quant_platform.data.providers.base import read_cache_manifest

    lagging = RangeTushareAPI(omit_after=CACHED_END)
    _seed_cache(tmp_path, api=lagging, end_date=REQUESTED_END)
    old_file = tmp_path / (
        f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    )
    old_manifest = read_cache_manifest(old_file)
    assert old_manifest["last_complete_date"] == CACHED_END

    # The cache file is named for 2024-01-11, but its marker is 2024-01-08.
    assert (
        tmp_path / f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    ).exists()

    updated = tushare_provider(tmp_path, end_date="2024-01-10", api=RangeTushareAPI())
    status = updated.update("2024-01-01", "2024-01-10")

    assert status["performed"] is True
    assert status["incomplete"] is False
    assert status["new_sessions"] == 2
    assert status["last_complete_date"] == "2024-01-10"
    dates = pd.DatetimeIndex(
        updated._prices.index.get_level_values("date").unique()
    )
    assert dates.max().date().isoformat() == "2024-01-10"

    new_file = tmp_path / "tushare_000300.SH_2024-01-01_2024-01-10.h5"
    assert new_file.exists()
    assert read_cache_manifest(new_file)["last_complete_date"] == "2024-01-10"


def test_empty_new_session_window_marks_incomplete(tmp_path):
    """Sessions exist between marker and end, but the provider returns none:
    a partial response, not a silently fresh panel."""
    _seed_cache(tmp_path)
    api = RangeTushareAPI(omit_after=CACHED_END)
    updated = tushare_provider(tmp_path, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["new_sessions"] == 0
    assert status["last_complete_date"] == CACHED_END
    assert updated.data_status()["incomplete"] is True


def test_single_missing_symbol_on_last_day_is_incomplete_large_universe(
    tmp_path,
):
    """Zero tolerance: one name missing on the final session (60-name panel)
    must not advance the marker."""
    codes = [f"60{i:04d}.SH" for i in range(1, 61)]
    api = RangeTushareAPI(codes=codes)
    provider = tushare_provider(tmp_path, end_date=CACHED_END, api=api)
    provider._load_or_fetch_all()

    update_api = RangeTushareAPI(
        codes=codes, omit_last_day_for={"600001.SH"},
    )
    updated = tushare_provider(tmp_path, api=update_api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["last_complete_date"] == CACHED_END
    assert "600001.SH" in status["missing_symbols"]


def test_full_fetch_marker_holds_back_on_partial_last_day(tmp_path):
    api = RangeTushareAPI(omit_last_day_for={"600003.SH"})
    provider = tushare_provider(tmp_path, api=api)
    provider._load_or_fetch_all()

    manifest = _cache_marker_for(provider)
    assert manifest["last_complete_date"] == "2024-01-10"  # not the 11th
    assert manifest["incomplete"] is True
    assert provider.data_status()["incomplete"] is True


def test_missing_symbol_in_factor_refresh_is_incomplete(tmp_path):
    """A partial full-range adj_factor refresh must not leave old rows on one
    factor and appended rows on another (mixed adjustment basis)."""
    _seed_cache(tmp_path)
    api = RangeTushareAPI(adj_full_omit_codes={"600002.SH"})
    updated = tushare_provider(tmp_path, api=api)
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["last_complete_date"] == CACHED_END
    assert not (
        tmp_path / f"tushare_000300.SH_2024-01-01_{REQUESTED_END}.h5"
    ).exists()


def test_old_factors_are_refreshed_over_the_cached_range(tmp_path):
    """A vendor revision of an old cumulative factor must reach cached rows."""
    revised = {day: 1.1 for day in SESSIONS}
    smooth = {
        day: 10.0 + 0.1 * index for index, day in enumerate(SESSIONS)
    }
    _seed_cache(tmp_path, api=RangeTushareAPI(raw_close=smooth))
    updated = tushare_provider(
        tmp_path,
        api=RangeTushareAPI(raw_close=smooth, native_factor=revised),
    )
    status = updated.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is False
    old_factor = updated._prices.loc[
        (pd.Timestamp("2024-01-02"), "600001.SH"), "adj_factor"
    ]
    assert old_factor == pytest.approx(1.0 / 1.1)
    adjusted = (updated._prices["close"] / updated._prices["adj_factor"])
    adjusted = adjusted.unstack("asset")
    assert float(
        adjusted.pct_change(fill_method=None).abs().max().max()
    ) < 0.05


# ----------------------------------------------------------------------
# Session arithmetic
# ----------------------------------------------------------------------


def test_incremental_window_starts_after_the_marker_with_overlap():
    # Marker is Friday 2024-01-05; the next session is Monday 2024-01-08.
    start = incremental_fetch_start(
        "2024-01-05", "2024-01-09", overlap=1,
    )
    assert pd.Timestamp(start) == pd.Timestamp("2024-01-05")  # overlap session
    start_no_overlap = incremental_fetch_start(
        "2024-01-05", "2024-01-09", overlap=0,
    )
    assert pd.Timestamp(start_no_overlap) == pd.Timestamp("2024-01-08")
    assert incremental_fetch_start("2024-01-09", "2024-01-09") is None


def test_data_age_is_counted_in_exchange_sessions_not_calendar_days():
    # Friday marker -> Tuesday expected: Monday + Tuesday = 2 sessions, while
    # the calendar distance is 4 days (weekend included).
    assert sessions_after("2024-01-05", "2024-01-09") == 2
    assert sessions_after("2024-01-09", "2024-01-09") == 0
    assert expected_data_end("2030-01-01", today="2024-01-09") == (
        pd.Timestamp("2024-01-09")
    )
    extra = incremental_marker_extra(
        last_complete_date="2024-01-05", requested_end="2024-01-09",
    )
    assert extra["data_age_days"] == 2
    assert extra["last_complete_date"] == "2024-01-05"


# ----------------------------------------------------------------------
# Baostock append (fake module, same contract)
# ----------------------------------------------------------------------


BS_FIELDS = [
    "date", "code", "open", "high", "low", "close", "preclose", "volume",
    "amount", "turn", "pctChg", "peTTM", "pbMRQ",
]
BS_CODES = ["600001", "600002", "000001", "300001"]


class FakeBsResultSet:
    def __init__(self, rows, fields=None, error_code="0"):
        self._rows = list(rows)
        self.fields = list(fields or BS_FIELDS)
        self.error_code = error_code
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class FakeBaostockRange:
    """Minimal baostock fake: stock list + date-windowed daily history."""

    def __init__(self):
        self.raw_close = {
            "2024-01-02": 10.0, "2024-01-03": 10.1, "2024-01-04": 10.2,
            "2024-01-05": 10.3, "2024-01-08": 10.4,
            "2024-01-09": 5.25, "2024-01-10": 5.30, "2024-01-11": 5.35,
        }
        self.factor = {
            day: (1.0 if day < "2024-01-09" else 2.0)
            for day in self.raw_close
        }
        self.history_queries: list[tuple] = []
        #: ``(six_digit_code, day)`` pairs the provider fails to return.
        self.omit: set[tuple[str, str]] = set()
        #: Same, but only for the hfq series (adjustflag=1).
        self.omit_adj: set[tuple[str, str]] = set()

    def login(self):
        return types.SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        return None

    def query_stock_basic(self, code="", code_name=""):
        rows = [
            [f"sh.{code_}" if code_.startswith("6") else f"sz.{code_}",
             f"name_{code_}", "2020-01-01", "", "1", "1"]
            for code_ in BS_CODES
        ]
        return FakeBsResultSet(
            rows,
            fields=["code", "code_name", "ipoDate", "outDate", "type", "status"],
        )

    def query_stock_industry(self):
        return FakeBsResultSet([])

    def query_history_k_data_plus(
        self, bs_code, fields, start_date=None, end_date=None,
        frequency="d", adjustflag="1",
    ):
        self.history_queries.append(
            (bs_code, start_date, end_date, adjustflag)
        )
        code = bs_code.split(".")[-1]
        if not code:
            code = bs_code
        rows = []
        for day, raw in self.raw_close.items():
            if start_date and day < start_date:
                continue
            if end_date and day > end_date:
                continue
            if (code, day) in self.omit:
                continue
            if adjustflag == "1" and (code, day) in self.omit_adj:
                continue
            close = raw * (self.factor[day] if adjustflag == "1" else 1.0)
            rows.append([
                day, bs_code, f"{close}", f"{close}", f"{close}", f"{close}",
                f"{close}", "10000", f"{close * 10000}", "2.5", "1.0",
                "20.0", "2.0",
            ])
        return FakeBsResultSet(rows)


def _baostock_provider(monkeypatch, cache_dir, fake):
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr(bs_mod, "CACHE_DIR", cache_dir)
    provider = BaostockDataProvider(cache_enabled=True)
    provider._bs = fake
    return provider


def test_baostock_append_extends_panel_and_keeps_adjustment_continuous(
    tmp_path, monkeypatch,
):
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()

    old = _baostock_provider(monkeypatch, cache_dir, fake)
    old.cache_universe(start_date="2024-01-01", end_date=CACHED_END)
    old_prices = old.get_prices("2024-01-01", CACHED_END)
    assert len(old_prices.index.get_level_values("date").unique()) == 5
    assert old.data_status()["last_complete_date"] == CACHED_END

    new = _baostock_provider(monkeypatch, cache_dir, fake)
    new.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)
    status = new.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is False
    assert status["new_sessions"] == 3
    assert status["last_complete_date"] == REQUESTED_END

    prices = new.get_prices("2024-01-01", REQUESTED_END)
    dates = prices.index.get_level_values("date").unique()
    assert len(dates) == 8
    assert not prices.index.duplicated().any()
    adjusted = (prices["close"] / prices["adj_factor"]).unstack("asset")
    assert float(
        adjusted.pct_change(fill_method=None).abs().max().max()
    ) < 0.05
    assert prices.loc[(pd.Timestamp("2024-01-09"), "600001"), "close"] == (
        pytest.approx(5.25)
    )

    # Byte-identical to a full fetch over the same range.
    fresh_dir = tmp_path / "bs_fresh"
    fresh_dir.mkdir()
    fresh = _baostock_provider(monkeypatch, fresh_dir, FakeBaostockRange())
    fresh.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)
    full_prices = fresh.get_prices("2024-01-01", REQUESTED_END)
    pd.testing.assert_frame_equal(prices, full_prices, check_exact=True)


def test_baostock_write_failure_keeps_old_cache_and_marker(
    tmp_path, monkeypatch,
):
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()
    old = _baostock_provider(monkeypatch, cache_dir, fake)
    old.cache_universe(start_date="2024-01-01", end_date=CACHED_END)
    old.get_prices("2024-01-01", CACHED_END)

    new = _baostock_provider(monkeypatch, cache_dir, fake)
    new.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)

    real_replace = bs_mod.os.replace

    def failing_replace(src, dst, *args, **kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(bs_mod.os, "replace", failing_replace)
    status = new.update("2024-01-01", REQUESTED_END)

    assert status["incomplete"] is True
    assert status["last_complete_date"] == CACHED_END
    # Restore the publish primitive (keeping the cache-dir/module patches) and
    # verify the old complete panel is still served, with its marker intact.
    monkeypatch.setattr(bs_mod.os, "replace", real_replace)
    reloaded = _baostock_provider(monkeypatch, cache_dir, fake)
    reloaded.cache_universe(start_date="2024-01-01", end_date=CACHED_END)
    panel = reloaded.get_prices("2024-01-01", CACHED_END)
    assert len(panel.index.get_level_values("date").unique()) == 5
    assert reloaded.data_status()["last_complete_date"] == CACHED_END


def test_baostock_partial_retry_refetches_the_window(tmp_path, monkeypatch):
    """A partial window must not be cached; a retry for the same end date has
    to re-fetch and succeed (previously the cached partial window wedged it)."""
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()
    old = _baostock_provider(monkeypatch, cache_dir, fake)
    old.cache_universe(start_date="2024-01-01", end_date=CACHED_END)
    old.get_prices("2024-01-01", CACHED_END)

    new = _baostock_provider(monkeypatch, cache_dir, fake)
    new.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)

    fake.omit = {
        ("000001", day) for day in ("2024-01-09", "2024-01-10", "2024-01-11")
    }
    first = new.update("2024-01-01", REQUESTED_END)
    assert first["incomplete"] is True
    assert first["last_complete_date"] == CACHED_END
    # No partial window was persisted under any incremental window key.
    assert not list(cache_dir.glob("daily_*_2024-01-08_2024-01-11_*"))

    fake.omit.clear()
    second = new.update("2024-01-01", REQUESTED_END)
    assert second["incomplete"] is False
    assert second["new_sessions"] == 3
    assert second["last_complete_date"] == REQUESTED_END


def test_baostock_csv_cache_preserves_zero_filled_asset_labels(
    tmp_path, monkeypatch,
):
    """The CSV fallback must not turn ``000001`` into integer ``1``: the append
    path and hfq reindex compare asset labels exactly."""
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()
    provider = _baostock_provider(monkeypatch, cache_dir, fake)
    provider.cache_universe(start_date="2024-01-01", end_date=CACHED_END)

    def no_parquet(*_args, **_kwargs):
        raise RuntimeError("simulated missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_parquet)
    provider.get_prices("2024-01-01", CACHED_END)

    key = provider._daily_cache_key(
        provider._pinned_price_codes, "2024-01-01", CACHED_END, "3",
    )
    assert provider._cache_path(key).with_suffix(".csv").exists()
    loaded = provider._load_cache(key)
    labels = set(loaded.index.get_level_values("asset"))
    assert labels == set(BS_CODES)
    assert "000001" in labels

    # The append path works on the CSV caches too.
    new = _baostock_provider(monkeypatch, cache_dir, fake)
    new.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)
    status = new.update("2024-01-01", REQUESTED_END)
    assert status["incomplete"] is False
    assert status["new_sessions"] == 3

def test_baostock_panel_older_than_marker_triggers_full_refetch(
    tmp_path, monkeypatch,
):
    """A concurrent publish can leave the data file older than the marker;
    appending from it would skip sessions, so the update refetches."""
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()
    old = _baostock_provider(monkeypatch, cache_dir, fake)
    old.cache_universe(start_date="2024-01-01", end_date=CACHED_END)
    old.get_prices("2024-01-01", CACHED_END)

    key = old._daily_cache_key(
        old._pinned_price_codes, "2024-01-01", CACHED_END, "3",
    )
    panel = old._load_cache(key)
    truncated = panel.loc[:pd.Timestamp("2024-01-04")]
    extra = incremental_marker_extra(
        last_complete_date=CACHED_END,
        requested_end=CACHED_END,
        source="full_fetch",
    )
    old._save_cache(key, truncated, extra=extra, raise_errors=True)

    new = _baostock_provider(monkeypatch, cache_dir, fake)
    new.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)
    status = new.update("2024-01-01", REQUESTED_END)

    assert status["full_refetch"] is True
    assert "marker" in status["reason"]
    assert status["last_complete_date"] == REQUESTED_END

    reloaded = _baostock_provider(monkeypatch, cache_dir, fake)
    reloaded.cache_universe(start_date="2024-01-01", end_date=REQUESTED_END)
    prices = reloaded.get_prices("2024-01-01", REQUESTED_END)
    assert len(prices.index.get_level_values("date").unique()) == 8

def test_baostock_bad_hfq_cache_is_evicted_and_retry_succeeds(
    tmp_path, monkeypatch,
):
    """The hfq panel is saved before the divisor merge validates it. A panel
    that cannot yield a divisor must be evicted, not replayed on retries."""
    fake = FakeBaostockRange()
    cache_dir = tmp_path / "bs_cache"
    cache_dir.mkdir()
    provider = _baostock_provider(monkeypatch, cache_dir, fake)
    provider.cache_universe(start_date="2024-01-01", end_date=CACHED_END)

    fake.omit_adj = {("000001", day) for day in fake.raw_close}
    with pytest.raises(DataFetchIncompleteError):
        provider.get_prices("2024-01-01", CACHED_END)

    hfq_key = provider._daily_cache_key(
        provider._pinned_price_codes, "2024-01-01", CACHED_END, "1",
    )
    base = provider._cache_path(hfq_key)
    assert not base.exists()
    assert not base.with_suffix(".csv").exists()

    fake.omit_adj.clear()
    prices = provider.get_prices("2024-01-01", CACHED_END)
    assert len(prices.index.get_level_values("date").unique()) == 5
