"""R7a: Baostock ST episodes derived from the daily ``isST`` field.

Pins the new behaviour:

* ``isST`` rides along with the single daily history request (no extra
  network calls) and never enters the public price frame;
* consecutive observed ``isST == "1"`` days form one episode with
  ``announce_date`` = first observed ST day (conservative PIT), ``end_date`` =
  last ST day (or sample end), ``trigger_date`` = announce date;
* two disjoint runs stay two episodes and the gap is never flagged;
* a response without ``isST`` (missing column or only blank values) yields
  the empty contract table and ``st_available == False`` -- nothing is
  fabricated;
* the observed flag is persisted with the daily cache, so a reload/regenerate
  derives the same episodes (and the cache version was bumped);
* the pipeline's per-date mask keeps the stock eligible before
  ``announce_date`` and ineligible from it.

No network access: baostock is replaced with a fake.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import quant_platform.data.providers.baostock_provider as mod
from quant_platform.data.pipeline import DataPipeline
from quant_platform.data.providers.baostock_provider import BaostockDataProvider

START = "2024-01-01"
END = "2024-03-29"
SESSIONS = pd.bdate_range(START, END)
CODE = "600001"
OTHER = "600002"

BASIC_FIELDS = ["code", "code_name", "ipoDate", "outDate", "type", "status"]


class FakeResultSet:
    def __init__(self, rows, fields=(), error_code="0", error_msg=""):
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


class FakeBaostockST:
    """Daily history fake with per-symbol ``isST`` flags.

    ``include_is_st=False`` mimics an older interface that ignores the
    requested field list and returns the thirteen legacy columns.
    ``default_flag`` lets a test return blank values instead of ``"0"``.
    ``omit_is_st_codes`` makes only those symbols return 13 columns, i.e. a
    mixed flag column with blanks (the CSV-round-trip trap).
    ``fail_history=True`` makes every history call raise (transient outage).
    """

    def __init__(
        self,
        flags: dict[str, dict[pd.Timestamp, str]] | None = None,
        *,
        include_is_st: bool = True,
        default_flag: str = "0",
        omit_is_st_codes: tuple[str, ...] = (),
        fail_history: bool = False,
        codes: tuple[str, ...] = (CODE, OTHER),
    ):
        self.flags = flags or {}
        self.include_is_st = include_is_st
        self.default_flag = default_flag
        self.omit_is_st_codes = set(omit_is_st_codes)
        self.fail_history = fail_history
        self.codes = list(codes)
        self.history_calls: list[tuple[str, str, list[str]]] = []
        self.basic_query_count = 0

    def login(self):
        return types.SimpleNamespace(error_code="0", error_msg="")

    def logout(self):
        return None

    def query_stock_basic(self, code="", code_name=""):
        self.basic_query_count += 1
        rows = []
        for code_ in self.codes:
            prefix = "sh" if code_.startswith(("6", "9")) else "sz"
            rows.append([
                f"{prefix}.{code_}", f"name_{code_}", "2020-01-01", "",
                "1", "1",
            ])
        return FakeResultSet(rows, fields=BASIC_FIELDS)

    def query_stock_industry(self):
        return FakeResultSet(
            [], fields=["code", "code_name", "industry", "industryClassification"],
        )

    def query_history_k_data_plus(self, code, fields, start_date=None,
                                  end_date=None, frequency="d", adjustflag="3"):
        requested = [part for part in str(fields).split(",") if part]
        six = code.split(".")[-1]
        self.history_calls.append((six, adjustflag, requested))
        if self.fail_history:
            raise RuntimeError("simulated history outage")
        start = pd.Timestamp(start_date) if start_date else None
        end = pd.Timestamp(end_date) if end_date else None
        window = [
            day for day in SESSIONS
            if (start is None or day >= start) and (end is None or day <= end)
        ]
        if code == "sh.000300":
            return FakeResultSet(
                [[str(day.date()), "1000.0", "990.0", "0.1"] for day in window],
                fields=["date", "close", "preclose", "pctChg"],
            )
        rows = []
        for day in window:
            # Raw and hfq series coincide (adj_factor = 1): only the index,
            # the close column and the ST side field matter here.
            price = 10.0
            row = [
                str(day.date()), code, f"{price}", f"{price + 0.2}",
                f"{price - 0.2}", f"{price}", f"{price}", "10000",
                f"{price * 10000}", "2.5", "1.0", "15.0", "2.0",
            ]
            if (
                self.include_is_st
                and "isST" in requested
                and six not in self.omit_is_st_codes
            ):
                row.append(self.flags.get(six, {}).get(day, self.default_flag))
            rows.append(row)
        return FakeResultSet(rows, fields=requested)

    def query_profit_data(self, code="", year=2024, quarter=4):
        return FakeResultSet(
            [], fields=["code", "pubDate", "statDate", "roeAvg", "npMargin",
                       "gpMargin", "netProfit", "epsTTM", "MBRevenue",
                       "totalShare", "liqaShare"],
        )

    def query_balance_data(self, code="", year=2024, quarter=4):
        return FakeResultSet(
            [], fields=["code", "pubDate", "statDate", "liabilityToAsset",
                       "assetToEquity"],
        )


def make_provider(fake, monkeypatch, cache_dir, *, cache_enabled=False,
                  adjustment="raw"):
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr(mod, "CACHE_DIR", cache_dir)
    provider = BaostockDataProvider(
        cache_enabled=cache_enabled,
        adjustment=adjustment,
        retry_backoff=0.0,
    )
    provider._bs = fake
    return provider


def _flags_for(days) -> dict[pd.Timestamp, str]:
    return {pd.Timestamp(day).normalize(): "1" for day in days}


# ----------------------------------------------------------------------
# Episode derivation
# ----------------------------------------------------------------------

def test_single_episode_uses_first_observed_st_day(tmp_path, monkeypatch):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    provider = make_provider(fake, monkeypatch, tmp_path)

    prices = provider.get_prices(START, END)
    st = provider.get_st_timeseries()

    assert list(prices.columns) == mod._PRICE_COLUMNS
    assert mod._ST_OBSERVED_COLUMN not in prices.columns

    assert len(st) == 1
    row = st.iloc[0]
    assert row["asset"] == CODE
    assert row["announce_date"] == SESSIONS[9]   # first observed ST day
    assert row["trigger_date"] == SESSIONS[9]
    assert row["end_date"] == SESSIONS[19]       # last day of the run
    assert bool(row["is_st"]) is True
    # Day 9 (1-based) is before the announcement: no episode covers it.
    assert SESSIONS[8] < row["announce_date"]
    assert provider.st_available is True

    # One history request per symbol (raw only in adjustment="raw" mode), and
    # isST is part of that same request.
    raw_calls = [call for call in fake.history_calls if call[1] == "3"]
    assert len(raw_calls) == len(fake.codes)
    assert all("isST" in call[2] for call in raw_calls)


def test_two_disjoint_runs_stay_two_episodes(tmp_path, monkeypatch):
    days = list(SESSIONS[9:20]) + list(SESSIONS[39:50])
    fake = FakeBaostockST(flags={CODE: _flags_for(days)})
    provider = make_provider(fake, monkeypatch, tmp_path)

    provider.get_prices(START, END)
    st = provider.get_st_timeseries().sort_values("announce_date").reset_index(drop=True)

    assert len(st) == 2
    assert st.loc[0, "announce_date"] == SESSIONS[9]
    assert st.loc[0, "end_date"] == SESSIONS[19]
    assert st.loc[1, "announce_date"] == SESSIONS[39]
    assert st.loc[1, "end_date"] == SESSIONS[49]

    # The gap between the runs (sessions 21-39, 1-based) is not flagged.
    for day in SESSIONS[20:39]:
        covered = (st["announce_date"] <= day) & (st["end_date"] >= day)
        assert not bool(covered.any()), day


def test_open_ended_episode_ends_at_sample_last_day(tmp_path, monkeypatch):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[-5:])})
    provider = make_provider(fake, monkeypatch, tmp_path)

    provider.get_prices(START, END)
    st = provider.get_st_timeseries()

    assert len(st) == 1
    assert st.iloc[0]["announce_date"] == SESSIONS[-5]
    assert st.iloc[0]["end_date"] == SESSIONS[-1]


# ----------------------------------------------------------------------
# Downgrade path and metadata consistency
# ----------------------------------------------------------------------

def test_missing_is_st_yields_empty_table_without_error(tmp_path, monkeypatch):
    fake = FakeBaostockST(include_is_st=False)
    provider = make_provider(fake, monkeypatch, tmp_path)

    prices = provider.get_prices(START, END)
    assert list(prices.columns) == mod._PRICE_COLUMNS

    st = provider.get_st_timeseries()
    assert st.empty
    assert list(st.columns) == mod._ST_EPISODE_COLUMNS
    assert provider.st_available is False
    assert provider.get_metadata().attrs["st_available"] is False


def test_blank_is_st_values_are_not_treated_as_st(tmp_path, monkeypatch):
    fake = FakeBaostockST(default_flag="")
    provider = make_provider(fake, monkeypatch, tmp_path)

    provider.get_prices(START, END)

    assert provider.get_st_timeseries().empty
    assert provider.st_available is False
    assert provider.get_metadata().attrs["st_available"] is False


def test_metadata_st_available_matches_the_episode_table(tmp_path, monkeypatch):
    """get_metadata() runs before get_prices() in the pipeline: with a pinned
    range it must resolve the same episode table (cache/fetch), so its
    ``st_available`` cannot disagree with ``get_st_timeseries()``."""
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    provider = make_provider(fake, monkeypatch, tmp_path)
    provider.cache_universe(start_date=START, end_date=END)

    meta = provider.get_metadata()
    st = provider.get_st_timeseries()

    assert not st.empty
    assert provider.st_available is True
    assert meta.attrs["st_available"] is True
    assert st.iloc[0]["announce_date"] == SESSIONS[9]

    # Same consistency for the no-episode case.
    fake2 = FakeBaostockST()
    provider2 = make_provider(fake2, monkeypatch, tmp_path / "empty")
    provider2.cache_universe(start_date=START, end_date=END)
    meta2 = provider2.get_metadata()
    assert provider2.get_st_timeseries().empty
    assert provider2.st_available is False
    assert meta2.attrs["st_available"] is False


# ----------------------------------------------------------------------
# Cache persistence
# ----------------------------------------------------------------------

def test_cached_flag_round_trip_keeps_episodes(tmp_path, monkeypatch):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    first = make_provider(fake, monkeypatch, cache_dir, cache_enabled=True)
    first.cache_universe(start_date=START, end_date=END)
    first.get_prices(START, END)
    episodes = first.get_st_timeseries()
    calls_after_fetch = len(fake.history_calls)

    # The raw daily cache carries the observed flag (persistence).
    key = first._daily_cache_key(first._pinned_price_codes, START, END, "3")
    cached = first._load_cache(key)
    assert cached is not None
    assert mod._ST_OBSERVED_COLUMN in cached.columns

    # A fresh provider resolves the episodes from the cache alone: no new
    # history query, same table.
    second = make_provider(fake, monkeypatch, cache_dir, cache_enabled=True)
    second.cache_universe(start_date=START, end_date=END)
    reloaded = second.get_st_timeseries()

    assert len(fake.history_calls) == calls_after_fetch
    pd.testing.assert_frame_equal(reloaded, episodes)


def test_cache_version_was_bumped(tmp_path, monkeypatch):
    """v5 panels have no ST evidence; they must never be replayed."""
    assert mod._CACHE_VERSION == "v6"
    fake = FakeBaostockST()
    provider = make_provider(fake, monkeypatch, tmp_path)
    assert f"{mod._CACHE_VERSION}-s" in provider._cache_version()


# ----------------------------------------------------------------------
# Pipeline PIT mask
# ----------------------------------------------------------------------

def test_pipeline_mask_is_effective_from_announce_date_only(
    tmp_path, monkeypatch,
):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    provider = make_provider(fake, monkeypatch, tmp_path)
    provider.cache_universe(start_date=START, end_date=END)
    prices = provider.get_prices(START, END)

    pipe = DataPipeline(
        provider, START, END, exclude_st=True, exclude_suspended=False,
    )
    pipe.prices = prices
    pipe._st_timeseries = provider.get_st_timeseries()
    mask = pipe._build_st_eligibility_mask()

    assert bool(mask.loc[SESSIONS[8], CODE]) is True    # before announcement
    assert bool(mask.loc[SESSIONS[9], CODE]) is False   # announcement day
    assert bool(mask.loc[SESSIONS[19], CODE]) is False  # last ST day
    assert bool(mask.loc[SESSIONS[20], CODE]) is True   # day after the run

    assert pipe.get_st_status(SESSIONS[8]) == set()
    assert CODE in pipe.get_st_status(SESSIONS[9])
    assert CODE not in pipe.get_st_status(SESSIONS[20])


def test_pipeline_run_end_to_end_applies_st_episodes(tmp_path, monkeypatch):
    """The real pipeline order (metadata -> PIT load -> prices) must still
    screen ST: the pre-price resolution fetches the raw panel and the later
    get_prices is a cache/memo hit."""
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    provider = make_provider(fake, monkeypatch, tmp_path)
    provider.cache_universe(start_date=START, end_date=END)

    pipe = DataPipeline(
        provider, START, END, exclude_st=True, exclude_suspended=False,
    )
    pipe.run()

    canonical = f"{CODE}.SH"
    assert len(pipe._st_timeseries) == 1
    assert pipe.metadata.attrs.get("st_available") is True
    assert pipe.st_eligible is not None
    assert bool(pipe.st_eligible.loc[SESSIONS[8], canonical]) is True
    assert bool(pipe.st_eligible.loc[SESSIONS[9], canonical]) is False
    assert bool(pipe.st_eligible.loc[SESSIONS[19], canonical]) is False
    assert bool(pipe.st_eligible.loc[SESSIONS[20], canonical]) is True


# ----------------------------------------------------------------------
# Reviewer-driven hardening: CSV fallback, transient failures, re-pins,
# request counts and hfq field projection
# ----------------------------------------------------------------------

def _no_parquet(*_args, **_kwargs):
    raise ImportError("pyarrow unavailable (simulated)")


def test_csv_fallback_round_trip_with_mixed_blanks_keeps_episodes(
    tmp_path, monkeypatch,
):
    """A mixed "1"/blank flag column read back from CSV must not be inferred
    as float64 (the reload would silently lose every ST episode)."""
    fake = FakeBaostockST(
        flags={CODE: _flags_for(SESSIONS[9:20])},
        omit_is_st_codes=(OTHER,),  # mixed column: "0"/"1" plus blanks
    )
    cache_dir = tmp_path / "csv_cache"
    cache_dir.mkdir()
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _no_parquet)

    first = make_provider(fake, monkeypatch, cache_dir, cache_enabled=True)
    first.cache_universe(start_date=START, end_date=END)
    first.get_prices(START, END)
    assert len(first.get_st_timeseries()) == 1

    key = first._daily_cache_key(first._pinned_price_codes, START, END, "3")
    csv_path = first._cache_path(key).with_suffix(".csv")
    assert csv_path.exists()
    cached = first._load_cache(key)
    assert str(
        cached.loc[(SESSIONS[9], CODE), mod._ST_OBSERVED_COLUMN]
    ) == "1"

    second = make_provider(fake, monkeypatch, cache_dir, cache_enabled=True)
    second.cache_universe(start_date=START, end_date=END)
    reloaded = second.get_st_timeseries()

    assert len(reloaded) == 1
    assert reloaded.iloc[0]["announce_date"] == SESSIONS[9]
    assert second.st_available is True


def test_transient_st_fetch_failure_is_retried_not_latched(
    tmp_path, monkeypatch,
):
    fake = FakeBaostockST(
        flags={CODE: _flags_for(SESSIONS[9:20])}, fail_history=True,
    )
    provider = make_provider(fake, monkeypatch, tmp_path)
    provider.cache_universe(start_date=START, end_date=END)

    assert provider.get_st_timeseries().empty
    assert provider.st_available is False
    calls_after_failure = len(fake.history_calls)

    fake.fail_history = False
    episodes = provider.get_st_timeseries()

    assert len(episodes) == 1
    assert provider.st_available is True
    assert len(fake.history_calls) > calls_after_failure


def test_repin_invalidates_episodes_from_the_previous_range(
    tmp_path, monkeypatch,
):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[-4:])})
    provider = make_provider(fake, monkeypatch, tmp_path)

    provider.cache_universe(start_date=START, end_date=END)
    assert len(provider.get_st_timeseries()) == 1

    # The new range ends before the ST run: episodes from the old range must
    # not leak into it.
    provider.cache_universe(
        start_date=START, end_date=str(SESSIONS[30].date()),
    )
    assert provider.get_st_timeseries().empty


def test_pinned_pipeline_order_fetches_each_symbol_once(
    tmp_path, monkeypatch,
):
    for cache_enabled in (True, False):
        fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
        cache_dir = tmp_path / f"cache_{cache_enabled}"
        cache_dir.mkdir()
        provider = make_provider(
            fake, monkeypatch, cache_dir,
            cache_enabled=cache_enabled, adjustment="hfq",
        )
        provider.cache_universe(start_date=START, end_date=END)

        provider.get_metadata()  # pre-price ST resolution
        prices = provider.get_prices(START, END)

        assert mod._ST_OBSERVED_COLUMN not in prices.columns
        raw_calls = [call for call in fake.history_calls if call[1] == "3"]
        hfq_calls = [call for call in fake.history_calls if call[1] == "1"]
        assert len(raw_calls) == len(fake.codes), cache_enabled
        assert len(hfq_calls) == len(fake.codes), cache_enabled


def test_hfq_fields_projection_excludes_the_side_column(tmp_path, monkeypatch):
    fake = FakeBaostockST(flags={CODE: _flags_for(SESSIONS[9:20])})
    provider = make_provider(
        fake, monkeypatch, tmp_path, adjustment="hfq",
    )

    sub = provider.get_prices(START, END, fields=["close", "volume"])

    assert list(sub.columns) == ["close", "volume"]
    assert len(provider.get_st_timeseries()) == 1

