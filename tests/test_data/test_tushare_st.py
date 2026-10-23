"""R14: Tushare ST episodes derived from the ``namechange`` endpoint.

Pins the new behaviour:

* the episode contract is identical to Synthetic/Baostock
  (``asset, trigger_date, announce_date, end_date, is_st``);
* ``announce_date`` is ``ann_date`` when present and parseable, otherwise the
  effective ``start_date`` (PIT-conservative: never earlier than announced);
* ``trigger_date`` is ``start_date``; ``end_date`` is the recorded end, capped
  at the next rename or the price sample's last day when empty -- never
  invented beyond the evidence;
* adjacent/overlapping ST intervals merge; non-ST renames are ignored;
* one symbol's failure retries twice, is recorded, and never aborts the rest;
  a completely unavailable endpoint yields the empty contract table and
  ``st_available == False`` (no fabrication);
* ``get_metadata().attrs["st_available"]`` always agrees with the episode
  table, and the raw feed survives a cache round-trip; a cache built while
  the endpoint was unavailable stores no ``/st_namechange`` key and is NOT
  replayed as "no ST names";
* ``DATA_SCHEMA_VERSION`` and the provider cache version were bumped.

No network access: the Tushare API is replaced with a fake.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
import quant_platform.data.providers.tushare_loader as mod
from quant_platform.data.providers.tushare_loader import TushareProvider

DATES = ["20240102", "20240103", "20240104"]
CODES = ["600001.SH", "600002.SZ"]
SAMPLE_END = pd.Timestamp("2024-01-04")


class FakeTushareSTAPI:
    """Minimal Tushare fake: full-fetch endpoints plus per-symbol namechange."""

    def __init__(
        self,
        namechange_rows: dict[str, list[dict]] | None = None,
        *,
        failing_codes: tuple[str, ...] = (),
        fail_all_namechange: bool = False,
    ):
        self.namechange_rows = {
            key: list(rows) for key, rows in (namechange_rows or {}).items()
        }
        self.failing_codes = set(failing_codes)
        self.fail_all_namechange = fail_all_namechange
        self.namechange_calls: list[str] = []

    @staticmethod
    def _batch(ts_code):
        return [code for code in str(ts_code).split(",") if code]

    def daily(self, ts_code, start_date, end_date, fields=None):
        rows = []
        for code in self._batch(ts_code):
            for day in DATES:
                rows.append({
                    "ts_code": code, "trade_date": day,
                    "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
                    "vol": 100.0, "amount": 1000.0, "turnover_rate": 1.0,
                })
        return pd.DataFrame(rows)

    def adj_factor(self, ts_code, start_date, end_date):
        return pd.DataFrame([
            {"ts_code": code, "trade_date": day, "adj_factor": 1.0}
            for code in self._batch(ts_code) for day in DATES
        ])

    def fina_indicator(self, ts_code, start_date, end_date, fields=None):
        return pd.DataFrame([{
            "ts_code": code, "end_date": "20231231", "ann_date": "20240102",
            "f_ann_date": "20240102", "total_mv": 1e10, "total_assets": 2e10,
            "total_hldr_eqy_inc_min_int": 1e10, "revenue": 1e9,
            "net_profit": 1e8, "roe": 10.0, "pb": 2.0, "pe": 20.0, "eps": 1.0,
        } for code in self._batch(ts_code)])

    def index_weight(self, index_code, start_date, end_date):
        return pd.DataFrame([
            {"trade_date": day, "ts_code": code}
            for day in DATES for code in CODES
        ])

    def stock_basic(self, exchange="", list_status="L", fields=None):
        if list_status == "L":
            return pd.DataFrame([
                {"ts_code": "600001.SH", "name": "AAA", "industry": "银行",
                 "list_date": "20000101", "delist_date": None, "is_hs": "H"},
                {"ts_code": "600002.SZ", "name": "BBB", "industry": "医药",
                 "list_date": "20000101", "delist_date": None, "is_hs": "S"},
            ])
        return pd.DataFrame(columns=[
            "ts_code", "name", "industry", "list_date", "delist_date", "is_hs",
        ])

    def index_daily(self, ts_code, start_date, end_date, fields=None):
        return pd.DataFrame([{"trade_date": day, "pct_chg": 1.0} for day in DATES])

    def namechange(self, ts_code, fields=None):
        self.namechange_calls.append(ts_code)
        if self.fail_all_namechange or ts_code in self.failing_codes:
            raise RuntimeError("namechange permission denied")
        return pd.DataFrame(self.namechange_rows.get(ts_code, []))


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


def row(ts_code, name, start, end=None, ann=None):
    return {
        "ts_code": ts_code, "name": name, "start_date": start,
        "end_date": end, "ann_date": ann, "change_reason": "ST",
    }


def price_panel() -> pd.DataFrame:
    """Tiny (date, asset) panel so the sample end is 2024-01-04."""
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex([pd.Timestamp(day) for day in DATES]), CODES],
        names=["date", "asset"],
    )
    return pd.DataFrame({"close": 10.0, "adj_factor": 1.0}, index=index)


def seed_full_provider(prov: TushareProvider, api: FakeTushareSTAPI) -> None:
    """Populate every table ``_save_to_cache`` writes, then fetch ST."""
    prov._prices = price_panel()
    prov._financials = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "asset"]),
        columns=["market_cap"],
    )
    prov._benchmark = pd.Series(
        [0.01] * len(DATES),
        index=pd.DatetimeIndex([pd.Timestamp(day) for day in DATES]),
        name="benchmark",
    )
    prov._metadata = prov._fetch_stock_basic(api, CODES)
    prov._constituent_history = pd.DataFrame([
        {"trade_date": pd.Timestamp(day), "ts_code": code}
        for day in DATES for code in CODES
    ])
    prov._fetch_st_episodes(api, CODES)


# ----------------------------------------------------------------------
# Episode mapping (a/b): announce timing and end fill
# ----------------------------------------------------------------------

def test_st_episode_maps_announce_trigger_and_end(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "*ST AAA", "20220425", "20230510", "20220420"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episodes = prov._fetch_st_episodes(api, ["600001.SH"])

    assert list(episodes.columns) == mod._ST_EPISODE_COLUMNS
    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["asset"] == "600001.SH"
    assert episode["announce_date"] == pd.Timestamp("2022-04-20")
    assert episode["trigger_date"] == pd.Timestamp("2022-04-25")
    assert episode["end_date"] == pd.Timestamp("2023-05-10")
    assert bool(episode["is_st"]) is True
    assert prov.st_available is True
    assert prov._st_fetch_status == "ok"


def test_missing_announce_date_falls_back_to_effective_start(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20220425", "20230510", ann=None),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episode = prov._fetch_st_episodes(api, ["600001.SH"]).iloc[0]

    assert episode["announce_date"] == pd.Timestamp("2022-04-25")
    assert episode["trigger_date"] == pd.Timestamp("2022-04-25")


def test_missing_end_date_fills_the_price_sample_last_day(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20230501", ann="20230425"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episode = prov._fetch_st_episodes(api, ["600001.SH"]).iloc[0]

    assert episode["end_date"] == SAMPLE_END


def test_unparseable_announce_date_conservatively_falls_back_to_start(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20220425", "20230510", ann="not-a-date"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episode = prov._fetch_st_episodes(api, ["600001.SH"]).iloc[0]

    assert episode["announce_date"] == pd.Timestamp("2022-04-25")


# ----------------------------------------------------------------------
# (c) merging, plain renames, and the next-rename cap
# ----------------------------------------------------------------------

def test_overlapping_and_adjacent_intervals_merge_and_plain_renames_ignored(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            # Overlapping ST -> *ST: one merged episode.
            row("600001.SH", "ST AAA", "20220101", "20220630", "20211220"),
            row("600001.SH", "*ST AAA", "20220601", "20221231", "20220525"),
            # A plain rename after the ST period must be ignored.
            row("600001.SH", "AAA", "20230101", "20231231", "20221220"),
        ],
        "600002.SZ": [
            # Adjacent (day-after) intervals merge too.
            row("600002.SZ", "ST BBB", "20220101", "20220630", "20211220"),
            row("600002.SZ", "*ST BBB", "20220701", "20221231", "20220625"),
            # A disjoint episode stays separate.
            row("600002.SZ", "ST BBB", "20230601", "20231231", "20230525"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episodes = prov._fetch_st_episodes(api, CODES)

    first = episodes.loc[episodes["asset"] == "600001.SH"].reset_index(drop=True)
    assert len(first) == 1
    assert first.loc[0, "trigger_date"] == pd.Timestamp("2022-01-01")
    assert first.loc[0, "announce_date"] == pd.Timestamp("2021-12-20")
    assert first.loc[0, "end_date"] == pd.Timestamp("2022-12-31")

    second = episodes.loc[episodes["asset"] == "600002.SZ"].reset_index(drop=True)
    assert len(second) == 2
    assert second.loc[0, "trigger_date"] == pd.Timestamp("2022-01-01")
    assert second.loc[0, "end_date"] == pd.Timestamp("2022-12-31")
    assert second.loc[1, "trigger_date"] == pd.Timestamp("2023-06-01")


def test_missing_end_is_capped_by_next_rename_not_sample_end(tmp_path):
    """An empty end on a superseded ST name must not fabricate ST through 2024."""
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20220101", end=None, ann="20211220"),
            row("600001.SH", "AAA", "20220601", "20231231", "20220520"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episode = prov._fetch_st_episodes(api, ["600001.SH"]).iloc[0]

    assert episode["end_date"] == pd.Timestamp("2022-05-31")
    assert episode["end_date"] < SAMPLE_END


# ----------------------------------------------------------------------
# (d) retries, per-symbol failure isolation, full outage
# ----------------------------------------------------------------------

def test_single_symbol_failure_retries_then_the_rest_continue(tmp_path):
    api = FakeTushareSTAPI(
        {
            "600001.SH": [
                row("600001.SH", "ST AAA", "20220425", "20230510", "20220420"),
            ],
        },
        failing_codes=("600002.SZ",),
    )
    prov = provider(tmp_path)
    prov._prices = price_panel()

    episodes = prov._fetch_st_episodes(api, CODES)

    assert api.namechange_calls.count("600002.SZ") == mod._ST_FETCH_ATTEMPTS
    assert prov.st_failed_symbols == ["600002.SZ"]
    assert prov._st_fetch_status == "partial"
    assert len(episodes) == 1
    assert episodes.iloc[0]["asset"] == "600001.SH"
    assert prov.st_available is True


def test_all_symbols_failing_yields_empty_contract_and_unavailable(tmp_path, caplog):
    api = FakeTushareSTAPI(fail_all_namechange=True)
    prov = provider(tmp_path)
    prov._prices = price_panel()

    with caplog.at_level(logging.WARNING):
        episodes = prov._fetch_st_episodes(api, CODES)

    assert episodes.empty
    assert list(episodes.columns) == mod._ST_EPISODE_COLUMNS
    assert prov.st_available is False
    assert prov._st_fetch_status == "unavailable"
    assert prov._st_namechange is None
    messages = [record.getMessage().lower() for record in caplog.records]
    assert any("unavailable" in message for message in messages)
    assert any("not evidence that every stock is non-st" in message
               for message in messages)


def test_large_universe_logs_estimate_but_fetches_every_symbol(tmp_path, caplog):
    api = FakeTushareSTAPI()
    prov = provider(tmp_path)
    many = [f"{600000 + i}.SH" for i in range(mod._ST_FETCH_WARN_THRESHOLD + 1)]

    with caplog.at_level(logging.WARNING):
        prov._fetch_namechange_raw(api, many)

    assert len(api.namechange_calls) == len(many)  # nothing silently skipped
    messages = [record.getMessage() for record in caplog.records]
    assert any("one API call per symbol" in message for message in messages)


# ----------------------------------------------------------------------
# (e) metadata attrs and cache round-trip
# ----------------------------------------------------------------------

def test_metadata_st_available_matches_the_episode_table(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20220425", "20230510", "20220420"),
        ],
    })
    prov = provider(tmp_path)
    prov._prices = price_panel()
    prov._metadata = prov._fetch_stock_basic(api, CODES)
    prov._fetch_st_episodes(api, CODES)

    meta = prov.get_metadata()
    assert meta.attrs["st_available"] is True
    assert prov.st_available is True
    assert not prov.get_st_timeseries().empty

    empty_api = FakeTushareSTAPI()
    prov2 = provider(tmp_path)
    prov2._prices = price_panel()
    prov2._metadata = prov2._fetch_stock_basic(empty_api, CODES)
    prov2._fetch_st_episodes(empty_api, CODES)

    assert prov2.get_metadata().attrs["st_available"] is False
    assert prov2.st_available is False
    assert prov2.get_st_timeseries().empty


def test_episodes_survive_a_cache_round_trip(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "*ST AAA", "20220425", "20230510", "20220420"),
        ],
    })
    prov = provider(tmp_path)
    seed_full_provider(prov, api)
    cache_file = prov._cache_file()
    prov._save_to_cache(cache_file)

    reloaded = provider(tmp_path)
    reloaded._load_from_cache(cache_file)

    assert reloaded.st_available is True
    assert reloaded._st_fetch_status == "ok"
    assert reloaded.st_failed_symbols == []
    first = prov.get_st_timeseries()
    second = reloaded.get_st_timeseries()
    pd.testing.assert_frame_equal(first, second)
    assert reloaded.get_metadata().attrs["st_available"] is True


def test_unavailable_feed_is_not_replayed_from_cache(tmp_path, caplog):
    api = FakeTushareSTAPI(fail_all_namechange=True)
    prov = provider(tmp_path)
    seed_full_provider(prov, api)
    cache_file = prov._cache_file()
    prov._save_to_cache(cache_file)

    with pd.HDFStore(cache_file, mode="r") as store:
        stored_keys = store.keys()
        assert "/st_namechange" not in stored_keys

    reloaded = provider(tmp_path)
    with caplog.at_level(logging.WARNING):
        reloaded._load_from_cache(cache_file)

    assert reloaded.st_available is False
    assert reloaded._st_fetch_status == "unavailable"
    assert reloaded._st_namechange is None
    assert reloaded.get_st_timeseries().empty
    assert reloaded.get_metadata().attrs["st_available"] is False
    messages = [record.getMessage().lower() for record in caplog.records]
    assert any("no usable st history" in message for message in messages)
    assert any("does not mean every stock is non-st" in message
               for message in messages)


def test_full_fetch_derives_episodes_and_caches_them(tmp_path):
    api = FakeTushareSTAPI({
        "600001.SH": [
            row("600001.SH", "ST AAA", "20220425", "20230510", "20220420"),
        ],
    })
    prov = provider(tmp_path)
    prov._ts_api = api

    prov._load_or_fetch_all()

    assert prov.st_available is True
    assert len(prov.get_st_timeseries()) == 1
    assert prov.get_metadata().attrs["st_available"] is True

    reloaded = provider(tmp_path)
    reloaded._ts_api = api
    reloaded._load_or_fetch_all()  # cache hit, no namechange calls
    assert reloaded.st_available is True
    assert len(reloaded.get_st_timeseries()) == 1
    assert api.namechange_calls.count("600001.SH") == 1


# ----------------------------------------------------------------------
# (f) version pins
# ----------------------------------------------------------------------

def test_cache_and_schema_versions_are_bumped():
    from quant_platform.utils.cache import DATA_SCHEMA_VERSION

    assert DATA_SCHEMA_VERSION == 8
    assert mod._PROVIDER_CACHE_VERSION == "v5"
