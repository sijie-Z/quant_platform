"""Provider-level cache versioning (D13).

Every provider cache (Tushare HDF5, Baostock parquet/CSV/JSON) must carry the
data-schema generation plus its adjustment/cache-format version and reject
(and refetch) a cache written under different semantics. The same
``DATA_SCHEMA_VERSION`` bump that retires the pipeline pickle must retire
these caches too.
"""

from __future__ import annotations

import json
import shutil
import sys

import pandas as pd
import pytest
import quant_platform.data.providers.baostock_provider as bs_mod
from quant_platform.data.providers.baostock_provider import (
    BaostockDataProvider,
    _from_bs_code,
)
from quant_platform.data.providers.base import (
    cache_manifest_matches,
    cache_manifest_path,
    read_cache_manifest,
)
from quant_platform.utils import cache as cache_mod
from quant_platform.utils.cache import DATA_SCHEMA_VERSION

from tests.test_data.test_baostock_provider import FakeBaostock, _universe
from tests.test_data.test_tushare_loader_regressions import (
    FakeTushareAPI,
)
from tests.test_data.test_tushare_loader_regressions import (
    provider as tushare_provider,
)


def _cache_data_path(provider, key):
    """The data file the provider actually wrote (parquet, else CSV).

    pyarrow is optional; CI installs without it and the provider correctly
    falls back to CSV, so tests must not assume the parquet sibling exists.
    """
    base = provider._cache_path(key)
    return base if base.exists() else base.with_suffix(".csv")


def _cache_manifest_path(provider, key):
    base = provider._cache_path(key)
    return base.with_name(f"{base.name}.manifest.json")


def _bump_schema(monkeypatch) -> int:
    bumped = DATA_SCHEMA_VERSION + 1
    monkeypatch.setattr(cache_mod, "DATA_SCHEMA_VERSION", bumped)
    return bumped


# ---------------------------------------------------------------------------
# Tushare HDF5
# ---------------------------------------------------------------------------


class _ExplodingTushareAPI:
    def __getattr__(self, name):
        raise AssertionError(f"Tushare API must not be called ({name})")


def _tushare_cache_file(tmp_path):
    return tmp_path / "tushare_000300.SH_2024-01-01_2024-01-04.h5"


class TestTushareCacheVersioning:
    def test_cache_writes_a_version_manifest_and_is_reused(self, tmp_path):
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()

        cache_file = _tushare_cache_file(tmp_path)
        assert cache_file.exists()
        manifest = read_cache_manifest(cache_file)
        assert manifest is not None
        assert manifest["schema_version"] == DATA_SCHEMA_VERSION
        assert manifest["adjustment"] == "raw"
        assert manifest["cache_version"] == "v6"
        # The fake has no namechange endpoint: the manifest must record the
        # degraded ST feed instead of letting an empty table look complete.
        assert manifest["st_fetch_status"] == "unavailable"

        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = _ExplodingTushareAPI()
        reloaded._load_or_fetch_all()  # must be a pure cache hit
        assert reloaded._prices is not None and len(reloaded._prices) > 0

    def test_missing_manifest_forces_refetch(self, tmp_path):
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()
        cache_file = _tushare_cache_file(tmp_path)
        cache_manifest_path(cache_file).unlink()

        api = FakeTushareAPI(ann_date="20240102")
        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = api
        reloaded._load_or_fetch_all()

        assert api.daily_attempts, "an unversioned cache must not be served"
        assert read_cache_manifest(cache_file)["schema_version"] == DATA_SCHEMA_VERSION

    def test_schema_bump_forces_refetch_and_rewrites_manifest(
        self, tmp_path, monkeypatch
    ):
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()

        bumped = _bump_schema(monkeypatch)
        api = FakeTushareAPI(ann_date="20240102")
        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = api
        reloaded._load_or_fetch_all()

        assert api.daily_attempts, "a schema bump must invalidate the HDF5 cache"
        assert read_cache_manifest(_tushare_cache_file(tmp_path))[
            "schema_version"
        ] == bumped

    def test_adjustment_mismatch_forces_refetch(self, tmp_path):
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()
        cache_file = _tushare_cache_file(tmp_path)
        manifest_path = cache_manifest_path(cache_file)
        manifest = read_cache_manifest(cache_file)
        manifest["adjustment"] = "hfq"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        api = FakeTushareAPI(ann_date="20240102")
        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = api
        reloaded._load_or_fetch_all()

        assert api.daily_attempts

    def test_cache_identity_and_constituent_universe(self, tmp_path):
        prov = tushare_provider(tmp_path)
        identity = prov.cache_identity()
        assert identity["provider"] == "tushare"
        assert identity["adjustment"] == "raw"
        assert identity["adjusted_price_basis"] == "hfq"
        assert identity["index_code"] == "000300.SH"

        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()

        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = _ExplodingTushareAPI()
        universe = reloaded.cache_universe(
            start_date="2024-01-01", end_date="2024-01-04",
        )
        assert universe == ["600001.SH", "600002.SH"]

    def test_cache_universe_fetches_when_no_cache_exists(self, tmp_path):
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI()
        universe = prov.cache_universe(
            start_date="2024-01-01", end_date="2024-01-04",
        )
        assert universe == ["600001.SH", "600002.SH"]
        assert prov._constituent_history is not None

    def test_memory_error_preserves_the_hdf5_cache(self, tmp_path, monkeypatch):
        """D34 for provider caches: a transient read error must not delete."""
        prov = tushare_provider(tmp_path)
        prov._ts_api = FakeTushareAPI(ann_date="20240102")
        prov._load_or_fetch_all()
        cache_file = _tushare_cache_file(tmp_path)
        assert cache_file.exists()

        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = _ExplodingTushareAPI()

        def oom(path):
            raise MemoryError("simulated transient failure")

        monkeypatch.setattr(reloaded, "_load_from_cache", oom)
        with pytest.raises(MemoryError):
            reloaded._load_or_fetch_all()
        assert cache_file.exists()

    def test_benchmark_fallback_is_recorded_and_restored(self, tmp_path):
        class NoIndexAPI(FakeTushareAPI):
            def index_daily(self, *args, **kwargs):
                raise RuntimeError("index endpoint down")

        prov = tushare_provider(tmp_path)
        prov._ts_api = NoIndexAPI(ann_date="20240102")
        prov._load_or_fetch_all()
        assert prov.benchmark_source == "equal_weight_fallback"
        manifest = read_cache_manifest(_tushare_cache_file(tmp_path))
        assert manifest["benchmark_source"] == "equal_weight_fallback"

        reloaded = tushare_provider(tmp_path)
        reloaded._ts_api = _ExplodingTushareAPI()
        reloaded._load_or_fetch_all()
        assert reloaded.benchmark_source == "equal_weight_fallback"


# ---------------------------------------------------------------------------
# Baostock parquet/CSV and constant-key JSON caches
# ---------------------------------------------------------------------------


@pytest.fixture()
def baostock_provider(monkeypatch, tmp_path):
    fake = FakeBaostock(basic_rows=_universe())
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr(bs_mod, "CACHE_DIR", tmp_path)
    provider = BaostockDataProvider(cache_enabled=True)
    provider._bs = fake
    return provider, fake, tmp_path


def _price_frame():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "000001")], names=["date", "asset"],
    )
    return pd.DataFrame({"close": [10.0], "adj_factor": [1.0]}, index=index)


class TestBaostockCacheVersioning:
    def test_generic_cache_roundtrip_carries_manifest(self, baostock_provider):
        provider, _fake, tmp_path = baostock_provider
        key = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_abc_adj1"
        provider._save_cache(key, _price_frame())

        path = tmp_path / f"{key}.parquet"
        manifest = read_cache_manifest(path)
        assert manifest["schema_version"] == DATA_SCHEMA_VERSION
        assert manifest["adjustment"] == "hfq"
        assert cache_manifest_matches(
            path, adjustment="hfq", cache_version=provider._cache_version(),
        )

        loaded = provider._load_cache(key)
        assert loaded is not None
        assert float(loaded.iloc[0]["close"]) == 10.0

    def test_schema_bump_invalidates_generic_cache(
        self, baostock_provider, monkeypatch
    ):
        provider, _fake, tmp_path = baostock_provider
        key = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_abc_adj1"
        provider._save_cache(key, _price_frame())

        _bump_schema(monkeypatch)

        assert provider._load_cache(key) is None
        assert not (tmp_path / f"{key}.parquet").exists()

    def test_missing_manifest_invalidates_generic_cache(self, baostock_provider):
        provider, _fake, tmp_path = baostock_provider
        key = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_abc_adj1"
        provider._save_cache(key, _price_frame())
        cache_manifest_path(tmp_path / f"{key}.parquet").unlink()

        assert provider._load_cache(key) is None
        assert not (tmp_path / f"{key}.parquet").exists()

    def test_daily_key_embeds_schema_and_changes_on_bump(
        self, baostock_provider, monkeypatch
    ):
        provider, _fake, _tmp_path = baostock_provider
        keys: list[str] = []
        monkeypatch.setattr(provider, "_load_cache", lambda key: keys.append(key) or None)
        monkeypatch.setattr(provider, "_save_cache", lambda key, df: None)

        provider._fetch_daily_data(["000001"], "2024-01-01", "2024-01-04")
        first = keys[-1]
        assert f"s{DATA_SCHEMA_VERSION}" in first
        assert first.endswith("_adj1")

        bumped = _bump_schema(monkeypatch)
        provider._fetch_daily_data(["000001"], "2024-01-01", "2024-01-04")
        second = keys[-1]
        assert f"s{bumped}" in second
        assert first != second

    def test_constant_key_metadata_cache_is_versioned(
        self, baostock_provider, monkeypatch
    ):
        provider, fake, tmp_path = baostock_provider
        provider.get_metadata()
        old_key = f"metadata_{provider._cache_version()}"
        assert _cache_data_path(provider, old_key).exists()
        assert read_cache_manifest(provider._cache_path(old_key)) is not None
        queries_after_first = fake.basic_query_count

        _bump_schema(monkeypatch)
        provider.get_metadata()
        new_key = f"metadata_{provider._cache_version()}"
        assert _cache_data_path(provider, new_key).exists()
        assert fake.basic_query_count > queries_after_first

    def test_stock_universe_json_cache_is_versioned(
        self, baostock_provider, monkeypatch
    ):
        provider, fake, tmp_path = baostock_provider
        records = [{
            "code": "600001", "code_name": "AAA", "ipoDate": "2020-01-01",
            "outDate": "", "status": "1",
        }]
        provider._save_stock_universe_cache(records)
        assert provider._load_stock_universe_cache() == records

        _bump_schema(monkeypatch)

        assert provider._load_stock_universe_cache() is None
        before = fake.basic_query_count
        provider._get_universe_records()
        assert fake.basic_query_count == before + 1
        # The refreshed JSON carries the new schema manifest.
        manifest = read_cache_manifest(tmp_path / bs_mod._STOCK_BASIC_CACHE_FILE)
        assert manifest["schema_version"] == DATA_SCHEMA_VERSION + 1

    def test_manifest_is_bound_to_its_cache_key(self, baostock_provider):
        provider, _fake, tmp_path = baostock_provider
        first = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_aaa_adj1"
        second = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_bbb_adj1"
        provider._save_cache(first, _price_frame())
        first_data = _cache_data_path(provider, first)
        second_data = provider._cache_path(second)
        if first_data.suffix == ".csv":
            second_data = second_data.with_suffix(".csv")
        shutil.copy(first_data, second_data)
        shutil.copy(
            _cache_manifest_path(provider, first),
            _cache_manifest_path(provider, second),
        )

        # The copied manifest says it belongs to `first`; serving it under
        # `second` would return a different symbol set's data.
        assert provider._load_cache(second) is None
        assert not provider._cache_path(second).exists()
        assert not provider._cache_path(second).with_suffix(".csv").exists()

    def test_cache_universe_matches_the_fetched_price_sample(
        self, baostock_provider
    ):
        """The key's symbol set is exactly what get_prices/get_financials use.

        get_prices always samples 500 stocks (it ignores universe.n_stocks),
        so cache_universe must describe that sample, not the config's
        n_stocks (reviewer BLOCKER).
        """
        provider, fake, _tmp_path = baostock_provider
        universe = provider.cache_universe(
            start_date="2024-01-01", end_date="2024-01-04", n_symbols=100,
        )
        assert universe is not None
        assert len(provider._pinned_price_codes) == 500
        assert set(provider._pinned_price_codes) <= set(universe)

        provider.get_prices("2024-01-01", "2024-01-04")
        queried = {
            _from_bs_code(args[0])
            for args, _kwargs in fake.history_queries
        }
        assert queried == set(provider._pinned_price_codes)

    def test_pinned_universe_matches_compact_iso_dates(self, baostock_provider):
        provider, _fake, _tmp_path = baostock_provider
        provider.cache_universe(
            start_date="20240101", end_date="20240104", n_symbols=100,
        )
        assert provider._pinned_codes(
            "2024-01-01", "2024-01-04", financial=False,
        ) == provider._pinned_price_codes

    def test_cache_universe_disables_caching_without_a_real_universe(
        self, baostock_provider, monkeypatch
    ):
        provider, _fake, _tmp_path = baostock_provider
        monkeypatch.setattr(provider, "_get_universe_records", lambda: None)
        assert provider.cache_universe(
            start_date="2024-01-01", end_date="2024-01-04",
        ) is None

    def test_csv_fallback_does_not_serve_a_stale_parquet(
        self, baostock_provider, monkeypatch
    ):
        provider, _fake, tmp_path = baostock_provider
        key = f"daily_{provider._cache_version()}_2024-01-01_2024-01-04_1_abc_adj1"
        provider._save_cache(key, _price_frame())
        assert _cache_data_path(provider, key).exists()

        def parquet_unavailable(*args, **kwargs):
            raise RuntimeError("parquet engine unavailable")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", parquet_unavailable)
        newer = _price_frame()
        newer["close"] = 20.0
        provider._save_cache(key, newer)

        assert not provider._cache_path(key).exists()
        assert provider._cache_path(key).with_suffix(".csv").exists()
        loaded = provider._load_cache(key)
        assert loaded is not None
        assert float(loaded.iloc[0]["close"]) == 20.0

    def test_cache_identity_declares_adjustment(self, baostock_provider):
        provider, _fake, _tmp_path = baostock_provider
        identity = provider.cache_identity()
        assert identity["provider"] == "baostock"
        assert identity["adjustment"] == "raw"
        assert identity["adjustment_mode"] == "hfq"
        assert provider.price_adjustment == "raw"
