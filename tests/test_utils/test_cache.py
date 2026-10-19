"""Tests for pipeline caching."""

import tempfile

import pandas as pd

from quant_platform.utils.cache import DATA_SCHEMA_VERSION, PipelineCache


class TestPipelineCache:
    """Test the PipelineCache class."""

    def test_make_config_hash_deterministic(self):
        config1 = {"universe": {"n_stocks": 500}, "data": {"start_date": "2021-01-01"}}
        config2 = {"universe": {"n_stocks": 500}, "data": {"start_date": "2021-01-01"}}
        h1 = PipelineCache.make_config_hash(config1)
        h2 = PipelineCache.make_config_hash(config2)
        assert h1 == h2
        assert len(h1) == 12

    def test_make_config_hash_different(self):
        config1 = {"universe": {"n_stocks": 500}}
        config2 = {"universe": {"n_stocks": 300}}
        h1 = PipelineCache.make_config_hash(config1)
        h2 = PipelineCache.make_config_hash(config2)
        assert h1 != h2


class TestConfigHashCoversTheRun:
    """The hash used to cover the raw YAML only. The provider actually used,
    explicit overrides and the data schema all change the output without
    touching the YAML, so a warm cache could serve the wrong run's data."""

    config = {"universe": {"n_stocks": 500}, "data": {"provider": "synthetic"}}

    def test_different_providers_hash_differently(self):
        baostock = PipelineCache.make_config_hash(self.config, provider="baostock")
        tushare = PipelineCache.make_config_hash(self.config, provider="tushare")
        synthetic = PipelineCache.make_config_hash(self.config, provider="synthetic")

        assert len({baostock, tushare, synthetic}) == 3

    def test_a_silent_fallback_does_not_land_in_the_requested_slot(self):
        """Baostock requested but unavailable -> synthetic data. That run must
        not read or write the baostock cache entry."""
        requested = PipelineCache.make_config_hash(self.config, provider="baostock")
        fell_back = PipelineCache.make_config_hash(self.config, provider="synthetic")

        assert requested != fell_back

    def test_provider_affects_the_key_even_when_config_names_one(self):
        # --use-baostock overrides the config's provider.
        from_config = PipelineCache.make_config_hash(
            self.config, provider=self.config["data"]["provider"],
        )
        overridden = PipelineCache.make_config_hash(self.config, provider="baostock")
        assert from_config != overridden

    def test_explicit_overrides_change_the_key(self):
        base = PipelineCache.make_config_hash(self.config)
        with_flag = PipelineCache.make_config_hash(
            self.config, overrides={"use_baostock": True},
        )
        assert base != with_flag

    def test_schema_version_changes_the_key(self):
        current = PipelineCache.make_config_hash(self.config)
        bumped = PipelineCache.make_config_hash(
            self.config, schema_version=DATA_SCHEMA_VERSION + 1,
        )
        assert current != bumped

    def test_default_schema_version_matches_the_explicit_constant(self):
        assert PipelineCache.make_config_hash(
            self.config, schema_version=DATA_SCHEMA_VERSION,
        ) == PipelineCache.make_config_hash(self.config)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(tmpdir)
            key = "test_key_123"
            data = {"a": 1, "b": [2, 3], "c": pd.DataFrame({"x": [1, 2, 3]})}

            # Should not exist yet
            assert cache.load_stage("test_stage", key) is None

            # Save and reload
            cache.save_stage("test_stage", key, data)
            loaded = cache.load_stage("test_stage", key)
            assert loaded is not None
            assert loaded["a"] == 1
            assert loaded["b"] == [2, 3]
            assert loaded["c"].equals(pd.DataFrame({"x": [1, 2, 3]}))

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(tmpdir)
            assert cache.load_stage("nonexistent", "bad_key") is None

    def test_list_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(tmpdir)
            assert len(cache.list_cached()) == 0

            cache.save_stage("stage_a", "key1", {"data": 1})
            cache.save_stage("stage_b", "key1", {"data": 2})

            entries = cache.list_cached()
            assert len(entries) == 2
            assert any("stage_a" in e for e in entries)
            assert any("stage_b" in e for e in entries)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(tmpdir)
            cache.save_stage("s1", "k1", {"a": 1})
            cache.save_stage("s2", "k1", {"b": 2})
            assert len(cache.list_cached()) == 2

            count = cache.clear()
            assert count == 2
            assert len(cache.list_cached()) == 0

    def test_corrupted_cache_handled(self):
        """Corrupted cache files should not crash, just return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(tmpdir)
            key = "corrupt_test"

            # Write a corrupted file
            path = cache._make_path("stage", key)
            path.write_text("this is not valid pickle data")

            result = cache.load_stage("stage", key)
            assert result is None  # Should handle gracefully
            assert not path.exists()  # Should clean up corrupted file
