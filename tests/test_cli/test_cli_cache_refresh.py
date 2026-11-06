"""CLI data-stage cache behaviour (D12/D33/D35).

Pins ``--force`` overwriting (not just bypassing) the cache entry, the
``cache refresh`` command, provenance served from a cache hit (a synthetic
fallback must not masquerade as a real-provider run), and ``--no-cache``
semantics. Uses a fake provider resolution and a fake data loader so no
network or full pipeline is involved.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import quant_platform.main as cli
from quant_platform.utils.cache import DATA_SCHEMA_VERSION, PipelineCache
from quant_platform.utils.config import load_config

INDEX = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])


def _config(**data_overrides):
    raw = {
        "data": {
            "provider": "synthetic",
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            **data_overrides,
        },
        "universe": {"n_stocks": 2},
    }
    return load_config(raw=raw)


def _data_tuple(value: float):
    prices = pd.DataFrame(
        {"A": [value], "B": [value + 1.0]}, index=INDEX[:1],
    )
    returns = prices.pct_change().fillna(0.0)
    return prices, returns, None, None, None, None


def _resolution(
    config,
    *,
    requested: str = "synthetic",
    effective: str = "synthetic",
    identity: dict | None = None,
    universe: list[str] | None = None,
):
    if identity is None:
        synthetic = effective == "synthetic"
        identity = {
            "provider": effective,
            "adjustment": "synthetic_adj_factor" if synthetic else "hfq",
            "synthetic": synthetic,
        }
    if universe is None:
        universe = [f"{i:06d}.SZ" for i in range(1, config.universe.n_stocks + 1)]
    return {
        "provider": object(),
        "provider_name": effective,
        "requested_provider": requested,
        "effective_provider": effective,
        "identity": identity,
        "universe": universe,
        "allow_fallback": requested != effective,
        "failures": [],
        "validated": False,
    }


def _install_fake_load(monkeypatch, data, resolution, config):
    calls: list[int] = []

    def fake_load(config_, use_tushare=False, use_baostock=False, resolution=None):
        calls.append(1)
        cli._LAST_DATA_PROVENANCE = cli._build_data_manifest(
            resolution, config_, data_date="2024-01-03",
            payload_assets=list(data[0].columns),
        )
        return data

    monkeypatch.setattr(cli, "_load_data", fake_load)
    return calls


class TestForceRefresh:
    def test_force_overwrites_the_stale_entry_for_the_next_run(
        self, tmp_path, monkeypatch
    ):
        """D33: old entry present -> --force writes new payload -> normal run
        serves the new payload (the pre-fix code left the old one in place)."""
        config = _config()
        resolution = _resolution(config)
        cache = PipelineCache(tmp_path)
        key = cli._data_cache_key(config, resolution)

        old = _data_tuple(1.0)
        old_manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(old[0].columns),
        )
        assert cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": old, "provenance": old_manifest},
            manifest=old_manifest,
        )

        new = _data_tuple(9.0)
        calls = _install_fake_load(monkeypatch, new, resolution, config)
        _tuple, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=False, write_cache=True,
        )
        assert hit is False
        assert calls == [1]
        assert provenance["cache"]["hit"] is False

        # The next normal run must not recompute: the forced payload replaced
        # the old one.
        monkeypatch.setattr(
            cli, "_load_data",
            lambda *args, **kwargs: pytest.fail("normal run recomputed"),
        )
        loaded, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=False,
        )
        assert hit is True
        assert float(loaded[0].iloc[0]["A"]) == 9.0
        assert provenance["synthetic"] is True

    def test_no_cache_neither_reads_nor_writes(self, tmp_path, monkeypatch):
        config = _config()
        resolution = _resolution(config)
        cache = PipelineCache(tmp_path)
        key = cli._data_cache_key(config, resolution)

        old = _data_tuple(1.0)
        old_manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(old[0].columns),
        )
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": old, "provenance": old_manifest},
            manifest=old_manifest,
        )

        new = _data_tuple(7.0)
        calls = _install_fake_load(monkeypatch, new, resolution, config)
        loaded, _provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=False, write_cache=False,
        )
        assert hit is False
        assert calls == [1]
        assert float(loaded[0].iloc[0]["A"]) == 7.0
        # The old entry is untouched (no read, no write).
        cached = cache.load_stage(cli.DATA_STAGE, key)
        assert float(cached["data"][0].iloc[0]["A"]) == 1.0

    def test_force_with_no_cache_does_not_write(self, tmp_path, monkeypatch):
        """--force --no-cache must recompute without touching the cache."""
        config = _config()
        resolution = _resolution(config)
        cache = PipelineCache(tmp_path)
        key = cli._data_cache_key(config, resolution)

        old = _data_tuple(1.0)
        old_manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(old[0].columns),
        )
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": old, "provenance": old_manifest},
            manifest=old_manifest,
        )

        new = _data_tuple(6.0)
        calls = _install_fake_load(monkeypatch, new, resolution, config)
        loaded, _provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=False, write_cache=False,
        )
        assert hit is False
        assert calls == [1]
        assert float(loaded[0].iloc[0]["A"]) == 6.0
        cached = cache.load_stage(cli.DATA_STAGE, key)
        assert float(cached["data"][0].iloc[0]["A"]) == 1.0

    def test_cache_refresh_refuses_an_uncacheable_provider(
        self, tmp_path, monkeypatch, capsys
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n  provider: synthetic\nuniverse:\n  n_stocks: 2\n",
            encoding="utf-8",
        )
        config = _config()
        resolution = _resolution(config)
        resolution["universe"] = None
        monkeypatch.setattr(
            cli, "_resolve_provider", lambda *args, **kwargs: resolution,
        )
        monkeypatch.setattr(
            cli, "_load_data",
            lambda *args, **kwargs: pytest.fail("must not compute"),
        )
        monkeypatch.setattr(sys, "argv", [
            "main.py", "cache", "refresh",
            "--config", str(config_path),
            "--cache-dir", str(tmp_path / "cache"),
        ])

        assert cli.main() == 1
        assert "not cacheable" in capsys.readouterr().err

    def test_cache_refresh_command_recomputes_only_the_data_stage(
        self, tmp_path, monkeypatch, capsys
    ):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "data:\n"
            "  provider: synthetic\n"
            "  start_date: '2024-01-01'\n"
            "  end_date: '2024-01-05'\n"
            "universe:\n"
            "  n_stocks: 2\n",
            encoding="utf-8",
        )
        config = load_config(raw={
            "data": {
                "provider": "synthetic",
                "start_date": "2024-01-01",
                "end_date": "2024-01-05",
            },
            "universe": {"n_stocks": 2},
        })
        resolution = _resolution(config)
        monkeypatch.setattr(
            cli, "_resolve_provider", lambda *args, **kwargs: resolution,
        )
        data = _data_tuple(3.0)
        calls = _install_fake_load(monkeypatch, data, resolution, config)

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(sys, "argv", [
            "main.py", "cache", "refresh",
            "--config", str(config_path),
            "--cache-dir", str(cache_dir),
        ])

        assert cli.main() == 0
        out = capsys.readouterr().out
        assert "Refreshed cache entry" in out
        assert calls == [1]

        cache = PipelineCache(cache_dir, auto_prune=False)
        expected_key = cli._data_cache_key(config, resolution)
        assert cache.list_cached() == [f"{cli.DATA_STAGE}_{expected_key}"]
        payload = cache.load_stage(cli.DATA_STAGE, expected_key)
        assert float(payload["data"][0].iloc[0]["A"]) == 3.0
        stored_manifest = cache.load_stage_with_manifest(
            cli.DATA_STAGE, expected_key,
        )[1]
        assert stored_manifest["schema_version"] == DATA_SCHEMA_VERSION
        assert stored_manifest["synthetic"] is True


class TestCacheHitProvenance:
    def test_synthetic_fallback_hit_does_not_masquerade_as_real(
        self, tmp_path, monkeypatch
    ):
        """D35: the cached manifest reports the original source (tushare
        requested, synthetic effective), not today's requested provider."""
        config = _config(provider="tushare", allow_synthetic_fallback=True)
        resolution = _resolution(
            config, requested="tushare", effective="synthetic",
            identity={
                "provider": "synthetic",
                "adjustment": "synthetic_adj_factor",
                "synthetic": True,
            },
        )
        key = cli._data_cache_key(config, resolution)
        cache = PipelineCache(tmp_path)
        data = _data_tuple(2.0)
        manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(data[0].columns),
        )
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": data, "provenance": manifest},
            manifest=manifest,
        )

        monkeypatch.setattr(
            cli, "_load_data",
            lambda *args, **kwargs: pytest.fail("cache hit recomputed data"),
        )
        _tuple, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=False,
        )
        assert hit is True
        assert provenance["requested_provider"] == "tushare"
        assert provenance["effective_provider"] == "synthetic"
        assert provenance["synthetic"] is True
        assert provenance["source"] == "synthetic"
        assert provenance["cache"]["hit"] is True

    def test_real_provider_hit_reports_the_original_manifest(
        self, tmp_path, monkeypatch
    ):
        config = _config(provider="baostock")
        resolution = _resolution(
            config, requested="baostock", effective="baostock",
            identity={
                "provider": "baostock",
                "adjustment": "hfq",
                "synthetic": False,
            },
        )
        key = cli._data_cache_key(config, resolution)
        cache = PipelineCache(tmp_path)
        data = _data_tuple(4.0)
        manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(data[0].columns),
        )
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": data, "provenance": manifest},
            manifest=manifest,
        )

        monkeypatch.setattr(
            cli, "_load_data",
            lambda *args, **kwargs: pytest.fail("cache hit recomputed data"),
        )
        _tuple, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=False,
        )
        assert hit is True
        for field in (
            "requested_provider", "effective_provider", "source", "synthetic",
            "start_date", "end_date", "data_date", "symbol_hash",
            "adjustment", "schema_version", "created_at", "app_version",
        ):
            assert provenance[field] == manifest[field]

    def test_payload_manifest_mismatch_is_not_served(
        self, tmp_path, monkeypatch
    ):
        """A reader that pairs an old payload with a newer sidecar must not
        report the sidecar's provenance for the old data."""
        config = _config()
        resolution = _resolution(config)
        key = cli._data_cache_key(config, resolution)
        cache = PipelineCache(tmp_path)
        data = _data_tuple(1.0)
        manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(data[0].columns),
        )
        stale_provenance = dict(manifest)
        stale_provenance["created_at"] = "1999-01-01T00:00:00+00:00"
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": data, "provenance": stale_provenance},
            manifest=manifest,
        )

        new = _data_tuple(5.0)
        calls = _install_fake_load(monkeypatch, new, resolution, config)
        loaded, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=True,
        )
        assert hit is False
        assert calls == [1]
        assert float(loaded[0].iloc[0]["A"]) == 5.0
        assert provenance["created_at"] != "1999-01-01T00:00:00+00:00"

    def test_validated_runs_are_not_cached(self, tmp_path, monkeypatch):
        """The key cannot attest which validated sources contributed."""
        config = _config()
        resolution = _resolution(config)
        resolution["validated"] = True
        key = cli._data_cache_key(config, resolution)
        cache = PipelineCache(tmp_path)
        data = _data_tuple(1.0)
        calls = _install_fake_load(monkeypatch, data, resolution, config)

        _tuple, _provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=True,
        )
        assert hit is False
        assert calls == [1]
        assert cache.list_cached() == []

    def test_incompatible_manifest_is_evicted_and_recomputed(
        self, tmp_path, monkeypatch
    ):
        config = _config()
        resolution = _resolution(config)
        key = cli._data_cache_key(config, resolution)
        cache = PipelineCache(tmp_path)
        data = _data_tuple(1.0)
        stale_manifest = cli._build_data_manifest(
            resolution, config, "2024-01-03", list(data[0].columns),
        )
        stale_manifest["effective_provider"] = "baostock"
        stale_manifest["synthetic"] = False
        cache.save_stage(
            cli.DATA_STAGE, key,
            {"data": data, "provenance": stale_manifest},
            manifest=stale_manifest,
        )

        new = _data_tuple(8.0)
        calls = _install_fake_load(monkeypatch, new, resolution, config)
        _tuple, provenance, hit = cli._load_data_stage(
            cache, key, config, resolution,
            read_cache=True, write_cache=True,
        )
        assert hit is False
        assert calls == [1]
        assert provenance["effective_provider"] == "synthetic"
        # The stale entry was replaced with a compatible one.
        replacement = cache.load_stage_with_manifest(cli.DATA_STAGE, key)[1]
        assert replacement["effective_provider"] == "synthetic"
