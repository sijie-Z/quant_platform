"""Cache provenance, versioning, corruption and concurrency (D12/D34/D35/D36).

Pins the cache-wave fixes:

* the data-stage key is computed from the materialized config, the effective
  provider chain, the sampled symbol set, the adjustment basis, the date
  range and ``DATA_SCHEMA_VERSION`` (D12);
* only genuine corruption evicts an entry, and eviction re-checks the file so
  a concurrent writer's fresh payload is not deleted (D34);
* every save can carry a provenance manifest that is exposed at load (D35);
* saves/evictions are serialised by a cross-process lock, failed saves keep
  the previous entry, are logged at WARNING and counted, temp files are
  cleaned and concurrent processes leave a valid file (D36).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from quant_platform.utils import cache as cache_mod
from quant_platform.utils.cache import (
    DATA_SCHEMA_VERSION,
    PipelineCache,
    _CrossProcessLock,
)

MANIFEST = {
    "requested_provider": "tushare",
    "effective_provider": "synthetic",
    "synthetic": True,
    "source": "synthetic",
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
    "data_date": "2025-12-31",
    "symbol_hash": "abc123",
    "adjustment": "synthetic_adj_factor",
    "schema_version": DATA_SCHEMA_VERSION,
    "app_version": "0.1.0",
}


def _config(n_stocks: int | None = None):
    from quant_platform.utils.config import load_config

    raw = {"data": {"provider": "synthetic"}}
    if n_stocks is not None:
        raw["universe"] = {"n_stocks": n_stocks}
    return load_config(raw=raw)


# ---------------------------------------------------------------------------
# D12: key completeness
# ---------------------------------------------------------------------------


class TestDataCacheKey:
    def test_materialized_defaults_are_in_the_key(self):
        """Same YAML + schema default vs explicit default: one key.

        A different value (n_stocks 300) must produce a different key even
        though neither YAML names every nested field.
        """
        implicit = _config()
        explicit_default = _config(500)
        changed = _config(300)

        def key(config):
            return PipelineCache.make_data_cache_key(
                config,
                {"provider": "synthetic", "adjustment": "synthetic_adj_factor"},
                symbols=[f"{i:06d}.SH" for i in range(config.universe.n_stocks)],
            )

        assert key(implicit) == key(explicit_default)
        assert key(implicit) != key(changed)

    def test_effective_provider_and_secondary_change_the_key(self):
        config = _config(2)
        symbols = ["000001.SH", "000002.SH"]
        plain = {
            "provider": "baostock",
            "adjustment": "hfq",
            "synthetic": False,
        }
        validated_adata = {
            "provider": "validated(baostock+adata)",
            "wrapper": "ValidatedProvider",
            "primary": "primary",
            "sources": {
                "primary": {"provider": "baostock", "adjustment": "hfq"},
                "secondary": {"provider": "adata", "adjustment": "provider-defined"},
            },
            "adjustment": "mixed",
            "synthetic": False,
        }
        validated_tushare = {
            "provider": "validated(baostock+tushare)",
            "wrapper": "ValidatedProvider",
            "primary": "primary",
            "sources": {
                "primary": {"provider": "baostock", "adjustment": "hfq"},
                "secondary": {"provider": "tushare", "adjustment": "hfq"},
            },
            "adjustment": "hfq",
            "synthetic": False,
        }
        keys = {
            PipelineCache.make_data_cache_key(config, identity, symbols=symbols)
            for identity in (plain, validated_adata, validated_tushare)
        }
        assert len(keys) == 3

    def test_symbol_set_and_order(self):
        config = _config(2)
        identity = {"provider": "synthetic", "adjustment": "synthetic_adj_factor"}
        ab = PipelineCache.make_data_cache_key(
            config, identity, symbols=["000001.SH", "000002.SZ"],
        )
        ba = PipelineCache.make_data_cache_key(
            config, identity, symbols=["000002.SZ", "000001.SH"],
        )
        ac = PipelineCache.make_data_cache_key(
            config, identity, symbols=["000001.SH", "000003.SZ"],
        )
        assert ab == ba, "the symbol hash must be order-insensitive"
        assert ab != ac

    def test_date_range_and_adjustment_change_the_key(self):
        config = _config(2)
        identity = {"provider": "tushare", "adjustment": "hfq"}
        base = PipelineCache.make_data_cache_key(
            config, identity, symbols=["600000.SH"],
            start_date="2021-01-01", end_date="2025-12-31",
        )
        other_range = PipelineCache.make_data_cache_key(
            config, identity, symbols=["600000.SH"],
            start_date="2022-01-01", end_date="2025-12-31",
        )
        raw_identity = {"provider": "tushare", "adjustment": "raw"}
        raw = PipelineCache.make_data_cache_key(
            config, raw_identity, symbols=["600000.SH"],
            start_date="2021-01-01", end_date="2025-12-31",
        )
        assert base != other_range
        assert base != raw

    def test_schema_bump_changes_the_key(self):
        config = _config(2)
        identity = {"provider": "synthetic"}
        current = PipelineCache.make_data_cache_key(config, identity)
        bumped = PipelineCache.make_data_cache_key(
            config, identity, schema_version=DATA_SCHEMA_VERSION + 1,
        )
        assert current != bumped

    def test_hash_symbols_is_stable(self):
        assert PipelineCache.hash_symbols(["b", "a"]) == PipelineCache.hash_symbols(
            ["a", "b"]
        )
        assert PipelineCache.hash_symbols(None) is None


# ---------------------------------------------------------------------------
# D34/D35: load, validate, evict, manifest
# ---------------------------------------------------------------------------


class TestLoadAndEviction:
    def test_manifest_roundtrip_and_load_exposes_it(self, tmp_path):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"payload": 1}, manifest=MANIFEST)

        data, manifest = cache.load_stage_with_manifest("data", "k")
        assert data == {"payload": 1}
        assert manifest == MANIFEST
        assert cache.load_stage("data", "k") == {"payload": 1}

    def test_transient_memory_error_preserves_the_file(self, tmp_path, monkeypatch):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"payload": 1})
        path = cache._make_path("data", "k")

        def boom(*args, **kwargs):
            raise MemoryError("simulated transient failure")

        monkeypatch.setattr(cache_mod.pickle, "load", boom)
        assert cache.load_stage("data", "k") is None
        assert path.exists(), "a transient read error must not evict a valid entry"

    def test_corrupt_pickle_is_evicted(self, tmp_path):
        cache = PipelineCache(tmp_path)
        path = cache._make_path("stage", "corrupt")
        path.write_text("this is not pickle data")
        assert cache.load_stage("stage", "corrupt") is None
        assert not path.exists()

    def test_schema_mismatch_is_evicted(self, tmp_path):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "old", {"payload": 1})
        path = cache._make_path("data", "old")
        cache._meta_path(path).write_text(
            json.dumps({"schema": 0, "stage": "data", "created": 0}),
            encoding="utf-8",
        )
        assert cache.load_stage("data", "old") is None
        assert not path.exists()

    def test_stage_mismatch_is_evicted(self, tmp_path):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"payload": 1})
        path = cache._make_path("data", "k")
        cache._meta_path(path).write_text(
            json.dumps(
                {"schema": DATA_SCHEMA_VERSION, "stage": "other", "created": 0}
            ),
            encoding="utf-8",
        )
        assert cache.load_stage("data", "k") is None
        assert not path.exists()

    def test_eviction_skips_a_concurrently_replaced_file(self, tmp_path):
        cache = PipelineCache(tmp_path)
        path = cache._make_path("data", "race")
        path.write_bytes(b"old-bytes")
        stat_before = path.stat()

        time.sleep(0.02)
        path.write_bytes(b"new-bytes-from-another-writer")

        assert cache._remove_if_unchanged(path, stat_before) is False
        assert path.exists()
        assert path.read_bytes() == b"new-bytes-from-another-writer"

    def test_unreadable_metadata_sidecar_is_evicted(self, tmp_path):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "meta", {"payload": 1})
        path = cache._make_path("data", "meta")
        cache._meta_path(path).write_text("{not json", encoding="utf-8")
        assert cache.load_stage("data", "meta") is None
        assert not path.exists()

    def test_second_save_overwrites_the_first(self, tmp_path):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"v": 1}, manifest=MANIFEST)
        cache.save_stage("data", "k", {"v": 2}, manifest=MANIFEST)
        data, manifest = cache.load_stage_with_manifest("data", "k")
        assert data == {"v": 2}
        assert manifest == MANIFEST


# ---------------------------------------------------------------------------
# D36: locking, save failures, temp cleanup, cross-process writes
# ---------------------------------------------------------------------------


class _DeniedLock:
    def __enter__(self):
        return False

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSaveSafety:
    def test_unpicklable_payload_keeps_the_previous_entry(self, tmp_path, caplog):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"v": 1})

        caplog.set_level(logging.WARNING)
        assert cache.save_stage("data", "k", lambda: None) is False
        assert cache.save_failures == 1
        assert any("Failed to save cache" in rec.message for rec in caplog.records)
        assert cache.load_stage("data", "k") == {"v": 1}

    def test_lock_timeout_keeps_the_previous_entry(self, tmp_path, caplog):
        cache = PipelineCache(tmp_path)
        cache.save_stage("data", "k", {"v": 1})

        caplog.set_level(logging.WARNING)
        cache._lock = lambda: _DeniedLock()  # type: ignore[assignment]
        assert cache.save_stage("data", "k", {"v": 2}) is False
        assert cache.save_failures == 1
        assert any("Failed to save cache" in rec.message for rec in caplog.records)
        assert cache.load_stage("data", "k") == {"v": 1}

    def test_stale_tmp_files_are_cleaned(self, tmp_path):
        cache = PipelineCache(tmp_path, auto_prune=False)
        stale = tmp_path / "data_k.pkl.tmp.123.456"
        fresh = tmp_path / "data_k.pkl.tmp.789.101"
        stale.write_bytes(b"x")
        fresh.write_bytes(b"x")
        old = time.time() - 7200
        os.utime(stale, (old, old))

        removed = cache._clean_stale_tmp()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists()

    def test_fallback_lock_is_exclusive(self, tmp_path):
        path = tmp_path / ".cache.lock"
        first = _CrossProcessLock(path, timeout=0.2)
        assert first._acquire_fallback(time.monotonic() + 0.2) is True
        try:
            second = _CrossProcessLock(path, timeout=0.2)
            deadline = time.monotonic() + 0.2
            assert second._acquire_fallback(deadline) is False
        finally:
            first.release()
        third = _CrossProcessLock(path, timeout=0.2)
        assert third._acquire_fallback(time.monotonic() + 0.2) is True
        third.release()

    def test_lock_is_cross_process(self, tmp_path):
        lock_path = tmp_path / ".cache.lock"
        code = (
            "import sys\n"
            "from quant_platform.utils.cache import _CrossProcessLock\n"
            "lock = _CrossProcessLock(sys.argv[1], timeout=0.3)\n"
            "with lock as locked:\n"
            "    print('locked' if locked else 'blocked')\n"
        )
        repo_root = Path(cache_mod.__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([
            str(repo_root.parent),
            str(repo_root / "packages" / "quant_core" / "src"),
            env.get("PYTHONPATH", ""),
        ])

        holder = _CrossProcessLock(lock_path, timeout=5.0)
        assert holder.__enter__() is True
        try:
            blocked = subprocess.run(
                [sys.executable, "-c", code, str(lock_path)],
                capture_output=True, text=True, env=env, timeout=60,
            )
            assert blocked.returncode == 0, blocked.stderr
            assert blocked.stdout.strip() == "blocked"
        finally:
            holder.release()

        unlocked = subprocess.run(
            [sys.executable, "-c", code, str(lock_path)],
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert unlocked.returncode == 0, unlocked.stderr
        assert unlocked.stdout.strip() == "locked"

    def test_two_processes_writing_leave_a_valid_file(self, tmp_path):
        cache_dir = tmp_path / "cache"
        code = (
            "import sys\n"
            "from quant_platform.utils.cache import PipelineCache\n"
            "cache = PipelineCache(sys.argv[1], auto_prune=False)\n"
            "tag = sys.argv[2]\n"
            "for i in range(20):\n"
            "    cache.save_stage('data', 'shared', "
            "{'writer': tag, 'i': i, 'blob': tag * 4096})\n"
        )
        repo_root = Path(cache_mod.__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([
            str(repo_root.parent),
            str(repo_root / "packages" / "quant_core" / "src"),
            env.get("PYTHONPATH", ""),
        ])
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(cache_dir), tag],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            )
            for tag in ("a", "b")
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, stderr.decode(errors="replace")

        cache = PipelineCache(cache_dir, auto_prune=False)
        value = cache.load_stage("data", "shared")
        assert isinstance(value, dict)
        assert value["writer"] in {"a", "b"}
        assert value["blob"] == value["writer"] * 4096
        assert cache.list_cached() == ["data_shared"]
        assert list(cache_dir.glob("*.tmp.*")) == []
