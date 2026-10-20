"""Pipeline result caching for fast iteration.

Caches intermediate results (data pipeline output, factors, alpha signal)
to avoid recomputation when running the same config repeatedly. Uses pickle
with config hash as cache key.

Retention (issue S11): the cache is bounded by default. Every entry carries a
small ``*.meta.json`` sidecar recording the data-schema generation and the
creation time. ``PipelineCache`` prunes on construction (and exposes
``prune()`` for the CLI):

- entries written under a retired ``DATA_SCHEMA_VERSION`` are deleted
  (they can never be served again: the schema version is part of the key);
- entries older than ``max_age_days`` are deleted;
- the oldest entries beyond ``max_entries`` are deleted (LRU by mtime).

Limits are configurable per instance and through the environment:
``QUANT_CACHE_MAX_AGE_DAYS`` and ``QUANT_CACHE_MAX_ENTRIES``.

Usage:
    from quant_platform.utils.cache import PipelineCache

    cache = PipelineCache("./.cache")
    data = cache.load_stage("data", config_hash)
    if data is None:
        data = run_data_pipeline()
        cache.save_stage("data", config_hash, data)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("./.quant_cache")
DEFAULT_MAX_AGE_DAYS = 30.0
DEFAULT_MAX_ENTRIES = 64

# Bump whenever the data layer changes the meaning or layout of what it
# produces (columns, adjustment convention, PIT handling, ...). Cached stages
# written under an older schema must not be replayed as if identical.
# 4: benchmarks aligned to the forward-return convention; metadata may carry
#    a PIT sector panel.
# 5: synthetic panels use the exchange calendar (exchange holidays removed
#    from prices/benchmark/financial visibility), suspension flags are derived
#    from the data, and the tradability mask covers listing/delisting/
#    suspension. Panels cached under 4 are business-day grids and must not be
#    replayed.
DATA_SCHEMA_VERSION = 5

# Bump when the key recipe itself changes, so entries written by an older
# recipe cannot collide with new ones.
CACHE_KEY_VERSION = 2


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected a number)", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (expected an integer)", name, raw)
        return default


class PipelineCache:
    """Caches pipeline stages to disk for fast re-runs.

    Cache key = MD5 hash of the config subset plus everything that changes the
    produced data but lives outside the YAML: the data provider actually used
    (a silent fallback to synthetic must not share the requested provider's
    slot), explicit CLI/environment overrides, and the data schema version.
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        max_age_days: float | None = None,
        max_entries: int | None = None,
        auto_prune: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = (
            _env_float("QUANT_CACHE_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)
            if max_age_days is None else float(max_age_days)
        )
        self.max_entries = (
            _env_int("QUANT_CACHE_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)
            if max_entries is None else int(max_entries)
        )
        if auto_prune:
            try:
                self.prune()
            except Exception as exc:  # pruning must never break a run
                logger.warning("Cache prune failed: %s", exc)

    def _make_path(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / f"{stage}_{cache_key}.pkl"

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.meta.json")

    @staticmethod
    def make_config_hash(
        config_dict: dict[str, Any],
        provider: str | None = None,
        overrides: dict[str, Any] | None = None,
        schema_version: int | str | None = None,
    ) -> str:
        """Create a deterministic hash from everything that determines a run.

        Args:
            config_dict: The raw config mapping (as loaded from YAML).
            provider: The provider that will actually produce the data. Pass
                the *effective* provider -- if a silent fallback to synthetic
                happened, hashing the requested provider would make the run
                read the requested provider's cached output.
            overrides: Explicit overrides applied on top of the YAML (CLI
                flags, environment-derived settings). Their presence and values
                both matter, so ``{"use_baostock": False}`` and no overrides at
                all produce different keys only if they change behaviour.
            schema_version: Data-layer schema version; defaults to
                ``DATA_SCHEMA_VERSION``.

        The raw YAML alone is not a run identity: ``--use-baostock``, an
        environment token enabling a real provider, a silent fallback, or a
        code change to the data schema all alter the output while leaving the
        YAML untouched.
        """
        payload = {
            "key_version": CACHE_KEY_VERSION,
            "config": config_dict,
            "provider": provider,
            "overrides": overrides or {},
            "schema_version": (
                DATA_SCHEMA_VERSION if schema_version is None else schema_version
            ),
        }
        config_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def load_stage(self, stage: str, cache_key: str) -> Any | None:
        """Load a cached stage. Returns None if not found."""
        path = self._make_path(stage, cache_key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                logger.info("Cache hit: %s (%s)", stage, cache_key)
                return data
            except Exception as e:
                logger.debug("Failed to load cache %s: %s", stage, e)
                # Another writer may hold the file on Windows; a failed
                # cleanup must not escape as PermissionError.
                self._remove_entry(path)
        return None

    def save_stage(self, stage: str, cache_key: str, data: Any) -> None:
        """Save a stage to cache atomically.

        Writing the target in place let a concurrent reader deserialize a
        half-written pickle (or hit a Windows sharing violation). Write to a
        unique temp file, fsync, then os.replace.
        """
        path = self._make_path(stage, cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            self._write_meta(path, stage)
            logger.info("Cached: %s (%s)", stage, cache_key)
        except Exception as e:
            logger.debug("Failed to save cache %s: %s", stage, e)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> int:
        """Clear all cached files (and metadata). Returns files removed."""
        count = 0
        for p in self.cache_dir.glob("*.pkl"):
            self._remove_entry(p)
            count += 1
        for meta in self.cache_dir.glob("*.pkl.meta.json"):
            try:
                meta.unlink()
            except OSError:
                pass
        logger.info("Cleared %d cache files", count)
        return count

    def list_cached(self) -> list[str]:
        """List cached stages."""
        return sorted(p.stem for p in self.cache_dir.glob("*.pkl"))

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def prune(self) -> int:
        """Enforce schema-version, TTL and LRU limits. Returns files removed.

        Files without a metadata sidecar are pre-retention entries whose key
        can no longer be reproduced after a schema/key bump; they are aged by
        their mtime against the same TTL and count limits.
        """
        now = time.time()
        removed = 0
        survivors: list[tuple[float, Path]] = []

        for path in self.cache_dir.glob("*.pkl"):
            meta = self._read_meta(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue

            if meta is not None and meta.get("schema") != DATA_SCHEMA_VERSION:
                self._remove_entry(path)
                removed += 1
                continue

            age_days = (now - mtime) / 86400.0
            if self.max_age_days is not None and age_days > self.max_age_days:
                self._remove_entry(path)
                removed += 1
                continue

            survivors.append((mtime, path))

        if self.max_entries is not None and self.max_entries >= 0:
            survivors.sort(key=lambda item: item[0], reverse=True)
            for _mtime, path in survivors[self.max_entries:]:
                self._remove_entry(path)
                removed += 1

        if removed:
            logger.info(
                "Cache prune removed %d entr%s (schema=%d, max_age=%sd, max_entries=%s)",
                removed, "y" if removed == 1 else "ies",
                DATA_SCHEMA_VERSION, self.max_age_days, self.max_entries,
            )
        return removed

    def _remove_entry(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            self._meta_path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def _write_meta(self, path: Path, stage: str) -> None:
        meta = {
            "schema": DATA_SCHEMA_VERSION,
            "stage": stage,
            "created": time.time(),
        }
        meta_path = self._meta_path(path)
        tmp = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            os.replace(tmp, meta_path)
        except Exception as exc:
            logger.debug("Failed to write cache metadata for %s: %s", path, exc)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_meta(self, path: Path) -> dict | None:
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
