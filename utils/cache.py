"""Pipeline result caching for fast iteration.

Caches intermediate results (data pipeline output, factors, alpha signal)
to avoid recomputation when running the same config repeatedly. Uses pickle
with config hash as cache key.

Provenance (D35): every entry written by :meth:`PipelineCache.save_stage` can
carry a manifest (requested/effective provider, synthetic flag, source, date
range, symbol-set hash, adjustment basis, data date, schema version,
creation time, app version). It is stored in the ``*.meta.json`` sidecar and
returned by :meth:`PipelineCache.load_stage_with_manifest`, so a cache hit can
report the provenance of the *original* fetch instead of the current run's
requested provider.

Versioning (D13): the sidecar records ``DATA_SCHEMA_VERSION``; an entry whose
schema does not match is never returned and is evicted. Provider caches use
the same version (see ``data.providers.base``), so one schema bump invalidates
both layers.

Concurrency (D36): saves and evictions are serialised by a cross-process file
lock (``msvcrt.locking`` on Windows, ``fcntl.flock`` on POSIX, atomic lock file
with stale detection as a fallback). Writes go to unique temp files and are
published with ``os.replace``; a failed save leaves the previous entry intact,
is logged at WARNING and counted in ``save_failures``. Stale ``*.tmp.*`` files
older than one hour are removed by ``prune()``.

Corruption handling (D34): only genuine corruption (unpickling error, EOF,
schema/manifest validation failure) evicts an entry. Transient read errors
(MemoryError, I/O errors) leave the file in place. Eviction re-stats the file
and skips when a concurrent writer replaced it in the meantime.

Retention (issue S11): the cache is bounded by default. ``PipelineCache``
prunes on construction (and exposes ``prune()`` for the CLI):

- entries written under a retired ``DATA_SCHEMA_VERSION`` are deleted
  (they can never be served again: the schema version is part of the key);
- entries older than ``max_age_days`` are deleted;
- the oldest entries beyond ``max_entries`` are deleted (LRU by mtime).

Limits are configurable per instance and through the environment:
``QUANT_CACHE_MAX_AGE_DAYS``, ``QUANT_CACHE_MAX_ENTRIES`` and
``QUANT_CACHE_LOCK_TIMEOUT``.

Usage:
    from quant_platform.utils.cache import PipelineCache

    cache = PipelineCache("./.cache")
    data = cache.load_stage("data", config_hash)
    if data is None:
        data = run_data_pipeline()
        cache.save_stage("data", config_hash, data)
"""

from __future__ import annotations

import dataclasses
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
DEFAULT_LOCK_TIMEOUT = 10.0

#: Temp files older than this are leftovers from a crashed writer.
STALE_TMP_SECONDS = 3600.0

# Bump whenever the data layer changes the meaning or layout of what it
# produces (columns, adjustment convention, PIT handling, ...). Cached stages
# written under an older schema must not be replayed as if identical, and
# provider-level caches (HDF5/parquet/JSON) carry the same version.
# 4: benchmarks aligned to the forward-return convention; metadata may carry
#    a PIT sector panel.
# 5: synthetic panels use the exchange calendar (exchange holidays removed
#    from prices/benchmark/financial visibility), suspension flags are derived
#    from the data, and the tradability mask covers listing/delisting/
#    suspension. Panels cached under 4 are business-day grids and must not be
#    replayed.
# 6: universe PIT semantics changed (R6b): index membership was split out of
#    `tradable_mask` into a separate `universe_mask` consulted by the buy gate
#    only, and the provider's snapshot forward fill now propagates constituent
#    *exits* instead of keeping every former member forever. Panels cached
#    under 5 carry the old "once a member, always a member" universe and the
#    fused mask; they must not be replayed.
# 7: two reasons. (a) R6c changed the `excluded_mask` semantics: factor levels
#    and signal membership are now computed from the uncropped price history
#    before any exclusion, so a cached factor/signal stage produced under the
#    old cropped panel must not be replayed. (b) R7a adds the Baostock ST
#    cache fields: the daily panel persists the observed `isST` side column
#    and providers derive ST episodes from it; provider caches written under
#    6 have no ST evidence and would silently disable ST screening on real
#    data, so they must not be replayed either.
DATA_SCHEMA_VERSION = 7

# Bump when the key recipe itself changes, so entries written by an older
# recipe cannot collide with new ones.
# 3: the data key is computed after provider resolution from the *materialized*
#    config (schema defaults included), the effective provider chain (incl.
#    ValidatedProvider wrappers), the sampled symbol set, the date range and
#    the adjustment basis (D12).
CACHE_KEY_VERSION = 3


class CacheValidationError(Exception):
    """A cache entry is structurally valid pickle but fails schema checks."""


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


def app_version() -> str:
    """Version of the running application, for the provenance manifest."""
    try:
        from importlib.metadata import version as _dist_version

        return str(_dist_version("quant_platform"))
    except Exception:  # not installed as a distribution (source checkout)
        pass
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with open(pyproject, "rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except Exception:
        return "unknown"


def materialize_config(config: Any) -> Any:
    """Return a JSON-friendly, fully-defaulted form of a config.

    A typed ``Config`` dataclass is converted with ``dataclasses.asdict`` so
    schema defaults are part of the cache identity; raw mappings are used
    as-is (callers that want default materialisation must pass the parsed
    config object).
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        try:
            return dataclasses.asdict(config)
        except Exception as exc:  # defensive: never let hashing break a run
            logger.warning("Could not materialize config for hashing: %s", exc)
            return repr(config)
    return config


class _CrossProcessLock:
    """Best-effort cross-process lock with a portable fallback.

    ``msvcrt.locking`` (Windows) and ``fcntl.flock`` (POSIX) release the lock
    automatically when the process dies. When neither is available, an atomic
    lock file (``O_CREAT|O_EXCL``) with stale detection is used instead.

    ``__enter__`` returns ``True`` when the lock was acquired and ``False`` on
    timeout; it never raises for lock contention (a cache miss must not change
    research results).
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = DEFAULT_LOCK_TIMEOUT,
        stale_after: float = 300.0,
    ):
        self.path = Path(path)
        self.timeout = max(0.0, float(timeout))
        self.stale_after = max(1.0, float(stale_after))
        self._handle = None
        self._locked = False
        self._fallback_path: Path | None = None

    def __enter__(self) -> bool:
        try:
            self._handle = open(self.path, "a+b")
        except OSError as exc:
            logger.warning("Could not open cache lock %s: %s", self.path, exc)
            return False
        deadline = time.monotonic() + self.timeout
        if os.name == "nt":
            return self._acquire_windows(deadline)
        try:
            return self._acquire_posix(deadline)
        except ImportError:
            return self._acquire_fallback(deadline)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self._handle is None and self._fallback_path is None:
            return
        if self._locked:
            try:
                if self._fallback_path is not None:
                    try:
                        self._fallback_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                elif os.name == "nt":
                    import msvcrt

                    self._handle.seek(0)
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        self._locked = False
        self._fallback_path = None
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    # -- platform implementations -------------------------------------

    def _acquire_windows(self, deadline: float) -> bool:
        import msvcrt

        # Ensure the locked byte range exists even on an empty file.
        try:
            self._handle.seek(0, os.SEEK_END)
            if self._handle.tell() == 0:
                self._handle.write(b"\0")
                self._handle.flush()
        except OSError:
            pass
        while True:
            try:
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
                self._locked = True
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)

    def _acquire_posix(self, deadline: float) -> bool:
        import fcntl

        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)

    def _acquire_fallback(self, deadline: float) -> bool:
        lock_path = self.path.with_name(f"{self.path.name}.atomic")
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode())
                finally:
                    os.close(fd)
                self._fallback_path = lock_path
                self._locked = True
                return True
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > self.stale_after:
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.05)


class PipelineCache:
    """Caches pipeline stages to disk for fast re-runs.

    Cache key = MD5 hash of the config subset plus everything that changes the
    produced data but lives outside the YAML: the data provider actually used
    (a silent fallback to synthetic must not share the requested provider's
    slot), explicit CLI/environment overrides, and the data schema version.

    ``make_data_cache_key`` is the D12 recipe: it hashes the *materialized*
    config, the effective provider chain, the sampled symbol set (hash), the
    adjustment basis and the date range.
    """

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        max_age_days: float | None = None,
        max_entries: int | None = None,
        auto_prune: bool = True,
        lock_timeout: float | None = None,
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
        self.lock_timeout = (
            _env_float("QUANT_CACHE_LOCK_TIMEOUT", DEFAULT_LOCK_TIMEOUT)
            if lock_timeout is None else float(lock_timeout)
        )
        #: Number of failed saves since construction (D36). Saves never raise.
        self.save_failures = 0
        self._lock_path = self.cache_dir / ".cache.lock"
        if auto_prune:
            try:
                self.prune()
            except Exception as exc:  # pruning must never break a run
                logger.warning("Cache prune failed: %s", exc)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _make_path(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / f"{stage}_{cache_key}.pkl"

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.meta.json")

    @staticmethod
    def hash_symbols(symbols: list[str] | tuple[str, ...] | set[str] | None) -> str | None:
        """Stable hash of a symbol set (order-insensitive)."""
        if symbols is None:
            return None
        values = sorted({str(symbol) for symbol in symbols})
        return hashlib.md5(",".join(values).encode()).hexdigest()[:12]

    @staticmethod
    def make_config_hash(
        config_dict: dict[str, Any],
        provider: str | None = None,
        overrides: dict[str, Any] | None = None,
        schema_version: int | str | None = None,
        data_identity: dict[str, Any] | None = None,
    ) -> str:
        """Create a deterministic hash from everything that determines a run.

        Args:
            config_dict: The config mapping (raw YAML or materialized config).
            provider: The provider that will actually produce the data. Pass
                the *effective* provider -- if a silent fallback to synthetic
                happened, hashing the requested provider would make the run
                read the requested provider's cached output. Prefer
                :meth:`make_data_cache_key`, which also covers the wrapper
                chain, symbol set and adjustment basis.
            overrides: Explicit overrides applied on top of the YAML (CLI
                flags, environment-derived settings). Their presence and values
                both matter, so ``{"use_baostock": False}`` and no overrides at
                all produce different keys only if they change behaviour.
            schema_version: Data-layer schema version; defaults to
                ``DATA_SCHEMA_VERSION``.
            data_identity: Optional provider/selection identity (D12).

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
            "data_identity": data_identity,
        }
        config_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    @staticmethod
    def make_data_cache_key(
        config: Any,
        provider_identity: dict[str, Any],
        *,
        symbols: list[str] | tuple[str, ...] | set[str] | None = None,
        overrides: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        schema_version: int | str | None = None,
    ) -> str:
        """Data-stage cache key computed *after* provider resolution (D12).

        Inputs: the materialized config (schema defaults included), the
        effective provider chain (including a ``ValidatedProvider`` wrapper
        and its secondary source), the sampled symbol set hash, the date
        range, explicit overrides, and ``DATA_SCHEMA_VERSION``.
        """
        symbol_hash = PipelineCache.hash_symbols(symbols)
        payload = {
            "key_version": CACHE_KEY_VERSION,
            "config": materialize_config(config),
            "effective_provider": provider_identity,
            "symbol_hash": symbol_hash,
            "symbol_count": None if symbols is None else len(set(symbols)),
            "start_date": start_date,
            "end_date": end_date,
            "overrides": overrides or {},
            "schema_version": (
                DATA_SCHEMA_VERSION if schema_version is None else schema_version
            ),
        }
        return hashlib.md5(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load_stage(self, stage: str, cache_key: str) -> Any | None:
        """Load a cached stage. Returns None if not found/stale/corrupt."""
        data, _manifest = self.load_stage_with_manifest(stage, cache_key)
        return data

    def load_stage_with_manifest(
        self, stage: str, cache_key: str
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Load a stage and its provenance manifest (if one was recorded)."""
        path = self._make_path(stage, cache_key)
        if not path.exists():
            return None, None
        try:
            stat_before = path.stat()
        except OSError:
            return None, None
        try:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            meta = self._read_meta_strict(path)
            if meta is not None:
                stored_schema = meta.get("schema")
                if stored_schema is not None:
                    try:
                        stored_schema = int(stored_schema)
                    except (TypeError, ValueError):
                        raise CacheValidationError(
                            f"malformed schema {meta.get('schema')!r}"
                        ) from None
                    if stored_schema != DATA_SCHEMA_VERSION:
                        raise CacheValidationError(
                            f"schema {stored_schema} != {DATA_SCHEMA_VERSION}"
                        )
                stored_stage = meta.get("stage")
                if stored_stage is not None and str(stored_stage) != str(stage):
                    raise CacheValidationError(
                        f"stage {stored_stage!r} != {stage!r}"
                    )
            manifest = (meta or {}).get("manifest")
            if not isinstance(manifest, dict):
                manifest = None
            logger.info("Cache hit: %s (%s)", stage, cache_key)
            return data, manifest
        except CacheValidationError as exc:
            logger.warning(
                "Cache entry %s failed validation (%s); evicting", path, exc,
            )
            self._remove_if_unchanged(path, stat_before)
        except (pickle.UnpicklingError, EOFError) as exc:
            logger.warning("Cache entry %s is corrupt (%s); evicting", path, exc)
            self._remove_if_unchanged(path, stat_before)
        except Exception as exc:  # noqa: BLE001 - transient errors keep the file
            # MemoryError / I/O errors are not evidence of corruption: unlink
            # here would destroy a valid entry that a later run could use.
            logger.warning(
                "Failed to read cache %s (keeping file): %s", path, exc,
            )
        return None, None

    def save_stage(
        self,
        stage: str,
        cache_key: str,
        data: Any,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> bool:
        """Save a stage to cache atomically. Returns True on success.

        A unique temp file is written (and fsynced) outside the lock, then
        published with ``os.replace`` under the cross-process lock. On any
        failure the previous entry is left untouched, the failure is logged at
        WARNING and counted in ``save_failures``.
        """
        path = self._make_path(stage, cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:  # noqa: BLE001 - save failures never raise
            self._record_save_failure(stage, exc)
            self._discard_tmp(tmp)
            return False

        try:
            with self._lock() as locked:
                if not locked:
                    self._record_save_failure(
                        stage,
                        TimeoutError(
                            f"timed out acquiring {self._lock_path} after "
                            f"{self.lock_timeout:g}s"
                        ),
                    )
                    self._discard_tmp(tmp)
                    return False
                self._replace_with_retries(tmp, path)
                self._write_meta(path, stage, manifest)
            logger.info("Cached: %s (%s)", stage, cache_key)
            return True
        except Exception as exc:  # noqa: BLE001 - save failures never raise
            self._record_save_failure(stage, exc)
            self._discard_tmp(tmp)
            return False

    def evict_stage(self, stage: str, cache_key: str) -> bool:
        """Remove one entry (used when its manifest no longer matches)."""
        return self._remove_entry(self._make_path(stage, cache_key))

    def _record_save_failure(self, stage: str, exc: BaseException) -> None:
        self.save_failures += 1
        logger.warning(
            "Failed to save cache %s (failure #%d): %s",
            stage, self.save_failures, exc,
        )

    @staticmethod
    def _discard_tmp(tmp: Path) -> None:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    def _replace_with_retries(
        self, tmp: Path, path: Path, attempts: int = 5
    ) -> None:
        """``os.replace`` with retries for Windows sharing violations."""
        last_exc: OSError | None = None
        for attempt in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:  # another process holds the file open
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(0.05 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Locking
    # ------------------------------------------------------------------

    def _lock(self) -> _CrossProcessLock:
        return _CrossProcessLock(
            self._lock_path, timeout=self.lock_timeout,
        )

    # ------------------------------------------------------------------
    # Retention / maintenance
    # ------------------------------------------------------------------

    def _clean_stale_tmp(self, max_age_seconds: float = STALE_TMP_SECONDS) -> int:
        """Remove ``*.tmp.*`` leftovers from writers that crashed (D36)."""
        now = time.time()
        removed = 0
        for tmp in self.cache_dir.glob("*.tmp.*"):
            try:
                if now - tmp.stat().st_mtime > max_age_seconds:
                    tmp.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info("Removed %d stale cache temp file(s)", removed)
        return removed

    def clear(self) -> int:
        """Clear all cached files (and metadata). Returns files removed."""
        count = 0
        for p in list(self.cache_dir.glob("*.pkl")):
            try:
                stat_before = p.stat()
            except OSError:
                continue
            # Re-stat under the lock so a concurrent writer's fresh entry is
            # not deleted (D36).
            if self._remove_if_unchanged(p, stat_before):
                count += 1
        # Only drop metadata whose payload no longer exists: removing the
        # sidecar of an entry whose .pkl survived would make it load
        # unvalidated.
        for meta in self.cache_dir.glob("*.pkl.meta.json"):
            data_path = meta.with_name(meta.name.removesuffix(".meta.json"))
            if not data_path.exists():
                try:
                    meta.unlink()
                except OSError:
                    pass
        self._clean_stale_tmp()
        logger.info("Cleared %d cache files", count)
        return count

    def list_cached(self) -> list[str]:
        """List cached stages."""
        return sorted(p.stem for p in self.cache_dir.glob("*.pkl"))

    def prune(self) -> int:
        """Enforce schema-version, TTL and LRU limits. Returns files removed.

        Files without a metadata sidecar are pre-retention entries whose key
        can no longer be reproduced after a schema/key bump; they are aged by
        their mtime against the same TTL and count limits.
        """
        self._clean_stale_tmp()
        now = time.time()
        removed = 0
        survivors: list[tuple[os.stat_result, Path]] = []

        for path in self.cache_dir.glob("*.pkl"):
            meta = self._read_meta(path)
            try:
                stat_before = path.stat()
            except OSError:
                continue

            if meta is not None and meta.get("schema") != DATA_SCHEMA_VERSION:
                if self._remove_if_unchanged(path, stat_before):
                    removed += 1
                continue

            age_days = (now - stat_before.st_mtime) / 86400.0
            if self.max_age_days is not None and age_days > self.max_age_days:
                if self._remove_if_unchanged(path, stat_before):
                    removed += 1
                continue

            survivors.append((stat_before, path))

        if self.max_entries is not None and self.max_entries >= 0:
            survivors.sort(key=lambda item: item[0].st_mtime, reverse=True)
            for stat_before, path in survivors[self.max_entries:]:
                if self._remove_if_unchanged(path, stat_before):
                    removed += 1

        if removed:
            logger.info(
                "Cache prune removed %d entr%s (schema=%d, max_age=%sd, max_entries=%s)",
                removed, "y" if removed == 1 else "ies",
                DATA_SCHEMA_VERSION, self.max_age_days, self.max_entries,
            )
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_if_unchanged(self, path: Path, stat_before: os.stat_result) -> bool:
        """Evict ``path`` only if no concurrent writer replaced it meanwhile."""
        with self._lock() as locked:
            if not locked:
                logger.warning(
                    "Could not lock cache; leaving %s in place", path,
                )
                return False
            try:
                stat_now = path.stat()
            except OSError:
                return False
            unchanged = (
                stat_now.st_mtime_ns == stat_before.st_mtime_ns
                and stat_now.st_size == stat_before.st_size
                and stat_now.st_ino == stat_before.st_ino
            )
            if not unchanged:
                logger.info(
                    "Cache entry %s changed while reading; not evicting", path,
                )
                return False
            return self._remove_entry_unlocked(path)

    def _remove_entry(self, path: Path) -> bool:
        try:
            with self._lock() as locked:
                if not locked:
                    logger.warning(
                        "Could not lock cache; leaving %s in place", path,
                    )
                    return False
                return self._remove_entry_unlocked(path)
        except Exception as exc:  # noqa: BLE001 - eviction must never raise
            logger.warning("Failed to remove cache entry %s: %s", path, exc)
            return False

    @staticmethod
    def _remove_entry_unlocked(path: Path) -> bool:
        removed = False
        try:
            path.unlink(missing_ok=True)
            removed = True
        except OSError:
            pass
        try:
            PipelineCache._meta_path(path).unlink(missing_ok=True)
        except OSError:
            pass
        return removed

    def _write_meta(
        self, path: Path, stage: str, manifest: dict[str, Any] | None = None,
    ) -> None:
        meta: dict[str, Any] = {
            "schema": DATA_SCHEMA_VERSION,
            "stage": stage,
            "created": time.time(),
            "cache_key_version": CACHE_KEY_VERSION,
        }
        if manifest:
            meta["manifest"] = manifest
        meta_path = self._meta_path(path)
        tmp = meta_path.with_name(
            f"{meta_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, meta_path)
        except Exception as exc:
            self._discard_tmp(tmp)
            raise exc

    def _read_meta(self, path: Path) -> dict | None:
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _read_meta_strict(self, path: Path) -> dict | None:
        """Read the sidecar for a load; an unparseable sidecar is corruption.

        A missing sidecar remains a legacy entry (returns None). An I/O error
        propagates as a transient failure so the caller keeps the file (D34);
        a schema/manifest that cannot be parsed is a validation failure.
        """
        meta_path = self._meta_path(path)
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise CacheValidationError(
                f"unreadable metadata sidecar: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise CacheValidationError(
                "metadata sidecar is not a JSON object"
            )
        return data
