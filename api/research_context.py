"""Bounded, TTL-cached research context shared by analysis/monitor endpoints.

The analysis endpoints that used to answer ``available: false`` did so because
a stored run keeps only ``chart_data`` (equity curve + IC summaries), not the
factor panel or per-period returns.  Fabricating the missing numbers was the
original defect; the honest alternative is to recompute them from the same
inputs ``python main.py run`` uses.

This module builds that input set once -- data, processed factors, the alpha
signal -- through the canonical ``_load_data`` / ``_compute_factors`` /
``_generate_signal`` path in :mod:`quant_platform.main`, so PIT masks, price
basis and provenance all come from the run pipeline rather than being
re-derived here.

Cost control:

* The full pipeline is CPU-heavy.  ``get_research_context`` runs it on the
  API's analytics executor through ``routes._run_analytics``, which rejects
  requests with ``429`` + ``Retry-After`` when its two workers are busy.
* Built contexts are cached for ``CONTEXT_TTL_SECONDS`` (default 300 s),
  keyed by the materialized config, effective provider identity, universe
  settings and date range.  At most ``CONTEXT_MAX_ENTRIES`` contexts are kept.
* Concurrent requests for the same key are single-flighted with a per-key
  lock: the second caller waits, then re-reads the cache instead of running
  the pipeline twice.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.main import (
    _compute_factors,
    _generate_signal,
    _load_data,
    _resolve_provider,
)
from quant_platform.utils.cache import PipelineCache, materialize_config
from quant_platform.utils.config import load_config
from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

#: How long a built context may be served without recomputation.
CONTEXT_TTL_SECONDS = 300.0
#: Maximum number of cached contexts (the task book asks for 1-2).
CONTEXT_MAX_ENTRIES = 2
#: Maximum number of per-key single-flight locks kept around.
_KEY_LOCK_ENTRIES = 8
#: Trailing sessions used by the factor-statistics endpoints (IC decay,
#: correlation, factor status).  A full five-year panel made these endpoints
#: take minutes on the default 500-name universe; a stated trailing year is
#: the standard monitoring horizon and is reported back to the client as
#: ``window_sessions``/``panel_start``.
ANALYSIS_WINDOW_SESSIONS = 252

#: Bounds for request-supplied date overrides. Without them a single query
#: (``start_date=0001-01-01&end_date=9999-12-31``) built a multi-million-row
#: panel before any cap and could OOM the process.
_MIN_WINDOW_DATE = date(2000, 1, 1)
_MAX_WINDOW_DATE = date(2100, 12, 31)
_MAX_WINDOW_DAYS = 3653  # 10 years


@dataclass(frozen=True)
class ResearchContext:
    """Everything the analysis endpoints need, computed by the run pipeline."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark: Any
    metadata: Any
    financials: Any
    turnover: pd.DataFrame | None
    processed_factors: dict[str, pd.DataFrame]
    ic_results: dict[str, dict]
    sector_map: dict
    fin_unstacked: pd.DataFrame | None
    signal: pd.DataFrame
    config: Any
    provider: str
    synthetic: bool
    config_hash: str
    as_of: str
    computed_at: str
    start_date: str
    end_date: str
    cached: bool = False

    @property
    def n_sessions(self) -> int:
        try:
            return int(len(self.prices.index))
        except Exception:  # defensive: provenance must not break a response
            return 0

    @property
    def n_assets(self) -> int:
        try:
            return int(len(self.prices.columns))
        except Exception:
            return 0

    def provenance(self) -> dict[str, Any]:
        """Fields every wired endpoint echoes so the UI can label its data."""
        return {
            "as_of": self.as_of,
            "provider": self.provider,
            "synthetic": bool(self.synthetic),
            "config_hash": self.config_hash,
            "computed_at": self.computed_at,
            "cached": bool(self.cached),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_sessions": self.n_sessions,
            "n_assets": self.n_assets,
        }


# ── TTL cache ──

_cache: OrderedDict[str, tuple[float, ResearchContext]] = OrderedDict()
_cache_lock = threading.Lock()
_key_locks: OrderedDict[str, threading.Lock] = OrderedDict()

# Derived endpoint results (IC decay, walk-forward folds, ...) are as
# expensive as the context itself, so they share the same TTL.  Bounded and
# keyed by the context hash plus the endpoint's own parameters.
_derived_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
_DERIVED_MAX_ENTRIES = 8

# Provider resolution can be expensive (a Postgres provider opens a
# connection, a validated setup probes a secondary source), so the resolved
# provider is memoized per (materialized config, dates, config-file identity).
# The parsed YAML itself is cached until the file changes. Both keep the
# event loop free of provider construction.
_config_cache: dict[tuple, Any] = {}
_config_cache_lock = threading.Lock()
_resolution_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_RESOLUTION_MAX_ENTRIES = 4


def _default_config_identity() -> tuple:
    path = os.environ.get("QUANT_CONFIG") or str(
        Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    )
    try:
        stat = os.stat(path)
        file_key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        file_key = (0, 0)
    return (path, file_key, os.environ.get("QUANT_DATA_PROVIDER", ""))


def _load_base_config(identity: tuple) -> Any:
    with _config_cache_lock:
        cached = _config_cache.get(identity)
        if cached is not None:
            return cached
    config = load_config()
    with _config_cache_lock:
        _config_cache.clear()
        _config_cache[identity] = config
    return config


def _resolution_get(key: str) -> dict | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _resolution_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if now >= expires_at:
            _resolution_cache.pop(key, None)
            return None
        _resolution_cache.move_to_end(key)
        return value


def _resolution_put(key: str, resolution: dict, context_key: str) -> None:
    with _cache_lock:
        _resolution_cache.pop(key, None)
        _resolution_cache[key] = (
            time.monotonic() + CONTEXT_TTL_SECONDS,
            {"resolution": resolution, "context_key": context_key},
        )
        while len(_resolution_cache) > _RESOLUTION_MAX_ENTRIES:
            _resolution_cache.popitem(last=False)


def derived_key(context: ResearchContext, name: str, params: Any = None) -> str:
    payload = json.dumps(
        [context.config_hash, name, params], sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def derived_cache_get(key: str) -> Any | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _derived_cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if now >= expires_at:
            _derived_cache.pop(key, None)
            return None
        _derived_cache.move_to_end(key)
        return value


def derived_cache_put(key: str, value: Any) -> None:
    with _cache_lock:
        _derived_cache.pop(key, None)
        _derived_cache[key] = (time.monotonic() + CONTEXT_TTL_SECONDS, value)
        while len(_derived_cache) > _DERIVED_MAX_ENTRIES:
            _derived_cache.popitem(last=False)


def _cache_get(key: str) -> ResearchContext | None:
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, context = entry
        if now >= expires_at:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return context


def _cache_put(key: str, context: ResearchContext) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _cache[key] = (time.monotonic() + CONTEXT_TTL_SECONDS, context)
        while len(_cache) > CONTEXT_MAX_ENTRIES:
            _cache.popitem(last=False)


def _key_lock_for(key: str) -> threading.Lock:
    """Per-key single-flight lock, bounded like the context cache."""
    with _cache_lock:
        lock = _key_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _key_locks[key] = lock
        _key_locks.move_to_end(key)
        # Evicting an in-use lock is safe: the holder/waiters keep their
        # reference; a later request simply gets a fresh lock.  Worst case is
        # one duplicate load, still bounded by the analytics semaphore.
        while len(_key_locks) > _KEY_LOCK_ENTRIES:
            _key_locks.popitem(last=False)
        return lock


def reset_context_cache() -> None:
    """Drop every cached context, resolution and lock (tests / config reload)."""
    with _cache_lock:
        _cache.clear()
        _key_locks.clear()
        _derived_cache.clear()
        _resolution_cache.clear()
    with _config_cache_lock:
        _config_cache.clear()


# ── Config / key ──


def parse_window(
    start_date: str | None, end_date: str | None,
) -> tuple[str | None, str | None]:
    """Validate optional ISO date overrides from a request.

    Raises ``ValueError`` with a client-safe message; the endpoints translate
    it to HTTP 422. Dates are bounded to 2000-01-01..2100-12-31 and a window
    to at most 10 years: ``?start_date=0001-01-01&end_date=9999-12-31`` used
    to build a multi-million-row panel before any cap (OOM).
    """
    parsed: list[str | None] = []
    for label, value in (("start_date", start_date), ("end_date", end_date)):
        if value is None or value == "":
            parsed.append(None)
            continue
        try:
            parsed_date = date.fromisoformat(str(value))
        except ValueError:
            raise ValueError(f"{label} must be an ISO date (YYYY-MM-DD)") from None
        if not _MIN_WINDOW_DATE <= parsed_date <= _MAX_WINDOW_DATE:
            raise ValueError(
                f"{label} must be between {_MIN_WINDOW_DATE.isoformat()} and "
                f"{_MAX_WINDOW_DATE.isoformat()}"
            )
        parsed.append(parsed_date.isoformat())
    start, end = parsed
    # One-sided overrides are resolved against the config they will be merged
    # into before the span check: `?end_date=2100-12-31` alone used to pass
    # validation and then expand the default 2021 start into a 29k-day window.
    if start is not None or end is not None:
        effective_start, effective_end = start, end
        if effective_start is None or effective_end is None:
            try:
                base = _load_base_config(_default_config_identity())
                effective_start = effective_start or str(base.data.start_date)
                effective_end = effective_end or str(base.data.end_date)
            except Exception:
                effective_start, effective_end = start, end
        if effective_start is not None and effective_end is not None:
            try:
                start_day = date.fromisoformat(str(effective_start)[:10])
                end_day = date.fromisoformat(str(effective_end)[:10])
            except ValueError:
                raise ValueError(
                    "effective data window is not an ISO date pair",
                ) from None
            if start_day >= end_day:
                raise ValueError("start_date must be earlier than end_date")
            if (end_day - start_day).days > _MAX_WINDOW_DAYS:
                raise ValueError("date span must be at most 10 years")
    return start, end


def _prepare_config(
    *,
    config: Any | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[Any, tuple]:
    """Load a private config copy for this request and its file identity.

    The parsed YAML is reused until the config file changes; each request
    still gets its own copy because the date overrides are applied in place.
    """
    if config is not None:
        cfg = copy.deepcopy(config)
        identity: tuple = ("caller",)
    else:
        identity = _default_config_identity()
        cfg = copy.deepcopy(_load_base_config(identity))
    if start_date is not None:
        cfg.data.start_date = start_date
    if end_date is not None:
        cfg.data.end_date = end_date
    return cfg, identity


def _resolution_memo_key(config: Any, identity: tuple) -> str:
    payload = json.dumps(
        [materialize_config(config), identity], sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def make_context_key(config: Any, resolution: dict) -> str:
    """Cache identity: materialized config, provider identity, universe, dates."""
    return PipelineCache.make_data_cache_key(
        config=config,
        provider_identity=resolution.get("identity") or {},
        symbols=None,
        overrides={"stage": "research_context"},
        start_date=str(config.data.start_date),
        end_date=str(config.data.end_date),
    )


# ── Build ──


def _load_context_uncached(config: Any, resolution: dict, key: str) -> ResearchContext:
    """Run data -> factors -> signal. Always called on an analytics worker."""
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, resolution=resolution,
    )
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config,
    )
    signal = _generate_signal(
        config, processed_factors, returns, prices=prices, volume=turnover,
    )

    identity = resolution.get("identity") or {}
    as_of = ""
    if prices is not None and len(prices.index) > 0:
        as_of = str(prices.index[-1])[:10]

    logger.info(
        "Research context built: key=%s provider=%s sessions=%d assets=%d "
        "factors=%d as_of=%s",
        key,
        resolution.get("effective_provider") or resolution.get("provider_name"),
        0 if prices is None else len(prices.index),
        0 if prices is None else len(prices.columns),
        len(processed_factors),
        as_of,
    )

    return ResearchContext(
        prices=prices,
        returns=returns,
        benchmark=benchmark,
        metadata=metadata,
        financials=financials,
        turnover=turnover,
        processed_factors=processed_factors,
        ic_results=ic_results,
        sector_map=sector_map,
        fin_unstacked=fin_unstacked,
        signal=signal,
        config=config,
        provider=str(
            resolution.get("effective_provider")
            or resolution.get("provider_name")
            or ""
        ),
        synthetic=bool(identity.get("synthetic")),
        config_hash=key,
        as_of=as_of,
        computed_at=datetime.now(UTC).isoformat(),
        start_date=str(config.data.start_date),
        end_date=str(config.data.end_date),
    )


def _load_and_cache(config: Any, resolution: dict, key: str) -> ResearchContext:
    """Single-flight load for one key; re-checks the cache under the key lock."""
    with _key_lock_for(key):
        cached = _cache_get(key)
        if cached is not None:
            return replace(cached, cached=True)
        context = _load_context_uncached(config, resolution, key)
        _cache_put(key, context)
        return context


def _resolve_and_load(config: Any, memo_key: str) -> ResearchContext:
    """Resolve the provider, compute the context key and load (worker only)."""
    resolution = _resolve_provider(config, with_universe=False)
    key = make_context_key(config, resolution)
    _resolution_put(memo_key, resolution, key)
    cached = _cache_get(key)
    if cached is not None:
        return replace(cached, cached=True)
    return _load_and_cache(config, resolution, key)


#: One in-flight build per key. A dashboard that opens correlation and
#: ic-decay (or a reload while the first build runs) must not spend both
#: analytics slots on the same context: followers await the leader's future
#: without occupying a worker.
_inflight_lock = threading.Lock()
_inflight: dict[str, asyncio.Future] = {}


async def _single_flight(key: str, factory):
    loop = asyncio.get_running_loop()
    with _inflight_lock:
        future = _inflight.get(key)
        leader = future is None
        if leader:
            future = loop.create_future()
            _inflight[key] = future
    if not leader:
        return await asyncio.shield(future)
    try:
        result = await factory()
    except BaseException as exc:
        if not future.done():
            future.set_exception(exc)
        raise
    else:
        if not future.done():
            future.set_result(result)
        return result
    finally:
        with _inflight_lock:
            if _inflight.get(key) is future:
                _inflight.pop(key, None)


async def get_research_context(
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    config: Any | None = None,
) -> ResearchContext:
    """Return the (possibly cached) research context for the configured data.

    Runs the build on the shared analytics executor, so the event loop is
    never blocked and saturation surfaces as ``429`` with ``Retry-After``.
    Provider resolution (which can open a database connection or probe a
    network source) also runs on a worker, only when the resolution memo is
    missing or expired; cache hits stay cheap.
    """
    start, end = parse_window(start_date, end_date)
    cfg, identity = _prepare_config(
        config=config, start_date=start, end_date=end,
    )
    memo_key = _resolution_memo_key(cfg, identity)

    memo = _resolution_get(memo_key)
    if memo is not None:
        context_key = memo["context_key"]
        cached = _cache_get(context_key)
        if cached is not None:
            return replace(cached, cached=True)
        # The resolution is still valid but the context expired: rebuild with
        # the memoized provider on a worker.
        from quant_platform.api import routes as routes_module

        return await _single_flight(
            f"ctx:{context_key}",
            lambda: routes_module._run_analytics(
                _load_and_cache, cfg, memo["resolution"], context_key,
            ),
        )

    # Imported lazily: routes imports this module at startup.
    from quant_platform.api import routes as routes_module

    return await _single_flight(
        f"memo:{memo_key}",
        lambda: routes_module._run_analytics(_resolve_and_load, cfg, memo_key),
    )
