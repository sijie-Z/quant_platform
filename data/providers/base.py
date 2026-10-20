"""Abstract base class for data providers.

Defines the contract that all data providers must implement, enabling
swappable backends (synthetic, akshare, tushare, custom CSV, etc.).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)

# Sidecar manifest appended to every versioned provider cache file. It records
# the data-schema generation, the cache-format version and the adjustment
# convention, so a cache written under old semantics is rejected (and
# refetched) instead of being replayed as current data.
CACHE_MANIFEST_SUFFIX = ".manifest.json"

# Sidecar recording the outcome of the most recent *incremental attempt*.
# A partial response never touches the data file or its version manifest (D32
# completion-marker rule: the marker only advances after a complete publish),
# but the failed attempt must still be visible to provenance and operators.
INCREMENTAL_STATUS_SUFFIX = ".incremental.json"

#: Sessions re-fetched before the first new session on every incremental
#: update. One overlap session is enough to detect a provider that revised
#: the boundary row; the merge is last-wins on ``(date, asset)``.
INCREMENTAL_OVERLAP_SESSIONS = 1


def schema_version() -> int:
    """Current data-layer schema generation (single source of truth)."""
    from quant_platform.utils.cache import DATA_SCHEMA_VERSION

    return int(DATA_SCHEMA_VERSION)


def cache_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}{CACHE_MANIFEST_SUFFIX}")


def write_cache_manifest(
    path: str | Path,
    *,
    provider: str,
    adjustment: str = "n/a",
    cache_version: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the version manifest for a provider cache file atomically."""
    manifest: dict[str, Any] = {
        "provider": provider,
        "schema_version": schema_version(),
        "adjustment": adjustment,
        "cache_version": str(cache_version),
        "created_at": time.time(),
    }
    if extra:
        manifest.update(extra)
    target = cache_manifest_path(path)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return manifest


def read_cache_manifest(path: str | Path) -> dict[str, Any] | None:
    target = cache_manifest_path(path)
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def cache_manifest_matches(
    path: str | Path,
    *,
    adjustment: str | None = None,
    cache_version: str | None = None,
    key: str | None = None,
) -> bool:
    """Whether ``path`` carries a manifest valid for the current semantics.

    A missing manifest means the cache predates versioning (or was written by
    an older release): it is *not* trusted. The data file itself is left in
    place; callers decide whether to delete/refetch.
    """
    manifest = read_cache_manifest(path)
    if manifest is None:
        return False
    try:
        stored_schema = int(manifest.get("schema_version", -1))
    except (TypeError, ValueError):
        return False
    if stored_schema != schema_version():
        return False
    if cache_version is not None and str(manifest.get("cache_version", "")) != str(
        cache_version
    ):
        return False
    if key is not None and str(manifest.get("key", "")) != str(key):
        return False
    return adjustment is None or str(manifest.get("adjustment", "")) == str(
        adjustment
    )


def exchange_sessions(start, end) -> pd.DatetimeIndex:
    """Exchange sessions in ``[start, end]`` (never calendar/business days)."""
    from quant_platform.utils.exchange_calendar import trading_days

    return trading_days(start, end)


def incremental_fetch_start(
    last_complete_date,
    requested_end,
    *,
    overlap: int = INCREMENTAL_OVERLAP_SESSIONS,
) -> pd.Timestamp | None:
    """First session of the incremental fetch window.

    The window starts at ``last_complete_date + 1 exchange session`` but is
    extended backwards by ``overlap`` sessions so a provider that revised the
    boundary rows is re-read. Returns ``None`` when the marker already covers
    the requested end (nothing to fetch).

    Calendar arithmetic is never used: ``last + 1 day`` can be a weekend or an
    exchange holiday, and treating it as a session is the classic off-by-one
    that either skips a real session or forges a non-session row.
    """
    last = pd.Timestamp(last_complete_date).normalize()
    end = pd.Timestamp(requested_end).normalize()
    if end <= last:
        return None
    overlap = max(0, int(overlap))
    # Enough lookback to always contain `overlap` sessions before `last`.
    lookback = pd.Timedelta(days=max(21, 4 * overlap + 7))
    sessions = exchange_sessions(last - lookback, end)
    position = int(sessions.searchsorted(last, side="right"))
    if position >= len(sessions):
        return None
    return sessions[max(0, position - overlap)]


def sessions_after(last_date, until_date) -> int:
    """Number of exchange sessions strictly after ``last_date`` up to ``until``.

    This is the unit of ``data_age_days``: a marker on Friday and a requested
    end on the following Tuesday is 2 exchange sessions old (Monday, Tuesday),
    not 4 calendar days. Returns 0 when ``until`` is not later than
    ``last_date`` (a panel can cover a future requested end).
    """
    last = pd.Timestamp(last_date).normalize()
    until = pd.Timestamp(until_date).normalize()
    if until <= last:
        return 0
    return int(len(exchange_sessions(last + pd.Timedelta(days=1), until)))


def expected_data_end(requested_end, today=None) -> pd.Timestamp:
    """The last session a panel can be expected to contain right now.

    ``min(requested_end, today)``: a historical backtest is not stale because
    wall-clock time moved on, while a daily refresh with an end date in the
    future can only be as complete as today's session.
    """
    end = pd.Timestamp(requested_end).normalize()
    current = pd.Timestamp(today).normalize() if today is not None else (
        pd.Timestamp.now().normalize()
    )
    return min(end, current)


def panel_last_date(prices: pd.DataFrame | None) -> pd.Timestamp | None:
    """Last date of a ``(date, asset)`` panel (None when empty)."""
    if prices is None or len(prices) == 0:
        return None
    if isinstance(prices.index, pd.MultiIndex):
        return pd.Timestamp(prices.index.get_level_values("date").max())
    return pd.Timestamp(prices.index.max())


def canonicalize_panel_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a ``(date, asset)`` MultiIndex to canonical levels.

    Two panels with identical row values can still differ in *level order*
    (pandas keeps factorisation order from ``set_index``/``concat``), and
    ``unstack`` turns that order into the column order of the pipeline
    panels. Rebuilding the levels sorted -- and dropping inferred
    ``freq=<BusinessDay>`` metadata that ``pd.concat`` adds when dates happen
    to form a contiguous business-day range -- makes an incremental panel
    byte-identical to a full fetch over the same range.
    """
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        return df
    if list(df.index.names) != ["date", "asset"]:
        return df
    dates = df.index.get_level_values("date")
    assets = df.index.get_level_values("asset")
    date_codes, date_levels = pd.factorize(dates, sort=True)
    asset_codes, asset_levels = pd.factorize(assets, sort=True)
    df = df.copy(deep=False)
    df.index = pd.MultiIndex(
        levels=[
            pd.DatetimeIndex(date_levels, name="date"),
            pd.Index(asset_levels, name="asset"),
        ],
        codes=[date_codes, asset_codes],
        names=["date", "asset"],
        verify_integrity=False,
    )
    return df


def deduplicate_panel(
    df: pd.DataFrame,
    dataset: str = "panel",
    *,
    keep: str = "last",
) -> pd.DataFrame:
    """Drop duplicate ``(date, asset)`` rows and assert none remain.

    Incremental appends overlap the cached range by at least one session, so
    the merged frame is expected to contain duplicates. ``keep="last"`` is the
    documented policy: freshly fetched rows supersede cached ones on the
    boundary. The post-condition is part of the contract -- a provider that
    still returns duplicates would otherwise fail later inside ``unstack``.
    """
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        return df
    duplicated = df.index.duplicated(keep=keep)
    count = int(duplicated.sum())
    if count:
        logger.warning(
            "%s: dropped %d duplicate (date, asset) row(s) while appending "
            "(keep=%r)", dataset, count, keep,
        )
        df = df[~duplicated]
    if df.index.duplicated(keep=keep).any():
        raise ValueError(
            f"{dataset}: duplicate (date, asset) rows remain after "
            f"deduplication (keep={keep!r})"
        )
    return canonicalize_panel_index(df)


def symbols_on_date(frame: pd.DataFrame, date) -> set[str]:
    """Asset labels present on one date of a ``(date, asset)`` panel."""
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return set()
    target = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(
        frame.index.get_level_values("date")
    ).normalize()
    if not (dates == target).any():
        return set()
    return {str(asset) for asset in frame.index[dates == target].get_level_values("asset")}


def last_day_missing_symbols(
    frame: pd.DataFrame,
    *,
    delisted: set[str] | None = None,
) -> set[str]:
    """Symbols present on the previous session but missing on the last one.

    A partial provider response can truncate the last day of the window
    without failing a batch outright (the symbols appeared earlier in the
    window). The completion marker must not advance past such a day. Symbols
    whose provider metadata says they were delisted are exempt.
    """
    if frame is None or frame.empty or not isinstance(frame.index, pd.MultiIndex):
        return set()
    dates = pd.DatetimeIndex(
        frame.index.get_level_values("date").unique()
    ).sort_values()
    if len(dates) < 2:
        return set()
    previous = symbols_on_date(frame, dates[-2])
    current = symbols_on_date(frame, dates[-1])
    exempt = {str(code) for code in (delisted or set())}
    return previous - current - exempt


def complete_panel_marker(
    frame: pd.DataFrame,
    *,
    delisted: set[str] | None = None,
) -> tuple[pd.Timestamp | None, bool]:
    """Completion marker for a freshly fetched panel.

    The last row alone is not proof of completeness: a provider can return a
    truncated final cross-section. When the last session is missing symbols
    present on the previous one (delisted names exempt), the marker is held
    back to the previous session and ``incomplete`` is reported. Without a
    suspension feed a legitimately suspended name looks the same as a
    truncation, so the conservative reading wins -- the next append re-reads
    the boundary and advances once the symbol reappears.
    """
    last = panel_last_date(frame)
    if last is None:
        return None, False
    missing = last_day_missing_symbols(frame, delisted=delisted)
    if not missing:
        return last, False
    dates = pd.DatetimeIndex(
        frame.index.get_level_values("date").unique()
    ).sort_values()
    return (dates[-2] if len(dates) >= 2 else None), True


def incremental_marker_extra(
    *,
    last_complete_date,
    requested_end,
    previous_last_complete_date=None,
    new_sessions: int = 0,
    incomplete: bool = False,
    source: str = "full_fetch",
) -> dict[str, Any]:
    """Manifest fields of the D32 completion marker (JSON-friendly)."""
    marker = (
        pd.Timestamp(last_complete_date).normalize()
        if last_complete_date is not None else None
    )
    end = pd.Timestamp(requested_end).normalize()
    age = None
    if marker is not None:
        age = sessions_after(marker, expected_data_end(end))
    previous = (
        pd.Timestamp(previous_last_complete_date).normalize()
        if previous_last_complete_date is not None else None
    )
    return {
        "last_complete_date": marker.date().isoformat() if marker is not None else None,
        "previous_last_complete_date": (
            previous.date().isoformat() if previous is not None else None
        ),
        "requested_end": end.date().isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "data_age_days": age,
        "incomplete": bool(incomplete),
        "new_sessions": int(new_sessions),
        "incremental_source": str(source),
    }


def incremental_status_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.name}{INCREMENTAL_STATUS_SUFFIX}")


def write_incremental_status(path: str | Path, status: dict[str, Any]) -> None:
    """Atomically record the outcome of an incremental attempt.

    This sidecar is deliberately separate from the data file and its version
    manifest: an incomplete attempt must never modify (or advance) either.
    """
    target = incremental_status_path(path)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(status, handle, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("Could not write incremental status %s: %s", target, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def read_incremental_status(path: str | Path) -> dict[str, Any] | None:
    target = incremental_status_path(path)
    try:
        with open(target, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def provider_cache_lock(directory: str | Path, *, timeout: float | None = None):
    """Cross-process lock serialising provider-cache publication.

    Reuses the pipeline cache's ``_CrossProcessLock`` (``msvcrt.locking`` on
    Windows, ``fcntl.flock`` on POSIX) so an append and a pipeline-cache save
    share the same well-tested primitives. ``__enter__`` returns ``True`` when
    the lock was acquired and ``False`` on timeout; it never raises for
    contention.
    """
    from quant_platform.utils.cache import DEFAULT_LOCK_TIMEOUT, _CrossProcessLock

    return _CrossProcessLock(
        Path(directory) / ".provider_cache.lock",
        timeout=(
            DEFAULT_LOCK_TIMEOUT if timeout is None else float(timeout)
        ),
    )


def _provider_label(provider_name: str) -> str:
    labels = {
        "SyntheticDataProvider": "synthetic",
        "TushareProvider": "tushare",
        "BaostockDataProvider": "baostock",
        "PostgresDataProvider": "postgres",
        "AdataProvider": "adata",
        "ValidatedProvider": "validated",
    }
    return labels.get(provider_name, provider_name.lower())


# Provider-specific adjustment conventions when the provider does not declare
# one on the instance.
_ADJUSTMENT_BY_CLASS = {
    "SyntheticDataProvider": "synthetic_adj_factor",
    "TushareProvider": "hfq",
    "BaostockDataProvider": "hfq",
}


def provider_identity(provider: DataProvider) -> dict[str, Any]:
    """Stable description of the *effective* provider chain.

    For a :class:`ValidatedProvider` wrapper the dict nests one entry per
    source, so adding/removing the secondary source (or swapping it) changes
    the pipeline cache key. Providers can override ``cache_identity()`` to add
    generation parameters; the base implementation falls back to conventional
    attributes (n_stocks/seed/index_code/...), which pins the synthetic
    generator's inputs without editing that module.
    """
    providers = getattr(provider, "providers", None)
    if isinstance(providers, dict) and providers:
        sources = {
            str(name): provider_identity(sub) for name, sub in providers.items()
        }
        adjustments = {str(src.get("adjustment")) for src in sources.values()}
        return {
            "provider": "validated(" + "+".join(sorted(sources)) + ")",
            "class": f"{type(provider).__module__}.{type(provider).__name__}",
            "wrapper": type(provider).__name__,
            "primary": getattr(provider, "primary", None),
            "sources": sources,
            "adjustment": (
                next(iter(adjustments)) if len(adjustments) == 1 else "mixed"
            ),
            "synthetic": any(bool(src.get("synthetic")) for src in sources.values()),
        }

    identity: dict[str, Any] = {}
    hook = getattr(provider, "cache_identity", None)
    if callable(hook):
        try:
            value = hook()
            if isinstance(value, dict):
                identity.update(value)
        except Exception as exc:  # identity is diagnostic, never fatal
            logger.warning(
                "Provider cache_identity() failed for %s: %s",
                type(provider).__name__, exc,
            )

    class_name = type(provider).__name__
    identity.setdefault("provider", _provider_label(class_name))
    identity.setdefault("class", f"{type(provider).__module__}.{class_name}")
    identity.setdefault(
        "adjustment",
        getattr(
            provider,
            "price_adjustment",
            _ADJUSTMENT_BY_CLASS.get(class_name, "provider-defined"),
        ),
    )
    identity.setdefault("synthetic", _provider_label(class_name) == "synthetic")
    # Conventional generation/selection attributes. ``str`` keeps the payload
    # JSON-serialisable (Timestamps, Paths, numpy scalars).
    for attr in (
        "n_stocks", "seed", "start_date", "end_date", "index_code",
        "embedded_alpha", "alpha_strength", "cache_enabled",
    ):
        if hasattr(provider, attr):
            identity.setdefault(attr, str(getattr(provider, attr)))
    return identity


def sampled_universe(
    provider: DataProvider,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    n_symbols: int | None = None,
) -> list[str] | None:
    """The provider-level symbol sample that determines a run's panel.

    Returns ``None`` when the provider cannot describe its sample cheaply;
    callers must then treat the provider cache identity as incomplete (and
    must not serve a pipeline-cache entry). Providers can implement
    ``cache_universe()`` (Tushare, Baostock do); synthetic's asset list is
    reproduced deterministically from ``n_stocks`` without generating data.
    """
    providers = getattr(provider, "providers", None)
    if isinstance(providers, dict) and providers:
        # A validated wrapper's panel is its primary source's panel (the
        # secondary only validates/fuses); the primary defines the symbols.
        primary_name = getattr(provider, "primary", None)
        primary = providers.get(primary_name) if primary_name else None
        if primary is None:
            primary = next(iter(providers.values()))
        return sampled_universe(
            primary,
            start_date=start_date,
            end_date=end_date,
            n_symbols=n_symbols,
        )

    hook = getattr(provider, "cache_universe", None)
    if callable(hook):
        try:
            codes = hook(
                start_date=start_date, end_date=end_date, n_symbols=n_symbols,
            )
        except Exception as exc:  # universe resolution is best effort
            logger.warning(
                "Could not determine the provider universe for cache identity: %s",
                exc,
            )
            return None
        if codes is None:
            return None
        return sorted({str(code) for code in codes})

    # SyntheticDataProvider assigns the assets during `_generate_all`, but the
    # formula is deterministic in n_stocks. Reproducing it avoids generating
    # the whole panel just to compute a cache key.
    n_stocks = getattr(provider, "n_stocks", None)
    if n_stocks is not None and type(provider).__name__ == "SyntheticDataProvider":
        return [
            f"{i:06d}.SH" if i % 3 != 0 else f"{i:06d}.SZ"
            for i in range(1, int(n_stocks) + 1)
        ]

    assets = getattr(provider, "_assets", None)
    if assets:
        return sorted({str(asset) for asset in assets})
    return None


class DataFetchIncompleteError(RuntimeError):
    """Raised when a provider could not fetch every requested symbol.

    Providers must never return (or cache) a silently truncated panel.  A
    caller that gets this exception knows the result is unusable as a
    complete cross-section and must retry, fall back, or fail closed.
    """


class DataProvider(ABC):
    """Abstract interface for market data providers.

    Each provider must supply:
    - Daily price/volume data (OHLCV) as a multi-indexed DataFrame (date, asset)
    - Financial statement data as a multi-indexed DataFrame (date, asset)
    - Index/benchmark data
    - Asset metadata (sector, market cap, ST status, etc.)
    """

    #: How ``close`` relates to the traded price. Providers that embed the
    #: corporate-action factor in the price level (hfq) or ship a divisor
    #: column must declare it; it is part of every cache identity.
    price_adjustment: str = "provider-defined"

    #: Whether :meth:`update` can append new sessions to an existing cache.
    #: Callers (the CLI) must fall back to a full fetch -- with a warning --
    #: when this is False or when ``update`` raises ``NotImplementedError``.
    #: A provider that does not advertise this must never be treated as
    #: "nothing new": stale data is reported by the freshness gate, not
    #: assumed fresh because an append was skipped.
    supports_incremental: bool = False

    def update(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Append the sessions after the cached completion marker, if any.

        Contract:

        * Return a JSON-friendly status dict with at least ``supported``,
          ``performed``, ``full_refetch``, ``new_sessions``,
          ``last_complete_date``, ``requested_end``, ``incomplete`` and
          ``reason``.
        * Never advance ``last_complete_date`` past data that was not
          completely fetched. A partial response returns ``incomplete=True``
          (or raises :class:`DataFetchIncompleteError`) and leaves the data
          file *and* its manifest untouched.
        * Publish the appended panel atomically (temp file + ``os.replace``
          under :func:`provider_cache_lock`) and write the version manifest
          only after the data write succeeded.
        * The merged panel must be byte-identical to a full fetch over the
          same range: overlap rows win (``keep="last"``), no duplicate
          ``(date, asset)`` rows remain, and corporate-action factors are
          refreshed so adjusted series stay continuous.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support incremental updates; "
            "perform a full fetch instead"
        )

    def data_status(self) -> dict[str, Any]:
        """Completion marker/provenance of the currently loaded data.

        Keys (any may be absent): ``last_complete_date``, ``requested_end``,
        ``updated_at``, ``incomplete`` and ``source``. Providers that do not
        maintain a marker return an empty dict; callers then derive freshness
        from the panel's last date.
        """
        return {}

    def cache_identity(self) -> dict[str, Any]:
        """Describe the inputs that determine this provider's fetched data.

        The pipeline cache key includes this description. Providers with
        generation parameters (seeds, index codes, cache formats) should
        override it; the default labels the class and adjustment convention.
        """
        adjustment = self.price_adjustment
        if adjustment == "provider-defined":
            adjustment = _ADJUSTMENT_BY_CLASS.get(
                type(self).__name__, adjustment,
            )
        return {
            "provider": _provider_label(type(self).__name__),
            "adjustment": adjustment,
        }

    @abstractmethod
    def get_prices(
        self,
        start_date: str,
        end_date: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return daily OHLCV data.

        Returns DataFrame with MultiIndex (date, asset) and columns like
        open, high, low, close, volume, turnover, adj_factor.

        ``fields`` is a column projection with one contract across all
        providers: unknown names raise :class:`ValueError` (never silently
        dropped) and the returned columns appear in the requested order.

        Units are fixed at this boundary: ``volume`` is in shares, ``amount``
        in CNY and ``turnover`` is a fraction (0.025 == 2.5%).  ``close`` is
        the adjusted price series and ``adj_factor`` the divisor the pipeline
        applies to obtain the return-generating price (adjustment applied
        exactly once: ``close_adj = close / adj_factor``).

        If any requested symbol cannot be fetched after bounded retries the
        provider must raise :class:`DataFetchIncompleteError` rather than return a
        truncated panel.
        """
        ...

    @abstractmethod
    def get_financials(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Return quarterly financial statement data.

        Returns DataFrame with MultiIndex (date, asset) and columns like
        market_cap, total_assets, net_assets, revenue, net_profit, roe, pb, pe.
        Data is forward-filled from report dates.
        """
        ...

    @abstractmethod
    def get_benchmark(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """Return benchmark daily returns (e.g., CSI 300).

        Returns Series indexed by date with daily return values.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> pd.DataFrame:
        """Return static asset metadata.

        Returns DataFrame indexed by asset with columns like
        sector, market_cap_group, is_st, listing_date.
        """
        ...
