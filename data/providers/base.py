"""Abstract base class for data providers.

Defines the contract that all data providers must implement, enabling
swappable backends (synthetic, akshare, tushare, custom CSV, etc.).
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
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
