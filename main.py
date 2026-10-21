#!/usr/bin/env python3
"""Entry point for the A-Share Multi-Factor Quant Platform.

Usage:
    python main.py run                      # Run full pipeline with default config
    python main.py run --config myconf.yaml # Run with custom config
    python main.py analyze                  # Analyze existing results
"""

from __future__ import annotations

import argparse
import inspect
import signal
import sys
from pathlib import Path

# Allow running directly via python main.py (pip install -e . also supported)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_platform.utils.config import load_config
from quant_platform.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Effective provider used by the most recent `_load_data()` call, including a
# ValidatedProvider wrapper chain (e.g. "validated(baostock+adata)"). The data
# cache key is computed *after* provider resolution from this identity, so a
# silent fallback can never land in the requested provider's slot.
_LAST_DATA_PROVIDER = ""

# Provenance of the most recent `_load_data()` call, recorded in metrics.json
# and the HTML report so synthetic output can never masquerade as a real feed.
_LAST_DATA_PROVENANCE: dict = {}


class DataProviderUnavailableError(RuntimeError):
    """A configured real data provider could not be used (fail closed)."""


class DataStalenessError(RuntimeError):
    """The loaded panel is older than ``data.max_staleness_days`` sessions.

    Raised by the freshness gate for live/decisive runs; research runs log the
    same message as a warning instead of failing.
    """


def _install_shutdown_handlers(cleanup):
    """Install SIGTERM/SIGINT/atexit shutdown for a CLI entry point (S16).

    Called only from CLI command functions, never from library import paths.
    The signal handler raises ``SystemExit`` so an in-flight run unwinds
    through its normal ``finally`` blocks (the LiveRunner session is closed
    there); ``cleanup`` runs from the caller's finally/atexit exactly once.
    """
    import atexit

    state = {"done": False}

    def _cleanup_once():
        if state["done"]:
            return
        state["done"] = True
        try:
            cleanup()
        except Exception:
            logger.exception("Shutdown cleanup failed")

    def _handler(signum, _frame):
        raise SystemExit(128 + int(signum))

    for name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError) as exc:  # not the main thread / unsupported
            logger.warning("Could not install %s handler: %s", name, exc)
    atexit.register(_cleanup_once)
    return _cleanup_once


# ---------------------------------------------------------------------------
# Shared pipeline core — used by run / compare / sweep
# ---------------------------------------------------------------------------

def _resolve_config_path(config_path_arg) -> Path:
    """Resolve config path from arg, env, or default."""
    if config_path_arg is not None:
        return Path(config_path_arg)
    import os
    env_path = os.environ.get("QUANT_CONFIG")
    if env_path is not None:
        return Path(env_path)
    return Path(__file__).resolve().parent / "config" / "default.yaml"


def _sector_panel_is_usable(panel, has_history: bool) -> bool:
    """Whether a sector panel should be attached to the metadata frame.

    An all-NaN panel with no industry history (the provider ships no
    classification at all) must not be attached: `neutralize` treats any
    supplied panel as authoritative, which would turn the old no-op static
    path into intercept-only demeaning. But when the provider *does* ship
    history -- even one whose entries all fall outside the sample -- the
    all-NaN panel is meaningful: it forbids falling back to the whole-sample
    static map, which would leak future classifications.
    """
    if panel is None:
        return False
    return has_history or bool(panel.notna().to_numpy().any())


def _resolve_provider(
    config,
    use_tushare: bool = False,
    use_baostock: bool = False,
    *,
    with_universe: bool = True,
) -> dict:
    """Select the effective provider and describe its data identity (D12).

    Provider resolution happens *before* the pipeline cache key is computed:
    a silent fallback to synthetic, a ``ValidatedProvider`` wrapper, or a
    changed secondary source all alter the produced data while leaving the
    YAML untouched. The returned dict carries the provider instance plus the
    identity inputs (requested/effective names, wrapper chain, sampled symbol
    set, adjustment basis) used by :func:`_data_cache_key`.
    """
    from quant_platform.data.providers.baostock_provider import BaostockDataProvider
    from quant_platform.data.providers.synthetic import SyntheticDataProvider
    from quant_platform.data.providers.tushare_loader import TushareProvider
    from quant_platform.utils.metrics import get_metrics

    metrics = get_metrics()
    provider = None
    provider_name = ""
    configured_provider = getattr(config.data, 'provider', 'synthetic')
    requested_provider = (
        "baostock" if use_baostock
        else "tushare" if use_tushare
        else configured_provider
    )
    allow_fallback = bool(
        getattr(config.data, "allow_synthetic_fallback", False)
    )
    failures: list[str] = []

    def _record_failure(name: str, error: Exception) -> None:
        failures.append(f"{name}: {error}")
        metrics.counter(
            "data_provider_failures_total",
            labels={"provider": name},
            help="Data-provider initialization failures",
        ).inc()

    # Try configured provider first
    if configured_provider == "baostock" or use_baostock:
        try:
            provider = BaostockDataProvider()
            provider_name = "baostock"
            logger.info("Using Baostock real data provider (free, no API key)")
        except Exception as e:
            _record_failure("baostock", e)
            logger.warning("Baostock unavailable (%s)", e)

    if provider is None and (configured_provider == "tushare" or use_tushare):
        try:
            provider = TushareProvider(
                start_date=config.data.start_date,
                end_date=config.data.end_date,
            )
            provider_name = "tushare"
            logger.info("Using Tushare real data provider")
        except (RuntimeError, ImportError) as e:
            _record_failure("tushare", e)
            logger.warning("Tushare unavailable (%s)", e)

    if provider is None and configured_provider == "postgres":
        try:
            from quant_platform.data.providers.postgres_provider import PostgresDataProvider
            provider = PostgresDataProvider()
            provider_name = "postgres"
            logger.info("Using PostgreSQL data provider")
        except Exception as e:
            _record_failure("postgres", e)
            logger.warning("PostgreSQL unavailable (%s)", e)

    # Fail closed: a configured real provider that is unavailable must not
    # silently become synthetic research data. Synthetic is allowed only for
    # an explicitly synthetic request (config or CLI) or an explicit
    # development opt-in.
    if provider is None:
        if requested_provider != "synthetic" and not allow_fallback:
            detail = "; ".join(failures) or "provider initialization failed"
            raise DataProviderUnavailableError(
                f"Configured data provider '{requested_provider}' is "
                f"unavailable ({detail}). Refusing to fall back to synthetic "
                "data. Fix the provider/credentials, or pass "
                "--allow-synthetic-fallback (or set "
                "data.allow_synthetic_fallback: true) for a development run."
            )
        embedded = getattr(config.data.synthetic, 'embedded_alpha', False)
        alpha_strength = getattr(config.data.synthetic, 'alpha_strength', 0.03)
        if requested_provider != "synthetic":
            logger.warning(
                "Configured provider '%s' unavailable, falling back to "
                "synthetic (explicitly allowed)", requested_provider,
            )
        provider = SyntheticDataProvider(
            n_stocks=config.universe.n_stocks,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            embedded_alpha=embedded,
            alpha_strength=alpha_strength,
        )
        provider_name = "synthetic"

    # Optional: wrap in ValidatedProvider for cross-source validation
    validated = getattr(config.data, 'validated', False)
    if validated and configured_provider != "synthetic":
        try:
            from quant_platform.data.providers.adata_provider import AdataProvider
            from quant_platform.data.providers.validated_provider import ValidatedProvider

            secondary = AdataProvider()
            # Quick test if adata works
            _ = secondary.get_metadata()

            provider = ValidatedProvider(
                providers={"primary": provider, "secondary": secondary},
                primary="primary",
            )
            logger.info("Multi-source validation enabled: primary + adata")
        except Exception:
            logger.info("Secondary provider (adata) unavailable — running without validation")

    from quant_platform.data.providers.base import provider_identity, sampled_universe

    identity = provider_identity(provider)
    if not isinstance(identity.get("sources"), dict) and provider_name:
        # The selection code is authoritative for the effective provider name
        # (test doubles and custom providers may not have a known class name).
        identity["provider"] = provider_name
        identity["synthetic"] = provider_name == "synthetic"
    effective_provider = str(
        identity.get("provider") or provider_name or requested_provider
    )
    universe = None
    if with_universe:
        universe = sampled_universe(
            provider,
            start_date=str(config.data.start_date),
            end_date=str(config.data.end_date),
            n_symbols=getattr(
                getattr(config, "universe", None), "n_stocks", None,
            ),
        )
    return {
        "provider": provider,
        "provider_name": provider_name,
        "requested_provider": requested_provider,
        "effective_provider": effective_provider,
        "identity": identity,
        "universe": universe,
        "allow_fallback": allow_fallback,
        "failures": failures,
        "validated": isinstance(identity.get("sources"), dict),
    }


def _effective_provider_attr(provider, attribute: str):
    """Read a provider attribute through a ValidatedProvider wrapper."""
    providers = getattr(provider, "providers", None)
    if isinstance(providers, dict) and providers:
        primary = providers.get(getattr(provider, "primary", None))
        if primary is None:
            primary = next(iter(providers.values()))
        return _effective_provider_attr(primary, attribute)
    return getattr(provider, attribute, None)


def _build_data_manifest(
    resolution: dict,
    config,
    data_date: str = "",
    payload_assets=None,
    freshness: dict | None = None,
    incremental: dict | None = None,
) -> dict:
    """Provenance record stored alongside a cached data stage (D35).

    Fields: requested/effective provider, synthetic flag, source, date range,
    symbol-set hash, adjustment basis, data date, schema version, creation
    time and app version. A cache hit serves this record, not the current
    run's requested provider, so a synthetic fallback can never masquerade as
    a real-provider result. D32 adds the completion marker/freshness record
    (``last_complete_date``, ``data_age_days`` in exchange sessions,
    ``incomplete``) and the incremental-update summary.
    """
    from datetime import UTC, datetime

    from quant_platform.utils.cache import (
        CACHE_KEY_VERSION,
        DATA_SCHEMA_VERSION,
        PipelineCache,
        app_version,
    )

    identity = resolution.get("identity") or {}
    universe = resolution.get("universe")
    assets = sorted({str(asset) for asset in (payload_assets or [])})
    sources = identity.get("sources")
    if isinstance(sources, dict):
        validation = {
            "wrapper": identity.get("wrapper"),
            "primary": identity.get("primary"),
            "sources": sorted(str(name) for name in sources),
        }
    else:
        validation = {"wrapper": None}
    freshness = dict(freshness or {})
    incremental_summary = incremental or resolution.get("incremental")
    return {
        "requested_provider": resolution.get("requested_provider"),
        "effective_provider": resolution.get("effective_provider"),
        "synthetic": bool(identity.get("synthetic")),
        "source": resolution.get("effective_provider"),
        "start_date": str(config.data.start_date),
        "end_date": str(config.data.end_date),
        "data_date": data_date,
        "symbol_hash": PipelineCache.hash_symbols(universe),
        "symbol_count": None if universe is None else len(set(universe)),
        "payload_asset_hash": PipelineCache.hash_symbols(assets) if assets else None,
        "payload_asset_count": len(assets),
        "adjustment": identity.get("adjustment"),
        "benchmark_source": _effective_provider_attr(
            resolution.get("provider"), "benchmark_source",
        ),
        "schema_version": DATA_SCHEMA_VERSION,
        "cache_key_version": CACHE_KEY_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "app_version": app_version(),
        "fallback_allowed": bool(resolution.get("allow_fallback")),
        "failures": list(resolution.get("failures") or []),
        "validation": validation,
        "last_complete_date": freshness.get("last_complete_date"),
        "data_age_days": freshness.get("data_age_days"),
        "incomplete": bool(freshness.get("incomplete", False)),
        "freshness": freshness,
        "incremental": incremental_summary,
    }


def _data_cache_key(
    config,
    resolution: dict,
    *,
    use_baostock: bool = False,
    use_tushare: bool = False,
) -> str:
    """Pipeline data-stage key: materialized config + effective provider (D12)."""
    from quant_platform.utils.cache import PipelineCache

    return PipelineCache.make_data_cache_key(
        config=config,
        provider_identity=resolution.get("identity") or {},
        symbols=resolution.get("universe"),
        overrides={
            "use_baostock": bool(use_baostock),
            "use_tushare": bool(use_tushare),
        },
        start_date=str(config.data.start_date),
        end_date=str(config.data.end_date),
    )


def _manifest_is_compatible(manifest: dict | None, resolution: dict) -> bool:
    """Whether a stored data manifest describes the current resolved run."""
    from quant_platform.utils.cache import DATA_SCHEMA_VERSION, PipelineCache

    if not isinstance(manifest, dict) or not manifest:
        return False
    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        return False
    if manifest.get("requested_provider") != resolution.get("requested_provider"):
        return False
    if manifest.get("effective_provider") != resolution.get("effective_provider"):
        return False
    identity = resolution.get("identity") or {}
    if manifest.get("adjustment") != identity.get("adjustment"):
        return False
    if bool(manifest.get("synthetic")) != bool(identity.get("synthetic")):
        return False
    expected_symbols = PipelineCache.hash_symbols(resolution.get("universe"))
    return expected_symbols is None or manifest.get("symbol_hash") == expected_symbols


DATA_STAGE = "data_v4"


def _payload_manifest_coherent(payload, manifest: dict | None) -> bool:
    """Whether the payload's embedded provenance matches its sidecar manifest.

    Saves write the pickle and the sidecar under the lock, but a reader can
    still pair an old payload with a newer sidecar (or vice versa) when a
    writer replaces both between the two reads. The manifest is built from
    the payload provenance, so a coherent pair matches on every identity
    field -- including ``created_at``.
    """
    if not isinstance(manifest, dict) or not manifest:
        return False
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        return False
    fields = (
        "requested_provider", "effective_provider", "synthetic", "source",
        "start_date", "end_date", "data_date", "symbol_hash",
        "payload_asset_hash", "adjustment", "benchmark_source",
        "schema_version", "created_at",
    )
    return all(provenance.get(field) == manifest.get(field) for field in fields)


def _max_staleness_days(config) -> int:
    """Configured freshness threshold in exchange sessions (default 3)."""
    value = getattr(getattr(config, "data", None), "max_staleness_days", 3)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid data.max_staleness_days=%r; using 3", value,
        )
        return 3


def _freshness_from_values(
    *,
    last_complete_date,
    data_date,
    requested_end,
    max_staleness_days: int,
    incomplete: bool = False,
    updated_at=None,
    source: str = "provenance",
) -> dict:
    """Measure panel age in exchange sessions and decide staleness."""
    from quant_platform.data.providers.base import (
        expected_data_end,
        sessions_after,
    )

    anchor = last_complete_date or data_date
    age: int | None = None
    if anchor:
        try:
            age = sessions_after(anchor, expected_data_end(requested_end))
        except (TypeError, ValueError) as exc:
            logger.warning("Could not compute data age from %r: %s", anchor, exc)
    stale = bool(age is not None and age > max_staleness_days)
    return {
        "last_complete_date": last_complete_date,
        "data_date": data_date,
        "requested_end": str(requested_end),
        "data_age_days": age,
        "max_staleness_days": int(max_staleness_days),
        "stale": stale,
        "incomplete": bool(incomplete),
        "updated_at": updated_at,
        "source": source,
    }


def _freshness_from_provenance(provenance: dict | None, config) -> dict:
    """Recompute freshness from a (possibly cached) provenance record.

    Age is recomputed at load time against the current wall clock/requested
    end: a stored ``data_age_days`` from a week ago must not make yesterday's
    cached panel look fresh.
    """
    provenance = provenance or {}
    stored = provenance.get("freshness") or {}
    return _freshness_from_values(
        last_complete_date=(
            stored.get("last_complete_date")
            or provenance.get("last_complete_date")
        ),
        data_date=stored.get("data_date") or provenance.get("data_date"),
        requested_end=(
            stored.get("requested_end")
            or provenance.get("end_date")
            or getattr(getattr(config, "data", None), "end_date", "")
        ),
        max_staleness_days=_max_staleness_days(config),
        incomplete=bool(
            stored.get("incomplete", provenance.get("incomplete", False))
        ),
        updated_at=stored.get("updated_at"),
        source=stored.get("source", "provenance"),
    )


def _attach_freshness(provenance: dict | None, config) -> dict:
    """Store the freshness record in provenance with top-level mirrors."""
    freshness = _freshness_from_provenance(provenance, config)
    if isinstance(provenance, dict):
        provenance["freshness"] = freshness
        provenance["last_complete_date"] = freshness["last_complete_date"]
        provenance["data_age_days"] = freshness["data_age_days"]
        provenance["stale"] = freshness["stale"]
        provenance["incomplete"] = freshness["incomplete"]
    return freshness


def _enforce_freshness(
    provenance: dict | None,
    config,
    *,
    context: str = "research",
    fail_closed: bool = False,
) -> dict:
    """Apply the freshness gate: warn for research, fail closed for live.

    ``data.max_staleness_days`` (default 3) is measured in exchange sessions
    between the completion marker and ``min(requested_end, today)``. A stale
    research panel is served with a WARNING; a live/decisive run raises
    :class:`DataStalenessError` rather than trading on aged data.
    """
    freshness = _attach_freshness(provenance, config)
    message = (
        "Data is stale: last complete session "
        f"{freshness['last_complete_date']} is "
        f"{freshness['data_age_days']} exchange session(s) behind the "
        f"expected end {freshness['requested_end']} (max "
        f"{freshness['max_staleness_days']})"
    )
    if freshness["stale"]:
        if fail_closed:
            raise DataStalenessError(
                message + f"; refusing to run {context} on stale data"
            )
        logger.warning("%s; continuing with a warning (%s run)", message, context)
    if freshness["incomplete"]:
        logger.warning(
            "The provider reported an incomplete data update; the panel is "
            "the last complete snapshot (marker %s)",
            freshness["last_complete_date"],
        )
    return freshness


def _incremental_summary_line(provenance: dict | None, freshness: dict) -> str:
    """One-line CLI summary required by D32.

    ``data: <n> new sessions appended, data_date=..., stale=...``
    """
    provenance = provenance or {}
    appended = int((provenance.get("incremental") or {}).get("new_sessions") or 0)
    return (
        f"data: {appended} new sessions appended, "
        f"data_date={provenance.get('data_date') or ''}, "
        f"stale={'yes' if freshness.get('stale') else 'no'}"
    )


def _run_incremental_update(resolution: dict, config) -> dict:
    """Ask the effective provider to append new sessions (D32).

    Returns a JSON-friendly summary; never raises for an unsupported provider
    (the CLI falls back to a full fetch with a WARNING). A provider that
    raises ``NotImplementedError`` despite advertising support is treated the
    same way -- an append is optional, silently treating stale data as fresh
    is not (the freshness gate still applies).
    """
    provider = resolution.get("provider")
    requested_end = str(config.data.end_date)
    unsupported = {
        "supported": False,
        "performed": False,
        "full_refetch": True,
        "new_sessions": 0,
        "requested_end": requested_end,
        "incomplete": False,
        "reason": "",
    }
    if resolution.get("validated"):
        logger.warning(
            "--incremental: validated multi-source runs cannot attest which "
            "sources contribute to an append; performing a full fetch"
        )
        return {
            **unsupported,
            "reason": "validated multi-source run cannot append",
        }
    if not bool(getattr(provider, "supports_incremental", False)):
        logger.warning(
            "--incremental: provider %s does not support incremental updates; "
            "performing a full fetch",
            type(provider).__name__,
        )
        return {
            **unsupported,
            "reason": "provider does not support incremental updates",
        }
    try:
        summary = provider.update(
            str(config.data.start_date), requested_end,
        )
    except NotImplementedError as exc:
        logger.warning(
            "--incremental: provider %s refused to append (%s); performing "
            "a full fetch", type(provider).__name__, exc,
        )
        return {
            **unsupported,
            "reason": f"provider refused to append: {exc}",
        }
    summary = dict(summary or {})
    summary.setdefault("supported", True)
    summary.setdefault("requested_end", requested_end)
    summary.setdefault("incomplete", False)
    summary.setdefault("performed", False)
    summary.setdefault("new_sessions", 0)
    if summary.get("incomplete"):
        logger.warning(
            "Incremental data update incomplete: %s (cached panel kept; "
            "freshness gate applies)", summary.get("reason"),
        )
    elif summary.get("full_refetch"):
        logger.info(
            "Incremental update fell back to a full refetch: %s",
            summary.get("reason"),
        )
    else:
        logger.info("Incremental update: %s", summary.get("reason") or "ok")
    return summary


def _load_data_stage(
    cache,
    cache_key: str,
    config,
    resolution: dict,
    *,
    read_cache: bool,
    write_cache: bool,
    use_tushare: bool = False,
    use_baostock: bool = False,
    incremental: bool = False,
):
    """Load the data stage from cache or recompute and (over)write it.

    ``--force`` is ``read_cache=False, write_cache=True``: the stale entry is
    atomically replaced instead of being left behind (D33). ``--no-cache`` is
    both False. A provider whose universe cannot be determined cheaply is not
    cached at all (a key without the symbol-set hash could serve stale data).

    ``incremental`` asks the effective provider to append new exchange
    sessions to its own cache before the pipeline runs (D32). Providers that
    cannot append raise ``NotImplementedError`` / advertise
    ``supports_incremental=False``; the CLI then falls back to a full fetch
    with a WARNING and records that in provenance. The pipeline-cache read is
    skipped by the caller when ``incremental`` is set, so a stale data-stage
    pickle can never mask the append.

    Returns ``(data_tuple, provenance, cache_hit)``.
    """
    stage = DATA_STAGE
    validated = bool(resolution.get("validated"))
    # A validated multi-source run may silently drop a configured source at
    # fetch time (e.g. adata returns empty). The key records the configured
    # chain, not which sources actually contributed, so caching such a run
    # could serve primary-only data under a "validated" entry. Skip caching
    # until the wrapper can attest the observed sources.
    can_cache = resolution.get("universe") is not None and not validated

    if read_cache and can_cache:
        payload, manifest = cache.load_stage_with_manifest(stage, cache_key)
        if isinstance(payload, dict) and payload.get("data") is not None:
            compatible = _manifest_is_compatible(manifest, resolution)
            coherent = _payload_manifest_coherent(payload, manifest)
            if compatible and coherent:
                data_tuple = payload["data"]
                provenance = dict(manifest or payload.get("provenance") or {})
                provenance["cache"] = {
                    "hit": True, "stage": stage, "key": cache_key,
                }
                logger.info("Data loaded from cache (key %s)", cache_key)
                _attach_freshness(provenance, config)
                return data_tuple, provenance, True
            if not compatible:
                logger.warning(
                    "Cache entry %s failed provenance validation; recomputing",
                    cache_key,
                )
                cache.evict_stage(stage, cache_key)
            else:
                # A concurrent writer is replacing this entry: do not evict
                # the fresh file, just recompute and let the last save win.
                logger.warning(
                    "Cache entry %s has mismatched payload/manifest "
                    "provenance (concurrent writer?); recomputing without "
                    "evicting", cache_key,
                )
    elif read_cache and not can_cache:
        logger.info(
            "Pipeline cache disabled for this run: %s",
            "validated multi-source runs cannot attest which sources "
            "contributed" if validated
            else "the provider universe cannot be determined before loading",
        )

    if incremental:
        # Append to the *provider* cache first; the pipeline then reads the
        # extended panel. The provider keeps its marker unchanged on a
        # partial response, and the freshness gate below still sees the age.
        resolution["incremental"] = _run_incremental_update(resolution, config)

    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config,
        use_tushare=use_tushare,
        use_baostock=use_baostock,
        resolution=resolution,
    )
    data_tuple = (
        prices, returns, benchmark, metadata, financials, turnover,
    )
    provenance = dict(_LAST_DATA_PROVENANCE)
    provenance["cache"] = {"hit": False, "stage": stage, "key": cache_key}
    _attach_freshness(provenance, config)

    if write_cache and can_cache:
        manifest = {
            key: value for key, value in provenance.items() if key != "cache"
        }
        cache.save_stage(
            stage, cache_key,
            {"data": data_tuple, "provenance": provenance},
            manifest=manifest,
        )
    elif write_cache:
        logger.info(
            "Not caching data: the effective provider's symbol set is unknown"
        )
    return data_tuple, provenance, False


def _load_data(
    config,
    use_tushare: bool = False,
    use_baostock: bool = False,
    resolution: dict | None = None,
):
    """Load and clean data. Returns (prices, returns, benchmark, metadata, financials, turnover).

    Respects the config.data.provider setting:
    - "baostock":  Baostock free A-share data
    - "tushare":   Tushare Pro (requires TUSHARE_TOKEN env var)
    - "postgres":  PostgreSQL stored data
    - "synthetic": Synthetic data (for testing only)

    `use_tushare`/`use_baostock` are explicit CLI overrides. They default to
    False so the config decides; the old `use_tushare=True` default silently
    replaced a configured synthetic/postgres provider whenever a Tushare
    token happened to be present in the environment.

    ``resolution`` may be passed by callers that resolved the provider
    earlier to compute the pipeline cache key (D12); otherwise the provider is
    resolved here.
    """
    from quant_platform.data.pipeline import DataPipeline
    from quant_platform.utils.metrics import get_metrics

    if resolution is None:
        resolution = _resolve_provider(
            config, use_tushare=use_tushare, use_baostock=use_baostock,
        )
    provider = resolution["provider"]
    provider_name = resolution.get("provider_name") or ""
    requested_provider = resolution.get("requested_provider", "")
    metrics = get_metrics()

    pipeline = DataPipeline(
        provider=provider,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        exclude_st=config.universe.exclude_st,
        exclude_suspended=config.universe.exclude_suspended,
    )
    try:
        pipeline.run()
    except Exception:
        metrics.counter(
            "data_provider_failures_total",
            labels={"provider": provider_name or requested_provider},
            help="Data-provider initialization failures",
        ).inc()
        raise
    metrics.counter(
        "data_provider_success_total",
        labels={"provider": provider_name},
        help="Successful data-provider loads",
    ).inc()

    metadata = pipeline.metadata
    # The point-in-time sector panel rides along on the metadata frame so the
    # 6-tuple return contract stays intact (api/routes.py, backtest/distributed
    # and risk/lookahead_detector unpack it). `_compute_factors` reads it from
    # `metadata.attrs` and falls back to the static map when it is absent.
    # Only pay for the panel when sector neutralization is configured.
    factors_cfg = getattr(config, "factors", None)
    processing_cfg = (
        getattr(factors_cfg, "processing", {}) if factors_cfg is not None else {}
    )
    if not isinstance(processing_cfg, dict):
        processing_cfg = {}
    neutralize_cfg = processing_cfg.get("neutralize", {}) or {}
    wants_sector = neutralize_cfg.get("enabled", True) and "sector" in (
        neutralize_cfg.get("by", ["sector", "log_market_cap"])
    )
    sector_panel = None
    if wants_sector:
        try:
            panel = pipeline.get_sector_panel()
        except Exception as exc:  # defensive: panel failure must not kill the run
            logger.warning(
                "Sector panel unavailable (%s); using static sector map", exc,
            )
            panel = None
        # An all-NaN panel (no sector metadata and no industry history) would
        # otherwise flip the no-op static-map path into an intercept-only
        # demean; leave metadata untouched in that case. History that simply
        # does not cover the sample still counts as "panel usable": it must
        # keep neutralize from falling back to the static map.
        if _sector_panel_is_usable(panel, pipeline.has_industry_history()):
            sector_panel = panel
    if metadata is not None and sector_panel is not None:
        metadata.attrs["sector_panel"] = sector_panel

    # Execution-day tradability travels on metadata (not on the price panel:
    # DataFrame attrs propagate into derived frames and any DataFrame-valued
    # attr can crash pandas' internals). `_run_backtest` forwards these to the
    # engine, which checks them on the fill date, not the signal date.
    if metadata is not None and hasattr(metadata, "attrs"):
        try:
            gates = pipeline.get_execution_gates()
        except Exception as exc:  # defensive: gates must not kill the run
            logger.warning("Execution gates unavailable (%s); fills are ungated", exc)
            gates = {}
        if gates:
            metadata.attrs["execution_gates"] = gates
        try:
            ohlc = pipeline.get_factor_ohlc()
        except Exception as exc:  # defensive: OHLC must not kill the run
            logger.warning("Adjusted OHLC unavailable (%s); candle factors skipped", exc)
            ohlc = {}
        if ohlc:
            metadata.attrs["factor_ohlc"] = ohlc

    global _LAST_DATA_PROVIDER, _LAST_DATA_PROVENANCE
    _LAST_DATA_PROVIDER = resolution.get("effective_provider") or provider_name

    factor_prices = pipeline.get_factor_prices()
    # Money paths (trade execute, smart-route, orchestrator fills) need the
    # traded CNY panel; the adjusted research panel above is not tradeable.
    # It rides on metadata (a DataFrame-valued attr on the *price* panel would
    # propagate into derived frames, per the gates comment above).
    if metadata is not None and hasattr(metadata, "attrs"):
        from quant_platform.data.schema import DataQualityError

        try:
            raw_close = pipeline.get_raw_close()
        except DataQualityError as exc:
            logger.warning(
                "Raw close unavailable; money paths must use broker/real-time "
                "quotes instead: %s", exc,
            )
        else:
            metadata.attrs["raw_prices"] = {
                "close": raw_close, "price_basis": "raw",
            }
    data_date = ""
    payload_assets: list[str] = []
    if factor_prices is not None and len(factor_prices.index) > 0:
        data_date = str(factor_prices.index[-1])[:10]
        try:
            payload_assets = [str(asset) for asset in factor_prices.columns]
        except Exception:  # defensive: provenance must not kill the run
            payload_assets = []
    freshness = None
    freshness_hook = getattr(pipeline, "get_data_freshness", None)
    if callable(freshness_hook):  # test doubles may not implement it
        try:
            freshness = freshness_hook()
        except Exception as exc:  # provenance must not kill a run
            logger.warning("Data freshness unavailable: %s", exc)
    _LAST_DATA_PROVENANCE = _build_data_manifest(
        resolution, config, data_date=data_date, payload_assets=payload_assets,
        freshness=freshness,
        incremental=(resolution or {}).get("incremental"),
    )

    return (
        factor_prices,
        pipeline.returns,
        pipeline.benchmark,
        metadata,
        pipeline.financials,
        pipeline.get_turnover(),
    )


def _compute_factors(prices, returns, financials, metadata, turnover=None, config=None,
                     return_raw: bool = False):
    """Compute and process all registered factors.

    Returns ``(processed_factors, ic_results, sector_map, fin_unstacked)``, or
    a fifth element ``raw_factors`` when ``return_raw`` is true. Screening
    rules are written against raw economic units (``pe_ratio < 30``), so the
    CLI needs the unstandardized panels; everything else consumes the
    processed ones.
    """
    from quant_platform.factors.base import mask_financials
    from quant_platform.factors.evaluation import ic_summary, rank_ic
    from quant_platform.factors.fundamental import register_all as register_fundamental
    from quant_platform.factors.processing import process_factor
    from quant_platform.factors.registry import get_registry
    from quant_platform.factors.technical import register_all as register_technical

    register_technical()
    register_fundamental()
    registry = get_registry()

    # Build set of enabled factor names from config (if provided)
    enabled_factors: set[str] | None = None
    if config is not None and hasattr(config, 'factors'):
        enabled_factors = set()
        enabled_factors.update(getattr(config.factors, 'enabled_technicals', ()))
        enabled_factors.update(getattr(config.factors, 'enabled_fundamentals', ()))

    # Point-in-time asset-day eligibility (ST, pre-listing, delisted,
    # suspended) travels with the factor price panel.  It must reach the
    # fundamental inputs too: masking only the price columns left financial
    # factors live for ST / not-yet-listed / delisted names, which then
    # received a signal and a target weight.
    excluded = prices.attrs.get("excluded_mask") if prices is not None else None

    fin_unstacked = financials.unstack("asset") if financials is not None else None
    if fin_unstacked is not None and excluded is not None:
        fin_unstacked = mask_financials(fin_unstacked, excluded)

    raw_factors = {}
    factor_params = (
        getattr(config.factors, "params", {}) if config is not None else {}
    )
    # Adjusted OHLC for factors that declare `required_inputs` (candle
    # factors). It rides on metadata like the sector panel and execution
    # gates so the 6-tuple load contract stays intact.
    factor_ohlc = getattr(metadata, "attrs", {}).get("factor_ohlc", {}) if metadata is not None else {}
    for factor_name in registry.list_all():
        if enabled_factors is not None and factor_name not in enabled_factors:
            continue
        cls = registry.get(factor_name)
        try:
            inst = cls(**factor_params.get(factor_name, {}))
        except TypeError as exc:
            logger.warning(
                "Invalid params for %s in config, using defaults: %s",
                factor_name, exc,
            )
            inst = cls()
        try:
            kwargs = {}
            if turnover is not None:
                kwargs["turnover"] = turnover
            for field in getattr(inst, "required_inputs", ()):
                panel = factor_ohlc.get(field) if factor_ohlc else None
                if panel is not None:
                    kwargs[field] = panel
            if inst.category.value == "fundamental" and fin_unstacked is not None:
                result = inst.run(prices, fin_unstacked, **kwargs)
            else:
                result = inst.run(prices, **kwargs)
            raw_factors[result.name] = result.values
        except Exception as e:
            logger.warning("Failed to compute %s: %s", factor_name, e)

    logger.info("Computed %d factors: %s", len(raw_factors), list(raw_factors.keys()))

    # Belt-and-braces for factors that do not go through the price panel's
    # attrs (or override `compute`): blank every raw factor, whatever its
    # category, on ineligible asset-days. ST names, pre-listing assets,
    # delisted names and suspended days must never reach the processed
    # cross-section, the signal or the optimizer.
    if excluded is not None:
        raw_factors = {
            name: values.mask(
                excluded.reindex(
                    index=values.index, columns=values.columns, fill_value=True,
                )
            )
            for name, values in raw_factors.items()
        }

    sector_map = metadata.get("sector", {}) if metadata is not None else {}
    # Point-in-time sector panel attached by `_load_data`; absent when the
    # caller built metadata itself (lookahead detector, cached results), in
    # which case the static map is the documented fallback.
    sector_panel = (
        getattr(metadata, "attrs", {}).get("sector_panel")
        if metadata is not None else None
    )
    mcap = fin_unstacked.get("market_cap") if fin_unstacked is not None else None

    # `factors.processing` used to be dropped during parsing, so changing
    # winsorize/standardize/neutralize settings produced identical factors.
    processing = (
        getattr(config.factors, "processing", {}) if config is not None else {}
    )
    w_cfg = processing.get("winsorize", {}) or {}
    s_cfg = processing.get("standardize", {}) or {}
    n_cfg = processing.get("neutralize", {}) or {}
    neutralize_by = n_cfg.get("by", ["sector", "log_market_cap"])

    processed_factors = {}
    for name, factor in raw_factors.items():
        processed_factors[name] = process_factor(
            factor,
            winsorize_enabled=w_cfg.get("enabled", True),
            winsorize_lower=w_cfg.get("lower", 0.01),
            winsorize_upper=w_cfg.get("upper", 0.99),
            standardize_enabled=s_cfg.get("enabled", True),
            standardize_method=s_cfg.get("method", "zscore"),
            neutralize_enabled=n_cfg.get("enabled", True),
            sector_map=sector_map if "sector" in neutralize_by else None,
            market_cap=(
                mcap
                if ("log_market_cap" in neutralize_by or "market_cap" in neutralize_by)
                else None
            ),
            # Point-in-time classification when the pipeline supplied one;
            # otherwise process_factor uses the static sector_map above.
            sector_panel=(
                sector_panel
                if sector_panel is not None and "sector" in neutralize_by
                else None
            ),
            factor_name=name,
        )

    ic_results = {}
    for name, factor in processed_factors.items():
        ic = rank_ic(factor, returns, name=name)
        summary = ic_summary(ic)
        ic_results[name] = summary
        if summary.get("n_obs", 0) < 10:
            # A factor whose processed panel yields no evaluable
            # cross-sections contributes nothing to the signal. Say so
            # instead of reporting a NaN row as if it were a weak factor.
            logger.warning(
                "Factor %s has only %d evaluable IC observations and will "
                "not contribute to the alpha signal",
                name, summary.get("n_obs", 0),
            )

    # Producer for the Grafana "Factor IC Decay" panel.
    try:
        import math

        from quant_platform.utils.metrics import get_metrics

        metrics = get_metrics()
        for name, summary in ic_results.items():
            mean_ic = summary.get("mean_ic")
            if mean_ic is None or not math.isfinite(float(mean_ic)):
                continue
            metrics.gauge(
                "factor_ic_value",
                labels={"factor": name},
                help="Mean Rank IC of the computed factor",
            ).set(float(mean_ic))
    except Exception as exc:  # monitoring must not break the pipeline
        logger.debug("Factor metric publish failed: %s", exc)

    if return_raw:
        return processed_factors, ic_results, sector_map, fin_unstacked, raw_factors
    return processed_factors, ic_results, sector_map, fin_unstacked


def _execution_gates(metadata):
    """Execution-day tradability panels attached by `_load_data`, if any."""
    attrs = getattr(metadata, "attrs", None)
    return attrs.get("execution_gates") if attrs else None


def _run_backtest(config, signal, prices, returns, benchmark, sector_map, financials,
                  optimizer_override=None, frequency_override=None, gates=None):
    """Set up and run backtest engine. Returns results dict.

    ``gates`` is the pipeline's execution-day tradability set
    (``tradable_mask``/``limit_up``/``limit_down``, normally read from
    ``metadata.attrs["execution_gates"]``). Without it the engine fills at
    any next-bar close, including limit-up opens and suspension days.
    """
    from quant_platform.backtest.cost_model import CostModel
    from quant_platform.backtest.engine import BacktestEngine
    from quant_platform.portfolio.constraints import PortfolioConstraints

    constraints = PortfolioConstraints(
        long_only=config.portfolio.constraints.long_only,
        max_weight=config.portfolio.constraints.max_weight,
        max_sector_exposure=config.portfolio.constraints.max_sector_exposure,
        max_turnover=config.portfolio.constraints.max_turnover,
        lot_size=config.portfolio.constraints.lot_size,
        risk_aversion=config.portfolio.risk_aversion,
    )

    cost_model = CostModel(
        commission=config.costs.commission,
        stamp_tax=config.costs.stamp_tax,
        slippage=config.costs.slippage,
        slippage_model=config.costs.slippage_model,
    )

    engine = BacktestEngine(
        initial_capital=config.backtest.initial_capital,
        rebalance_frequency=frequency_override or config.backtest.rebalance_frequency,
        cost_model=cost_model,
        constraints=constraints,
        optimizer=optimizer_override or config.portfolio.optimizer,
        benchmark=config.backtest.benchmark,
        covariance_method=config.portfolio.covariance.method,
        covariance_lookback=config.portfolio.covariance.lookback,
        covariance_ewma_half_life=config.portfolio.covariance.ewma_half_life,
    )

    return engine.run(
        signal=signal, prices=prices, returns=returns,
        benchmark_returns=benchmark, sector_map=sector_map,
        financials=financials,
        **(gates or {}),
    )


def _generate_signal(config, processed_factors, returns, prices=None, volume=None):
    """Generate alpha signal from processed factors.

    The factor combination fills a missing factor value with 0 (the
    cross-sectional mean of a z-scored factor) by design, so masking the
    factor panels alone is not enough: an ST / pre-listing / delisted /
    suspended name whose factors are all NaN still receives a neutral score
    and can be selected.  The pipeline's eligibility mask (carried on the
    price panel's ``attrs``) is therefore re-applied to the final signal, so
    those names drop out of the optimizer's cross-section entirely.
    """
    from quant_platform.alpha.pipeline import AlphaPipeline

    alpha_pipe = AlphaPipeline(
        method=config.alpha.method,
        lookback=config.alpha.lookback,
        min_icir=config.alpha.min_icir,
        tradability_gate=getattr(config.alpha, 'tradability_gate', False),
        min_tradability=getattr(config.alpha, 'min_tradability', 0.3),
    )
    signal = alpha_pipe.run(processed_factors, returns, prices=prices, volume=volume)

    excluded = prices.attrs.get("excluded_mask") if prices is not None else None
    if excluded is not None:
        signal = signal.mask(
            excluded.reindex(
                index=signal.index, columns=signal.columns, fill_value=True,
            )
        )
    return signal


def _chart_payload(results: dict) -> dict:
    """Chart data for the standalone HTML report (CLI path).

    Mirrors the subset `reporting.html_report` reads. The API builds a richer
    ChartData model for the dashboard; the CLI only needs the time series.
    """
    import numpy as np
    import pandas as pd

    strategy_returns = results["daily_returns"].dropna()
    equity = (1 + strategy_returns).cumprod()
    dates = [str(d.date()) for d in equity.index]

    benchmark_returns = results.get("benchmark_returns")
    bench_equity = None
    if benchmark_returns is not None:
        br = benchmark_returns.reindex(strategy_returns.index).dropna()
        bench_equity = (1 + br).cumprod().tolist()

    wealth = np.concatenate([[1.0], equity.to_numpy()])
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth / running_max - 1)[1:].tolist()

    rolling_sharpe = None
    if len(strategy_returns) >= 252:
        from quant_platform.backtest.metrics import TRADING_DAYS_PER_YEAR

        rolling_ret = strategy_returns.rolling(252).mean() * TRADING_DAYS_PER_YEAR
        rolling_vol = (
            strategy_returns.rolling(252).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        rs = (rolling_ret / rolling_vol.replace(0, np.nan)).dropna()
        rolling_sharpe = rs.tolist()

    monthly = None
    if len(strategy_returns) >= 21:
        monthly_ret = strategy_returns.resample(
            pd.offsets.MonthEnd()
        ).apply(lambda x: (1 + x).prod() - 1)
        monthly_ret.index = monthly_ret.index.to_period("M")
        mdf = monthly_ret.groupby(
            [monthly_ret.index.year, monthly_ret.index.month]
        ).first().unstack()
        mdf = mdf.reindex(columns=range(1, 13))
        if not mdf.empty:
            monthly = {
                "years": [str(y) for y in mdf.index.tolist()],
                "months": list(range(1, 13)),
                "data": [
                    [None if pd.isna(value) else round(float(value), 6)
                     for value in row]
                    for row in mdf.values.tolist()
                ],
            }

    return {
        "dates": dates,
        "equity": equity.tolist(),
        "benchmark": bench_equity,
        "drawdown": drawdown,
        "rolling_sharpe": rolling_sharpe,
        "monthly_returns": monthly or {},
    }


def _exposure_payload(results: dict, metadata) -> dict:
    """Latest-weights exposure for the HTML report."""
    weights_history = results.get("weights_history") or {}
    if not weights_history:
        return {"sectors": {}, "n_assets": 0}
    latest_weights = weights_history[max(weights_history)]
    sector_map = metadata.get("sector", {}) if metadata is not None else {}
    try:
        from quant_platform.risk.exposure import exposure_report

        exp = exposure_report(latest_weights, sector_map)
    except Exception as exc:  # exposure is presentation-only
        logger.warning("Exposure report unavailable: %s", exc)
        return {"sectors": {}, "n_assets": 0}
    sectors = exp.get("sector_exposure", {})
    if hasattr(sectors, "to_dict"):
        sectors = sectors.to_dict()
    return {
        "sectors": {str(k): float(v) for k, v in dict(sectors).items()},
        "n_assets": exp.get("n_assets", 0),
        "effective_n": exp.get("effective_n", 0),
        "top5_concentration": exp.get("top5_concentration", 0),
        "top10_concentration": exp.get("top10_concentration", 0),
    }


def _json_safe(value):
    """Recursively replace non-finite floats and numpy scalars for strict JSON."""
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_run_artifacts(
    results: dict,
    ic_results: dict,
    metadata,
    config,
    output_dir: str | Path | None = None,
    provenance: dict | None = None,
) -> tuple[Path, Path]:
    """Write the standalone HTML report and metrics.json for a CLI run.

    Both were promised by the docs but never produced: `run` only wrote CSVs
    and PNGs, and `research report` required a metrics.json nobody generated.

    ``output_dir`` overrides ``config.output.results_dir`` so each run can
    write into its own per-run directory (issue S15). ``provenance`` records
    the requested/effective data provider, synthetic flag and data date in
    both metrics.json and the HTML report (issue S8).
    """
    import json
    from datetime import datetime

    from quant_platform.reporting.html_report import generate_html_report

    output_dir = Path(output_dir or config.output.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = provenance or {}

    strategy_returns = results["daily_returns"].dropna()
    summary = results.get("summary", {})

    risk = {}
    try:
        from quant_platform.risk.var import var_summary

        risk = var_summary(
            strategy_returns,
            confidence=config.risk.var.confidence,
            horizon=config.risk.var.horizon,
        )
    except Exception as exc:
        logger.warning("Risk summary unavailable for the report: %s", exc)

    stress_rows = []
    try:
        from quant_platform.risk.stress import run_all_stress_tests

        stress_df = run_all_stress_tests(strategy_returns)
        stress_rows = [
            {
                "scenario": str(row.name),
                "cumulative_return": float(row["estimated_cumulative_return"]),
                "max_drawdown": float(row["estimated_max_drawdown"]),
            }
            for _, row in stress_df.iterrows()
        ]
    except Exception as exc:
        logger.warning("Stress tests unavailable for the report: %s", exc)

    factors = [
        {
            "name": name,
            "mean_ic": summary_row.get("mean_ic", 0.0),
            "std_ic": summary_row.get("std_ic", 0.0),
            "icir": summary_row.get("icir", 0.0),
            "ic_positive_ratio": summary_row.get("ic_positive_ratio", 0.0),
            "n_obs": summary_row.get("n_obs", 0),
        }
        for name, summary_row in (ic_results or {}).items()
    ]

    payload = {
        "performance": summary,
        "risk": risk,
        "factors": factors,
        "chart_data": _chart_payload(results),
        "exposure": _exposure_payload(results, metadata),
        "stress_tests": stress_rows,
        "provenance": provenance,
    }
    html_path = Path(generate_html_report(
        payload,
        output_path=str(output_dir / "backtest_report.html"),
    ))

    metrics = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "provenance": provenance,
        "performance": summary,
        "risk": risk,
        "factors": factors,
        "stress_tests": stress_rows,
        "daily_returns": [float(x) for x in strategy_returns.to_numpy()],
        "daily_return_dates": [str(d.date()) for d in strategy_returns.index],
        # A stored backtest keeps no factor-return regressions, so genuine
        # factor attribution cannot be produced here. An empty mapping tells
        # `research report` to print the measured metrics instead of inventing
        # contributions.
        "factor_contributions": {},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_json_safe(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return html_path, metrics_path


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    """Execute full pipeline: data -> factors -> alpha -> portfolio -> backtest -> report."""
    import uuid

    import yaml

    from quant_platform.utils.cache import PipelineCache
    from quant_platform.utils.metrics import pipeline_stage

    config_path = _resolve_config_path(args.config)
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    config = load_config(raw=raw_config)
    if getattr(args, "allow_synthetic_fallback", False):
        # Explicit development opt-in for a silent synthetic fallback.
        config.data.allow_synthetic_fallback = True

    run_id = uuid.uuid4().hex[:12]
    use_baostock = bool(getattr(args, "use_baostock", False))
    use_tushare = bool(getattr(args, "use_tushare", False))
    # D32: --incremental appends new exchange sessions to the provider cache
    # before the pipeline runs. --force still means "full refetch", which wins
    # over --incremental; the data-stage pickle is not read for an incremental
    # run, otherwise stale cached output would mask the append.
    incremental = bool(getattr(args, "incremental", False))
    if incremental and args.force:
        logger.warning(
            "--force overrides --incremental; performing a full refetch"
        )
        incremental = False
    # D12: resolve the provider *before* computing the key, so a silent
    # fallback, a ValidatedProvider wrapper or a changed secondary source
    # cannot reuse another run's entry. The universe (sampled symbols) is
    # needed only when the cache is used at all.
    resolution = _resolve_provider(
        config,
        use_tushare=use_tushare,
        use_baostock=use_baostock,
        with_universe=not args.no_cache or incremental,
    )
    cache_key = _data_cache_key(
        config, resolution,
        use_baostock=use_baostock, use_tushare=use_tushare,
    )
    config_hash = cache_key
    cache = PipelineCache(args.cache_dir)
    # D33: --force skips the *read* but still writes: a forced recomputation
    # must replace the stale entry, otherwise the next normal run serves the
    # old payload again. --no-cache disables both. --incremental also skips
    # the read (but writes) so the refreshed provider panel is used.
    read_cache = not args.force and not args.no_cache and not incremental
    write_cache = not args.no_cache

    def _save_config_version() -> None:
        """Persist the config snapshot only for successful runs (issue S11)."""
        try:
            from quant_platform.utils.version_manager import VersionManager

            vm = VersionManager()
            desc = args.description or (
                f"Run: alpha={config.alpha.method} "
                f"optimizer={config.portfolio.optimizer}"
            )
            version_id = vm.save(raw_config, description=desc, run_id=run_id)
            logger.info("Config auto-saved as version %s", version_id)
        except Exception as exc:  # version history must not fail a run
            logger.warning("Failed to save the config version: %s", exc)

    logger.info("=" * 60)
    logger.info("Quant Platform: Starting full pipeline run")
    logger.info("  Config: %s  Cache key: %s  Read cache: %s  Write cache: %s",
                 config_path, cache_key, read_cache, write_cache)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading data...")
    (prices, returns, benchmark, metadata, financials, turnover), provenance, cache_hit = (
        _load_data_stage(
            cache, cache_key, config, resolution,
            read_cache=read_cache, write_cache=write_cache,
            use_tushare=use_tushare, use_baostock=use_baostock,
            incremental=incremental,
        )
    )
    if cache_hit:
        logger.info("Data loaded from cache")

    freshness = _enforce_freshness(
        provenance, config, context="research", fail_closed=False,
    )
    if incremental:
        print(_incremental_summary_line(provenance, freshness))

    if provenance:
        logger.info(
            "Data provenance: requested=%s effective=%s synthetic=%s data_date=%s",
            provenance.get("requested_provider"),
            provenance.get("effective_provider"),
            provenance.get("synthetic"),
            provenance.get("data_date"),
        )

    # ------------------------------------------------------------------
    # 2. Factors
    # ------------------------------------------------------------------
    logger.info("[2/6] Computing factors...")
    with pipeline_stage("factors"):
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

    for name, summary in ic_results.items():
        logger.info("  %-20s Rank IC=%.4f  ICIR=%.2f", name, summary["mean_ic"], summary["icir"])

    # Save factor evaluations to Factor Research Store if requested
    if getattr(args, 'save_factors', False):
        try:
            from quant_platform.factors.store import FactorEvalRecord, FactorResearchStore
            fs = FactorResearchStore()
            saved = 0
            for name, summary in ic_results.items():
                fs.save_evaluation(FactorEvalRecord(
                    factor_name=name,
                    signal_date=config.data.end_date,
                    rank_ic=summary["mean_ic"],
                    pearson_ic=summary.get("pearson_ic", 0),
                    icir=summary["icir"],
                    coverage=summary.get("coverage", 0),
                    n_assets=summary.get("n_assets", 0),
                    run_id=f"run_{config.data.start_date}_{config.data.end_date}",
                ))
                saved += 1
            logger.info("Saved %d factor evaluations to Factor Research Store", saved)
        except Exception as e:
            logger.warning("Failed to save factor evaluations: %s", e)

    # ------------------------------------------------------------------
    # 3. Alpha Signal (or Screener mode)
    # ------------------------------------------------------------------
    screener_enabled = getattr(config, 'screener', None) and config.screener.enabled

    if screener_enabled:
        logger.info("[3/6] Running Factor Screener (boolean rules)...")
        from quant_platform.portfolio.screener import FactorScreener, ScreenConfig, ScreenRule

        # Build screener from config rules
        rules = []
        for r in config.screener.rules:
            rules.append(ScreenRule(
                factor=r.factor,
                operator=r.operator,
                value=r.value,
            ))
        screener_config = ScreenConfig(
            enabled=True,
            rules=rules,
            logic=config.screener.logic,
            min_stocks=config.screener.min_stocks,
            max_stocks=config.screener.max_stocks,
        )
        screener = FactorScreener(screener_config)
        qualifiers = screener.screen(processed_factors)

        if not qualifiers:
            logger.warning("Screener returned no qualifying stocks — skipping backtest")
            print("\n  SCREENER RESULT: No stocks matched the rules.\n")
            _save_config_version()
            return 0

        # Build equal-weight signal from qualifying stocks
        import pandas as pd
        logger.info("Screener: %d qualifying stocks — using equal weight", len(qualifiers))
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in signal.index:
            w = 1.0 / len(qualifiers)
            for asset in qualifiers:
                if asset in signal.columns:
                    signal.loc[date, asset] = w
    else:
        logger.info("[3/6] Generating alpha signal...")
        with pipeline_stage("alpha"):
            signal = _generate_signal(config, processed_factors, returns,
                                      prices=prices, volume=turnover)

    # ------------------------------------------------------------------
    # 4-5. Portfolio + Backtest
    # ------------------------------------------------------------------
    logger.info("[4/6] Portfolio optimization (integrated in backtest)...")
    logger.info("[5/6] Running backtest...")
    with pipeline_stage("backtest"):
        results = _run_backtest(config, signal, prices, returns, benchmark,
                                sector_map, fin_unstacked,
                                gates=_execution_gates(metadata))

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    logger.info("[6/6] Generating report...")
    import json
    from datetime import datetime

    from quant_platform.reporting.dashboard import (
        create_run_dir,
        generate_dashboard,
        publish_run_artifacts,
        write_manifest,
    )

    results_base = Path(config.output.results_dir)
    run_dir = create_run_dir(results_base, run_id, config_hash)
    logger.info("Run artifacts: %s", run_dir)

    report = generate_dashboard(
        results=results, metadata=metadata,
        output_dir=str(run_dir),
        save_plots=config.output.save_plots,
        plot_format=config.output.plot_format,
        ic_results=ic_results,
        var_confidence=config.risk.var.confidence,
        var_horizon=config.risk.var.horizon,
    )

    print(report)
    try:
        html_path, metrics_path = _write_run_artifacts(
            results, ic_results, metadata, config,
            output_dir=run_dir, provenance=provenance,
        )
    except Exception as exc:
        logger.error("Failed to write the HTML report / metrics.json: %s", exc)
        return 1

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": config_hash,
        "created_at": datetime.now().isoformat(),
        "requested_provider": provenance.get("requested_provider"),
        "provider": provenance.get("effective_provider"),
        "synthetic": bool(provenance.get("synthetic")),
        "date_range": {
            "start": provenance.get("start_date"),
            "end": provenance.get("end_date"),
            "data_date": provenance.get("data_date"),
        },
        "metrics": {"performance": results.get("summary", {})},
        "artifacts": sorted(
            p.name for p in run_dir.iterdir()
            if p.is_file() and p.name != "manifest.json"
        ),
    }
    try:
        write_manifest(run_dir, manifest)
        published = publish_run_artifacts(results_base, run_dir, manifest)
    except Exception as exc:
        logger.error("Failed to write the run manifest: %s", exc)
        return 1

    logger.info("HTML report: %s", html_path)
    logger.info("Metrics: %s", metrics_path)
    logger.info("Results saved to: %s", run_dir)
    if not published:
        logger.warning(
            "Top-level results copies were skipped (another run holds the "
            "publish lock); use the run directory or latest_manifest.json.",
        )
    _save_config_version()
    logger.info("Pipeline complete.")
    return 0


def cmd_analyze(args) -> int:
    """Analyze existing results from output directory.

    Loads saved CSV files (daily_returns.csv, benchmark_returns.csv,
    weights_history.csv) and regenerates the full analysis dashboard.
    """
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        print(f"Error: Results directory '{results_dir}' does not exist.")
        print("Run 'python main.py run' first to generate results.")
        return 1

    # A run now writes into results/runs/<id>/ with a manifest; analyze the
    # published/latest run directory instead of mixing top-level CSVs from
    # different runs (issue S15). The legacy flat layout still works.
    from quant_platform.reporting.dashboard import resolve_run_dir

    run_dir = resolve_run_dir(results_dir)
    if run_dir != results_dir:
        print(f"Analyzing run: {run_dir}")

    returns_file = run_dir / "daily_returns.csv"
    bench_file = run_dir / "benchmark_returns.csv"
    weights_file = run_dir / "weights_history.csv"

    if not returns_file.exists():
        logger.error("daily_returns.csv not found in %s", results_dir)
        print(f"Error: 'daily_returns.csv' not found in {results_dir}")
        return 1

    import pandas as pd

    # Load strategy returns
    strategy_returns = pd.read_csv(returns_file, index_col=0, parse_dates=True).squeeze()
    strategy_returns.name = "strategy_return"
    logger.info("Loaded strategy returns: %d days", len(strategy_returns))

    # Load benchmark returns (optional)
    benchmark_returns = None
    if bench_file.exists():
        benchmark_returns = pd.read_csv(bench_file, index_col=0, parse_dates=True).squeeze()
        benchmark_returns.name = "benchmark"
        logger.info("Loaded benchmark returns: %d days", len(benchmark_returns))

    # Load weights history (optional)
    weights_history = {}
    if weights_file.exists():
        weights_df = pd.read_csv(weights_file, index_col=0, parse_dates=True)
        for date, row in weights_df.iterrows():
            weights_history[pd.Timestamp(date)] = row.dropna()
        logger.info("Loaded weights history: %d rebalance dates", len(weights_history))

    # Build results dict
    results = {
        "daily_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "weights_history": weights_history,
    }

    # --- Run full analysis ---
    logger.info("=" * 60)
    logger.info("Analyzing existing results...")
    logger.info("=" * 60)

    from quant_platform.reporting.dashboard import generate_dashboard

    report = generate_dashboard(
        results=results,
        metadata=None,
        output_dir=str(run_dir),
        save_plots=True,
        plot_format="png",
    )

    print(report)
    logger.info("Analysis complete. Charts regenerated in: %s", run_dir)
    return 0


def cmd_compare(args) -> int:
    """Compare multiple strategy configurations side by side."""
    import pandas as pd

    optimizers = args.optimizers.split(",") if args.optimizers else [
        "equal_weight", "mean_variance", "risk_parity"
    ]
    optimizers = [o.strip() for o in optimizers]

    config = load_config(args.config)
    # Smaller universe for faster comparison
    config.universe.n_stocks = min(config.universe.n_stocks, 300)

    # Load data once, reuse across optimizers
    prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)
    signal = _generate_signal(config, processed_factors, returns, prices=prices)

    results_table = []

    for opt in optimizers:
        logger.info("=" * 50)
        logger.info("Comparing optimizer: %s", opt)
        logger.info("=" * 50)

        try:
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked, optimizer_override=opt,
                gates=_execution_gates(metadata),
            )
            summary = bt_results["summary"]
            results_table.append({
                "Optimizer": opt,
                "Total Return %": f"{summary.get('total_return', 0)*100:.2f}",
                "Ann. Return %": f"{summary.get('annual_return', 0)*100:.2f}",
                "Ann. Vol %": f"{summary.get('annual_volatility', 0)*100:.2f}",
                "Sharpe": f"{summary.get('sharpe_ratio', 0):.2f}",
                "Sortino": f"{summary.get('sortino_ratio', 0):.2f}",
                "Max DD %": f"{summary.get('max_drawdown', 0)*100:.2f}",
                "Calmar": f"{summary.get('calmar_ratio', 0):.2f}",
                "Win Rate %": f"{summary.get('win_rate', 0)*100:.1f}",
                "IR": f"{summary.get('information_ratio', 0):.2f}",
            })
        except Exception as e:
            logger.error("Optimizer %s failed: %s", opt, e)
            results_table.append({"Optimizer": opt, "Error": str(e)[:80]})

    if results_table:
        df = pd.DataFrame(results_table)
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    return 0


def cmd_sweep(args) -> int:
    """Parameter sweep: grid search over key parameters."""
    import itertools

    import pandas as pd

    config = load_config(args.config)

    optimizers = args.optimizers.split(",") if args.optimizers else [
        "equal_weight", "mean_variance", "risk_parity"
    ]
    optimizers = [o.strip() for o in optimizers]
    frequencies = args.frequencies.split(",") if args.frequencies else [
        "monthly", "weekly"
    ]
    frequencies = [f.strip() for f in frequencies]
    n_stocks_list = [int(n) for n in args.n_stocks.split(",")] if args.n_stocks else [200]

    sweep_results = []
    total = len(optimizers) * len(frequencies) * len(n_stocks_list)
    count = 0

    for opt, freq, n_stocks in itertools.product(optimizers, frequencies, n_stocks_list):
        count += 1
        logger.info("[%d/%d] Sweep: opt=%s freq=%s n=%d", count, total, opt, freq, n_stocks)

        config.portfolio.optimizer = opt
        config.backtest.rebalance_frequency = freq
        config.universe.n_stocks = n_stocks

        try:
            prices, returns, benchmark, metadata, financials, turnover = _load_data(config, use_tushare=False)
            processed_factors, _ic_results, sector_map, fin_unstacked = _compute_factors(
                prices, returns, financials, metadata, turnover, config=config)
            signal = _generate_signal(config, processed_factors, returns, prices=prices)
            bt_results = _run_backtest(
                config, signal, prices, returns, benchmark,
                sector_map, fin_unstacked,
                optimizer_override=opt, frequency_override=freq,
                gates=_execution_gates(metadata),
            )

            summary = bt_results["summary"]
            sweep_results.append({
                "Optimizer": opt,
                "Frequency": freq,
                "N Stocks": n_stocks,
                "Sharpe": round(summary.get("sharpe_ratio", 0), 2),
                "Ann. Ret %": round(summary.get("annual_return", 0) * 100, 1),
                "Max DD %": round(summary.get("max_drawdown", 0) * 100, 1),
                "Sortino": round(summary.get("sortino_ratio", 0), 2),
                "Win Rate %": round(summary.get("win_rate", 0) * 100, 1),
            })
        except Exception as e:
            logger.error("Sweep [%s/%s/%d] failed: %s", opt, freq, n_stocks, e)
            sweep_results.append({
                "Optimizer": opt, "Frequency": freq, "N Stocks": n_stocks,
                "Sharpe": "ERR", "Error": str(e)[:60],
            })

    if sweep_results:
        df = pd.DataFrame(sweep_results)
        if "Sharpe" in df.columns:
            df = df.sort_values("Sharpe", ascending=False,
                                key=lambda x: pd.to_numeric(x, errors="coerce"))
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

    return 0


def _cmd_cache_refresh(args) -> int:
    """Recompute the current config's data stage and overwrite its entry (D33).

    Unlike ``run --force`` this stops after the data stage: no factors,
    backtest or reports. Useful right after a provider/data fix to make sure
    the next normal run cannot replay the pre-fix payload.
    """
    import yaml

    from quant_platform.utils.cache import PipelineCache

    config_path = _resolve_config_path(getattr(args, "config", None))
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    config = load_config(raw=raw_config)
    if getattr(args, "allow_synthetic_fallback", False):
        config.data.allow_synthetic_fallback = True

    use_baostock = bool(getattr(args, "use_baostock", False))
    resolution = _resolve_provider(config, use_baostock=use_baostock)
    if resolution.get("universe") is None or resolution.get("validated"):
        print(
            "Error: this run's data stage is not cacheable "
            + (
                "(validated multi-source runs cannot attest which sources "
                "contributed)."
                if resolution.get("validated")
                else "(the effective provider's symbol set is unknown)."
            )
            + " There is nothing to refresh.",
            file=sys.stderr,
        )
        return 1
    cache_key = _data_cache_key(config, resolution, use_baostock=use_baostock)
    cache = PipelineCache(getattr(args, "cache_dir", None) or ".quant_cache")
    incremental = bool(getattr(args, "incremental", False))
    _data_tuple, provenance, _hit = _load_data_stage(
        cache, cache_key, config, resolution,
        read_cache=False, write_cache=True, use_baostock=use_baostock,
        incremental=incremental,
    )
    freshness = _attach_freshness(provenance, config)
    print(
        f"Refreshed cache entry {DATA_STAGE} {cache_key}: "
        f"provider={provenance.get('effective_provider')} "
        f"symbols={provenance.get('symbol_count')} "
        f"data_date={provenance.get('data_date')}"
    )
    if incremental:
        print(_incremental_summary_line(provenance, freshness))
    if cache.save_failures:
        print(
            f"Warning: {cache.save_failures} cache save failure(s); "
            "the previous entry may still be in place. See logs.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_cache(args) -> int:
    """Manage pipeline cache."""
    from quant_platform.utils.cache import PipelineCache

    cache_dir = getattr(args, "cache_dir", None) or ".quant_cache"
    subcommand = getattr(args, "subcommand", None)
    if subcommand is None:
        # argparse `required=True` normally prevents this; keep a defensive
        # nonzero exit so a programmatic caller cannot get a false success.
        print(
            "Error: a cache subcommand is required "
            "(list, clear, prune, refresh).",
            file=sys.stderr,
        )
        return 2
    if subcommand == "refresh":
        return _cmd_cache_refresh(args)

    cache = PipelineCache(cache_dir)

    if subcommand == "list":
        entries = cache.list_cached()
        if entries:
            print(f"Cached stages ({len(entries)}):")
            for e in entries:
                print(f"  {e}")
        else:
            print("Cache is empty.")
    elif subcommand == "clear":
        count = cache.clear()
        print(f"Cleared {count} cached files.")
    elif subcommand == "prune":
        if getattr(args, "max_age_days", None) is not None:
            cache.max_age_days = float(args.max_age_days)
        if getattr(args, "max_entries", None) is not None:
            cache.max_entries = int(args.max_entries)
        removed = cache.prune()
        print(
            f"Pruned {removed} cache entr{'y' if removed == 1 else 'ies'} "
            f"(max_age_days={cache.max_age_days}, max_entries={cache.max_entries})."
        )

    return 0


def cmd_runs(args) -> int:
    """Research run registry: list results and their validity.

    Exists so that "which results does this bug fix invalidate?" is a query
    rather than an exercise in reading commit messages. A fix that changes
    numerical output marks the earlier runs affected; it does not delete them.
    """
    from quant_platform.lab.registry import DEFAULT_DB, RunStore

    store = RunStore(getattr(args, "db", None) or DEFAULT_DB)
    sub = getattr(args, "subcommand", None) or "list"

    if sub == "mark":
        ok = store.mark_validity(args.run_id, args.state, args.reason or "")
        if not ok:
            print(f"Unknown run: {args.run_id}")
            return 1
        print(
            f"{args.run_id}: validity={args.state}"
            + (f"  affected_by={args.reason}" if args.reason else "")
        )
        return 0

    if sub == "import":
        inserted = store.import_runs(args.path)
        print(f"Imported {inserted} new run(s) from {args.path}")
        return 0

    if sub == "export":
        count = store.export_runs(args.path, validity=getattr(args, "validity", None))
        print(f"Exported {count} run(s) to {args.path}")
        return 0

    if sub == "affected":
        rows = store.list_affected(args.reason)
        title = "Runs affected" + (f" by {args.reason}" if args.reason else "")
    else:
        rows = store.list_runs(
            getattr(args, "limit", 50),
            validity=getattr(args, "validity", None),
        )
        title = "Recent runs"

    if not rows:
        print(f"{title}: none")
        return 0

    print(f"{title} ({len(rows)}):")
    print(f"  {'run_id':36s} {'validity':12s} {'factor':18s} affected_by")
    for r in rows:
        print(
            f"  {r['run_id']:36s} {r.get('validity') or 'valid':12s} "
            f"{(r.get('factor') or '-'):18s} {r.get('affected_by') or '[]'}"
        )
    return 0


def cmd_web(args) -> int:
    """Start the web server (FastAPI + Vue frontend)."""
    import uvicorn

    from quant_platform.app import create_app

    app = create_app(serve_frontend=not args.no_frontend)
    mode = "API + Frontend" if not args.no_frontend else "API only"
    print("Quant Platform Web Server starting...")
    print(f"  Mode: {mode}")
    print(f"  API:  http://{args.host}:{args.port}/api/docs")
    if not args.no_frontend:
        from quant_platform.app import _DIST_DIR
        if _DIST_DIR.exists():
            print(f"  UI:   http://{args.host}:{args.port}/")
        else:
            print("  UI:   Frontend not built. Run: cd frontend && npm run build")
    import logging as _logging

    from quant_platform.utils.logging import resolve_level

    uvicorn_log_level = _logging.getLevelName(resolve_level()).lower()
    uvicorn.run(app, host=args.host, port=args.port, log_level=uvicorn_log_level)
    return 0


def cmd_ml(args) -> int:
    """ML alpha signal operations."""
    config = load_config(_resolve_config_path(args.config))

    if args.subcommand == "train":
        from quant_platform.alpha.ml_signal import (
            InsufficientTrainingData,
            MLSignalConfig,
            MLSignalGenerator,
        )

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(
            model_type=args.model,
            n_splits=args.splits,
        )
        gen = MLSignalGenerator(config=ml_config)
        try:
            perf = gen.train(processed_factors, returns)
        except InsufficientTrainingData as exc:
            print(f"ML training failed: {exc}")
            return 1

        if perf.n_train_samples <= 0:
            print(
                "ML training produced no samples; refusing to report a "
                "zero-IC model as a success."
            )
            return 1

        # Producer for the Grafana "ML Model Performance" panel.
        try:
            import math

            from quant_platform.utils.metrics import get_metrics

            metrics = get_metrics()
            if math.isfinite(float(perf.test_ic)):
                metrics.gauge(
                    "ml_model_test_ic",
                    labels={"model_type": args.model},
                    help="Test-set IC of the trained ML model",
                ).set(float(perf.test_ic))
            if math.isfinite(float(perf.test_icir)):
                metrics.gauge(
                    "ml_model_test_icir",
                    labels={"model_type": args.model},
                    help="Test-set ICIR of the trained ML model",
                ).set(float(perf.test_icir))
        except Exception as exc:
            logger.debug("ML metric publish failed: %s", exc)

        print("=" * 60)
        print(f"  ML MODEL TRAINING — {args.model.upper()}")
        print("=" * 60)
        print(f"  Test IC:        {perf.test_ic:.4f}")
        print(f"  Test ICIR:      {perf.test_icir:.4f}")
        print(f"  Train IC:       {perf.train_ic:.4f}")
        print(f"  Train samples:  {perf.n_train_samples}")
        print(f"  Date:           {perf.date}")
        print("-" * 60)
        print("  Feature Importance:")
        for name, imp in sorted(perf.feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name:<20} {imp:.4f}")
        print("=" * 60)
        return 0

    elif args.subcommand == "signal":
        from quant_platform.alpha.ml_signal import (
            InsufficientTrainingData,
            MLSignalConfig,
            MLSignalGenerator,
        )

        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)

        ml_config = MLSignalConfig(model_type=args.model)
        gen = MLSignalGenerator(config=ml_config)
        try:
            signal = gen.generate(processed_factors, returns)
        except InsufficientTrainingData as exc:
            print(f"ML signal failed: {exc}")
            return 1

        # Show top/bottom 10 stocks by signal
        last_signal = signal.iloc[-1].dropna().sort_values()
        if last_signal.empty:
            print(
                "ML signal is empty on the last date; refusing to print an "
                "empty Top/Bottom list as a success."
            )
            return 1
        print("=" * 60)
        print(f"  ML SIGNAL — {args.model.upper()} (last date)")
        print("=" * 60)
        print("  Top 10 (most bullish):")
        for stock, val in last_signal.tail(10).items():
            print(f"    {stock:<15} {val:+.4f}")
        print("  Bottom 10 (most bearish):")
        for stock, val in last_signal.head(10).items():
            print(f"    {stock:<15} {val:+.4f}")
        print("=" * 60)
        return 0

    else:
        print("Usage: python main.py ml train [--model lightgbm|xgboost]")
        print("       python main.py ml signal [--model lightgbm|xgboost]")
        return 1


def _cmd_research_validate(args) -> int:
    """Run Research Validation Sprint."""
    from quant_platform.strategy.research_validation import ResearchValidator

    n_stocks = args.n_stocks if args.n_stocks > 0 else 200
    validator = ResearchValidator(
        n_stocks=n_stocks,
        timeout_seconds=args.timeout,
    )

    if args.mode == "quick":
        report = validator.run_quick(
            max_factors=args.max_factors,
            max_folds=args.max_folds,
            n_symbols=n_stocks,
        )
    else:
        report = validator.run_full(max_folds=args.max_folds)

    print(report.print_report())
    return 0 if report.summary.get("overall") != "needs_review" else 1


def cmd_research(args) -> int:
    """LLM research agent and validation operations."""
    if args.subcommand == "validate":
        return _cmd_research_validate(args)
    if args.subcommand == "report":
        import json
        from pathlib import Path

        from quant_platform.agent.research_agent import ResearchAgent

        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return 1

        # Try to load backtest results. `run` writes metrics.json; older
        # result directories only have the CSV, so fall back to the measured
        # daily returns instead of dead-ending.
        metrics_file = results_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
        else:
            returns_file = results_dir / "daily_returns.csv"
            if not returns_file.exists():
                print(
                    "No metrics.json or daily_returns.csv found. "
                    "Run 'python main.py run' first."
                )
                return 1
            import pandas as pd

            from quant_platform.backtest.metrics import all_metrics

            frame = pd.read_csv(returns_file, index_col=0)
            frame.index = pd.to_datetime(frame.index)
            daily = frame.iloc[:, 0].astype(float)
            metrics = {
                "performance": all_metrics(daily),
                "daily_returns": daily.tolist(),
                "factor_contributions": {},
            }
            print(
                "metrics.json not found; computed performance from "
                "daily_returns.csv (factor attribution needs a full run)."
            )

        agent = ResearchAgent(mode="keyword")

        # Generate attribution summary
        factor_contrib = metrics.get("factor_contributions", {})
        if factor_contrib:
            import pandas as pd
            returns_data = metrics.get("daily_returns", [])
            daily_returns = pd.Series(returns_data) if returns_data else None
            summary = agent.summarize_attribution(factor_contrib, daily_returns)
            print(summary)
        else:
            performance = metrics.get("performance") or {}
            print("No factor contributions in results. Measured performance:")
            printed = False
            for k, v in performance.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    print(f"  {k}: {v}")
                    printed = True
            if not printed:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        print(f"  {k}: {v}")

        return 0

    else:
        print("Usage: python main.py research report [--results-dir ./results]")
        return 1


def cmd_screen(args) -> int:
    """Run factor screener: boolean-rule stock selection.

    Computes factors for the configured universe, then applies user-defined
    screen rules. Prints qualifying stocks with their factor values.
    """
    import json
    from datetime import datetime

    import pandas as pd

    from quant_platform.portfolio.screener import (
        FactorScreener,
        ScreenRule,
    )

    config = load_config(_resolve_config_path(args.config))

    # Parse rules from CLI arg or use config defaults
    rules: list[ScreenRule] = []
    if args.rules:
        for rule_str in args.rules.split(","):
            rule_str = rule_str.strip()
            if not rule_str:
                continue
            # Format: factor_name=operator:value  e.g. pe_ratio=lt:30
            if "=" in rule_str:
                factor_expr = rule_str.split("=", 1)
                factor_name = factor_expr[0].strip()
                rest = factor_expr[1].strip()
                if ":" in rest:
                    op, val_str = rest.split(":", 1)
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
                    rules.append(ScreenRule(factor=factor_name, operator=op, value=val))
                else:
                    logger.warning("Invalid rule format (expected op:val): %s", rule_str)
            else:
                logger.warning("Invalid rule format (expected factor=op:val): %s", rule_str)
    else:
        # Use config rules
        if not config.screener.enabled:
            print(
                "  Note: screener.enabled=false in the config; 'screen' was "
                "invoked explicitly, so the configured rules are used."
            )
        for r in config.screener.rules:
            rules.append(ScreenRule(factor=r.factor, operator=r.operator, value=r.value))

    if not rules:
        print("No screen rules defined. Use --rules or configure screener.rules in config.")
        return 1

    # Load data & compute factors
    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")

    print("Computing factors...")
    _processed_factors, ic_results, sector_map, fin_unstacked, raw_factors = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config,
        return_raw=True,
    )

    # Screen on raw factor values, not the standardized/neutralized panels:
    # `pe_ratio < 30` means a PE of 30, while the processed panel is a
    # cross-sectional z-score (roughly -3..3) that passes every stock.
    screen_factors = raw_factors
    print("  Screen values: raw factor units (before winsorize/standardize)")

    # Run screener
    screener = FactorScreener({
        "enabled": True,
        "rules": [{"factor": r.factor, "operator": r.operator, "value": r.value}
                  for r in rules],
        "logic": args.logic or config.screener.logic,
        "min_stocks": args.min_stocks or config.screener.min_stocks,
        "max_stocks": args.max_stocks or config.screener.max_stocks,
    })
    qualifiers = screener.screen(screen_factors, rules)

    # Build output table
    if qualifiers:
        # Get latest factor values for qualifying stocks
        last_date = next(iter(screen_factors.values())).index[-1]
        print(f"\n{'='*80}")
        print(f"  SCREENER RESULTS — {last_date}")
        print(f"{'='*80}")
        print(f"  Rules: {', '.join(str(r.factor) + ' ' + r.operator + ' ' + str(r.value) for r in rules)}")
        print(f"  Logic: {args.logic or config.screener.logic}")
        print(f"  Passing: {len(qualifiers)} / {len(next(iter(screen_factors.values())).columns)} stocks")
        print(f"{'='*80}\n")

        # Build detail DataFrame
        detail_rows = []
        for asset in qualifiers:
            row = {"code": asset}
            for r in rules:
                if r.factor in screen_factors:
                    df = screen_factors[r.factor]
                    if last_date in df.index:
                        row[r.factor] = round(df.loc[last_date, asset], 4)
            detail_rows.append(row)

        df_out = pd.DataFrame(detail_rows)

        # Print table
        pd.set_option("display.max_rows", 120)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(df_out.to_string(index=False))

        # Save to CSV if requested
        if args.output:
            df_out.to_csv(args.output, index=False)
            print(f"\n  Saved to: {args.output}")

        # Print summary stats
        print(f"\n  {'─'*60}")
        print(f"  Total qualifying stocks: {len(qualifiers)}")
        for r in rules:
            if r.factor in screen_factors:
                df = screen_factors[r.factor]
                if last_date in df.index:
                    vals = df.loc[last_date, qualifiers].dropna()
                    print(f"  {r.factor}: mean={vals.mean():.4f}  "
                          f"min={vals.min():.4f}  max={vals.max():.4f}")
        print(f"  {'─'*60}\n")
    else:
        print("\n  No stocks matched the screen rules.\n")
        return 1

    return 0


def cmd_execute(args) -> int:
    """Execute full pipeline: factors -> signal -> orders -> fills -> positions."""
    config = load_config(_resolve_config_path(args.config))

    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    try:
        # Decisive run (orders/fills): stale data fails closed (D32).
        _enforce_freshness(
            _LAST_DATA_PROVENANCE, config,
            context="decisive execute", fail_closed=True,
        )
    except DataStalenessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"  Data: {len(prices)} days, {len(prices.columns)} assets")

    print("Computing factors...")
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config,
    )

    print("Generating signal...")
    signal = _generate_signal(config, processed_factors, returns, prices=prices, volume=turnover)

    print("Setting up execution engine...")
    from quant_platform.execution.engine import ExecutionEngine, OrderSide
    from quant_platform.strategy.multi_strategy import MultiStrategyManager, StrategyConfig
    from quant_platform.strategy.portfolio_orchestrator import PortfolioOrchestrator

    ms = MultiStrategyManager(total_capital=1_000_000)
    strat_id = ms.add_strategy(StrategyConfig(
        name="test_strat", optimizer=config.portfolio.optimizer,
        allocation_pct=1.0, is_active=True,
    ))
    orchestrator = PortfolioOrchestrator(ms)

    # Get latest **raw** prices. The `prices` first element of `_load_data`
    # is the adjusted research panel; filling or marking a paper book at hfq
    # levels would misstate cash and shares (D15).
    raw_panel = None
    if metadata is not None:
        raw_panel = (
            (getattr(metadata, "attrs", {}) or {})
            .get("raw_prices", {})
            .get("close")
        )
    if raw_panel is None or raw_panel.empty:
        print(
            "Error: the configured provider supplied no raw CNY price panel; "
            "refusing to execute at adjusted levels. Use a provider that "
            "returns raw OHLCV plus adj_factor.",
            file=sys.stderr,
        )
        return 1
    last_date_ts = prices.index[-1]
    last_prices = (
        raw_panel.loc[last_date_ts].dropna().to_dict()
        if last_date_ts in raw_panel.index
        else raw_panel.iloc[-1].dropna().to_dict()
    )
    if not last_prices:
        print(
            f"Error: no finite raw CNY price for {last_date_ts} in the "
            "provider panel; refusing to execute against an empty book of "
            "tradeable prices.",
            file=sys.stderr,
        )
        return 1
    orchestrator._last_prices.update(last_prices)

    last_date = str(prices.index[-1])[:10]
    print(f"Processing signal for {last_date}...")
    orchestrator.on_signal(last_date, signal, strategy_id=strat_id)

    print("Executing rebalance...")
    orders = orchestrator.rebalance()
    print(f"  Created {len(orders)} orders")

    print("Processing fills...")
    orchestrator.process_fills(last_prices)

    summary = orchestrator.portfolio_summary()
    print()
    print("=" * 70)
    print("  EXECUTION RESULT")
    print("=" * 70)
    print(f"  Date:          {last_date}")
    print(f"  Positions:     {summary['n_positions']}")
    print(f"  Cash:          {summary['cash_available']:,.2f}")
    print(f"  Position Val:  {summary['positions_value']:,.2f}")
    print(f"  Unrealized PnL:{summary['unrealized_pnl']:+,.2f}")
    print(f"  Realized PnL:  {summary['realized_pnl']:+,.2f}")
    print(f"  Total PnL:     {summary['total_pnl']:+,.2f}")
    print(f"  Alerts:        {summary['alerts']}")
    print("-" * 70)
    for p in summary['positions'][:10]:
        print(f"  {p['ticker']:<8} {p['quantity']:>5} @ {p['avg_cost']:<8.2f} = {p['market_value']:>10.2f}  PnL:{p['unrealized_pnl']:>+8.2f}")
    if len(summary['positions']) > 10:
        print(f"  ... and {len(summary['positions']) - 10} more")
    print("=" * 70)
    return 0

def cmd_config(args) -> int:
    """Manage configuration versions: list, show, diff, rollback."""
    from quant_platform.utils.version_manager import VersionManager

    vm = VersionManager()

    if args.subcommand == "list":
        versions = vm.list()
        if not versions:
            print("No config versions saved yet.")
            print("  Run 'python main.py run' to auto-save the first version.")
            return 0

        print(f"{'ID':<6} {'Timestamp':<22} {'Description'}")
        print("-" * 80)
        for v in versions:
            desc = (v.description[:60] + "..") if len(v.description) > 62 else v.description
            print(f"{v.id:<6} {v.timestamp:<22} {desc}")
        print(f"\n{len(versions)} versions total.")

    elif args.subcommand == "show":
        try:
            config = vm.show(args.version)
            import yaml
            print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "diff":
        try:
            diff_text = vm.diff(args.v1, args.v2)
            if diff_text:
                print(diff_text)
            else:
                print("Configs are identical.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "rollback":
        target = args.target
        try:
            vm.rollback(args.version, target_path=target)
            print(f"Config rolled back to {args.version}.")
            if target:
                print(f"  Target: {target}")
            else:
                print("  Target: config/default.yaml")
            print("  Run 'python main.py run' to verify.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1

    elif args.subcommand == "delete":
        confirm = args.force or input(f"Delete {args.version}? [y/N] ").lower() == "y"
        if confirm:
            vm.delete(args.version)
            print(f"Deleted {args.version}.")
        else:
            print("Cancelled.")

    elif args.subcommand == "prune":
        keep = int(getattr(args, "keep", 20) or 20)
        if keep < 1:
            print("Error: --keep must be >= 1.", file=sys.stderr)
            return 2
        removed = vm.prune(
            keep=keep,
            max_age_days=getattr(args, "max_age_days", None),
        )
        if removed:
            print(f"Pruned {len(removed)} version(s): {', '.join(removed)}")
        else:
            print("Nothing to prune.")

    return 0


def cmd_check_lookahead(args) -> int:
    """Run lookahead bias detection on the factor pipeline."""
    config = load_config(_resolve_config_path(args.config))
    from quant_platform.risk.lookahead_detector import LookaheadDetector

    print("Loading data...")
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )

    detector = LookaheadDetector(
        threshold=args.threshold,
        max_check_dates=args.max_dates,
    )
    result = detector.detect(prices, returns, financials, metadata, config)
    detector.print_report(result)

    suggestions = detector.suggest_fixes(result)
    if suggestions:
        print("Suggested fixes:")
        for s in suggestions:
            print(f"  - {s}")
    return 1 if result["has_bias"] else 0


def cmd_profile(args) -> int:
    """Profile pipeline performance — shows time per stage."""
    import time

    config = load_config(_resolve_config_path(args.config))

    timings = {}
    total_start = time.perf_counter()

    # Stage 1: Data
    t0 = time.perf_counter()
    prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
    timings["1_data"] = time.perf_counter() - t0

    # Stage 2: Factors
    t0 = time.perf_counter()
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)
    timings["2_factors"] = time.perf_counter() - t0

    # Stage 3: Alpha
    t0 = time.perf_counter()
    signal = _generate_signal(config, processed_factors, returns, prices=prices)
    timings["3_alpha"] = time.perf_counter() - t0

    # Stage 4: Backtest
    t0 = time.perf_counter()
    _run_backtest(config, signal, prices, returns, benchmark, sector_map, fin_unstacked,
                  gates=_execution_gates(metadata))
    timings["4_backtest"] = time.perf_counter() - t0

    total = time.perf_counter() - total_start

    print("=" * 60)
    print("  PIPELINE PERFORMANCE PROFILE")
    print("=" * 60)
    for stage, duration in timings.items():
        pct = duration / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {stage:<15} {duration:>6.2f}s  {pct:>5.1f}%  {bar}")
    print("-" * 60)
    print(f"  {'TOTAL':<15} {total:>6.2f}s")
    print("=" * 60)
    return 0


def _make_signal_history_provider(config):
    """Daily raw closes for a real-broker LiveRunner warm-up.

    A live runner sees one quote per cycle; without a history preload its
    3/6-month momentum factors stay NaN forever and the shared composite
    scores nothing. Baostock is the free history source (raw ``close``);
    failures return an empty mapping so the runner fails closed rather than
    trading on invented scores.
    """

    def _fetch(codes, sessions):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        from quant_platform.data.providers.baostock_provider import (
            BaostockDataProvider,
        )
        from quant_platform.shared.symbols import normalize_symbol

        end = datetime.now(ZoneInfo("Asia/Shanghai")).date()
        start = end - timedelta(days=int(sessions * 2.2) + 30)
        try:
            provider = BaostockDataProvider(cache_enabled=True)
            panel = provider.get_prices(start.isoformat(), end.isoformat())
        except Exception as exc:
            logger.warning("Signal history fetch failed: %s", exc)
            return {}
        if panel is None or panel.empty or "close" not in panel.columns:
            return {}
        close = panel["close"].unstack("asset")
        history: dict[str, list[float]] = {}
        for code in codes:
            canonical = normalize_symbol(code)
            series = None
            for candidate in (canonical, str(code)):
                if candidate in close.columns:
                    series = close[candidate]
                    break
            if series is None:
                continue
            values = [float(v) for v in series.dropna().tail(sessions)]
            if len(values) >= 20:
                history[code] = values
        return history

    return _fetch


def cmd_trade(args) -> int:
    """Execute live trading via LiveRunner (Paper or QMT).

    Exit codes: 2 for a usage error (e.g. ``--days 0``), 1 when the broker
    fails to connect, when every cycle failed, or when nothing filled, and 0
    only for a session that actually traded.
    """
    from quant_platform.trading.live_runner import LiveRunner

    broker_type = args.broker
    days = int(args.days)
    if days < 1:
        print("Error: --days must be >= 1.", file=sys.stderr)
        return 2

    universe = [c.strip() for c in args.universe.split(",")] if args.universe else [
        "600519", "000858", "000001", "002001", "300750",
    ]

    config = load_config(_resolve_config_path(getattr(args, "config", None)))

    broker_kwargs: dict = {}
    if broker_type in ("qmt", "qmt_sim", "qmt_live"):
        # Account/server/password/mode come from execution.qmt (password
        # env-first); `--broker qmt` no longer silently means "simulated".
        broker_kwargs = _resolve_qmt_kwargs(config)
        if broker_type == "qmt_live":
            broker_kwargs["mode"] = "live"
        elif broker_type == "qmt_sim":
            broker_kwargs["mode"] = "sim"
        elif broker_type == "qmt":
            broker_kwargs["mode"] = config.execution.qmt.mode
        mode = broker_kwargs.get("mode", "sim")
        print(f"  QMT mode:  {mode}  Account: {broker_kwargs.get('account', '') or '(unset)'}")

    print("=" * 60)
    print("  LIVE TRADING RUNNER")
    print("=" * 60)
    print(f"  Broker:    {broker_type}")
    print(f"  Universe:  {len(universe)} stocks")
    print(f"  Days:      {days}")
    print(f"  Capital:   {args.cash:,.0f} CNY")
    print(f"  DualTrack: {not args.no_dual_track}")
    print("=" * 60)

    try:
        runner = LiveRunner(
            broker_type=broker_type,
            initial_cash=args.cash,
            dual_track=not args.no_dual_track,
            **broker_kwargs,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    runner.set_universe(universe)
    # Real brokers see one quote per cycle; warm up the shared 7-factor
    # signal from daily history (simulated/paper runs seed their own).
    # `getattr` keeps injected test doubles without the new method working.
    if broker_type in ("qmt", "qmt_sim", "qmt_live", "xtp"):
        setter = getattr(runner, "set_history_provider", None)
        if callable(setter):
            setter(_make_signal_history_provider(config))

    def _cleanup_runner():
        """Flush and close the runner's store on normal exit or SIGTERM."""
        store = getattr(runner, "_store", None)
        if store is None:
            return
        try:
            store.checkpoint()
        except Exception as exc:
            logger.warning("Store checkpoint on shutdown failed: %s", exc)
        try:
            store.close()
        except Exception as exc:
            logger.warning("Store close on shutdown failed: %s", exc)

    # S16: SIGTERM/SIGINT unwinds through LiveRunner.run()'s finally block
    # (which completes the session row), then this cleanup flushes the WAL
    # and closes the store. Installed only from this CLI entry point.
    cleanup_once = _install_shutdown_handlers(_cleanup_runner)
    run_kwargs: dict = {"days": days, "seed": args.seed}
    # Simulated/paper runs generate a deterministic split/dividend demo feed
    # consistent with their synthetic price path; real brokers settle
    # corporate actions on the broker side and never get a local feed (the
    # runner refuses one).  `inspect` keeps injected/test-double runners
    # whose `run` predates the keyword working.
    try:
        accepts_actions = (
            "simulate_corporate_actions"
            in inspect.signature(runner.run).parameters
        )
    except (TypeError, ValueError):
        accepts_actions = False
    if accepts_actions and broker_type in ("simulated", "paper"):
        run_kwargs["simulate_corporate_actions"] = True
    try:
        report = runner.run(**run_kwargs)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup_once()
    d = report.to_dict()

    print(f"\nSession: {d['session_id']}")
    print(f"  Broker:         {d.get('broker_type', broker_type)}")
    print(f"  Status:         {d.get('status', 'completed')}")
    print(f"  Days traded:    {d['days_traded']}")
    print(f"  Failed cycles:  {d.get('failed_cycles', 0)}")
    print(f"  Total orders:   {d['total_orders']}")
    print(f"  Total fills:    {d['total_fills']}")
    print(f"  Initial:        {d['initial_capital']:,.0f} CNY")
    print(f"  Final:          {d['final_value']:,.0f} CNY")
    print(f"  Return:         {d['total_return_pct']:.2f}%")
    print(f"  Annualized:     {d['annualized_return_pct']:.2f}%")
    print(f"  Sharpe:         {d['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:   {d['max_drawdown_pct']:.2f}%")
    print(f"  Avg Volume:     {d['avg_daily_volume']:,.0f} CNY")
    ca = d.get("corporate_actions_applied") or {}
    print(
        f"  Corp actions:   {ca.get('count', 0)} applied"
        f" (last ex-date {ca.get('last_ex_date') or '-'}, "
        f"source {ca.get('source') or '-'})"
    )

    # Never report success for a session that did not trade.
    if d.get("failed_cycles", 0) > 0:
        print(
            f"Error: {d['failed_cycles']} trading cycle(s) failed; "
            "the session did not complete cleanly.",
            file=sys.stderr,
        )
        return 1
    if d["total_orders"] == 0 or d["total_fills"] == 0:
        print(
            "Error: no orders filled"
            + (" (every order was rejected)" if d["total_orders"] else "")
            + "; refusing to report success.",
            file=sys.stderr,
        )
        return 1
    if d.get("status") != "completed":
        print(
            f"Error: session status is '{d.get('status')}', not completed.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_data(args) -> int:
    """Data maintenance commands (S10)."""
    if getattr(args, "subcommand", "") != "retention":
        print("Error: unknown data subcommand", file=sys.stderr)
        return 2

    from quant_platform.core.store import RetentionPolicy, Store

    policy = RetentionPolicy(
        orders_days=args.orders_days,
        trades_days=args.trades_days,
        pnl_days=args.pnl_days,
        signals_days=args.signals_days,
        sessions_days=args.sessions_days,
        events_days=args.events_days,
        nav_days=args.nav_days,
    )
    store = Store(args.db)
    try:
        result = store.run_retention(
            policy=policy,
            archive_dir=args.archive_dir,
            vacuum=not args.no_vacuum,
        )
    except Exception as exc:
        print(f"Error: retention failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print("Retention complete")
    for table, entry in result["tables"].items():
        if not entry.get("enabled"):
            print(f"  {table:<14} disabled")
            continue
        print(
            f"  {table:<14} archived={entry['archived']:<6} "
            f"deleted={entry['deleted']:<6} cutoff={entry['cutoff']}"
        )
        if entry.get("archive_path"):
            print(f"                 archive: {entry['archive_path']}")
    print(
        f"  db size: {result['db_size_bytes']:,} bytes; "
        f"checkpointed={result['checkpointed']} vacuumed={result['vacuumed']}"
    )
    return 0


def cmd_factor(args) -> int:
    """Unified research entry: run any factor offline or through the real runner."""
    if args.factor == "list":
        try:
            from quant_platform.lab.runs.synthetic_factor_run import available_factors
            for name in available_factors():
                print(name)
        except Exception as e:
            logger.error("Failed to list factors: %s", e)
            return 1
        return 0

    try:
        if args.synthetic:
            from quant_platform.lab.runs.synthetic_factor_run import run_synthetic_factor
            run_synthetic_factor(
                args.factor,
                n_stocks=getattr(args, "n_stocks", 5),
                n_days=getattr(args, "n_days", 500),
            )
        else:
            from quant_platform.run_factor import run_factor_report
            run_factor_report(args.factor)
    except Exception as e:
        logger.error("Factor run failed: %s", e)
        return 1
    return 0


def cmd_factor_store(args) -> int:
    """Factor Research Store operations."""
    from quant_platform.factors.registry import get_registry
    from quant_platform.factors.store import FactorDefinition, FactorResearchStore

    store = FactorResearchStore()

    if args.subcommand == "stats":
        stats = store.get_stats()
        print("\n=== Factor Research Store ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    elif args.subcommand == "rank":
        ranking = store.get_factor_ranking()
        print("\n=== Factor Ranking by Health Score ===")
        print(f"  {'Rank':<4} {'Factor':<25} {'IC':<10} {'ICIR':<8} {'Coverage':<10} {'Health':<8}")
        print(f"  {'-'*4} {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
        for i, f in enumerate(ranking[:getattr(args, 'limit', 20)]):
            print(f"  {i+1:<4} {f['factor_name']:<25} {f['ic_mean']:<10.6f} {f['icir']:<8.4f} {f['coverage']:<10.4f} {f['health_score']:<8.4f}")
        return 0

    elif args.subcommand == "register-all":
        # Register all factors into registry first
        from quant_platform.factors.fundamental import register_all as register_fundamental
        from quant_platform.factors.technical import register_all as register_technical
        register_technical()
        register_fundamental()
        registry = get_registry()
        factor_names = registry.list_all()
        count = 0
        for name in factor_names:
            cls = registry.get(name)
            inst = cls()
            try:
                store.save_definition(FactorDefinition(
                    name=inst.name,
                    category=inst.category.value,
                    description=inst.__doc__ or "",
                    higher_is_better=True,
                    params=inst.params,
                ))
                count += 1
            except Exception as e:
                logger.warning("Failed to register %s: %s", name, e)
        print(f"Registered {count}/{len(factor_names)} factors")
        return 0

    elif args.subcommand == "history":
        factor = getattr(args, 'factor', '')
        if not factor:
            print("Please specify --factor")
            return 1
        limit = getattr(args, 'limit', 50)
        evals = store.get_evaluation_history(factor, limit=limit)
        print(f"\n=== Evaluation History: {factor} ===")
        print(f"  Records: {len(evals)}")
        if evals:
            print(f"  {'Date':<12} {'Rank IC':<10} {'ICIR':<8} {'Coverage':<10} {'N':<6}")
            print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")
            for e in evals[:20]:
                print(f"  {e.signal_date:<12} {e.rank_ic:<10.4f} {e.icir:<8.4f} {e.coverage:<10.4f} {e.n_assets:<6}")
        return 0

    elif args.subcommand == "walkforward-summary":
        factor = getattr(args, 'factor', '')
        if not factor:
            print("Please specify --factor")
            return 1
        summary = store.get_walk_forward_summary(factor)
        if summary:
            print(f"\n=== Walk-Forward Summary: {factor} ===")
            for key, value in summary.items():
                print(f"  {key}: {value}")
        else:
            print(f"No walk-forward data for {factor}")
        return 0

    elif args.subcommand == "clear":
        confirm = input("Clear ALL factor store data? (y/N): ")
        if confirm.lower() == 'y':
            store.clear()
            print("Factor store cleared")
        return 0

    else:
        print("Unknown subcommand. Use: stats, rank, register-all, history, walkforward-summary, clear")
        return 1


def cmd_gate(args) -> int:
    """Run Strategy Evaluation Gates on the current pipeline results.

    Integrates IC results, walk-forward, backtest metrics, and data coverage
    into a unified PASS/WARNING/FAIL/REJECTED gate report.
    """
    from quant_platform.strategy.gates import (
        GateConfig,
        GateRunner,
        count_assets_with_data,
    )

    config = load_config(_resolve_config_path(args.config))

    # Build gate config with optional overrides
    gate_config = GateConfig()
    if args.min_ic is not None:
        gate_config.minimum_ic = args.min_ic
    if args.max_dd is not None:
        gate_config.maximum_drawdown = args.max_dd

    # Load data
    prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
    # Coverage is measured from the panel, not assumed: passing the same
    # number for both sides made the coverage gate compare a value with itself.
    n_assets_total, n_assets_with_data = count_assets_with_data(prices)

    # Compute factors and IC results
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config)

    # Estimate factor params count
    from quant_platform.factors.fundamental import register_all as register_fundamental
    from quant_platform.factors.registry import get_registry
    from quant_platform.factors.technical import register_all as register_technical
    register_technical()
    register_fundamental()
    n_params = 0
    for name in get_registry().list_all():
        try:
            inst = get_registry().get(name)()
            n_params += len(inst.params)
        except Exception:
            pass
    n_factors = len(processed_factors)

    # Check for saved walk-forward results
    wf_result = None
    try:
        import glob
        from pathlib import Path
        wf_files = glob.glob(str(Path(config.output.results_dir) / "walkforward*.json"))
        if wf_files:
            import json
            with open(wf_files[-1]) as f:
                wf_result = json.load(f)
    except Exception:
        pass

    # The command's docstring promised backtest metrics, but it never loaded
    # or computed any, so the drawdown and trading-cost gates were permanently
    # SKIPPED. Run the backtest so those gates see real numbers.
    backtest_summary = None
    try:
        signal = _generate_signal(config, processed_factors, returns, prices=prices)
        bt_results = _run_backtest(
            config, signal, prices, returns, benchmark,
            sector_map, fin_unstacked,
            gates=_execution_gates(metadata),
        )
        backtest_summary = bt_results.get("summary")
    except Exception as exc:
        logger.warning("Gate backtest failed; drawdown/cost gates will be skipped: %s", exc)

    # Run gates
    runner = GateRunner(gate_config)
    report = runner.run(
        strategy_name=args.strategy,
        ic_results=ic_results,
        wf_results=wf_result,
        backtest_summary=backtest_summary,
        n_factors=n_factors,
        n_params=n_params,
        n_assets_total=n_assets_total,
        n_assets_with_data=n_assets_with_data,
    )

    print(report.print_report())

    # Return non-zero for FAIL/REJECTED
    if report.overall_status in ("FAIL", "REJECTED"):
        return 1
    return 0


def cmd_strategy(args) -> int:
    """Manage Strategy DSL definitions."""
    from quant_platform.factors.fundamental import register_all as register_fundamental
    from quant_platform.factors.technical import register_all as register_technical
    from quant_platform.strategy.dsl import (
        StrategyDefinition,
        dsl_to_config_overrides,
        validate_strategy,
    )
    from quant_platform.strategy.registry import StrategyRegistry

    registry = StrategyRegistry()

    if args.subcommand == "list":
        strategies = registry.list_strategies()
        print(f"\n=== Registered Strategies ({len(strategies)}) ===")
        print(f"  {'Name':<30} {'Version':<10} {'Created':<20}")
        print(f"  {'-'*30} {'-'*10} {'-'*20}")
        for s in strategies:
            print(f"  {s['name']:<30} {s['version']:<10} {s.get('created_at',''):<20}")
        return 0

    elif args.subcommand == "show":
        name = args.name
        if not name:
            print("Please specify --name")
            return 1
        version = args.version if args.version else ""
        s = registry.get(name, version)
        if s is None:
            print(f"Strategy '{name}' not found")
            return 1
        import yaml
        print(f"\n=== {s.name} v{s.version} ===")
        print(yaml.dump(s.to_dict(), default_flow_style=False, allow_unicode=True))
        return 0

    elif args.subcommand == "validate":
        s = StrategyDefinition.from_yaml(args.file)
        register_technical()
        register_fundamental()
        result = validate_strategy(s, check_factors_exist=True)
        print(f"\n=== Validation: {s.name} v{s.version} ===")
        if result.valid:
            print("  Status: VALID")
        else:
            print("  Status: INVALID")
        for err in result.errors:
            print(f"  ERROR: {err}")
        for warn in result.warnings:
            print(f"  WARNING: {warn}")
        return 0 if result.valid else 1

    elif args.subcommand == "register":
        s = StrategyDefinition.from_yaml(args.file)
        register_technical()
        register_fundamental()
        result = validate_strategy(s, check_factors_exist=True)
        if not result.valid:
            print(f"Cannot register: validation failed for {s.name}")
            for err in result.errors:
                print(f"  ERROR: {err}")
            return 1
        registry.save(s)
        print(f"Registered: {s.name} v{s.version}")
        return 0

    elif args.subcommand == "run":
        strategy = None
        if args.file:
            strategy = StrategyDefinition.from_yaml(args.file)
        elif args.name:
            strategy = registry.get(args.name, args.version or "")

        if strategy is None:
            print("Please specify --name or --file")
            return 1

        register_technical()
        register_fundamental()
        result = validate_strategy(strategy, check_factors_exist=True)
        if not result.valid:
            print(f"Cannot run: validation failed for {strategy.name}")
            for err in result.errors:
                print(f"  ERROR: {err}")
            return 1

        # Load base config
        config = load_config(_resolve_config_path(getattr(args, 'config', None)))

        # Apply DSL overrides to config for factor enablement
        if not hasattr(config, 'factors') or config.factors is None:
            from types import SimpleNamespace
            config.factors = SimpleNamespace()
        config.factors.enabled_technicals = list(strategy.factor_weights.keys())
        config.factors.enabled_fundamentals = []

        # Override portfolio settings
        if strategy.portfolio_method:
            config.portfolio.optimizer = strategy.portfolio_method

        # Run pipeline
        prices, returns, benchmark, metadata, financials, turnover = _load_data(config)
        processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
            prices, returns, financials, metadata, turnover, config=config)
        signal = _generate_signal(config, processed_factors, returns, prices=prices)
        bt_result = _run_backtest(config, signal, prices, returns, benchmark,
                                   sector_map, financials,
                                   gates=_execution_gates(metadata))

        # Generate report
        from pathlib import Path

        # `reporting.dashboard` exposes `generate_dashboard(results, ...)`, not
        # a `DashboardGenerator` class -- that name existed here and nowhere
        # else, so `main.py strategy run` raised ImportError every time.
        from quant_platform.reporting.dashboard import generate_dashboard

        report_path = Path(config.output.results_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        report = generate_dashboard(bt_result, output_dir=str(report_path))
        print(report)
        logger.info("Strategy run complete: %s v%s", strategy.name, strategy.version)

        # Record run
        import uuid
        registry.record_run(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            run_id=uuid.uuid4().hex[:12],
            config=strategy.to_dict(),
            summary=bt_result.get("summary", {}),
        )
        return 0

    elif args.subcommand == "history":
        name = args.name if args.name else ""
        history = registry.get_run_history(name, limit=20)
        if not history:
            print("No run history found")
            return 0
        print(f"\n=== Run History{f' for {name}' if name else ''} ===")
        print(f"  {'Run ID':<16} {'Strategy':<20} {'Version':<10} {'Time':<20}")
        print(f"  {'-'*16} {'-'*20} {'-'*10} {'-'*20}")
        for h in history:
            print(f"  {h['run_id']:<16} {h['strategy_name']:<20} {h['strategy_version']:<10} {h.get('started_at',''):<20}")
        return 0

    elif args.subcommand == "stats":
        stats = registry.get_stats()
        print("\n=== Strategy Registry Stats ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return 0

    else:
        print("Unknown subcommand. Use: list, show, validate, register, run, history, stats")
        return 1


def cmd_walkforward(args) -> int:
    """Walk-forward validation: the gold standard for overfitting control.

    Each fold recomputes factor weights using only train-period data.
    This eliminates look-ahead from IC/ICIR weighting.
    """
    import pandas as pd

    config = load_config(_resolve_config_path(args.config))

    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"  Method:    {args.method}")
    print(f"  Folds:     {args.folds}")
    print(f"  Train:     {args.train_days} days")
    print(f"  Test:      {args.test_days} days")
    print("=" * 70)

    # Load data
    prices, returns, benchmark, metadata, financials, turnover = _load_data(
        config, use_baostock=args.use_baostock
    )
    print(f"  Data:      {len(prices)} days, {len(prices.columns)} assets")

    # Compute factors (full period, for initial signal)
    processed_factors, ic_results, sector_map, fin_unstacked = _compute_factors(
        prices, returns, financials, metadata, turnover, config=config
    )

    # Generate full-period signal (used as reference)
    signal = _generate_signal(config, processed_factors, returns, prices=prices)

    # Run walk-forward with per-fold signal recomputation
    from quant_platform.backtest.walkforward import WalkForwardValidator

    validator = WalkForwardValidator(
        train_period=args.train_days,
        test_period=args.test_days,
        step_size=args.test_days,
        mode=args.method,
        n_folds=args.folds,
    )

    from quant_platform.backtest.cost_model import CostModel
    from quant_platform.portfolio.constraints import PortfolioConstraints

    # The fold engines must use the same execution model as the headline
    # backtest. Passing only capital/frequency/benchmark silently ignored
    # every configured cost, constraint, optimizer and covariance setting,
    # so OOS numbers were not comparable with the in-sample run (audit B3).
    engine_kwargs = {
        "initial_capital": config.backtest.initial_capital,
        "rebalance_frequency": config.backtest.rebalance_frequency,
        "benchmark": config.backtest.benchmark,
    }
    cost_model = CostModel(
        commission=config.costs.commission,
        stamp_tax=config.costs.stamp_tax,
        slippage=config.costs.slippage,
        slippage_model=config.costs.slippage_model,
    )
    constraints = PortfolioConstraints(
        long_only=config.portfolio.constraints.long_only,
        max_weight=config.portfolio.constraints.max_weight,
        max_sector_exposure=config.portfolio.constraints.max_sector_exposure,
        max_turnover=config.portfolio.constraints.max_turnover,
        lot_size=config.portfolio.constraints.lot_size,
        risk_aversion=config.portfolio.risk_aversion,
    )

    results = validator.run(
        signal=signal,
        prices=prices,
        returns=returns,
        benchmark_returns=benchmark,
        sector_map=sector_map,
        financials=fin_unstacked,
        engine_kwargs=engine_kwargs,
        factors=processed_factors,
        alpha_kwargs={
            "method": config.alpha.method,
            "lookback": config.alpha.lookback,
            "min_icir": config.alpha.min_icir,
        },
        cost_model=cost_model,
        constraints=constraints,
        optimizer=config.portfolio.optimizer,
        covariance_method=config.portfolio.covariance.method,
        covariance_lookback=config.portfolio.covariance.lookback,
        covariance_ewma_half_life=config.portfolio.covariance.ewma_half_life,
        execution_gates=_execution_gates(metadata),
    )

    # Print results
    fold_metrics = results.get("fold_metrics", [])
    print("\n" + "=" * 70)
    print("  WALK-FORWARD RESULTS (OOS)")
    print("=" * 70)
    print(f"  Requested folds: {args.folds}; executed: {len(fold_metrics)}")

    for i, m in enumerate(fold_metrics):
        print(f"\n  Fold {i+1}:")
        print(f"    OOS Sharpe:  {m.get('sharpe_ratio', 0):.3f}")
        print(f"    Ann. Ret:    {m.get('annual_return', 0)*100:.2f}%")
        print(f"    Max DD:      {m.get('max_drawdown', 0)*100:.2f}%")
        print(f"    Sortino:     {m.get('sortino_ratio', 0):.3f}")

    agg = results.get("aggregate_metrics", {})
    print("\n" + "-" * 70)
    print(f"  OOS Mean Sharpe:    {agg.get('sharpe_ratio', 0):.3f}")
    print(f"  Info Ratio:         {agg.get('information_ratio', 0):.3f}")
    print(f"  Annual Return:      {agg.get('annual_return', 0)*100:.2f}%")
    print(f"  Max Drawdown:       {agg.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Folds:              {len(fold_metrics)}")
    print("=" * 70)

    sharpe = agg.get('sharpe_ratio', 0)
    if sharpe < 0.5:
        print("\n  WARNING: Low OOS Sharpe — likely overfitting.")
        print("  Try: fewer factors, longer training window, equal-weight combination.")
    elif sharpe < 1.0:
        print("\n  NOTE: Moderate OOS Sharpe. Promising but needs more validation.")
    else:
        print("\n  Good OOS Sharpe. Consider forward testing (paper trading).")

    return 0


def _resolve_qmt_kwargs(config) -> dict:
    """Extract QMT kwargs from config, resolving password from env."""
    import os
    from typing import Any
    kwargs: dict[str, Any] = {}
    if hasattr(config, "execution") and hasattr(config.execution, "qmt"):
        qmt = config.execution.qmt
        kwargs["account"] = qmt.account
        kwargs["server"] = qmt.server
        kwargs["mode"] = qmt.mode
        kwargs["data_server"] = qmt.data_server or qmt.server
        password = qmt.password or os.environ.get("QMT_PASSWORD", "")
        if password:
            kwargs["password"] = password
    return kwargs


def _positive_int(value: str) -> int:
    """argparse type for counts: reject 0/negative at parse time (exit 2)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}",
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A-Share Multi-Factor Quant Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("--config", "-c", type=str, default=None,
                            help="Path to config YAML file")
    run_parser.add_argument("--use-baostock", action="store_true",
                            help="Use Baostock real A-share data instead of synthetic")
    run_parser.add_argument("--force", "-f", action="store_true",
                            help="Force recomputation and overwrite the pipeline "
                                 "cache entry (provider data caches are unaffected)")
    run_parser.add_argument("--no-cache", action="store_true",
                            help="Disable the pipeline result cache, read and "
                                 "write (provider data caches are unaffected)")
    run_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                            help="Cache directory path")
    run_parser.add_argument("--allow-synthetic-fallback", action="store_true",
                            help="Development only: allow a configured real data "
                                 "provider to fall back to synthetic data")
    run_parser.add_argument("--incremental", action="store_true",
                            help="Append only new exchange sessions to the provider "
                                 "cache (full fetch with a warning for providers that "
                                 "cannot append; --force still forces a full refetch)")
    run_parser.add_argument("--description", "-d", type=str, default=None,
                            help="Optional description for this run (saved in config version)")
    run_parser.add_argument("--save-factors", action="store_true",
                            help="Save factor values and evaluation history to Factor Research Store")

    # trade
    trade_parser = subparsers.add_parser("trade", help="Run live trading (Paper or QMT)")
    trade_parser.add_argument("--broker", "-b", type=str, default="simulated",
                            choices=["simulated", "paper", "qmt", "qmt_sim", "qmt_live"],
                            help="Broker type (qmt_live sets execution.qmt.mode=live)")
    trade_parser.add_argument("--config", "-c", type=str, default=None,
                            help="Path to config YAML file (execution.qmt settings)")
    trade_parser.add_argument("--universe", "-u", type=str, default=None,
                            help="Comma-separated stock codes (default: 5 blue chips)")
    trade_parser.add_argument("--days", "-d", type=_positive_int, default=30,
                            help="Number of exchange sessions to simulate (>= 1)")
    trade_parser.add_argument("--cash", type=float, default=10_000_000,
                            help="Initial capital in CNY")
    trade_parser.add_argument("--seed", "-s", type=int, default=42,
                            help="Random seed for price generation")
    trade_parser.add_argument("--no-dual-track", action="store_true",
                            help="Disable PaperBroker parallel tracking")

    # data (retention / maintenance)
    data_parser = subparsers.add_parser("data", help="Data maintenance (retention)")
    data_sub = data_parser.add_subparsers(
        dest="subcommand", required=True, help="Data action",
    )
    retention_parser = data_sub.add_parser(
        "retention",
        help="Archive and delete rows older than each table's max age",
    )
    retention_parser.add_argument(
        "--db", type=str, default="data/trading.db", help="SQLite database path",
    )
    retention_parser.add_argument(
        "--archive-dir", type=str, default="data/archive",
        help="Directory for dated CSV archives written before deletion",
    )
    retention_parser.add_argument(
        "--orders-days", type=int, default=5 * 365,
        help="Order retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--trades-days", type=int, default=5 * 365,
        help="Trade retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--pnl-days", type=int, default=5 * 365,
        help="P&L snapshot retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--signals-days", type=int, default=365,
        help="Signal retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--sessions-days", type=int, default=365,
        help="Session retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--events-days", type=int, default=180,
        help="Event/audit retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--nav-days", type=int, default=10 * 365,
        help="NAV history retention in days (0 disables)",
    )
    retention_parser.add_argument(
        "--no-vacuum", action="store_true",
        help="Skip the VACUUM after deletion",
    )

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("--results-dir", "-r", type=str, default="./results",
                                help="Results directory path")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies")
    compare_parser.add_argument("--config", "-c", type=str, default=None,
                                help="Path to config YAML file")
    compare_parser.add_argument("--optimizers", type=str, default=None,
                                help="Comma-separated optimizers (default: equal_weight,mean_variance,risk_parity)")
    compare_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                                help="Cache directory path")

    # sweep
    sweep_parser = subparsers.add_parser("sweep", help="Grid search over parameters")
    sweep_parser.add_argument("--config", "-c", type=str, default=None,
                              help="Path to config YAML file")
    sweep_parser.add_argument("--optimizers", type=str, default=None,
                              help="Comma-separated optimizers")
    sweep_parser.add_argument("--frequencies", type=str, default=None,
                              help="Comma-separated frequencies (default: monthly,weekly)")
    sweep_parser.add_argument("--n-stocks", type=str, default=None,
                              help="Comma-separated universe sizes (default: 200)")
    sweep_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                              help="Cache directory path")

    # web
    web_parser = subparsers.add_parser("web", help="Start web server (FastAPI + Vue)")
    web_parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    web_parser.add_argument("--host", type=str, default="127.0.0.1",
                           help="Server host (use 0.0.0.0 in a container, with QUANT_API_TOKEN set)")
    web_parser.add_argument("--no-frontend", action="store_true", help="API only mode")

    # cache
    cache_parser = subparsers.add_parser("cache", help="Manage pipeline cache")
    cache_sub = cache_parser.add_subparsers(
        dest="subcommand", required=True, help="Cache action",
    )
    cache_list = cache_sub.add_parser("list", help="List cached stages")
    cache_list.add_argument("--cache-dir", type=str, default=argparse.SUPPRESS,
                            help="Cache directory path")
    cache_clear = cache_sub.add_parser("clear", help="Clear all cached files")
    cache_clear.add_argument("--cache-dir", type=str, default=argparse.SUPPRESS,
                             help="Cache directory path")
    cache_prune = cache_sub.add_parser(
        "prune", help="Evict expired/undersized cache entries",
    )
    cache_prune.add_argument("--cache-dir", type=str, default=argparse.SUPPRESS,
                             help="Cache directory path")
    cache_prune.add_argument("--max-age-days", type=float, default=None,
                             help="Delete entries older than this (default: 30)")
    cache_prune.add_argument("--max-entries", type=int, default=None,
                             help="Keep at most this many entries (default: 64)")
    cache_refresh = cache_sub.add_parser(
        "refresh",
        help="Recompute the data stage for the current config and overwrite "
             "its pipeline cache entry (no factors/backtest; provider data "
             "caches are range-immutable and still used)",
    )
    cache_refresh.add_argument("--config", "-c", type=str, default=None,
                               help="Path to config YAML file")
    cache_refresh.add_argument("--use-baostock", action="store_true",
                               help="Use Baostock real A-share data")
    cache_refresh.add_argument("--allow-synthetic-fallback", action="store_true",
                               help="Development only: allow a real provider to "
                                    "fall back to synthetic data")
    cache_refresh.add_argument("--incremental", action="store_true",
                               help="Append only new exchange sessions to the "
                                    "provider cache before overwriting the data stage")
    cache_refresh.add_argument("--cache-dir", type=str, default=argparse.SUPPRESS,
                               help="Cache directory path")
    # Kept for backward compatibility with `cache --cache-dir X list`; the
    # subparser copies (SUPPRESS default) win when given after the subcommand.
    cache_parser.add_argument("--cache-dir", type=str, default=".quant_cache",
                              help="Cache directory path (prefer after the subcommand)")

    # runs — research run registry and result validity
    runs_parser = subparsers.add_parser(
        "runs", help="Research runs: list results and mark their validity"
    )
    runs_sub = runs_parser.add_subparsers(dest="subcommand", help="Runs action")
    runs_list = runs_sub.add_parser("list", help="List recent runs with validity")
    runs_list.add_argument("--limit", type=int, default=50, help="How many runs")
    runs_list.add_argument("--validity", type=str, default=None,
                           choices=["valid", "affected", "invalidated", "superseded"],
                           help="Only list runs in this validity state")
    runs_list.add_argument("--db", type=str, default=None, help="Registry path")
    runs_aff = runs_sub.add_parser(
        "affected", help="Runs marked affected/invalidated rather than valid"
    )
    runs_aff.add_argument("--reason", type=str, default=None,
                          help="Filter by cause, e.g. BUG-03")
    runs_aff.add_argument("--db", type=str, default=None, help="Registry path")
    runs_mark = runs_sub.add_parser("mark", help="Mark a run's result validity")
    runs_mark.add_argument("run_id", type=str, help="Run id to mark")
    runs_mark.add_argument("state", type=str,
                           choices=["valid", "affected", "invalidated", "superseded"],
                           help="New validity state")
    runs_mark.add_argument("--reason", type=str, default="",
                           help="Cause, e.g. BUG-03")
    runs_mark.add_argument("--db", type=str, default=None, help="Registry path")
    runs_import = runs_sub.add_parser(
        "import", help="Import runs from a JSON snapshot into the registry"
    )
    runs_import.add_argument("path", type=str, help="JSON snapshot path")
    runs_import.add_argument("--db", type=str, default=None, help="Registry path")
    runs_export = runs_sub.add_parser(
        "export", help="Export the registry to a JSON snapshot"
    )
    runs_export.add_argument("path", type=str, help="JSON snapshot path")
    runs_export.add_argument("--validity", type=str, default=None,
                             choices=["valid", "affected", "invalidated", "superseded"],
                             help="Only export runs in this validity state")
    runs_export.add_argument("--db", type=str, default=None, help="Registry path")

    # ml
    ml_parser = subparsers.add_parser("ml", help="ML alpha signal operations")
    ml_sub = ml_parser.add_subparsers(dest="subcommand", help="ML action")
    ml_train = ml_sub.add_parser("train", help="Train ML model and show performance")
    ml_train.add_argument("--model", type=str, default="lightgbm",
                          choices=["xgboost", "lightgbm"], help="Model type")
    ml_train.add_argument("--splits", type=int, default=5, help="CV splits")
    ml_train.add_argument("--config", "-c", type=str, default=None, help="Config path")
    ml_signal = ml_sub.add_parser("signal", help="Generate ML signals")
    ml_signal.add_argument("--model", type=str, default="lightgbm",
                           choices=["xgboost", "lightgbm"], help="Model type")
    ml_signal.add_argument("--config", "-c", type=str, default=None, help="Config path")

    # research
    research_parser = subparsers.add_parser("research", help="LLM research agent and validation")
    research_sub = research_parser.add_subparsers(dest="subcommand", help="Research action")
    research_analyze = research_sub.add_parser("report", help="Analyze backtest results with LLM")
    research_analyze.add_argument("--results-dir", "-r", type=str, default="./results",
                                  help="Results directory")
    research_validate = research_sub.add_parser("validate", help="Run Research Validation Sprint")
    research_validate.add_argument("--mode", type=str, default="quick",
                                    choices=["quick", "full"], help="Validation mode")
    research_validate.add_argument("--max-factors", type=int, default=3,
                                    help="Max factors to evaluate (quick mode)")
    research_validate.add_argument("--max-folds", type=int, default=2,
                                    help="Max walk-forward folds")
    research_validate.add_argument("--n-stocks", type=int, default=0,
                                    help="Number of stocks (0 = use config default)")
    research_validate.add_argument("--timeout", type=int, default=600,
                                    help="Timeout in seconds")

    # factor (unified research entry)
    factor_parser = subparsers.add_parser("factor", help="Run a factor research report")
    factor_parser.add_argument("factor", nargs="?", default="list", help="Factor name or 'list'")
    factor_parser.add_argument("--synthetic", action="store_true",
                               help="Run offline with synthetic data")
    factor_parser.add_argument("--n-stocks", type=int, default=5,
                               help="Synthetic universe size")
    factor_parser.add_argument("--n-days", type=int, default=500,
                               help="Synthetic history length")

    # factor-store
    fs_parser = subparsers.add_parser("factor-store", help="Factor Research Store operations")
    fs_sub = fs_parser.add_subparsers(dest="subcommand", help="Factor Store action")
    fs_sub.add_parser("stats", help="Show store statistics")
    fs_rank = fs_sub.add_parser("rank", help="Rank factors by health score")
    fs_rank.add_argument("--limit", type=int, default=20, help="Number of top factors")
    fs_sub.add_parser("register-all", help="Register all available factors from registry")
    fs_hist = fs_sub.add_parser("history", help="Show evaluation history for a factor")
    fs_hist.add_argument("--factor", type=str, default="", help="Factor name")
    fs_hist.add_argument("--limit", type=int, default=50, help="Number of records")
    fs_wf = fs_sub.add_parser("walkforward-summary", help="Show walk-forward summary for a factor")
    fs_wf.add_argument("--factor", type=str, required=True, help="Factor name")
    fs_sub.add_parser("clear", help="Clear all factor store data (for testing)")

    # walkforward (standalone)
    wf_parser = subparsers.add_parser("walkforward", help="Walk-forward validation")
    wf_parser.add_argument("--config", "-c", type=str, default=None,
                           help="Path to config YAML file")
    wf_parser.add_argument("--folds", type=int, default=6,
                           help="Number of walk-forward folds")
    wf_parser.add_argument("--method", type=str, default="expanding",
                           choices=["expanding", "rolling"],
                           help="Walk-forward method")
    wf_parser.add_argument("--train-days", type=int, default=504,
                           help="Training window length in days")
    wf_parser.add_argument("--test-days", type=int, default=126,
                           help="Test window length in days")
    wf_parser.add_argument("--use-baostock", action="store_true",
                           help="Use Baostock data")

    # strategy gates
    gate_parser = subparsers.add_parser("gate", help="Run Strategy Evaluation Gates")
    gate_parser.add_argument("--config", "-c", type=str, default=None,
                              help="Path to config YAML file")
    gate_parser.add_argument("--strategy", type=str, default="default",
                              help="Strategy name for the report")
    gate_parser.add_argument("--min-ic", type=float, default=None,
                              help="Override minimum IC threshold")
    gate_parser.add_argument("--max-dd", type=float, default=None,
                              help="Override maximum drawdown threshold")

    # screen
    screen_parser = subparsers.add_parser("screen", help="Screen stocks using boolean factor rules")
    screen_parser.add_argument("--config", "-c", type=str, default=None,
                               help="Path to config YAML file")
    screen_parser.add_argument("--rules", "-r", type=str, default=None,
                               help='Rules on RAW factor values: "factor=op:value,..." '
                                    'e.g. "pe_ratio=lt:30,roe=gt:0.15"')
    screen_parser.add_argument("--logic", "-l", type=str, default=None,
                               choices=["and", "or"], help="Rule combination logic")
    screen_parser.add_argument("--output", "-o", type=str, default=None,
                               help="Save results to CSV")
    screen_parser.add_argument("--min-stocks", type=int, default=None,
                               help="Minimum qualifying stocks (auto-relax)")
    screen_parser.add_argument("--max-stocks", type=int, default=None,
                               help="Maximum qualifying stocks (cap with scoring)")
    screen_parser.add_argument("--use-baostock", action="store_true",
                               help="Use Baostock real data")

    # config
    config_parser = subparsers.add_parser("config", help="Manage configuration versions")
    config_sub = config_parser.add_subparsers(
        dest="subcommand", required=True, help="Config action",
    )

    config_sub.add_parser("list", help="List all config versions")
    config_show = config_sub.add_parser("show", help="Show config for a version")
    config_show.add_argument("version", type=str, help="Version ID (e.g. v3)")
    config_diff = config_sub.add_parser("diff", help="Diff two config versions")
    config_diff.add_argument("v1", type=str, help="First version (before)")
    config_diff.add_argument("v2", type=str, help="Second version (after)")
    config_rollback = config_sub.add_parser("rollback", help="Restore a version as active config")
    config_rollback.add_argument("version", type=str, help="Version ID to restore (e.g. v3)")
    config_rollback.add_argument("--target", "-t", type=str, default=None,
                                  help="Target config path (default: config/default.yaml)")
    config_delete = config_sub.add_parser("delete", help="Delete a config version")
    config_delete.add_argument("version", type=str, help="Version ID to delete")
    config_delete.add_argument("--force", "-f", action="store_true",
                                help="Skip confirmation")
    config_prune = config_sub.add_parser(
        "prune", help="Delete old config versions by count/age",
    )
    config_prune.add_argument("--keep", type=int, default=20,
                              help="Keep at least this many newest versions")
    config_prune.add_argument("--max-age-days", type=float, default=None,
                              help="Also delete versions older than this many days")

    # lookahead
    la_parser = subparsers.add_parser("check-lookahead", help="Detect lookahead bias in factor pipeline")
    la_parser.add_argument("--config", "-c", type=str, default=None,
                           help="Path to config YAML file")
    la_parser.add_argument("--threshold", type=float, default=1e-4,
                           help="Signal diff threshold for bias detection")
    la_parser.add_argument("--max-dates", type=int, default=20,
                           help="Maximum number of dates to check")
    la_parser.add_argument("--use-baostock", action="store_true",
                           help="Use Baostock real data")

    # strategy DSL
    strat_parser = subparsers.add_parser("strategy", help="Manage Strategy DSL definitions")
    strat_sub = strat_parser.add_subparsers(dest="subcommand", help="Strategy action")
    strat_sub.add_parser("list", help="List all registered strategies")
    strat_show = strat_sub.add_parser("show", help="Show strategy definition")
    strat_show.add_argument("--name", type=str, default="", help="Strategy name")
    strat_show.add_argument("--version", type=str, default="", help="Version (optional)")
    strat_validate = strat_sub.add_parser("validate", help="Validate a strategy file")
    strat_validate.add_argument("file", type=str, help="Path to strategy YAML file")
    strat_register = strat_sub.add_parser("register", help="Register a strategy from file")
    strat_register.add_argument("file", type=str, help="Path to strategy YAML file")
    strat_run = strat_sub.add_parser("run", help="Run a strategy (pipeline execution)")
    strat_run.add_argument("--name", type=str, default="", help="Strategy name")
    strat_run.add_argument("--version", type=str, default="", help="Version")
    strat_run.add_argument("--file", type=str, default="", help="Path to strategy YAML file (overrides name)")
    strat_history = strat_sub.add_parser("history", help="Show strategy run history")
    strat_history.add_argument("--name", type=str, default="", help="Strategy name (optional)")
    strat_sub.add_parser("stats", help="Show registry statistics")

    # execute
    exec_parser = subparsers.add_parser("execute", help="Full pipeline: factors -> signal -> orders -> fills")
    exec_parser.add_argument("--config", "-c", type=str, default=None, help="Config path")
    exec_parser.add_argument("--use-baostock", action="store_true", help="Use Baostock data")

    # profile
    profile_parser = subparsers.add_parser("profile", help="Profile pipeline performance")
    profile_parser.add_argument("--config", "-c", type=str, default=None, help="Config path")

    args = parser.parse_args()

    # Configure logging once — all child loggers inherit handlers from the
    # "quant_platform" logger. The level honors QUANT_LOG_LEVEL / LOG_LEVEL.
    setup_logging()

    import yaml

    from quant_platform.utils.config import ConfigValidationError

    try:
        return _dispatch(parser, args)
    except FileNotFoundError as exc:
        # A missing config (or input file) is an operator error, not a crash:
        # print an actionable message instead of a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ConfigValidationError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    except DataProviderUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"Config error (invalid YAML): {exc}", file=sys.stderr)
        return 1


def _dispatch(parser, args) -> int:
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "trade":
        return cmd_trade(args)
    elif args.command == "data":
        return cmd_data(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "cache":
        return cmd_cache(args)
    elif args.command == "runs":
        return cmd_runs(args)
    elif args.command == "web":
        return cmd_web(args)
    elif args.command == "ml":
        return cmd_ml(args)
    elif args.command == "research":
        return cmd_research(args)
    elif args.command == "walkforward":
        return cmd_walkforward(args)
    elif args.command == "factor":
        return cmd_factor(args)
    elif args.command == "factor-store":
        return cmd_factor_store(args)
    elif args.command == "screen":
        return cmd_screen(args)
    elif args.command == "execute":
        return cmd_execute(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "gate":
        return cmd_gate(args)
    elif args.command == "check-lookahead":
        return cmd_check_lookahead(args)
    elif args.command == "strategy":
        return cmd_strategy(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
