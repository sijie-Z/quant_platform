"""Shared builders for the wired analysis/monitor endpoint tests.

These helpers build a :class:`ResearchContext` from hand-made panels so a
unit test can exercise the computations without paying for a full pipeline
run; the end-to-end tests in ``test_endpoint_wiring.py`` use the real
(cached) context built by the default synthetic config with a smaller
universe.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from quant_platform.api.research_context import ResearchContext
from quant_platform.utils.config import load_config

_context_counter = itertools.count(1)


def make_context(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    processed_factors: dict[str, pd.DataFrame] | None = None,
    *,
    benchmark: Any = None,
    metadata: Any = None,
    financials: Any = None,
    turnover: pd.DataFrame | None = None,
    fin_unstacked: pd.DataFrame | None = None,
    signal: pd.DataFrame | None = None,
    config: Any = None,
    config_hash: str | None = None,
    provider: str = "synthetic",
    synthetic: bool = True,
    start_date: str = "",
    end_date: str = "",
) -> ResearchContext:
    if processed_factors is None:
        processed_factors = {}
    if signal is None:
        signal = pd.DataFrame(
            0.0, index=prices.index, columns=prices.columns,
        )
    if config_hash is None:
        # Unique per built context: the endpoint caches are keyed by this
        # hash, and two hand-made contexts sharing one would poison each
        # other's cached derived results.
        config_hash = f"test-context-{next(_context_counter)}"
    as_of = ""
    if len(prices.index) > 0:
        as_of = str(prices.index[-1])[:10]
    return ResearchContext(
        prices=prices,
        returns=returns,
        benchmark=benchmark,
        metadata=metadata,
        financials=financials,
        turnover=turnover,
        processed_factors=processed_factors,
        ic_results={},
        sector_map={},
        fin_unstacked=fin_unstacked,
        signal=signal,
        config=config if config is not None else load_config(),
        provider=provider,
        synthetic=synthetic,
        config_hash=config_hash,
        as_of=as_of,
        computed_at=datetime.now(UTC).isoformat(),
        start_date=start_date or (
            str(prices.index[0])[:10] if len(prices.index) else ""
        ),
        end_date=end_date or as_of,
    )


def small_config(n_stocks: int = 150):
    """The default YAML with a smaller synthetic universe (test runtime only)."""
    config = load_config()
    config.universe.n_stocks = n_stocks
    return config
