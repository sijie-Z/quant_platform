"""MarketDataProvider — daily market data contract with explicit adjust + honesty.

Per CONSTITUTION.md Principle 1 (Truth First), the contract refuses to return
"prices" without declaring how adjusted they are. A non-adjusted provider is
allowed, but MUST surface its adjustment state via `data_meta()` so the
Registry can flag the run and the Report can warn — same discipline as
UniverseProvider. We never silently return raw close and let it be treated as
return-quality data.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import pandas as pd

AdjustKind = Literal[
    "none",      # raw close (adjustflag=3) — NOT trust-worthy as return input
    "front",     # 前复权 — basis date varies per stock (free-source caveat)
    "back",      # 后复权
]


@runtime_checkable
class MarketDataProvider(Protocol):
    @property
    def adjust(self) -> AdjustKind:
        """How prices are adjusted. Registry records this verbatim."""
        ...

    def data_meta(self) -> dict:
        """Machine-readable honesty declaration for the price series.

        Required keys:
            adjust: AdjustKind
            basis: str            — e.g. "date_varies_per_stock" (front-adj caveat) / "fixed"
            trust_warning: list[str]
            description: str
        """
        ...

    def get_daily(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Daily OHLCV + fundamentals, indexed by (date, asset)."""
        ...
