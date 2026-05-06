"""Portfolio exposure analysis.

Analyzes concentration and factor exposures of portfolio weights:
- Sector concentration (Herfindahl index)
- Market cap group exposure
- Top-N concentration
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sector_exposure(
    weights: pd.Series,
    sector_map: pd.Series,
) -> pd.Series:
    """Compute total weight per sector."""
    aligned_sectors = sector_map.reindex(weights.index)
    return weights.groupby(aligned_sectors).sum().sort_values(ascending=False)


def herfindahl_index(weights: pd.Series) -> float:
    """Herfindahl-Hirschman Index: sum(w_i^2).

    Measures concentration. 1/N = perfect diversification, 1 = single stock.
    """
    w = weights.dropna()
    if len(w) == 0:
        return 0.0
    return float((w ** 2).sum())


def effective_n(weights: pd.Series) -> float:
    """Effective number of stocks = 1 / HHI.

    A portfolio with equal weight in 100 stocks has effective N = 100.
    A portfolio with 50% in one stock and 50% in 99 others has effective N ~= 4.
    """
    hhi = herfindahl_index(weights)
    if hhi < 1e-10:
        return float(len(weights.dropna()))
    return 1.0 / hhi


def top_n_concentration(weights: pd.Series, n: int = 10) -> float:
    """Weight in top N holdings."""
    return weights.nlargest(n).sum()


def market_cap_exposure(
    weights: pd.Series,
    cap_groups: pd.Series,
) -> pd.Series:
    """Compute weight allocation by market cap group."""
    aligned = cap_groups.reindex(weights.index)
    return weights.groupby(aligned).sum()


def exposure_report(
    weights: pd.Series,
    sector_map: pd.Series | None = None,
    cap_groups: pd.Series | None = None,
) -> dict:
    """Generate a comprehensive exposure report for a weight vector."""
    report = {
        "n_assets": int((weights > 1e-6).sum()),
        "effective_n": effective_n(weights),
        "herfindahl": herfindahl_index(weights),
        "top5_concentration": top_n_concentration(weights, 5),
        "top10_concentration": top_n_concentration(weights, 10),
        "max_single_weight": weights.max(),
    }

    if sector_map is not None:
        se = sector_exposure(weights, sector_map)
        report["max_sector_exposure"] = se.iloc[0] if len(se) > 0 else 0.0
        report["top_sector"] = se.index[0] if len(se) > 0 else None
        report["n_sectors"] = int((se > 1e-6).sum())

    if cap_groups is not None:
        ce = market_cap_exposure(weights, cap_groups)
        for group in ["large", "mid", "small"]:
            report[f"weight_{group}_cap"] = ce.get(group, 0.0)

    return report
