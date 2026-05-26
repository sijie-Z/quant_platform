"""Factor processing: winsorization, standardization, neutralization.

These operations are applied cross-sectionally (per date) to raw factor
values to reduce noise and remove unwanted exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


def winsorize(
    factor: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Cross-sectional winsorization.

    Uses Numba JIT acceleration when available (5-15x speedup on wide panels).

    Args:
        factor: (date x asset) raw factor values.
        lower: Lower quantile (default 1%).
        upper: Upper quantile (default 99%).

    Returns:
        Winsorized factor values.
    """
    if lower <= 0 and upper >= 1:
        return factor

    from quant_platform.utils.numba_accelerator import (
        HAS_NUMBA,
        winsorize_numba,
        winsorize_pandas,
    )

    if HAS_NUMBA and factor.shape[1] >= 10:
        return winsorize_numba(factor, lower, upper)

    return winsorize_pandas(factor, lower, upper)


def standardize(
    factor: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """Cross-sectional standardization.

    Uses Numba JIT for zscore method when available (3-8x speedup).

    Args:
        factor: (date x asset) factor values.
        method: 'zscore' for (x - mean) / std, 'rank' for percentile rank.

    Returns:
        Standardized factor values (date x asset).
    """
    if method == "zscore":
        from quant_platform.utils.numba_accelerator import zscore_numba
        return zscore_numba(factor)

    elif method == "rank":
        result = factor.copy()
        for date in factor.index:
            row = result.loc[date]
            valid_mask = row.notna()
            if valid_mask.sum() < 10:
                continue
            result.loc[date, valid_mask] = row[valid_mask].rank(pct=True)
        return result

    else:
        raise ValueError(f"Unknown standardization method: {method}")


def neutralize(
    factor: pd.DataFrame,
    sector_map: pd.Series | None = None,
    market_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional sector and market-cap neutralization.

    Regresses factor values on sector dummies and market cap, returns
    residuals. This removes systematic biases:
    - Sector: prevents unintended sector bets
    - Market cap: removes size factor exposure

    Args:
        factor: (date x asset) post-standardized factor values.
        sector_map: Series mapping asset -> sector name.
        market_cap: (date x asset) market cap values.

    Returns:
        Neutralized factor values (residuals).
    """
    result = factor.copy()

    for date in factor.index:
        row = result.loc[date].copy()
        valid_mask = row.notna()

        if valid_mask.sum() < 20:
            continue

        assets = factor.columns[valid_mask]
        y = row[valid_mask].values

        # Build feature matrix X
        features = []

        if sector_map is not None:
            asset_sectors = sector_map.reindex(assets)
            sector_dummies = pd.get_dummies(asset_sectors, drop_first=True)
            features.append(sector_dummies.values)

        if market_cap is not None:
            mcap_row = market_cap.loc[date].reindex(assets)
            mcap_log = np.log(mcap_row.clip(lower=1e8))
            features.append(mcap_log.values.reshape(-1, 1))

        if not features:
            continue

        X = np.column_stack(features) if len(features) > 1 else features[0]

        # Handle any NaN in X
        valid_rows = ~np.isnan(X).any(axis=1)
        if valid_rows.sum() < 10:
            continue

        model = LinearRegression()
        model.fit(X[valid_rows], y[valid_rows])
        residuals = y - model.predict(X)

        result.loc[date, assets] = residuals

    return result


def process_factor(
    factor: pd.DataFrame,
    winsorize_enabled: bool = True,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    standardize_enabled: bool = True,
    standardize_method: str = "zscore",
    neutralize_enabled: bool = True,
    sector_map: pd.Series | None = None,
    market_cap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full factor processing pipeline.

    Applies winsorization -> standardization -> neutralization in order.
    """
    result = factor.copy()

    if winsorize_enabled:
        result = winsorize(result, winsorize_lower, winsorize_upper)

    if standardize_enabled:
        result = standardize(result, standardize_method)

    if neutralize_enabled:
        result = neutralize(result, sector_map, market_cap)

    return result
