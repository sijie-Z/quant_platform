"""Factor processing: winsorization, standardization, neutralization.

These operations are applied cross-sectionally (per date) to raw factor
values to reduce noise and remove unwanted exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    if isinstance(factor, pd.DataFrame) and factor.ndim == 2 and HAS_NUMBA and factor.shape[1] >= 10:
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
        if isinstance(factor, pd.Series):
            factor = factor.to_frame("factor")
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
    if isinstance(factor, pd.Series):
        factor = factor.to_frame('factor')
    result = factor.copy()

    # Ensure sector_map is a pandas Series (not dict)
    if sector_map is not None and isinstance(sector_map, dict):
        sector_map = pd.Series(sector_map)

    columns = factor.columns
    values = factor.to_numpy(dtype=float, copy=True)

    # Build the static sector design once, not once per date per factor.
    # `neutralize` is called for every factor; rebuilding get_dummies for
    # each date made this function ~62% of the whole factor stage.
    dummy = None
    sector_valid = np.ones(len(columns), dtype=bool)
    if sector_map is not None:
        sectors = sector_map.reindex(columns)
        sector_valid = sectors.notna().to_numpy()
        coded = pd.Categorical(sectors)
        codes = coded.codes
        n_categories = len(coded.categories)
        if n_categories > 1:
            dummy = np.zeros((len(columns), n_categories - 1))
            for position, code in enumerate(codes):
                if code > 0:  # first category is the baseline
                    dummy[position, code - 1] = 1.0
        else:
            dummy = np.zeros((len(columns), 0))

    mcap = None
    if market_cap is not None:
        mcap = market_cap.reindex(index=factor.index, columns=columns).to_numpy(dtype=float)
        mcap = np.log(np.clip(mcap, 1e8, None))

    if dummy is None and mcap is None:
        return result

    for row_index, _date in enumerate(factor.index):
        y = values[row_index]
        present = ~np.isnan(y)
        if present.sum() < 20:
            continue

        asset_idx = np.flatnonzero(present)
        features = [np.ones(len(asset_idx))]
        if dummy is not None:
            features.append(dummy[asset_idx])
        if mcap is not None:
            features.append(mcap[row_index, asset_idx].reshape(-1, 1))
        X = np.column_stack(features)

        valid = np.isfinite(X).all(axis=1)
        if dummy is not None:
            valid &= sector_valid[asset_idx]
        if valid.sum() < 10:
            continue

        y_valid = y[asset_idx][valid]
        # Same least-squares fit as sklearn's LinearRegression(intercept=True)
        # on sector dummies + log market cap, without constructing 20k models.
        beta, *_ = np.linalg.lstsq(X[valid], y_valid, rcond=None)
        residuals = np.full(len(asset_idx), np.nan)
        residuals[valid] = y_valid - X[valid] @ beta
        values[row_index, asset_idx] = residuals

    return pd.DataFrame(values, index=factor.index, columns=columns)


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
    if isinstance(factor, pd.Series):
        factor = factor.to_frame('factor')
    result = factor.copy()

    if winsorize_enabled:
        result = winsorize(result, winsorize_lower, winsorize_upper)

    if standardize_enabled:
        result = standardize(result, standardize_method)

    if neutralize_enabled:
        result = neutralize(result, sector_map, market_cap)

    return result
