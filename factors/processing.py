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
    sector_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Cross-sectional sector and market-cap neutralization.

    Regresses factor values on sector dummies and market cap, returns
    residuals. This removes systematic biases:
    - Sector: prevents unintended sector bets
    - Market cap: removes size factor exposure

    ``sector_panel`` (date x asset) takes precedence over ``sector_map``: for
    each date the row in effect on that date is used, so a historical
    reclassification does not leak backwards into earlier cross-sections.
    Assets the panel leaves unclassified on a date stay NaN (no sector
    neutralization for them); a date with no classification at all skips
    only the sector leg. Neither is filled in from the static map, because
    the static map is a whole-sample (typically latest) classification and
    doing so would reintroduce the look-ahead on providers whose industry
    history starts after the sample does. ``sector_map`` is used only when no
    panel is supplied at all; passing only ``sector_map`` reproduces the old
    static behaviour.

    Args:
        factor: (date x asset) post-standardized factor values.
        sector_map: Series mapping asset -> sector name (static).
        market_cap: (date x asset) market cap values.
        sector_panel: (date x asset) point-in-time sector names.

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

    # Point-in-time panel: encode every row once against a common category
    # set. Code -1 marks "no classification on this date". A single-category
    # panel is still meaningful: it degenerates to intercept-only (demeaning)
    # and must override a static map that claims several sectors.
    panel_codes = None
    panel_n_categories = 0
    if sector_panel is not None:
        panel = sector_panel.reindex(index=factor.index, columns=columns)
        flat = panel.to_numpy(dtype=object).ravel()
        codes_flat, uniques = pd.factorize(flat, sort=True)
        panel_codes = codes_flat.reshape(panel.shape)
        panel_n_categories = len(uniques)
    has_panel_sectors = panel_codes is not None and panel_n_categories > 0

    mcap = None
    if market_cap is not None:
        mcap = market_cap.reindex(index=factor.index, columns=columns).to_numpy(dtype=float)
        mcap = np.log(np.clip(mcap, 1e8, None))

    if dummy is None and mcap is None and not has_panel_sectors:
        return result

    for row_index, _date in enumerate(factor.index):
        y = values[row_index]
        present = ~np.isnan(y)
        if present.sum() < 20:
            continue

        asset_idx = np.flatnonzero(present)
        features = [np.ones(len(asset_idx))]
        valid = np.ones(len(asset_idx), dtype=bool)

        use_panel = has_panel_sectors
        row_codes = None
        if use_panel:
            row_codes = panel_codes[row_index]
            # A row the panel leaves entirely unclassified skips only the
            # sector leg (mcap, if any, still applies). Falling back to the
            # static map here would leak the whole-sample classification
            # into dates that predate the provider's industry history.
            if not (row_codes >= 0).any():
                use_panel = False

        if use_panel:
            codes_selected = row_codes[asset_idx]
            n_dummy = max(panel_n_categories - 1, 0)
            row_dummy = np.zeros((len(asset_idx), n_dummy))
            nonbase = codes_selected > 0  # first category is the baseline
            row_dummy[np.flatnonzero(nonbase), codes_selected[nonbase] - 1] = 1.0
            features.append(row_dummy)
            valid &= codes_selected >= 0
        elif dummy is not None and sector_panel is None:
            # Static map only when the caller supplied no panel at all.
            features.append(dummy[asset_idx])
            valid &= sector_valid[asset_idx]
        if mcap is not None:
            features.append(mcap[row_index, asset_idx].reshape(-1, 1))
        X = np.column_stack(features)

        valid &= np.isfinite(X).all(axis=1)
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
    factor_name: str | None = None,
    sector_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full factor processing pipeline.

    Applies winsorization -> standardization -> neutralization in order.

    Args:
        factor_name: Name of the factor. ``"log_market_cap"`` skips the
            market-cap neutralization: regressing the size factor on size
            removes the factor itself (residual std collapses to ~1% of the
            input), leaving noise. Sector neutralization still applies.
        sector_panel: (date x asset) point-in-time sector names. When given,
            neutralization uses each date's own classification; otherwise
            ``sector_map`` (static) is used, preserving the old signature.
    """
    if isinstance(factor, pd.Series):
        factor = factor.to_frame('factor')
    result = factor.copy()

    if winsorize_enabled:
        result = winsorize(result, winsorize_lower, winsorize_upper)

    if standardize_enabled:
        result = standardize(result, standardize_method)

    if neutralize_enabled:
        if factor_name == "log_market_cap":
            # A size factor regressed on size is self-neutralization: the
            # exposure we would remove is the signal. Keep sector only.
            market_cap = None
        result = neutralize(
            result, sector_map, market_cap, sector_panel=sector_panel,
        )

    return result
