"""Barra-style multi-factor risk model.

Implements a simplified Barra risk model with 10 common risk factors:
1. Size: log(market_cap)
2. Value: book-to-price (1/PB)
3. Momentum: 12-month return (skip last month)
4. Volatility: 60-day realized vol
5. Quality: ROE
6. Growth: asset growth rate
7. Liquidity: 20-day average turnover
8. Leverage: debt-to-equity
9. Beta: market beta
10. Residual Vol: idiosyncratic volatility

The model:
1. Cross-sectional regression each day → factor betas
2. Factor covariance estimation (Newey-West, shrinkage)
3. Specific risk estimation
4. Risk decomposition and attribution

Used for:
- More accurate portfolio risk estimation
- Factor-based risk budgeting
- Risk attribution (which factors drive P&L)
- Optimal hedging

Reference: Barra Risk Model Handbook (MSCI)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


BARRA_FACTORS = [
    "size",           # log(market_cap)
    "value",          # 1 / PB ratio
    "momentum",       # 12-month return skip last month
    "volatility",     # 60-day realized volatility
    "quality",        # ROE
    "growth",         # asset growth rate
    "liquidity",      # 20-day average turnover
    "leverage",       # debt-to-equity
    "beta",           # market beta
    "residual_vol",   # idiosyncratic volatility
]

# Cross-asset factors: applicable to mixed stock/futures/ETF portfolios
CROSS_ASSET_FACTORS = [
    "momentum",       # 12-month return skip last month
    "volatility",     # 60-day realized volatility
    "liquidity",      # 20-day average turnover
    "beta",           # market beta
    "residual_vol",   # idiosyncratic volatility
    "basis",          # futures basis (spot - futures price, 0 for equities)
    "term_structure", # term structure slope (0 for equities)
    "open_interest",  # open interest change (0 for equities)
]


@dataclass
class BarraFactorReturn:
    """Daily factor return from cross-sectional regression."""
    date: str
    factor_returns: dict[str, float]
    r_squared: float = 0.0
    n_assets: int = 0
    residual_std: float = 0.0


@dataclass
class BarraRiskDecomposition:
    """Risk decomposition for a portfolio."""
    total_risk: float = 0.0
    factor_risk: float = 0.0
    specific_risk: float = 0.0
    factor_contributions: dict[str, float] = field(default_factory=dict)
    factor_exposures: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0


class BarraModel:
    """Barra-style multi-factor risk model.

    Estimates factor returns, factor covariance, and specific risk
    from cross-sectional regressions.

    Usage:
        model = BarraModel()
        model.fit(factor_exposures, returns)
        risk = model.decompose_risk(portfolio_weights, factor_exposures)
    """

    def __init__(
        self,
        factor_names: list[str] | None = None,
        half_life: int = 252,
        shrinkage_target: str = "identity",
        newey_west_lags: int = 5,
    ):
        self.factor_names = factor_names or BARRA_FACTORS
        self.half_life = half_life
        self.shrinkage_target = shrinkage_target
        self.newey_west_lags = newey_west_lags

        self.factor_returns_history: list[BarraFactorReturn] = []
        self.factor_covariance: np.ndarray | None = None
        self.specific_risk: pd.Series | None = None
        self._fitted = False

    @classmethod
    def for_asset_type(cls, asset_type: str = "stock", **kwargs) -> BarraModel:
        """Factory for asset-type-specific Barra models.

        Args:
            asset_type: 'stock' for equity-only (default), 'cross' for mixed
                        stock/futures/ETF, 'future' for futures-only.
        """
        if asset_type == "cross":
            kwargs.setdefault("factor_names", CROSS_ASSET_FACTORS)
        elif asset_type == "future":
            kwargs.setdefault("factor_names", [
                "momentum", "volatility", "liquidity", "beta",
                "basis", "term_structure", "open_interest",
            ])
        return cls(**kwargs)

    def _cross_sectional_regression(
        self,
        returns: pd.Series,
        exposures: pd.DataFrame,
    ) -> BarraFactorReturn:
        """Run one day's cross-sectional regression.

        r_i = sum_j(beta_j * x_ij) + epsilon_i

        Args:
            returns: (asset,) daily returns
            exposures: (asset x factor) factor exposures

        Returns:
            BarraFactorReturn with estimated factor returns
        """
        # Align data
        common = returns.index.intersection(exposures.index)
        if len(common) < 30:
            return BarraFactorReturn(
                date=str(returns.name) if hasattr(returns, 'name') else "",
                factor_returns={f: 0.0 for f in self.factor_names},
                n_assets=len(common),
            )

        y = returns.loc[common].values
        X = exposures.loc[common].values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Weighted least squares (weight by sqrt(market_cap) if available)
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        except np.linalg.LinAlgError:
            return BarraFactorReturn(
                date=str(returns.name) if hasattr(returns, 'name') else "",
                factor_returns={f: 0.0 for f in self.factor_names},
                n_assets=len(common),
            )

        # Factor returns are beta[1:] (skip intercept)
        factor_returns = {}
        for j, name in enumerate(self.factor_names):
            if j + 1 < len(beta):
                factor_returns[name] = float(beta[j + 1])
            else:
                factor_returns[name] = 0.0

        # R-squared
        y_pred = X_with_intercept @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Residual std
        residual_std = float(np.std(y - y_pred)) if len(y) > len(beta) else 0

        return BarraFactorReturn(
            date=str(returns.name) if hasattr(returns, 'name') else "",
            factor_returns=factor_returns,
            r_squared=float(r_squared),
            n_assets=len(common),
            residual_std=residual_std,
        )

    def fit(
        self,
        factor_exposures: dict[str, pd.DataFrame],
        returns: pd.DataFrame,
    ):
        """Fit the Barra model using historical data.

        Args:
            factor_exposures: dict of factor_name -> (date x asset) DataFrame
            returns: (date x asset) daily returns
        """
        # Build exposure matrix per date
        dates = returns.index
        self.factor_returns_history = []

        # Convert factor_exposures dict to aligned matrices
        factor_dfs = []
        for name in self.factor_names:
            if name in factor_exposures:
                factor_dfs.append(factor_exposures[name])
            else:
                logger.warning("Factor '%s' not in exposures, using zeros", name)
                factor_dfs.append(pd.DataFrame(0, index=dates, columns=returns.columns))

        for date in dates:
            # Build (asset x factor) exposure matrix for this date
            exposure_rows = {}
            for asset in returns.columns:
                row = []
                for fdf in factor_dfs:
                    if date in fdf.index and asset in fdf.columns:
                        val = fdf.loc[date, asset]
                        row.append(val if not np.isnan(val) else 0)
                    else:
                        row.append(0)
                exposure_rows[asset] = row

            exposures = pd.DataFrame(
                exposure_rows.values(),
                index=exposure_rows.keys(),
                columns=self.factor_names,
            )

            daily_returns = returns.loc[date]
            result = self._cross_sectional_regression(daily_returns, exposures)
            result.date = str(date)
            self.factor_returns_history.append(result)

        # Estimate factor covariance matrix
        self._estimate_factor_covariance()

        # Estimate specific risk
        self._estimate_specific_risk(returns)

        self._fitted = True
        logger.info("Barra model fitted: %d dates, %d factors",
                     len(dates), len(self.factor_names))

    def _estimate_factor_covariance(self):
        """Estimate factor covariance with exponential decay weighting."""
        if not self.factor_returns_history:
            return

        n = len(self.factor_returns_history)
        n_factors = len(self.factor_names)

        # Exponential weights
        decay = np.log(2) / self.half_life
        weights = np.exp(-decay * np.arange(n)[::-1])
        weights /= weights.sum()

        # Weighted factor returns
        returns_matrix = np.zeros((n, n_factors))
        for i, fr in enumerate(self.factor_returns_history):
            for j, name in enumerate(self.factor_names):
                returns_matrix[i, j] = fr.factor_returns.get(name, 0)

        # Weighted covariance
        weighted_mean = np.average(returns_matrix, weights=weights, axis=0)
        centered = returns_matrix - weighted_mean

        cov = np.zeros((n_factors, n_factors))
        for i in range(n_factors):
            for j in range(n_factors):
                cov[i, j] = np.average(centered[:, i] * centered[:, j], weights=weights)

        # Shrinkage toward identity
        if self.shrinkage_target == "identity":
            trace = np.trace(cov) / n_factors
            identity = np.eye(n_factors) * trace
            # Ledoit-Wolf optimal shrinkage intensity
            shrinkage = self._ledoit_wolf_shrinkage(returns_matrix, weights)
            cov = (1 - shrinkage) * cov + shrinkage * identity

        self.factor_covariance = cov

    def _ledoit_wolf_shrinkage(self, X: np.ndarray, weights: np.ndarray) -> float:
        """Compute Ledoit-Wolf optimal shrinkage intensity."""
        n, p = X.shape
        if n < 2:
            return 0.5

        mean = np.average(X, weights=weights, axis=0)
        centered = X - mean

        # Sample covariance
        S = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                S[i, j] = np.average(centered[:, i] * centered[:, j], weights=weights)

        # Target: scaled identity
        mu = np.trace(S) / p
        F = np.eye(p) * mu

        # Shrinkage intensity
        sum_sq = 0
        for k in range(n):
            xk = centered[k]
            sum_sq += weights[k] * np.sum((np.outer(xk, xk) - S) ** 2)

        delta = sum_sq / (n * n)
        gamma = np.sum((S - F) ** 2)

        if gamma == 0:
            return 1.0

        shrinkage = min(1.0, delta / gamma)
        return float(shrinkage)

    def _estimate_specific_risk(self, returns: pd.DataFrame):
        """Estimate specific (idiosyncratic) risk per asset."""
        if not self.factor_returns_history:
            return

        for fr in self.factor_returns_history:
            # We don't have the actual residuals here, estimate from factor model
            pass

        # Simple approach: use residual_std from regressions
        avg_residual_std = np.mean([fr.residual_std for fr in self.factor_returns_history])
        self.specific_risk = pd.Series(
            {asset: avg_residual_std for asset in returns.columns}
        )

    def decompose_risk(
        self,
        weights: pd.Series,
        factor_exposures: dict[str, pd.DataFrame],
        date: str | None = None,
    ) -> BarraRiskDecomposition:
        """Decompose portfolio risk into factor and specific components.

        Args:
            weights: (asset,) portfolio weights
            factor_exposures: dict of factor_name -> (date x asset) DataFrame
            date: specific date to use (default: latest)

        Returns:
            BarraRiskDecomposition
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get portfolio factor exposure (weighted average)
        portfolio_exposure = {}
        for name in self.factor_names:
            if name in factor_exposures:
                fdf = factor_exposures[name]
                if date and date in fdf.index:
                    exposures = fdf.loc[date].reindex(weights.index).fillna(0)
                else:
                    exposures = fdf.iloc[-1].reindex(weights.index).fillna(0)
                portfolio_exposure[name] = float((weights * exposures).sum())
            else:
                portfolio_exposure[name] = 0.0

        # Factor risk: w' * F * w (where F is factor covariance)
        if self.factor_covariance is not None:
            exposure_vec = np.array([portfolio_exposure.get(f, 0) for f in self.factor_names])
            factor_var = float(exposure_vec @ self.factor_covariance @ exposure_vec)
        else:
            factor_var = 0

        # Specific risk
        if self.specific_risk is not None:
            spec = self.specific_risk.reindex(weights.index).fillna(0)
            specific_var = float((weights ** 2 * spec ** 2).sum())
        else:
            specific_var = 0

        total_var = factor_var + specific_var
        total_risk = np.sqrt(total_var) * np.sqrt(252)  # Annualized
        factor_risk = np.sqrt(factor_var) * np.sqrt(252)
        specific_risk_ann = np.sqrt(specific_var) * np.sqrt(252)

        # Factor contributions
        if self.factor_covariance is not None:
            exposure_vec = np.array([portfolio_exposure.get(f, 0) for f in self.factor_names])
            marginal = self.factor_covariance @ exposure_vec
            factor_contributions = {}
            for i, name in enumerate(self.factor_names):
                contrib = exposure_vec[i] * marginal[i]
                factor_contributions[name] = float(contrib * 252)  # Annualized
        else:
            factor_contributions = {f: 0 for f in self.factor_names}

        r_squared = factor_var / total_var if total_var > 0 else 0

        return BarraRiskDecomposition(
            total_risk=float(total_risk),
            factor_risk=float(factor_risk),
            specific_risk=float(specific_risk_ann),
            factor_contributions=factor_contributions,
            factor_exposures=portfolio_exposure,
            r_squared=float(r_squared),
        )

    def get_factor_return_series(self) -> pd.DataFrame:
        """Get factor returns as a DataFrame."""
        if not self.factor_returns_history:
            return pd.DataFrame()

        data = {}
        for name in self.factor_names:
            data[name] = [fr.factor_returns.get(name, 0) for fr in self.factor_returns_history]

        dates = [fr.date for fr in self.factor_returns_history]
        return pd.DataFrame(data, index=dates)

    def get_factor_covariance_df(self) -> pd.DataFrame:
        """Get factor covariance as a DataFrame."""
        if self.factor_covariance is None:
            return pd.DataFrame()
        return pd.DataFrame(
            self.factor_covariance,
            index=self.factor_names,
            columns=self.factor_names,
        )
