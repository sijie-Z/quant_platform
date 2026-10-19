"""Factor Orthogonalization — remove redundancy between correlated factors.

Implements three orthogonalization methods:
1. Gram-Schmidt: Sequential orthogonalization (order-dependent)
2. PCA: Principal Component Orthogonalization (order-independent)
3. Symmetric/Canonical: Pairwise symmetric decomposition (order-independent)

Orthogonalization is critical in multi-factor models because:
- Correlated factors double-count the same alpha signal
- Redundant factors increase portfolio concentration risk
- Orthogonal factors provide cleaner risk decomposition

The choice of method depends on the use case:
- Gram-Schmidt: When factor priority ordering matters (e.g., fundamental > technical)
- PCA: When you want maximum variance explained with minimum factors
- Symmetric: When no factor should be privileged over others

Reference:
- Menchero et al. (2010): "A Common Framework for Equity Risk Models"
- Bai & Ng (2008): "Large Dimensional Factor Analysis"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_platform.utils.logging import get_logger

logger = get_logger(__name__)


class OrthMethod(StrEnum):
    """Orthogonalization methods."""
    GRAM_SCHMIDT = "gram_schmidt"
    PCA = "pca"
    SYMMETRIC = "symmetric"


@dataclass
class OrthResult:
    """Result of factor orthogonalization."""
    method: OrthMethod
    factors: dict[str, pd.DataFrame]   # Orthogonalized factors (date × asset)
    correlation_before: pd.DataFrame   # Original correlation matrix
    correlation_after: pd.DataFrame    # Correlation matrix after orthogonalization
    variance_explained: pd.Series | None = None  # For PCA: variance per component
    n_components: int = 0              # For PCA: number of components kept
    transform_matrix: np.ndarray | None = None  # Transformation matrix


class FactorOrthogonalizer:
    """Remove redundancy between correlated factors.

    All methods operate on cross-sectional factor values (date × asset).
    For each date, factors are orthogonalized independently.

    Usage:
        orth = FactorOrthogonalizer()
        result = orth.orthogonalize(
            factors={"momentum": mom_df, "value": val_df, "quality": qual_df},
            method=OrthMethod.SYMMETRIC,
        )
        clean_factors = result.factors  # dict of orthogonalized DataFrames
    """

    def orthogonalize(
        self,
        factors: dict[str, pd.DataFrame],
        method: OrthMethod = OrthMethod.SYMMETRIC,
        n_components: int | None = None,
        variance_threshold: float = 0.95,
        priority_order: list[str] | None = None,
    ) -> OrthResult:
        """Orthogonalize a set of factors.

        Args:
            factors: Dict of {name: DataFrame(date × asset)} factor values.
            method: Orthogonalization method.
            n_components: For PCA, number of components to keep (overrides threshold).
            variance_threshold: For PCA, cumulative variance to explain (default 95%).
            priority_order: For Gram-Schmidt, factor ordering (first = highest priority).

        Returns:
            OrthResult with orthogonalized factors and diagnostics.
        """
        if len(factors) < 2:
            raise ValueError("Need at least 2 factors to orthogonalize")

        # Align all factors to common dates and assets
        names = list(factors.keys())
        common_dates = factors[names[0]].index
        common_assets = factors[names[0]].columns
        for name in names[1:]:
            common_dates = common_dates.intersection(factors[name].index)
            common_assets = common_assets.intersection(factors[name].columns)

        if len(common_dates) == 0 or len(common_assets) == 0:
            raise ValueError("No common dates or assets across factors")

        # Build aligned factor matrices
        aligned = {
            name: factors[name].loc[common_dates, common_assets].copy()
            for name in names
        }

        # Original correlation
        corr_before = self._cross_sectional_correlation(aligned)

        # Dispatch to method
        if method == OrthMethod.GRAM_SCHMIDT:
            result_factors, transform = self._gram_schmidt(
                aligned, priority_order or names
            )
            var_explained = None
            n_comp = len(names)
        elif method == OrthMethod.PCA:
            result_factors, transform, var_explained, n_comp = self._pca(
                aligned, n_components, variance_threshold
            )
        elif method == OrthMethod.SYMMETRIC:
            result_factors, transform = self._symmetric(aligned)
            var_explained = None
            n_comp = len(names)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Post-orthogonalization correlation
        corr_after = self._cross_sectional_correlation(result_factors)

        logger.info(
            "Orthogonalized %d factors via %s: max |corr| before=%.3f, after=%.3f",
            len(names), method.value,
            corr_before.values[np.triu_indices_from(corr_before.values, k=1)].max()
            if corr_before.shape[0] > 1 else 0,
            corr_after.values[np.triu_indices_from(corr_after.values, k=1)].max()
            if corr_after.shape[0] > 1 else 0,
        )

        return OrthResult(
            method=method,
            factors=result_factors,
            correlation_before=corr_before,
            correlation_after=corr_after,
            variance_explained=var_explained,
            n_components=n_comp,
            transform_matrix=transform,
        )

    def _cross_sectional_correlation(
        self, factors: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute average cross-sectional correlation between factors."""
        names = list(factors.keys())
        n = len(names)
        if n < 2:
            return pd.DataFrame([[1.0]], index=names, columns=names)

        # Stack factor values for each date, compute correlation
        corr_sum = np.zeros((n, n))
        count = 0

        dates = factors[names[0]].index
        for date in dates:
            # Build matrix: rows = assets, cols = factors
            cols = []
            for name in names:
                s = factors[name].loc[date].dropna()
                cols.append(s)

            # Find common assets
            common = cols[0].index
            for c in cols[1:]:
                common = common.intersection(c.index)

            if len(common) < 30:
                continue

            mat = np.column_stack([c[common].values for c in cols])
            corr = np.corrcoef(mat, rowvar=False)
            corr_sum += corr
            count += 1

        if count == 0:
            return pd.DataFrame(np.eye(n), index=names, columns=names)

        avg_corr = corr_sum / count
        np.fill_diagonal(avg_corr, 1.0)
        return pd.DataFrame(avg_corr, index=names, columns=names)

    def _gram_schmidt(
        self,
        factors: dict[str, pd.DataFrame],
        priority_order: list[str],
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
        """Gram-Schmidt sequential orthogonalization.

        Factors earlier in priority_order are preserved as-is.
        Later factors are made orthogonal to all earlier factors.

        This is order-dependent: the first factor keeps its original form,
        while subsequent factors are residuals after projecting out prior factors.

        Args:
            factors: Aligned factor DataFrames.
            priority_order: Factor names in priority order.

        Returns:
            Tuple of (orthogonalized factors dict, transformation matrix).
        """
        dates = list(factors.values())[0].index
        n_factors = len(priority_order)
        transform = np.eye(n_factors)

        # Store orthogonalized values
        result = {name: factors[name].copy() for name in priority_order}

        for date in dates:
            # Build matrix: rows = assets, cols = factors (in priority order)
            # Rows must be the SAME assets for every factor: independent
            # dropna() calls line up different assets positionally.
            common = None
            for name in priority_order:
                index = factors[name].loc[date].dropna().index
                common = index if common is None else common.intersection(index)
            min_len = len(common) if common is not None else 0
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([
                factors[name].loc[date, common].values for name in priority_order
            ])

            # Standardize each column
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # Gram-Schmidt
            orth = np.zeros_like(mat_std)
            for i in range(n_factors):
                v = mat_std[:, i].copy()
                for j in range(i):
                    proj = np.dot(v, orth[:, j]) / np.dot(orth[:, j], orth[:, j])
                    v -= proj * orth[:, j]
                    transform[i, j] = proj
                orth[:, i] = v

            # Write back to DataFrames (re-standardize for consistent scale)
            for i, name in enumerate(priority_order):
                col_std = orth[:, i].std()
                if col_std > 1e-10:
                    orth[:, i] = orth[:, i] / col_std
                assets = common
                result[name].loc[date, assets] = orth[:, i]

        return result, transform

    @staticmethod
    def _fix_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
        """Fix each eigenvector's sign so the largest-|component| is positive.

        Eigenvectors are defined up to sign.  Re-estimating a decomposition on
        a slightly different window can flip a component's sign, silently
        turning an "orthogonalized factor" into its negation.  Pinning the
        sign makes the output deterministic.
        """
        if eigenvectors.size == 0:
            return eigenvectors
        columns = np.arange(eigenvectors.shape[1])
        anchor = np.abs(eigenvectors).argmax(axis=0)
        signs = np.sign(eigenvectors[anchor, columns])
        signs[signs == 0] = 1.0
        return eigenvectors * signs

    def _pca(
        self,
        factors: dict[str, pd.DataFrame],
        n_components: int | None,
        variance_threshold: float,
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray, pd.Series, int]:
        """PCA-based orthogonalization, estimated causally.

        At each date t the covariance matrix is estimated over every valid
        cross-section up to and including t, and decomposed once.  Earlier
        implementations averaged eigenvectors over the whole sample (future
        data, and an average of arbitrary-sign vectors), so every date's
        output depended on the future and the components were not even
        guaranteed to be orthogonal.

        The number of kept components is fixed from the first valid
        cross-section: the output width must not change when later data
        changes, or past rows of the result frame would silently depend on
        the future.

        Args:
            factors: Aligned factor DataFrames.
            n_components: Fixed number of components (overrides threshold).
            variance_threshold: Cumulative variance to explain.

        Returns:
            Tuple of (orthogonalized factors, transform matrix, variance explained, n_components).
        """
        dates = list(factors.values())[0].index
        names = list(factors.keys())
        n_factors = len(names)

        cov_sum = np.zeros((n_factors, n_factors))
        count = 0
        n_comp: int | None = None
        out_names: list[str] = []
        last_eigenvalues: np.ndarray | None = None
        transform: np.ndarray | None = None
        # (date, common assets, scores) for valid cross-sections; frames are
        # assembled at the end because the component count is only known
        # after the first valid date.
        date_scores: list[tuple[pd.Timestamp, pd.Index, np.ndarray]] = []

        for date in dates:
            # Rows must be the SAME assets for every factor: independent
            # dropna() calls line up different assets positionally.
            common = None
            for name in names:
                index = factors[name].loc[date].dropna().index
                common = index if common is None else common.intersection(index)
            min_len = len(common) if common is not None else 0
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([
                factors[name].loc[date, common].values for name in names
            ])

            # Standardize
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # Covariance matrix
            cov = np.cov(mat_std, rowvar=False)
            if cov.ndim == 0 or not np.isfinite(cov).all():
                continue

            # Expanding window: the estimate on date t uses only dates <= t.
            cov_sum += cov
            count += 1
            avg_cov = cov_sum / count

            # Eigendecomposition of the as-of-t covariance matrix.
            eigenvalues, eigenvectors = np.linalg.eigh(avg_cov)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenvectors = self._fix_eigenvector_signs(eigenvectors)

            if n_comp is None:
                # Determine the number of components once, from the first
                # valid cross-section (causal by construction).
                total_var = eigenvalues.clip(min=0).sum()
                if total_var <= 0:
                    n_comp = n_factors
                elif n_components is not None:
                    n_comp = min(n_components, n_factors)
                else:
                    cum_var = (
                        np.cumsum(eigenvalues.clip(min=0)) / total_var
                    )
                    n_comp = int(np.searchsorted(cum_var, variance_threshold)) + 1
                    n_comp = max(1, min(n_comp, n_factors))
                out_names = names[:n_comp]

            # PCA scores: uncorrelated by construction (within the window the
            # transform was estimated on).
            scores = mat_std @ eigenvectors[:, :n_comp]

            # Re-standardize each component for consistent scale
            for i in range(n_comp):
                col_std = scores[:, i].std()
                if col_std > 1e-10:
                    scores[:, i] = scores[:, i] / col_std

            date_scores.append((date, common, scores))
            last_eigenvalues = eigenvalues
            transform = eigenvectors[:, :n_comp]

        if count == 0:
            raise ValueError("No valid dates for PCA")

        result = {name: factors[name].copy() for name in out_names}
        for date, common, scores in date_scores:
            for i, name in enumerate(out_names):
                result[name].loc[date, common] = scores[:, i]

        # Variance explained, reported from the latest (largest) window.
        total_var = float(last_eigenvalues.clip(min=0).sum())
        var_explained = pd.Series(
            last_eigenvalues / total_var if total_var > 0 else np.zeros(n_factors),
            index=[f"PC{i+1}" for i in range(n_factors)],
            name="variance_explained",
        )

        return result, transform, var_explained, n_comp

    def _symmetric(
        self,
        factors: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
        """Symmetric (canonical) orthogonalization, estimated causally.

        Uses eigendecomposition of the factor correlation matrix to
        create orthogonal factors that are symmetric transforms of
        the originals. No factor is privileged.

        The transform is: F_orth = F × V × D^{-1/2} × V'
        where V is the eigenvector matrix and D is the eigenvalue matrix.

        The correlation matrix is the average of the daily cross-sectional
        correlations observed up to and including the current date (an
        expanding window).  Estimating it on the full sample -- the old
        behaviour -- let a date's output depend on future returns.

        Args:
            factors: Aligned factor DataFrames.

        Returns:
            Tuple of (orthogonalized factors, transformation matrix).
        """
        dates = list(factors.values())[0].index
        names = list(factors.keys())
        n_factors = len(names)

        result = {name: factors[name].copy() for name in names}

        # Expanding-window correlation estimate.
        corr_sum = np.zeros((n_factors, n_factors))
        count = 0
        transform = np.eye(n_factors)

        for date in dates:
            # Rows must be the SAME assets for every factor: independent
            # dropna() calls line up different assets positionally.
            common = None
            for name in names:
                index = factors[name].loc[date].dropna().index
                common = index if common is None else common.intersection(index)
            min_len = len(common) if common is not None else 0
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([
                factors[name].loc[date, common].values for name in names
            ])
            corr = np.corrcoef(mat, rowvar=False)
            if not np.isfinite(corr).all():
                continue
            corr_sum += corr
            count += 1
            avg_corr = corr_sum / count
            np.fill_diagonal(avg_corr, 1.0)

            # Eigendecomposition of the as-of-date correlation matrix.
            eigenvalues, eigenvectors = np.linalg.eigh(avg_corr)
            eigenvectors = self._fix_eigenvector_signs(eigenvectors)

            # Clip small eigenvalues to avoid numerical issues
            eigenvalues = np.maximum(eigenvalues, 1e-8)

            # Transform: V × D^{-1/2} × V'
            # This maps correlated factors to uncorrelated space
            d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
            transform = eigenvectors @ d_inv_sqrt @ eigenvectors.T

            # Standardize
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # Apply symmetric transform
            orth = mat_std @ transform

            # Re-standardize for consistent scale
            for i, name in enumerate(names):
                col = orth[:, i]
                col_std = col.std()
                if col_std > 1e-10:
                    col = col / col_std
                result[name].loc[date, common] = col

        if count == 0:
            raise ValueError("No valid dates for symmetric orthogonalization")

        return result, transform


def orthogonalize_factors(
    factors: dict[str, pd.DataFrame],
    method: str = "symmetric",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Convenience function for factor orthogonalization.

    Args:
        factors: Dict of {name: DataFrame(date × asset)}.
        method: "gram_schmidt", "pca", or "symmetric".
        **kwargs: Additional arguments passed to orthogonalize().

    Returns:
        Dict of orthogonalized factor DataFrames.
    """
    orth = FactorOrthogonalizer()
    result = orth.orthogonalize(
        factors, method=OrthMethod(method), **kwargs
    )
    return result.factors
