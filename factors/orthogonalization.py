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
            vals = []
            for name in priority_order:
                vals.append(factors[name].loc[date].dropna().values)

            # Find common length
            min_len = min(len(v) for v in vals)
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([v[:min_len] for v in vals])

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
                assets = factors[name].loc[date].dropna().index[:min_len]
                result[name].loc[date, assets] = orth[:, i]

        return result, transform

    def _pca(
        self,
        factors: dict[str, pd.DataFrame],
        n_components: int | None,
        variance_threshold: float,
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray, pd.Series, int]:
        """PCA-based orthogonalization.

        Decomposes factors into principal components, then reconstructs
        using the top components that explain sufficient variance.

        This is order-independent and captures maximum information in
        minimum dimensions.

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

        # Accumulate PCA across dates
        all_components = None
        all_eigenvalues = None
        count = 0

        for date in dates:
            vals = []
            for name in names:
                vals.append(factors[name].loc[date].dropna().values)

            min_len = min(len(v) for v in vals)
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([v[:min_len] for v in vals])

            # Standardize
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # Covariance matrix
            cov = np.cov(mat_std, rowvar=False)
            if cov.ndim == 0:
                continue

            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            if all_eigenvalues is None:
                all_eigenvalues = eigenvalues
                all_components = eigenvectors
            else:
                all_eigenvalues += eigenvalues
                all_components += eigenvectors
            count += 1

        if count == 0:
            raise ValueError("No valid dates for PCA")

        avg_eigenvalues = all_eigenvalues / count
        avg_components = all_components / count

        # Determine number of components
        total_var = avg_eigenvalues.sum()
        if total_var <= 0:
            n_comp = n_factors
        else:
            cum_var = np.cumsum(avg_eigenvalues) / total_var
            if n_components is not None:
                n_comp = min(n_components, n_factors)
            else:
                n_comp = int(np.searchsorted(cum_var, variance_threshold)) + 1
                n_comp = max(1, min(n_comp, n_factors))

        # Variance explained
        var_explained = pd.Series(
            avg_eigenvalues / total_var if total_var > 0 else np.zeros(n_factors),
            index=[f"PC{i+1}" for i in range(n_factors)],
            name="variance_explained",
        )

        # Use top n_comp components
        top_components = avg_components[:, :n_comp]
        transform = top_components

        # Project factors into PCA space (scores are uncorrelated by construction)
        # Map first n_comp original factor names to PCA components
        out_names = names[:n_comp]
        result = {name: factors[names[0]].copy() for name in out_names}

        for date in dates:
            vals = []
            for name in names:
                vals.append(factors[name].loc[date].dropna().values)

            min_len = min(len(v) for v in vals)
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([v[:min_len] for v in vals])
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # PCA scores: uncorrelated by construction
            scores = mat_std @ top_components  # (assets × n_comp)

            # Re-standardize each component for consistent scale
            assets = factors[names[0]].loc[date].dropna().index[:min_len]
            for i, name in enumerate(out_names):
                col = scores[:, i]
                col_std = col.std()
                if col_std > 1e-10:
                    col = col / col_std
                result[name].loc[date, assets] = col

        return result, transform, var_explained, n_comp

    def _symmetric(
        self,
        factors: dict[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
        """Symmetric (canonical) orthogonalization.

        Uses eigendecomposition of the factor correlation matrix to
        create orthogonal factors that are symmetric transforms of
        the originals. No factor is privileged.

        This is the recommended method for multi-factor models where
        no factor has a natural priority ordering.

        The transform is: F_orth = F × V × D^{-1/2} × V'
        where V is the eigenvector matrix and D is the eigenvalue matrix.

        Args:
            factors: Aligned factor DataFrames.

        Returns:
            Tuple of (orthogonalized factors, transformation matrix).
        """
        dates = list(factors.values())[0].index
        names = list(factors.keys())
        n_factors = len(names)

        # Accumulate correlation matrix
        corr_sum = np.zeros((n_factors, n_factors))
        count = 0

        for date in dates:
            vals = []
            for name in names:
                vals.append(factors[name].loc[date].dropna().values)

            min_len = min(len(v) for v in vals)
            if min_len < n_factors + 1:
                continue

            mat = np.column_stack([v[:min_len] for v in vals])
            corr = np.corrcoef(mat, rowvar=False)
            corr_sum += corr
            count += 1

        if count == 0:
            raise ValueError("No valid dates for symmetric orthogonalization")

        avg_corr = corr_sum / count
        np.fill_diagonal(avg_corr, 1.0)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(avg_corr)

        # Clip small eigenvalues to avoid numerical issues
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Transform: V × D^{-1/2} × V'
        # This maps correlated factors to uncorrelated space
        d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        transform = eigenvectors @ d_inv_sqrt @ eigenvectors.T

        # Apply transformation
        result = {}
        for date in dates:
            vals = []
            for name in names:
                vals.append(factors[name].loc[date].dropna().values)

            min_len = min(len(v) for v in vals)
            if min_len < n_factors + 1:
                for name in names:
                    result.setdefault(name, factors[name].copy())
                continue

            mat = np.column_stack([v[:min_len] for v in vals])

            # Standardize
            means = mat.mean(axis=0)
            stds = mat.std(axis=0)
            stds[stds < 1e-10] = 1.0
            mat_std = (mat - means) / stds

            # Apply symmetric transform
            orth = mat_std @ transform

            # Re-standardize for consistent scale
            for i, name in enumerate(names):
                assets = factors[name].loc[date].dropna().index[:min_len]
                result.setdefault(name, factors[name].copy())
                col = orth[:, i]
                col_std = col.std()
                if col_std > 1e-10:
                    col = col / col_std
                result[name].loc[date, assets] = col

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
