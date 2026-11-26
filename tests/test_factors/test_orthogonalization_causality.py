"""Causal estimation for PCA and symmetric orthogonalization.

The old implementation accumulated the correlation / covariance matrix over
the *whole sample*, decomposed it once, and PCA additionally averaged the
per-date eigenvectors (whose signs and rotations are arbitrary).  Every
date's output therefore depended on future data, and the "orthogonalized"
PCA components were not even orthogonal.

The fix estimates the transform per date from an expanding window ending at
that date, decomposes that matrix once, and pins each eigenvector's sign.
Gram-Schmidt was already causal and is unaffected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.orthogonalization import (
    FactorOrthogonalizer,
    OrthMethod,
)

CUT = 60
N_DATES = 140
N_ASSETS = 60


def _factors(seed: int = 5) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=N_DATES)
    assets = [f"S{i:03d}" for i in range(N_ASSETS)]
    common = rng.normal(0, 1, (N_DATES, N_ASSETS))
    return {
        "a": pd.DataFrame(
            common + rng.normal(0, 0.3, common.shape),
            index=dates, columns=assets,
        ),
        "b": pd.DataFrame(
            common + rng.normal(0, 0.3, common.shape),
            index=dates, columns=assets,
        ),
        "c": pd.DataFrame(
            rng.normal(0, 1, common.shape), index=dates, columns=assets,
        ),
    }


def _replace_future(
    factors: dict[str, pd.DataFrame], cut: int = CUT,
) -> dict[str, pd.DataFrame]:
    """Same data before `cut`, entirely different draws at and after it."""
    rng = np.random.default_rng(999)
    replaced = {}
    for name, frame in factors.items():
        frame = frame.copy()
        future = frame.index >= frame.index[cut]
        frame.loc[future, :] = rng.normal(
            0, 5, (int(future.sum()), frame.shape[1])
        )
        replaced[name] = frame
    return replaced


@pytest.mark.parametrize("method", [OrthMethod.PCA, OrthMethod.SYMMETRIC])
def test_replacing_the_future_does_not_change_past_outputs(method):
    """A date's orthogonalized values may only use data up to that date."""
    factors = _factors()
    reference = FactorOrthogonalizer().orthogonalize(
        factors, method=method, n_components=3,
    )
    perturbed = FactorOrthogonalizer().orthogonalize(
        _replace_future(factors), method=method, n_components=3,
    )

    assert set(reference.factors) == set(perturbed.factors)
    for name in reference.factors:
        np.testing.assert_allclose(
            reference.factors[name].iloc[:CUT].to_numpy(),
            perturbed.factors[name].iloc[:CUT].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.parametrize("method", [OrthMethod.PCA, OrthMethod.SYMMETRIC])
def test_orthogonalized_factors_are_pairwise_uncorrelated(method):
    result = FactorOrthogonalizer().orthogonalize(
        _factors(), method=method, n_components=3,
    )
    corr = result.correlation_after.to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.1


def test_pca_eigenvector_signs_are_pinned():
    """Largest-|entry| of every returned component loading is positive.

    Eigenvectors are sign-ambiguous; without a convention the output can
    silently flip.  Replacing the future data must also leave signs alone --
    covered by the causality test -- and the returned transform itself must
    satisfy the convention.
    """
    result = FactorOrthogonalizer().orthogonalize(
        _factors(), method=OrthMethod.PCA, n_components=3,
    )
    transform = result.transform_matrix
    anchor = np.abs(transform).argmax(axis=0)
    columns = np.arange(transform.shape[1])
    assert (transform[anchor, columns] >= 0).all()


def test_gram_schmidt_unchanged_by_future_replacement():
    """Gram-Schmidt was already causal; the fix must not regress that."""
    factors = _factors()
    reference = FactorOrthogonalizer().orthogonalize(
        factors, method=OrthMethod.GRAM_SCHMIDT,
    )
    perturbed = FactorOrthogonalizer().orthogonalize(
        _replace_future(factors), method=OrthMethod.GRAM_SCHMIDT,
    )
    for name in reference.factors:
        np.testing.assert_allclose(
            reference.factors[name].iloc[:CUT].to_numpy(),
            perturbed.factors[name].iloc[:CUT].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
        )
