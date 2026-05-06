"""Tests for alpha signal generation."""

import numpy as np
import pandas as pd

from quant_platform.alpha.combination import (
    combine_equal_weight,
    combine_ic_weighted,
    combine_icir_weighted,
)


def _make_factors(n_dates=200, n_assets=50):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}.SH" for i in range(n_assets)]
    factors = {}
    for name in ["f1", "f2", "f3"]:
        data = np.random.randn(n_dates, n_assets)
        factors[name] = pd.DataFrame(data, index=dates, columns=assets)
    return factors


def _make_forward_returns(n_dates=200, n_assets=50):
    np.random.seed(99)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"{i:06d}.SH" for i in range(n_assets)]
    data = np.random.randn(n_dates, n_assets) * 0.02
    return pd.DataFrame(data, index=dates, columns=assets)


def test_combine_equal_weight():
    factors = _make_factors()
    result = combine_equal_weight(factors)
    assert result.shape == factors["f1"].shape
    assert not result.isnull().all().all()


def test_combine_empty():
    try:
        combine_equal_weight({})
        assert False, "Should have raised"
    except ValueError:
        pass


def test_combine_ic_weighted():
    factors = _make_factors()
    fwd_ret = _make_forward_returns()
    result = combine_ic_weighted(factors, fwd_ret)
    assert result.shape == factors["f1"].shape


def test_combine_icir_weighted():
    factors = _make_factors()
    fwd_ret = _make_forward_returns()
    result = combine_icir_weighted(factors, fwd_ret, min_icir=-99)
    assert result.shape == factors["f1"].shape
