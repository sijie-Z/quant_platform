"""Tests for alpha signal generation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.alpha.combination import (
    _align_factors,
    _build_row,
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
        raise AssertionError("Should have raised")
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


class TestAMisalignedFactorIsDropped:
    """A factor whose columns are not the asset universe used to union its
    labels into every cross-section, and pandas fills the misses with NaN:
    one bad column emptied the whole signal. Combined with the default
    21-factor config that was 0 of 546,376 cells populated and a backtest that
    never traded. The damage must stop at the factor itself."""

    @staticmethod
    def _with_a_broken_factor():
        factors = _make_factors()
        dates = factors["f1"].index
        # Exactly what `process_factor` built from a per-date Series.
        factors["broken"] = pd.DataFrame(
            {"factor": np.random.randn(len(dates))}, index=dates
        )
        return factors

    def test_equal_weight_keeps_the_universe_and_the_values(self):
        factors = self._with_a_broken_factor()
        result = combine_equal_weight(factors)
        assert result.shape == factors["f1"].shape
        assert result.notna().any().any()

    def test_ic_weighting_keeps_the_universe_and_the_values(self):
        factors = self._with_a_broken_factor()
        result = combine_ic_weighted(factors, _make_forward_returns())
        assert result.shape == factors["f1"].shape
        assert result.notna().any().any()

    def test_the_broken_factor_is_left_out_of_the_alignment(self):
        factors = self._with_a_broken_factor()
        aligned = _align_factors(factors)
        assert "broken" not in aligned
        assert set(aligned) == {"f1", "f2", "f3"}

    def test_a_factor_with_a_stray_column_is_aligned_to_the_universe(self):
        factors = _make_factors()
        widened = factors["f3"].copy()
        widened["not_an_asset"] = 1.0
        factors["widened"] = widened
        aligned = _align_factors(factors)
        assert list(aligned["widened"].columns) == list(factors["f1"].columns)

    def test_a_factor_with_no_asset_axis_is_rejected_loudly(self):
        factors = _make_factors()
        factors["series"] = factors["f1"].mean(axis=1)
        with pytest.raises(ValueError, match="Series"):
            _align_factors(factors)


class TestAnEmptyCrossSectionDoesNotEraseTheRow:
    """`row = row + vals * w` with an all-NaN `vals` makes every value NaN, so
    one factor with nothing to say about a date silenced every other factor on
    it -- and, if it came first in the dict, seeded the row that way."""

    def test_the_row_keeps_the_factors_that_have_values(self):
        factors = _make_factors()
        dates, assets = factors["f1"].index, factors["f1"].columns
        empty = pd.DataFrame(np.nan, index=dates, columns=assets)
        aligned = _align_factors({"empty": empty, **factors})

        row = _build_row(aligned, {name: 1.0 for name in aligned}, dates[10])

        assert row is not None
        assert row.notna().any()
        assert set(row.dropna().index) == set(assets)
