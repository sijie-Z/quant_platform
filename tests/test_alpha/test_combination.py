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


class TestPartialNaNCoverage:
    """A missing cell in one factor must not blank the asset."""

    def test_one_missing_cell_does_not_blank_the_asset(self):
        factors = _make_factors(n_dates=10, n_assets=5)
        factors["f1"].iloc[0, 0] = np.nan

        result = combine_equal_weight(factors)

        # Missing = 0 (the cross-sectional mean of a z-scored factor); the
        # denominator stays sum(|w|) = 1, so f1 contributes 0, not a
        # renormalisation of the other two factors.
        expected = (factors["f2"].iloc[0, 0] + factors["f3"].iloc[0, 0]) / 3
        assert result.iloc[0, 0] == pytest.approx(expected)

    def test_an_all_nan_factor_does_not_empty_the_signal(self):
        factors = _make_factors(n_dates=10, n_assets=5)
        factors["f1"][:] = np.nan

        result = combine_equal_weight(factors)

        assert result.shape == factors["f2"].shape
        assert result.notna().all().all()

    def test_disjoint_nan_masks_cover_the_union(self):
        factors = _make_factors(n_dates=10, n_assets=6)
        factors["f1"].iloc[:, :3] = np.nan
        factors["f2"].iloc[:, 3:] = np.nan

        result = combine_equal_weight(factors)

        assert result.notna().all().all()

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


class TestMissingFactorIsZeroNotARenormalisation:
    """A missing factor must not change the scale of the signal.

    The mask-aware denominator removed a missing factor from *both* numerator
    and denominator, so under 0.8/0.2 weights an asset missing the dominant
    factor was scored on the 0.2 factor alone (renormalised to 1.0) and could
    outrank an asset that had both. Missing values are now filled with 0 --
    the mean of a z-scored factor -- and the denominator is the fixed
    ``sum(abs(w))``.
    """

    @staticmethod
    def _aligned():
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        idx = dates[0]
        main = pd.DataFrame(
            {"A": [1.0, 1.0, 1.0], "B": [np.nan, np.nan, np.nan]}, index=dates
        )
        secondary = pd.DataFrame(
            {"A": [1.0, 1.0, 1.0], "B": [1.2, 1.2, 1.2]}, index=dates
        )
        return _align_factors({"main": main, "secondary": secondary}), idx

    def test_build_row_does_not_reward_the_missing_dominant_factor(self):
        aligned, date = self._aligned()

        row = _build_row(aligned, {"main": 0.8, "secondary": 0.2}, date)

        # Old behaviour: A = 1.0, B = 1.2 -> B ranked above A.
        # New: A = (0.8*1 + 0.2*1) / 1.0 = 1.0; B = (0 + 0.2*1.2) / 1.0 = 0.24.
        assert row["A"] == pytest.approx(1.0)
        assert row["B"] == pytest.approx(0.24)
        assert row["A"] > row["B"]

    def test_weighted_sum_uses_the_fixed_denominator(self):
        aligned, date = self._aligned()

        result = combine_equal_weight(dict(aligned))

        # Equal weights 0.5 each, denominator 1.0 for every asset.
        assert result.loc[date, "A"] == pytest.approx(1.0)
        assert result.loc[date, "B"] == pytest.approx(0.6)

    def test_full_coverage_is_unaffected(self):
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        aligned = _align_factors({
            "f1": pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=dates),
            "f2": pd.DataFrame({"A": [5.0, 6.0], "B": [7.0, 8.0]}, index=dates),
        })
        row = _build_row(aligned, {"f1": 0.8, "f2": 0.2}, dates[0])
        assert row["A"] == pytest.approx(0.8 * 1.0 + 0.2 * 5.0)
        assert row["B"] == pytest.approx(0.8 * 3.0 + 0.2 * 7.0)
