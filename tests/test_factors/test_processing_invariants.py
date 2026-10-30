"""Factor-processing invariants: partial winsorize bounds and breadth guard.

Regression for two audit survivors in ``factors/processing.py``:

- winsorize's early return used ``or`` instead of ``and``, so a partial bound
  such as ``lower=0.0, upper=0.99`` silently disabled all clipping.
- neutralization's breadth guard was loose enough that 10-19-name
  cross-sections were regressed on sector dummies plus log market cap,
  replacing the raw signal with in-sample noise. The documented guard is
  20 names.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.processing import neutralize, winsorize

WINSOR_ASSETS = 50
WINSOR_DATES = 5


def _factor_with_outliers() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2024-01-02", periods=WINSOR_DATES)
    assets = [f"S{i:03d}" for i in range(WINSOR_ASSETS)]
    data = rng.normal(size=(WINSOR_DATES, WINSOR_ASSETS))
    data[:, 0] = 10.0    # upper-tail outlier
    data[:, 1] = -10.0   # lower-tail outlier
    return pd.DataFrame(data, index=dates, columns=assets)


class TestPartialWinsorizeBounds:
    def test_upper_tail_is_clipped_when_lower_bound_is_zero(self):
        factor = _factor_with_outliers()

        out = winsorize(factor, lower=0.0, upper=0.99)

        for date in factor.index:
            raw = factor.loc[date]
            q99 = raw.quantile(0.99)
            assert raw.max() > q99, "fixture must actually need upper clipping"
            assert out.loc[date].max() == pytest.approx(q99, rel=0, abs=1e-12)
            assert out.loc[date].max() < raw.max()

    def test_lower_tail_is_clipped_when_upper_bound_is_one(self):
        factor = _factor_with_outliers()

        out = winsorize(factor, lower=0.01, upper=1.0)

        for date in factor.index:
            raw = factor.loc[date]
            q01 = raw.quantile(0.01)
            assert raw.min() < q01, "fixture must actually need lower clipping"
            assert out.loc[date].min() == pytest.approx(q01, rel=0, abs=1e-12)
            assert out.loc[date].min() > raw.min()

    def test_both_tails_still_clip_with_the_default_bounds(self):
        factor = _factor_with_outliers()

        out = winsorize(factor, lower=0.01, upper=0.99)

        for date in factor.index:
            raw = factor.loc[date]
            assert out.loc[date].max() < raw.max()
            assert out.loc[date].min() > raw.min()

    def test_a_fully_open_interval_is_an_exact_no_op(self):
        factor = _factor_with_outliers()

        out = winsorize(factor, lower=0.0, upper=1.0)

        pd.testing.assert_frame_equal(out, factor)


def _neutralize_inputs(n_assets: int, seed: int = 21):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=5)
    assets = [f"S{i:02d}" for i in range(n_assets)]
    labels = np.array(["Tech", "Finance", "Energy"])
    sectors = pd.Series(labels[np.arange(n_assets) % 3], index=assets)
    market_cap = pd.DataFrame(
        rng.lognormal(np.log(5e9), 0.5, size=(len(dates), n_assets)),
        index=dates,
        columns=assets,
    )
    factor = pd.DataFrame(rng.normal(size=(len(dates), n_assets)),
                          index=dates, columns=assets)
    factor = factor + 0.3 * np.log(market_cap)
    factor.loc[:, sectors == "Finance"] += 2.0
    return factor, sectors, market_cap


class TestNeutralizeBreadthGuard:
    def test_below_twenty_names_is_returned_bit_identical(self):
        factor, sectors, market_cap = _neutralize_inputs(15)

        out = neutralize(factor, sector_map=sectors, market_cap=market_cap)

        pd.testing.assert_frame_equal(out, factor)

    def test_nineteen_names_skip_and_twenty_names_neutralize(self):
        factor, sectors, market_cap = _neutralize_inputs(20)

        nineteen = neutralize(
            factor.iloc[:, :19],
            sector_map=sectors.iloc[:19],
            market_cap=market_cap.iloc[:, :19],
        )
        twenty = neutralize(factor, sector_map=sectors, market_cap=market_cap)

        pd.testing.assert_frame_equal(nineteen, factor.iloc[:, :19])
        assert not np.allclose(twenty.to_numpy(), factor.to_numpy())

    def test_fifty_names_are_actually_neutralized(self):
        factor, sectors, market_cap = _neutralize_inputs(50)

        out = neutralize(factor, sector_map=sectors, market_cap=market_cap)

        assert out.shape == factor.shape
        assert not np.allclose(out.to_numpy(), factor.to_numpy())
        for date in factor.index:
            row = out.loc[date].to_numpy()
            for sector in ("Tech", "Finance", "Energy"):
                mask = (sectors == sector).to_numpy()
                assert abs(row[mask].mean()) < 1e-8
