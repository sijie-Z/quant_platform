"""Regression tests for PureVolatilityFactor's AR(lags) filter.

`pure_volatility` is enabled in `config/default.yaml` and is in the default
factor set, so this runs on every pipeline invocation.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.technical import PureVolatilityFactor


def _panel(n_days: int = 260, n_assets: int = 8, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_assets)]
    steps = rng.normal(0, 0.012, (n_days, n_assets))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=cols
    )


class TestASuspensionDoesNotRemoveTheFactor:
    """The residuals were written to the wrong dates.

    They were assigned to `ivol.index[ivol.notna().any(axis=1)][lags:]` -- the
    dates where *any* asset has data -- while the series came from
    `ivol[asset].dropna()`, that asset's own dates. One suspended stock made
    the two lengths differ and the insert raised ValueError, which
    `except np.linalg.LinAlgError` did not catch. main.py logs any factor
    failure as a warning, so the factor disappeared from the run.
    """

    def test_a_suspended_stock_no_longer_raises(self):
        prices = _panel()
        prices.iloc[60:100, 1] = np.nan  # S01 suspended for 40 sessions

        out = PureVolatilityFactor().compute(prices)

        assert out.shape[0] == prices.shape[0]
        assert out.notna().to_numpy().any()

    def test_the_gap_is_preserved_in_the_output(self):
        prices = _panel()
        gap = prices.index[60:100]
        prices.loc[gap, "S01"] = np.nan

        out = PureVolatilityFactor().compute(prices)

        # No value may be invented for a date the stock did not trade.
        assert out.loc[gap, "S01"].isna().all(), (
            "the factor produced values while the stock was suspended"
        )

    @pytest.mark.parametrize("gap_start", [30, 60, 100, 150, 200, 240])
    def test_no_gap_position_crashes_the_factor(self, gap_start):
        """The shape mismatch depended on where the gap fell: the residual
        array came from the asset's own rows and the target index from the
        any-asset rows, so the two lengths agreed for some offsets and not
        others. Any offset used to raise."""
        prices = _panel()
        prices.iloc[gap_start:gap_start + 20, 1] = np.nan

        out = PureVolatilityFactor().compute(prices)

        assert out.shape[0] == prices.shape[0]
        assert out.notna().to_numpy().any()


class TestTheVectorizedWindowMatchesTheLoop:
    """The rolling regression used to be a python callback handed to
    `rolling().apply`, invoked once per asset per date -- 173 s for 300 assets
    x 1250 days, per pipeline run. It is now the same arithmetic done with
    window sums, which is only an improvement if it is genuinely the same
    arithmetic."""

    @staticmethod
    def _reference(ret: pd.DataFrame, window: int) -> pd.DataFrame:
        market_ret = ret.mean(axis=1)

        def _rolling_ivol(x):
            y = x.values
            m = market_ret.reindex(x.index).values
            if len(y) < window or np.std(m) < 1e-10:
                return np.nan
            beta = np.cov(y, m)[0, 1] / np.var(m)
            return float(np.std(y - beta * m, ddof=2))

        return ret.rolling(window).apply(_rolling_ivol, raw=False)

    @staticmethod
    def _ivol_stage(prices: pd.DataFrame) -> pd.DataFrame:
        """`compute` with an AR window longer than the sample, so the AR stage
        skips every asset and the rolling-vol result comes back untouched."""
        return PureVolatilityFactor(window=20, ar_lags=10_000).compute(prices)

    def test_matches_the_reference_on_clean_data(self):
        prices = _panel(n_days=160, n_assets=6)
        expected = self._reference(prices.pct_change(fill_method=None), 20)

        got = self._ivol_stage(prices)

        pd.testing.assert_frame_equal(got, expected, check_exact=False, atol=1e-12)

    def test_matches_the_reference_with_a_suspension(self):
        prices = _panel(n_days=160, n_assets=6)
        prices.iloc[70:95, 2] = np.nan
        expected = self._reference(prices.pct_change(fill_method=None), 20)

        got = self._ivol_stage(prices)

        pd.testing.assert_frame_equal(got, expected, check_exact=False, atol=1e-12)

    def test_the_window_is_a_day_wider_than_the_trading_week(self):
        """Guards against an off-by-one in the rolling sum alignment."""
        prices = _panel(n_days=80, n_assets=4)
        expected = self._reference(prices.pct_change(fill_method=None), 20)

        got = self._ivol_stage(prices)

        assert got.index[got.notna().to_numpy().any(axis=1)][0] == (
            expected.index[expected.notna().to_numpy().any(axis=1)][0]
        )


class TestTheFilterIsCausal:
    """The AR coefficients used to be fitted on the entire series, so the
    value at date t was computed with coefficients that had seen every later
    date -- in a factor whose docstring claims to remove 'cross-period
    information leakage'."""

    def test_rewriting_the_future_leaves_the_past_unchanged(self):
        full = _panel()
        factor = PureVolatilityFactor()

        whole = factor.compute(full)

        cut = full.index[180]
        truncated = full.loc[:cut]
        partial = factor.compute(truncated)

        overlap = partial.index
        pd.testing.assert_frame_equal(
            partial, whole.loc[overlap], check_exact=False, atol=1e-12,
        )

    def test_a_changed_future_leaves_the_past_unchanged(self):
        full = _panel()
        factor = PureVolatilityFactor()
        baseline = factor.compute(full)

        perturbed = full.copy()
        perturbed.iloc[200:] *= 1.5  # the future is different, the past is not
        after = factor.compute(perturbed)

        past = full.index[:200]
        pd.testing.assert_frame_equal(
            after.loc[past], baseline.loc[past], check_exact=False, atol=1e-12,
        )
