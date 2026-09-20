"""Regression tests for the live signal generator.

Its module docstring makes a strong claim: "if backtest and live use different
signals, the backtest P&L is meaningless". The IC/ICIR weighting path broke
that claim by handing AlphaPipeline a random forward-return panel.
"""

import numpy as np
import pandas as pd
import pytest

from quant_platform.alpha.pipeline import AlphaPipeline
from quant_platform.trading.signal_generator import DEFAULT_LIVE_FACTORS, LiveSignalGenerator


def _price_history(n_days: int = 320, n_assets: int = 40, seed: int = 11) -> pd.DataFrame:
    """A universe big enough for the IC path to run.

    `rank_ic` skips any date with fewer than 30 assets
    (factors/evaluation.py:62), so a smaller fixture would silently send
    `ic_weighted` down its equal-weight fallback and test nothing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    assets = [f"{600000 + i}" for i in range(n_assets)]
    # A persistent drift so factors have something real to rank on.
    drift = rng.normal(0.0004, 0.0002, n_assets)
    steps = rng.normal(0, 0.015, (n_days, n_assets)) + drift
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=assets)


def _generator(method: str) -> LiveSignalGenerator:
    return LiveSignalGenerator(
        factor_names=["momentum_1m", "momentum_3m", "volatility_20d", "rsi_14d"],
        alpha_method=method,
    )


class TestForwardReturnsAreReal:
    """The panel fed to AlphaPipeline must be derived from prices."""

    def test_pipeline_receives_price_derived_forward_returns(self, monkeypatch):
        prices = _price_history()
        gen = _generator("ic_weighted")
        gen.update_prices(prices)

        captured = {}
        original = AlphaPipeline.run

        def spy(self, factors, forward_returns, *args, **kwargs):
            captured["forward_returns"] = forward_returns
            return original(self, factors, forward_returns, *args, **kwargs)

        monkeypatch.setattr(AlphaPipeline, "run", spy)
        gen.generate()

        assert "forward_returns" in captured, "AlphaPipeline.run was never called"
        received = captured["forward_returns"]

        # The pipeline's convention: close(s) -> close(s+1), so the last row
        # is unobservable and NaN.
        expected = prices.pct_change(fill_method=None).shift(-1)
        pd.testing.assert_frame_equal(received, expected)
        assert received.index.max() == prices.index.max()
        assert received.iloc[:-1].notna().all().all(), "realized rows must carry data"

    def test_signals_are_not_redrawn_each_cycle(self):
        """A random panel is a fresh draw per call, so two calls on an
        unchanged buffer used to disagree."""
        prices = _price_history()
        gen = _generator("icir_weighted")
        gen.update_prices(prices)

        first = gen.generate()
        second = gen.generate()

        assert first, "the generator produced no signal"
        assert first == pytest.approx(second), "signals changed without new prices"

    def test_no_random_numbers_are_drawn(self, monkeypatch):
        prices = _price_history()
        gen = _generator("icir_weighted")
        gen.update_prices(prices)

        def explode(*args, **kwargs):
            raise AssertionError("the signal path drew random numbers")

        monkeypatch.setattr(np.random, "default_rng", explode)
        gen.generate()


class TestTheFixtureExercisesTheIcPath:
    """Guard against the tests above passing vacuously.

    `rank_ic` needs 30+ assets per date (factors/evaluation.py:62). Below that
    every IC series is empty, `combine_ic_weighted` falls back to equal weight,
    and the forward-return panel stops mattering -- which is exactly how a
    25-asset fixture hid the random panel.
    """

    def test_ic_weighting_differs_from_equal_weighting(self):
        prices = _price_history()
        equal = _generator("equal_weight")
        equal.update_prices(prices)
        ic = _generator("ic_weighted")
        ic.update_prices(prices)

        assert equal.generate() != pytest.approx(ic.generate()), (
            "ic_weighted produced equal-weight output -- the IC path did not run"
        )


class TestSignalShape:
    def test_scores_are_bounded_and_cover_the_universe(self):
        prices = _price_history()
        gen = _generator("equal_weight")
        gen.update_prices(prices)

        scores = gen.generate()

        assert scores, "the generator produced no signal"
        assert set(scores) <= set(prices.columns)
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert min(scores.values()) == pytest.approx(0.0)
        assert max(scores.values()) == pytest.approx(1.0)

    def test_insufficient_history_yields_nothing(self):
        gen = _generator("equal_weight")
        gen.update_prices(_price_history(n_days=5))
        assert gen.generate() == {}


def test_default_live_factors_are_all_registered():
    from quant_platform.factors.registry import get_registry

    registry = get_registry()
    for name in DEFAULT_LIVE_FACTORS:
        assert registry.get(name) is not None, f"{name} is not in the registry"
