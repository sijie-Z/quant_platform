"""Regression tests for the turnover factor's input contract (BUG-42).

The pipeline cached a 5-tuple without `turnover` and set `turnover = None` on a
cache hit, whereupon `TurnoverFactor` fell back to `prices.rolling(period).mean()`
-- a price-level moving average, which is not turnover. The same config
therefore produced a different `turnover_20d` depending on whether the cache was
warm, breaking the deterministic-cache contract the pipeline advertises.
"""

import numpy as np
import pandas as pd
import pytest
from quant_platform.factors.technical import TurnoverFactor
from quant_platform.utils.cache import PipelineCache


@pytest.fixture
def panels():
    idx = pd.bdate_range("2024-01-01", periods=40)
    cols = ["A", "B"]
    prices = pd.DataFrame(100.0, index=idx, columns=cols)
    turnover = pd.DataFrame(0.03, index=idx, columns=cols)
    return prices, turnover


class TestTurnoverFactorContract:
    def test_computes_from_turnover_when_given(self, panels):
        prices, turnover = panels
        out = TurnoverFactor(period=5).compute(prices, turnover=turnover)
        assert out.shape == turnover.shape
        assert out.iloc[-1].tolist() == pytest.approx([0.03, 0.03])

    def test_refuses_rather_than_substituting_a_price_average(self, panels):
        """A price SMA under the name `turnover_20d` mislabels whatever the
        alpha pipeline does with it."""
        prices, _ = panels
        with pytest.raises(ValueError, match="requires turnover data"):
            TurnoverFactor(period=5).compute(prices)

    def test_the_price_fallback_would_have_produced_a_different_value(self, panels):
        """Pins why the substitution mattered: the two series are unrelated."""
        prices, turnover = panels
        factor = TurnoverFactor(period=5)
        from_turnover = factor.compute(prices, turnover=turnover)
        price_sma = prices.rolling(5).mean()

        assert not np.allclose(from_turnover.values, price_sma.values)


class TestDataCachePreservesTurnover:
    def test_turnover_survives_a_save_and_load(self, panels, tmp_path):
        prices, turnover = panels
        cache = PipelineCache(tmp_path)
        benchmark = prices.mean(axis=1)
        key = cache.make_config_hash({"provider": "synthetic"})

        cache.save_stage(
            "data_v2", key,
            (prices, prices, benchmark, {"sector": {}}, None, turnover),
        )
        loaded = cache.load_stage("data_v2", key)

        assert len(loaded) == 6, "the cached entry lost turnover again"
        assert loaded[5] is not None
        pd.testing.assert_frame_equal(loaded[5], turnover)
