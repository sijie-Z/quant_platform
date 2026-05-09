"""Tests for backtest.walkforward — Walk-forward validation."""

import numpy as np
import pandas as pd
import pytest
from quant_platform.backtest.walkforward import WalkForwardValidator


class TestWalkForwardValidator:
    def test_basic_structure(self):
        """Test that WalkForwardValidator can be instantiated."""
        validator = WalkForwardValidator(
            train_period=252, test_period=63, step_size=63, mode="rolling",
        )
        assert validator.train_period == 252
        assert validator.test_period == 63
        assert validator.mode == "rolling"

    def test_expanding_mode(self):
        validator = WalkForwardValidator(mode="expanding")
        assert validator.mode == "expanding"

    def test_insufficient_data_raises(self):
        short_signal = pd.DataFrame(np.random.randn(100, 10))
        short_prices = pd.DataFrame(np.random.randn(100, 10))
        short_returns = pd.DataFrame(np.random.randn(100, 10))
        benchmark = pd.Series(np.random.randn(100))
        sector_map = pd.Series(["A"] * 10)

        validator = WalkForwardValidator(train_period=504, test_period=126)
        with pytest.raises(ValueError, match="Not enough data"):
            validator.run(short_signal, short_prices, short_returns, benchmark, sector_map)
